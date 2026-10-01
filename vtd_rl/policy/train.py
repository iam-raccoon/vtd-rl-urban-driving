"""행동 복제 학습 — 조향·가속은 가우시안 음의 로그가능도, 지시등은 교차엔트로피.

M4 의 PPO 가 같은 분포를 쓰므로 여기서도 평균제곱오차가 아니라 로그가능도로 배운다
(표준편차까지 배워 두면 PPO 시작점이 자연스럽다).
"""
import copy
import dataclasses
import math
import time
from dataclasses import dataclass

import torch
from torch import nn

from vtd_rl.policy.net import _tanh_log_det

LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 8
    turn_weight: float = 0.5        # 지시등은 대부분 '끔' 이라 가중치를 낮춘다
    grad_clip: float = 1.0
    seed: int = 0
    sigma_grad: bool = True         # False 면 NLL 을 log_std.detach() 로 계산한다(M4b: σ 만 푼다)
    squash: bool = False            # True 면 선생님 행동을 atanh 로 옮겨 스쿼시 밀도로 배운다
                                     # (net.py 의 squash=True 정책과 짝을 맞춘다). 기본값 False.
    atanh_eps: float = 1e-2          # squash=True 일 때 atanh 전에 선생님 행동을 자르는 여유.
                                     # M4c Task 5 리뷰 실측: 실데이터(200,511 행) 라벨의 10.48%가
                                     # 정확히 ±1 이다(가속 20.77%, 조향 0.19%). 1e-6 이면
                                     # atanh(±(1-1e-6))=±7.2477 이라 이 표본들이 NLL 을 지배하고,
                                     # 실데이터 BC 400 스텝에서 기울기 노름 중앙값이 1.59(비스쿼시
                                     # 기준)→54.72(34배)로 뛰어 100%가 grad_clip=1.0 에 걸린다.
                                     # ε∈{1e-2,1e-3,1e-4,1e-6}를 같은 시드·배치 순서로 실측한 결과
                                     # 행동공간 MAE(|tanh(mean)-control| 평균)가 가장 낮은 값은
                                     # 1e-2(0.1802) 였다(1e-3 0.1978, 1e-4 0.2175, 1e-6 0.2483 —
                                     # 손실 스케일 자체는 ε 마다 달라 비교 기준이 아니다). 전체 표는
                                     # `.superpowers/sdd/2026-09-23-m4c-action-box-and-reward/
                                     # task-5-report.md` 에 있다. squash=False 면 이 값을 안 쓴다.
    anchor_coef: float = 0.0         # >0 이면 손실에 **참조 정책과의 KL**(증류)을 더한다 —
                                     # "알던 것을 유지하라". 0 이면 앵커 계산 자체를 안 하므로
                                     # 기존 호출부·과거 성적표와 **비트 단위로 같다**(M4m).
    anchor_lr: float | None = None   # 주면 `lr` 대신 쓴다(웜스타트용 낮은 학습률).
                                     # `None` 이 "안 줬다" 다 — **0.0 은 진짜 0** 이라
                                     # `or` 가 아니라 `is None` 으로 가른다(effective_lr).

    def effective_lr(self) -> float:
        """실제로 Adam 에 넘길 학습률. `anchor_lr` 이 있으면 그것, 없으면 `lr`.

        M4l: `lr = 3e-4` 는 **처음부터 배우는 값**이다(8 에폭이면 Δθ = 11.0 으로 ‖θ‖ 의 44%
        를 움직인다). 웜스타트에는 과하지만 그 자체가 BC 손상의 원인은 아니다 — 단계 ③ 은
        `lr 3e-6` 에서도 완주율 0% 다. 그래서 기본값은 안 건드리고 **따로 주는 길**만 연다.
        """
        return self.lr if self.anchor_lr is None else self.anchor_lr


def make_reference(net):
    """앵커가 묶을 **고정된** 참조 정책 — 학습 시작 시점 정책의 얼린 사본.

    `deepcopy` 가 핵심이다. 얕은 복사나 같은 객체를 넘기면 참조가 학생과 같은 텐서를 보고
    KL 이 **항상 0** 이 된다 — 손실도 성적표도 멀쩡해 보이는데 앵커만 조용히 꺼진다.
    `policy_loss` 가 그 경우를 거부하지만(`_check_reference`), 만드는 길도 하나로 둔다.
    """
    ref = copy.deepcopy(net)
    ref.eval()
    ref.requires_grad_(False)
    return ref


def _check_reference(net, ref):
    """참조가 학생과 파라미터를 공유하면 앵커가 아무것도 안 묶는다 — 거부한다."""
    if ref is net:
        raise ValueError("참조가 학생과 같은 객체다 — KL 이 항상 0 이라 앵커가 꺼진 것과 같다."
                         " `make_reference(net)` 로 얼린 사본을 만들어 넘겨라.")
    net_params = getattr(net, "parameters", None)
    ref_params = getattr(ref, "parameters", None)
    if net_params is None or ref_params is None:
        return
    if {id(p) for p in net_params()} & {id(p) for p in ref_params()}:
        raise ValueError("참조가 학생과 같은 파라미터 텐서를 본다(얕은 복사?) — KL 이 항상 0 이다."
                         " `make_reference(net)` 를 써라.")


def anchor_kl(net_out, ref_out, cfg: TrainConfig):
    """`KL(ref ‖ net)` — 연속 머리는 두 가우시안의 해석적 KL, 지시등은 로짓 분포의 KL.

    **방향**은 참조에서 학생으로(순방향 KL, 질량 덮기)다. 참조가 질량을 둔 곳에 학생이 질량을
    안 두면 벌을 받는다 — 망각이 바로 그 모양이다. 반대 방향(`KL(net ‖ ref)`)은 모드 추종이라
    참조의 모드를 **버려도** 벌이 없어 우리가 막으려는 것을 못 막는다. 범주형에서 순방향 KL 은
    참조 확률을 부드러운 라벨로 쓰는 **증류**와 같은 식이다(참조 엔트로피 상수 차이).

    **스쿼시**: 두 정책이 **같은** tanh 를 통과하므로 여기서 재는 사전-스쿼시 KL 이 곧
    행동공간 KL 이다. KL 은 가역변환에 불변이다 — 두 밀도에 같은 `|det J|` 가 곱해져 로그 안의
    비에서 지워진다. 참조는 학생의 사본이라 변환이 반드시 같으므로 상쇄가 보장된다. 그래서
    야코비안을 안 더한다. 말로만 두지 않는다 —
    `tests/policy/test_train.py::test_앵커는_스쿼시_전에서_재도_같다_야코비안이_상쇄된다` 가
    `TransformedDistribution` 으로 뽑은 행동공간 KL 과 같음을 몬테카를로로 확인한다.
    `cfg.squash` 는 **데이터 항**의 설정이지 정책의 변환이 아니므로 여기서는 안 쓴다.

    **`turn_weight`**: 데이터 항이 `control + turn_weight * turn` 인 것과 **같은 균형**으로
    섞는다. 다른 균형을 쓰면 `anchor_coef` 하나가 두 머리에 서로 다른 세기로 걸려,
    "데이터가 미는 만큼의 몇 배로 붙잡는가" 라는 계수의 뜻이 머리마다 달라진다.

    **σ**: `cfg.sigma_grad` 는 **데이터 항**에서 σ 를 떼는 장치다(M4a). 앵커는 분포 전체를
    묶는 것이 목적이라 σ 를 떼지 않는다 — 순방향 KL 은 `σ_net` 이 0 으로 무너지면 발산하므로
    이 프로젝트가 겪은 log_std 붕괴를 덤으로 막는다.
    """
    mean, log_std, logits = net_out
    ref_mean, ref_log_std, ref_logits = ref_out
    kl_c = torch.distributions.kl_divergence(
        torch.distributions.Normal(ref_mean, ref_log_std.exp()),
        torch.distributions.Normal(mean, log_std.exp())).sum(dim=-1).mean()
    kl_t = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=ref_logits),
        torch.distributions.Categorical(logits=logits)).mean()
    return kl_c + cfg.turn_weight * kl_t


def policy_loss(net, batch, cfg: TrainConfig, ref=None):
    vec, objs, mask, control, turn = batch
    mean, log_std, logits = net(vec, objs, mask)
    # σ 를 떼면 평균·지시등에는 기울기가 그대로 흐르고 log_std 에만 안 흐른다. M4a 실측에서
    # 모방 손실이 log_std[0] 을 하한에 붙박아(기울기 +0.9392) 조향 탐색을 없앴는데, 모방을
    # 통째로 줄이면(run3) 평균까지 풀려 완주율이 0% 로 무너졌다 — 그래서 σ 만 뗀다.
    nll_log_std = log_std.detach() if not cfg.sigma_grad else log_std
    var = (2.0 * nll_log_std).exp()
    if cfg.squash:
        # 선생님 행동은 상자 안 값이다. 스쿼시 매개화에서는 그 행동의 로그가능도가
        # 가우시안 밀도(atanh 지점) 빼기 야코비안이다 — atanh 은 ±1 에서 발산하므로 잘라 준다.
        # 자르는 폭은 cfg.atanh_eps 다(실데이터 라벨의 10.48%가 정확히 ±1 이라 이 값이
        # 손실·기울기를 지배한다 — TrainConfig.atanh_eps 문서 참고).
        target = torch.atanh(control.clamp(-1.0 + cfg.atanh_eps, 1.0 - cfg.atanh_eps))
        jac = _tanh_log_det(target)          # 음수 — NLL 에 더한다
    else:
        target, jac = control, 0.0
    nll = 0.5 * (((target - mean) ** 2) / var + 2.0 * nll_log_std + LOG_2PI)
    control_loss = (nll.sum(dim=-1) + jac).mean()
    turn_loss = nn.functional.cross_entropy(logits, turn)
    total = control_loss + cfg.turn_weight * turn_loss
    # 앵커 — `ref` 가 있고 계수가 켜져 있을 때만 계산한다. 계수가 0 이면 아래 블록을 통째로
    # 건너뛰므로 `total` 이 예전과 **같은 텐서 그래프**다(비트 단위 동일, M4m Global Constraints).
    anchor = 0.0
    if ref is not None and cfg.anchor_coef > 0.0:
        _check_reference(net, ref)
        with torch.no_grad():        # 참조는 고정이다 — 평균·지시등 쪽 그래프를 아예 안 만든다
            ref_mean, ref_log_std, ref_logits = ref(vec, objs, mask)
        # ★ `no_grad` 는 **새 연산**만 막는다. `DrivePolicy.forward` 는 `log_std` 를 계산하지
        # 않고 Parameter 를 **그대로** 돌려주므로(net.py), 그 잎 텐서는 블록 안에서도
        # `requires_grad=True` 인 채로 나온다 — 실측: 안 떼면 `ref.log_std.grad` 에 기울기가
        # 쌓인다(옵티마이저가 참조를 안 보니 참조는 안 움직이고, 그래서 '참조가 안 움직였다'
        # 만 보는 테스트로는 안 잡힌다). 두 길을 다 막아야 참조가 진짜로 고정이다.
        anchor_t = anchor_kl((mean, log_std, logits),
                             (ref_mean, ref_log_std.detach(), ref_logits), cfg)
        total = total + cfg.anchor_coef * anchor_t
        anchor = float(anchor_t.item())
    return total, {"control": float(control_loss.item()), "turn": float(turn_loss.item()),
                   # `anchor` 는 **계수를 안 곱한** KL 이다 — 계수를 바꿔 가며 비교할 때
                   # 같은 자로 재야 한다(성적표가 이 값을 설정끼리 나란히 놓는다).
                   "anchor": anchor, "total": float(total.item())}


def train_epochs(net, dataset, cfg: TrainConfig = TrainConfig(), device=None, log=None,
                 ref=None) -> dict:
    """`ref` 를 주면 배치마다 그대로 넘긴다 — 앵커는 `cfg.anchor_coef > 0` 일 때만 켜진다.

    참조는 옵티마이저에 **안 들어간다**(`net.parameters()` 만 넘긴다) — 참조가 같이 학습되면
    앵커가 아무것도 안 묶는다.
    """
    device = device or net.device
    net.to(device).train()
    if ref is not None:
        ref.to(device)               # 학생과 같은 장치여야 한다(참조는 train() 으로 안 돌린다)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.effective_lr())
    gen = torch.Generator().manual_seed(cfg.seed)
    t0, last = time.perf_counter(), {}
    for epoch in range(cfg.epochs):
        sums, batches = {"total": 0.0, "control": 0.0, "turn": 0.0, "anchor": 0.0}, 0
        for batch in dataset.batches(cfg.batch_size, generator=gen, device=device):
            loss, parts = policy_loss(net, batch, cfg, ref=ref)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            net.clamp_log_std()      # forward 는 안 자르므로 최적화 한 걸음 뒤 여기서 지킨다
            for k in sums:
                sums[k] += parts[k]
            batches += 1
        last = {"epoch": epoch, "loss": sums["total"] / max(batches, 1),
                "control": sums["control"] / max(batches, 1),
                "turn": sums["turn"] / max(batches, 1),
                "anchor": sums["anchor"] / max(batches, 1)}
        if log is not None:
            log(last)
    net.eval()
    return {"epochs": cfg.epochs, "samples": len(dataset), "loss": last.get("loss", float("nan")),
            "control": last.get("control", float("nan")), "turn": last.get("turn", float("nan")),
            "anchor": last.get("anchor", float("nan")), "lr": cfg.effective_lr(),
            "seconds": time.perf_counter() - t0}


def squash_aligned(cfg: TrainConfig, net) -> TrainConfig:
    """손실 설정의 `squash` 를 정책 설정에 맞춘다 — 조용한 어긋남을 막는다."""
    want = bool(getattr(net, "cfg", None) is not None and net.cfg.squash)
    return cfg if cfg.squash == want else dataclasses.replace(cfg, squash=want)


@torch.no_grad()
def evaluate_labels(net, dataset, device=None, cfg: TrainConfig = TrainConfig()) -> dict:
    device = device or net.device
    net.to(device).eval()
    vec, objs, mask, control, turn = next(iter(dataset.batches(len(dataset), device=device)))
    mean, log_std, logits = net(vec, objs, mask)
    cfg = squash_aligned(cfg, net)
    loss, parts = policy_loss(net, (vec, objs, mask, control, turn), cfg)
    # 선생님 라벨은 **행동공간** 값이다. 스쿼시 정책이 실제로 내보내는 행동은 tanh(mean) 이므로
    # 사전-스쿼시 평균과 비교하면 MAE 보고가 틀린다.
    pred = torch.tanh(mean) if cfg.squash else mean
    return {"loss": parts["total"], "control": parts["control"], "turn": parts["turn"],
            "control_mae": float((pred - control).abs().mean().item()),
            "turn_acc": float((logits.argmax(dim=-1) == turn).float().mean().item())}
