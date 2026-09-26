"""행동 복제 학습 — 조향·가속은 가우시안 음의 로그가능도, 지시등은 교차엔트로피.

M4 의 PPO 가 같은 분포를 쓰므로 여기서도 평균제곱오차가 아니라 로그가능도로 배운다
(표준편차까지 배워 두면 PPO 시작점이 자연스럽다).
"""
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


def policy_loss(net, batch, cfg: TrainConfig):
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
    return total, {"control": float(control_loss.item()), "turn": float(turn_loss.item()),
                   "total": float(total.item())}


def train_epochs(net, dataset, cfg: TrainConfig = TrainConfig(), device=None, log=None) -> dict:
    device = device or net.device
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    t0, last = time.perf_counter(), {}
    for epoch in range(cfg.epochs):
        sums, batches = {"total": 0.0, "control": 0.0, "turn": 0.0}, 0
        for batch in dataset.batches(cfg.batch_size, generator=gen, device=device):
            loss, parts = policy_loss(net, batch, cfg)
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
                "turn": sums["turn"] / max(batches, 1)}
        if log is not None:
            log(last)
    net.eval()
    return {"epochs": cfg.epochs, "samples": len(dataset), "loss": last.get("loss", float("nan")),
            "control": last.get("control", float("nan")), "turn": last.get("turn", float("nan")),
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
