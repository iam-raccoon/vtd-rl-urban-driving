"""PPO 손실 + 모방 손실(감쇠) — 스펙 §6.3.

모방 손실은 M3 라벨(선생님)에 대한 로그가능도다. 처음에는 강하게 잡아 두고 반감기로 줄인다 —
PPO 단계에서는 선생님을 다시 돌리지 않고 모아 둔 데이터만 쓴다(스펙 §6.3).
"""
import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from vtd_rl.policy.train import TrainConfig, policy_loss


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch: int = 1024
    gamma: float = 0.99
    lam: float = 0.95
    imitation_coef0: float = 1.0
    imitation_half_life: int = 2_000_000
    imitation_sigma: str = "learn"   # "learn" | "detach" — detach 면 모방이 σ 를 안 건드린다
    target_kl: float = 0.03
    # 가치 클리핑 폭은 정책 비율 클리핑(clip)과 별개다 — 이 환경의 리턴은 O(100)
    # (progress 100 · goal 50 · collision -50, 보상 정규화 없음, 스펙 §5)인데 비율용 0.2 를
    # 그대로 재사용하면 비평가 수렴이 대략 4 배 느려진다(갱신 50 회에 V=63.8 vs 무클립 100.0).
    value_clip: float = 10.0


# 모방 손실은 매 미니배치 M3 학습 설정을 그대로 쓴다 — 새로 만들 이유가 없어 모듈 상수로 뺀다.
_IMITATION_TRAIN_CFG = TrainConfig()
# M4b: 평균은 선생님에 묶어 두고 σ 만 푸는 짝 — sigma_grad 만 다르다.
_IMITATION_TRAIN_CFG_DETACH = dataclasses.replace(_IMITATION_TRAIN_CFG, sigma_grad=False)
_IMITATION_CFGS = {"learn": _IMITATION_TRAIN_CFG, "detach": _IMITATION_TRAIN_CFG_DETACH}


def imitation_train_cfg(net, cfg: PPOConfig) -> TrainConfig:
    """모방 손실 설정 — σ 모드는 `cfg` 에서, 스쿼시 여부는 **정책에서** 가져온다.

    `TrainConfig()` 의 `squash=False` 를 그대로 쓰면, 스쿼시 체크포인트를 `--init` 로 받아
    `ActorCritic.from_policy` 가 `squash=True` 를 제대로 실어도 **모방만 조용히 비스쿼시로**
    계산된다. 그러면 모방은 `mean -> control` 로 끄는데 환경은 `tanh(mean)` 을 받는다
    (실측 어긋남 `|tanh(c)-c|` 평균 0.0428, 가속만 0.0813, 최대 0.2384).
    """
    try:
        base = _IMITATION_CFGS[cfg.imitation_sigma]
    except KeyError:
        raise ValueError(f"imitation_sigma 는 {sorted(_IMITATION_CFGS)} 중 하나여야 한다:"
                         f" {cfg.imitation_sigma!r}")
    squash = bool(net.policy.cfg.squash)
    return base if base.squash == squash else dataclasses.replace(base, squash=squash)


def imitation_coef(step: int, cfg: PPOConfig) -> float:
    return cfg.imitation_coef0 * 0.5 ** (step / max(cfg.imitation_half_life, 1))


def ppo_losses(net, batch, cfg: PPOConfig):
    """잘라낸 정책 손실 + 잘라낸 가치 손실 + 엔트로피 보너스.

    비율은 저장해 둔 **원표본**(`raw`)으로 계산한다 — 자른 값(`control`)으로 계산하면
    분포가 안 맞는다(`DrivePolicy.sample()`의 계약, 스펙 §6.3).
    """
    vec, objs, mask, raw, turn, old_log_prob, adv, ret, old_value = batch
    log_prob, entropy, value = net.evaluate_actions(vec, objs, mask, raw, turn)
    ratio = (log_prob - old_log_prob).exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv
    policy = -torch.min(unclipped, clipped).mean()
    v_clipped = old_value + (value - old_value).clamp(-cfg.value_clip, cfg.value_clip)
    value_loss = 0.5 * torch.max((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
    ent = entropy.mean()
    loss = policy + cfg.value_coef * value_loss - cfg.entropy_coef * ent
    with torch.no_grad():
        approx_kl = float(((ratio - 1.0) - (log_prob - old_log_prob)).mean())
        clip_frac = float(((ratio - 1.0).abs() > cfg.clip).float().mean())
    return loss, {"policy": policy.item(), "value": value_loss.item(), "entropy": ent.item(),
                  "approx_kl": approx_kl, "clip_frac": clip_frac}


def imitation_loss(net, dagger_batch, cfg: PPOConfig):
    """M3 라벨에 대한 로그가능도 — `vtd_rl.policy.train.policy_loss` 를 그대로 쓴다.

    `net` 은 `ActorCritic`(가치 머리 포함)이지만, `policy_loss` 는 `DrivePolicy` 를
    직접 호출하므로(`net(vec, objs, mask) -> mean, log_std, logits`) 반드시 `net.policy` 를
    넘긴다. `cfg.imitation_sigma` 가 `"detach"` 면 σ 에는 기울기를 안 보낸다.
    스쿼시 여부는 정책 설정을 따라간다(`imitation_train_cfg`).
    """
    return policy_loss(net.policy, dagger_batch, imitation_train_cfg(net, cfg))


def dagger_batches(dataset, batch_size: int, generator=None, device=None):
    """M3 라벨을 끝없이 돌린다 — 롤아웃이 수백 번이라 한 바퀴로는 모자란다.

    조각이 0 개인(행이 없는) 데이터셋을 넘기면 `dataset.batches()` 가 아무것도 안 내놓아
    `while True` 안에서 영원히 빈 순회만 도니, 여기서 미리 막는다.
    """
    if len(dataset) == 0:
        raise ValueError("dagger_batches: 데이터셋에 행이 없다")
    while True:
        for batch in dataset.batches(batch_size, generator=generator, device=device):
            yield batch


def update(net, opt, buffer, dagger_iter, cfg: PPOConfig, step: int, generator=None) -> dict:
    """에폭을 반복하며 PPO 손실 + 모방 손실(감쇠)을 합쳐 한 걸음씩 최적화한다.

    최적화 한 걸음마다 `net.clamp_log_std()` 를 불러 표준편차 범위를 지킨다(M3 가 σ 붕괴로
    학생이 브레이크만 밟았던 실패를 되풀이하지 않는다). `approx_kl` 이 `target_kl` 을 넘으면
    그 에폭에서 바로 멈춘다.
    """
    coef = imitation_coef(step, cfg)
    sums, count = {}, 0
    for _epoch in range(cfg.epochs):
        stop = False
        for batch in buffer.batches(cfg.minibatch, generator=generator):
            loss, parts = ppo_losses(net, batch, cfg)
            parts["imitation"] = 0.0
            if dagger_iter is not None and coef > 1e-4:
                dbatch = next(dagger_iter)
                imi, iparts = imitation_loss(net, dbatch, cfg)
                loss = loss + coef * imi
                parts["imitation"] = iparts["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            net.clamp_log_std()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
            if parts["approx_kl"] > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: v / max(count, 1) for k, v in sums.items()}
    out["imitation_coef"] = coef
    out["updates"] = count
    return out
