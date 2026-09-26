"""학생 정책망 — DeepSets 물체 인코더 + 몸통 + 두 머리(연속 조향·가속, 범주형 지시등).

물체는 공유 MLP 를 거쳐 **마스크를 씌운 뒤 최댓값으로** 합친다. 개수·순서가 달라져도 결과가 같아야
한다(스펙 §4.1). 머리 모양은 M4 의 PPO 가 그대로 쓴다 — 가우시안 평균·로그표준편차와 범주형 로짓.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from vtd_rl.env.action import TURNS
from vtd_rl.policy.encode import OBJ_DIM, VEC_DIM, flatten_obs, to_tensors

NEG_BIG = -1.0e9          # 마스크된 자리를 최댓값 합치기에서 제외하는 값
ENTROPY_MODES = ("gaussian", "squashed")


def _tanh_log_det(raw):
    """tanh 변환의 로그 야코비안 합 — `log(1 - tanh(u)^2)` 의 수치 안정형.

    항등식: log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u)).
    그대로 계산하면 |u| 가 조금만 커져도 1 - tanh(u)^2 이 0 으로 내려가 -inf 가 된다.
    """
    return (2.0 * (math.log(2.0) - raw - nn.functional.softplus(-2.0 * raw))).sum(dim=-1)


@dataclass(frozen=True)
class PolicyConfig:
    obj_hidden: int = 64
    obj_out: int = 64
    trunk: tuple = (256, 256)
    log_std_init: float = -1.0
    log_std_min: float = -2.0       # 조향 표준편차가 무너져 가속 머리를 굶기지 않도록(σ≥0.135, ~8배 차이로 제한)
    log_std_max: float = 0.5        # 탐색 폭이 무한정 커지지 않도록
    squash: bool = False            # True 면 tanh 로 행동을 상자 안에 가둔다(야코비안 보정 포함).
                                     # 기본값 False — 평균이 상자를 벗어나도 clamp 가 탐색을 먹던
                                     # 옛 동작을 그대로 유지해 M3·M4a·M4b 체크포인트가 그대로 읽힌다.
    entropy_mode: str = "gaussian"  # squash=True 일 때 엔트로피 보너스를 어떻게 잴지.
                                     #  "gaussian"  — 사전-스쿼시 Normal 의 닫힌형(탐색 보너스 대용).
                                     #                기울기 정확·분산 0·M4a/M4b 와 계수 의미 동일.
                                     #                단 포화를 벌하는 힘은 없다.
                                     #  "squashed"  — evaluate_actions 안에서 **새 재매개화 표본**으로
                                     #                -log p(tanh(u)) 를 추정한다. 야코비안이 살아 있어
                                     #                포화를 실제로 벌한다. 분산이 있고 계수 의미가 다르다.
                                     # squash=False 면 이 값과 무관하게 언제나 가우시안 닫힌형이다.

    def __post_init__(self):
        if self.entropy_mode not in ENTROPY_MODES:
            raise ValueError(f"entropy_mode 는 {sorted(ENTROPY_MODES)} 중 하나여야 한다:"
                             f" {self.entropy_mode!r}")


class DrivePolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        self.obj = nn.Sequential(nn.Linear(OBJ_DIM, cfg.obj_hidden), nn.Tanh(),
                                 nn.Linear(cfg.obj_hidden, cfg.obj_out), nn.Tanh())
        layers, last = [], VEC_DIM + cfg.obj_out
        for width in cfg.trunk:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(last, 2)
        self.turn = nn.Linear(last, len(TURNS))
        self.log_std = nn.Parameter(torch.full((2,), float(cfg.log_std_init)))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, vec, objs, mask):
        h = self.obj(objs)                                   # [B, N, obj_out]
        h = h.masked_fill(mask[..., None] <= 0.0, NEG_BIG)
        pooled = h.max(dim=1).values
        pooled = torch.where(mask.sum(dim=1, keepdim=True) > 0.0, pooled,
                             torch.zeros_like(pooled))        # 물체가 없으면 0
        z = self.trunk(torch.cat([vec, pooled], dim=-1))
        # 범위는 자르지 않는다 — 자르면 바닥·천장에서 기울기가 0 이 되어 탐색 폭이
        # 영원히 고정된다. 범위는 아래 clamp_log_std() 가 최적화 한 걸음 뒤에 지킨다.
        log_std = self.log_std
        return self.mean(z), log_std, self.turn(z)

    @torch.no_grad()
    def clamp_log_std(self):
        """최적화 한 걸음 뒤에 부른다 — 범위는 지키되 기울기는 살려 둔다."""
        self.log_std.clamp_(self.cfg.log_std_min, self.cfg.log_std_max)

    def _dists(self, vec, objs, mask):
        mean, log_std, logits = self(vec, objs, mask)
        normal = torch.distributions.Normal(mean, log_std.exp())
        cat = torch.distributions.Categorical(logits=logits)
        return normal, cat

    def _gauss_entropy(self, normal, generator=None):
        """조향·가속 머리의 엔트로피 보너스 — `cfg.entropy_mode` 를 따른다.

        스쿼시 분포에는 닫힌형 엔트로피가 없다. 그렇다고 **저장해 둔 표본**의 `-log_prob` 을
        쓰면 안 된다 — PPO 는 그 표본을 detach 해 버퍼에 담으므로 재매개화 경로가 끊기고,
        남는 기울기가 `∇θ H(π_old, π_θ)`, 즉 "방금 뽑은 행동에서 평균을 멀리 떼어놔라" 가
        된다(실측: 60 반복에 |mean| 0.188 → 2.584, 계수를 10 배로 키워도 그대로).
        그래서 두 갈래만 둔다 — 닫힌형(기울기 정확, 포화 무관심)이거나,
        **여기서 새로 뽑은 재매개화 표본**(야코비안이 살아 포화를 벌한다)이거나.
        """
        if not self.cfg.squash or self.cfg.entropy_mode == "gaussian":
            return normal.entropy().sum(dim=-1)
        # "squashed": u = mean + std*ξ 로 기울기가 mean·std 로 흐른다(ξ 는 여기서 새로 뽑는다)
        noise = torch.randn(normal.mean.shape, generator=generator).to(normal.mean.device)
        u = normal.mean + noise * normal.stddev
        return -(normal.log_prob(u).sum(dim=-1) - _tanh_log_det(u))

    def sample(self, vec, objs, mask, generator=None) -> dict:
        """PPO 용 표본 — **자르기 전** 원표본과 그 로그확률을 함께 준다.

        환경에는 자른 값을 넣지만, 로그확률·비율은 원표본으로 계산해야 분포가 일관된다.
        """
        normal, cat = self._dists(vec, objs, mask)
        noise = torch.randn(normal.mean.shape, generator=generator).to(normal.mean.device)
        raw = normal.mean + noise * normal.stddev
        turn = torch.multinomial(cat.probs.cpu(), 1, generator=generator).squeeze(-1).to(raw.device)
        if self.cfg.squash:
            control = torch.tanh(raw)
            gauss_lp = normal.log_prob(raw).sum(dim=-1) - _tanh_log_det(raw)
        else:
            control = raw.clamp(-1.0, 1.0)
            gauss_lp = normal.log_prob(raw).sum(dim=-1)
        gauss_ent = self._gauss_entropy(normal, generator=generator)
        log_prob = gauss_lp + cat.log_prob(turn)
        entropy = gauss_ent + cat.entropy()
        # `mean` = 정책의 결정적 의도(스쿼시 전) — 이미 계산돼 있던 `normal.mean` 을 그대로
        # 얹는다(추가 순전파 없음). M4c 의 `comfort_on_intent`·행동 상자 포화 진단이 쓴다
        # (`scripts/train_ppo.py`) — `squash=True` 면 호출부가 `tanh(mean)` 을 취한다.
        return {"raw": raw, "control": control, "turn": turn,
                "log_prob": log_prob, "entropy": entropy, "value": None, "mean": normal.mean}

    def evaluate_actions(self, vec, objs, mask, raw, turn, generator=None):
        """저장해 둔 원표본에 대한 현재 정책의 로그확률(PPO 비율 계산용).

        엔트로피는 `raw` 를 쓰지 않는다 — `raw` 는 버퍼에서 온 detach 된 상수라
        `-log p(raw)` 로 재면 기울기가 "평균을 그 표본에서 떼어놔라" 가 된다.
        `_gauss_entropy()` 의 설명을 함께 볼 것(쌍둥이가 `sample()` 에 있다).
        """
        normal, cat = self._dists(vec, objs, mask)
        if self.cfg.squash:
            gauss_lp = normal.log_prob(raw).sum(dim=-1) - _tanh_log_det(raw)
        else:
            gauss_lp = normal.log_prob(raw).sum(dim=-1)
        gauss_ent = self._gauss_entropy(normal, generator=generator)
        log_prob = gauss_lp + cat.log_prob(turn)
        entropy = gauss_ent + cat.entropy()
        return log_prob, entropy

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = True, generator=None) -> dict:
        vec, objs, mask = to_tensors(*flatten_obs(obs), self.device)
        if deterministic:
            mean, _log_std, logits = self(vec, objs, mask)
            control = torch.tanh(mean[0]) if self.cfg.squash else mean[0]
            turn = int(torch.argmax(logits[0]).item())
        else:
            # 표본 추출은 sample() 을 그대로 쓴다 — 내부에서 CPU 생성기를 CUDA 텐서에
            # 바로 쓰지 않도록 처리한다
            out = self.sample(vec, objs, mask, generator=generator)
            control = out["control"][0]
            turn = int(out["turn"][0].item())
        control = control.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        return {"control": control, "turn": turn}

    def save(self, path: str):
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device=None) -> "DrivePolicy":
        blob = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = PolicyConfig(**{**blob["cfg"], "trunk": tuple(blob["cfg"]["trunk"])})
        net = cls(cfg)
        net.load_state_dict(blob["state"])
        if device is not None:
            net.to(device)
        net.eval()
        return net
