"""행위자-비평가 — M3 정책망에 가치 머리를 붙인다.

가치 머리는 **정책 몸통을 공유하지 않는다**. 모방으로 이미 자리 잡은 정책을, 처음부터 배우는
가치 함수의 큰 기울기가 흔드는 것을 막기 위해서다(스펙 §6.3 "처음 몇 번은 가치 함수만 학습한다"와
같은 취지를 구조로도 지킨다).
"""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from vtd_rl.policy.encode import OBJ_DIM, VEC_DIM
from vtd_rl.policy.net import NEG_BIG, DrivePolicy, PolicyConfig


@dataclass(frozen=True)
class ActorCriticConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    value_hidden: tuple = (256, 256)
    obj_hidden: int = 64
    obj_out: int = 64


class ActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig = ActorCriticConfig()):
        super().__init__()
        self.cfg = cfg
        self.policy = DrivePolicy(cfg.policy)
        self.v_obj = nn.Sequential(nn.Linear(OBJ_DIM, cfg.obj_hidden), nn.Tanh(),
                                   nn.Linear(cfg.obj_hidden, cfg.obj_out), nn.Tanh())
        layers, last = [], VEC_DIM + cfg.obj_out
        for width in cfg.value_hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.v_trunk = nn.Sequential(*layers)
        self.v_head = nn.Linear(last, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def value(self, vec, objs, mask):
        h = self.v_obj(objs).masked_fill(mask[..., None] <= 0.0, NEG_BIG)
        pooled = h.max(dim=1).values
        pooled = torch.where(mask.sum(dim=1, keepdim=True) > 0.0, pooled, torch.zeros_like(pooled))
        return self.v_head(self.v_trunk(torch.cat([vec, pooled], dim=-1))).squeeze(-1)

    def act(self, vec, objs, mask, generator=None) -> dict:
        out = self.policy.sample(vec, objs, mask, generator=generator)
        out["value"] = self.value(vec, objs, mask)
        return out

    def evaluate_actions(self, vec, objs, mask, raw, turn, generator=None):
        log_prob, entropy = self.policy.evaluate_actions(vec, objs, mask, raw, turn,
                                                         generator=generator)
        return log_prob, entropy, self.value(vec, objs, mask)

    def clamp_log_std(self):
        self.policy.clamp_log_std()

    def save(self, path: str):
        torch.save({"cfg": {"policy": asdict(self.cfg.policy),
                            "value_hidden": list(self.cfg.value_hidden),
                            "obj_hidden": self.cfg.obj_hidden, "obj_out": self.cfg.obj_out},
                    "state": self.state_dict()}, path)

    @classmethod
    def _cfg_from(cls, blob) -> ActorCriticConfig:
        c = blob["cfg"]
        pol = PolicyConfig(**{**c["policy"], "trunk": tuple(c["policy"]["trunk"])})
        return ActorCriticConfig(policy=pol, value_hidden=tuple(c["value_hidden"]),
                                 obj_hidden=c["obj_hidden"], obj_out=c["obj_out"])

    @classmethod
    def load(cls, path: str, device=None) -> "ActorCritic":
        blob = torch.load(path, map_location=device or "cpu", weights_only=False)
        net = cls(cls._cfg_from(blob))
        net.load_state_dict(blob["state"])
        if device is not None:
            net.to(device)
        return net

    @classmethod
    def from_policy(cls, path: str, cfg: ActorCriticConfig | None = None, device=None) -> "ActorCritic":
        """M3 체크포인트(정책만)를 이식한다 — 가치 머리는 새로 시작한다."""
        blob = torch.load(path, map_location=device or "cpu", weights_only=False)
        pol_cfg = PolicyConfig(**{**blob["cfg"], "trunk": tuple(blob["cfg"]["trunk"])})
        base = cfg or ActorCriticConfig()
        net = cls(ActorCriticConfig(policy=pol_cfg, value_hidden=base.value_hidden,
                                    obj_hidden=base.obj_hidden, obj_out=base.obj_out))
        missing, unexpected = net.load_state_dict(
            {f"policy.{k}": v for k, v in blob["state"].items()}, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith(("v_obj", "v_trunk", "v_head")) for k in missing), missing
        # M3 체크포인트의 log_std[0] 은 -2.0024 로 하한 밖이다(그때는 forward 에서 잘랐다).
        # 이제 forward 가 자르지 않으므로 여기서 한 번 범위 안으로 넣는다.
        net.clamp_log_std()
        if device is not None:
            net.to(device)
        return net
