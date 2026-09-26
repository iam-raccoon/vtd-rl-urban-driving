import numpy as np
import pytest
import torch

from vtd_rl.env.action import action_space
from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.encode import flatten_obs, to_tensors
from vtd_rl.policy.evaluate import EpisodeOutcome, _summary
from vtd_rl.policy.net import DrivePolicy, PolicyConfig, _tanh_log_det


def test_log_std_는_기울기를_잃지_않는다(obs_batch):
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    vec, objs, mask = obs_batch()
    _mean, log_std, _logits = net(vec, objs, mask)
    log_std.sum().backward()
    assert net.log_std.grad is not None and torch.all(net.log_std.grad != 0.0)


def test_한_걸음_뒤에_잘라_범위를_지킨다(obs_batch):
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_min=-2.0, log_std_max=0.5))
    with torch.no_grad():
        net.log_std.copy_(torch.tensor([-5.0, 3.0]))
    net.clamp_log_std()
    assert torch.allclose(net.log_std.detach(), torch.tensor([-2.0, 0.5]))
    # 바닥에 있어도 다음 기울기는 살아 있다
    _m, log_std, _l = net(*obs_batch())
    log_std.sum().backward()
    assert torch.all(net.log_std.grad != 0.0)


def test_표본은_자르기_전_로그확률을_준다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_max=2.0))
    with torch.no_grad():
        net.log_std.copy_(torch.tensor([1.5, 1.5]))     # σ≈4.5 — 표본이 확실히 잘린다
    vec, objs, mask = obs_batch(6)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(1))
    assert out["raw"].shape == (6, 2) and out["control"].shape == (6, 2)
    assert torch.all(out["control"] <= 1.0) and torch.all(out["control"] >= -1.0)
    assert out["turn"].shape == (6,) and out["log_prob"].shape == (6,)
    assert not torch.allclose(out["raw"], out["control"])      # 실제로 잘린 표본이 있다
    again, entropy = net.evaluate_actions(vec, objs, mask, out["raw"], out["turn"])
    assert torch.allclose(again, out["log_prob"], atol=1e-5)
    assert torch.all(entropy > 0.0)
    # 잘린 값이 아니라 원표본으로 계산한다
    clipped_lp, _ = net.evaluate_actions(vec, objs, mask, out["control"], out["turn"])
    assert not torch.allclose(clipped_lp, out["log_prob"], atol=1e-6)


def test_행동은_환경_공간_안이다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    space = observation_space(ObsConfig())
    space.seed(3)
    obs = space.sample()
    a = net.act(obs, deterministic=False, generator=torch.Generator().manual_seed(0))
    assert action_space().contains(a)


def test_완주하지_못한_판은_0점():
    eps = [EpisodeOutcome("A", 0, "goal", 100, 1.0, 98.0, []),
           EpisodeOutcome("B", 0, "stalled", 600, -5.0, 99.0, []),
           EpisodeOutcome("D", 0, "offroad", 90, -50.0, 100.0, [])]
    s = _summary(eps)
    assert s["goal_rate"] == 1 / 3
    assert s["mean_score"] == 98.0 / 3                       # 완주한 판만 점수를 남긴다
    assert s["mean_score_raw"] == (98.0 + 99.0 + 100.0) / 3


def test_스쿼시하면_행동이_항상_상자_안이다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True, log_std_max=2.0))
    with torch.no_grad():
        net.log_std.copy_(torch.tensor([1.5, 1.5]))     # σ≈4.5 — 안 자르면 상자를 크게 벗어난다
    vec, objs, mask = obs_batch(64)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(1))
    assert torch.all(out["control"].abs() <= 1.0)
    assert torch.any(out["raw"].abs() > 1.0)            # 스쿼시 전은 상자 밖으로 나간다
    assert torch.allclose(out["control"], torch.tanh(out["raw"]), atol=1e-6)


def test_스쿼시_로그확률에_야코비안_보정이_들어간다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    vec, objs, mask = obs_batch(8)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(2))
    normal = torch.distributions.Normal(net(vec, objs, mask)[0], net.log_std.exp())
    base = normal.log_prob(out["raw"]).sum(dim=-1)
    corr = torch.log1p(-torch.tanh(out["raw"]) ** 2 + 1e-12).sum(dim=-1)
    turn_lp = torch.log_softmax(net(vec, objs, mask)[2], dim=-1).gather(
        1, out["turn"][:, None]).squeeze(1)
    assert torch.allclose(out["log_prob"], base - corr + turn_lp, atol=1e-4)
    # 보정이 없으면 값이 다르다 — 이 단언이 보정 누락을 잡는다
    assert not torch.allclose(out["log_prob"], base + turn_lp, atol=1e-3)


def test_스쿼시에서도_evaluate_actions_가_일치한다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    vec, objs, mask = obs_batch(8)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(3))
    again, entropy = net.evaluate_actions(vec, objs, mask, out["raw"], out["turn"])
    assert torch.allclose(again, out["log_prob"], atol=1e-5)
    assert torch.all(torch.isfinite(entropy))


def test_스쿼시_결정적_행동도_상자_안이다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    space = observation_space(ObsConfig()); space.seed(5)
    a = net.act(space.sample(), deterministic=True)
    assert abs(a["control"][0]) <= 1.0 and abs(a["control"][1]) <= 1.0


def test_스쿼시_결정적_행동은_tanh_평균이다():
    """`|a| <= 1` 만 보면 뒤에 붙은 clamp 가 tanh 누락을 가려 준다 — 값으로 잠근다.

    채점 경로(`policy.evaluate.run_policy_episode`)가 `deterministic=True` 를 쓰므로,
    여기서 `tanh(mean)` 대신 `clamp(mean)` 이 나가면 점수가 통째로 달라진다.
    """
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    with torch.no_grad():
        net.mean.bias.copy_(torch.tensor([1.4, -1.1]))   # tanh 와 clamp 가 확실히 갈리는 구간
    space = observation_space(ObsConfig()); space.seed(5)
    obs = space.sample()
    a = net.act(obs, deterministic=True)
    vec, objs, mask = to_tensors(*flatten_obs(obs), net.device)
    with torch.no_grad():
        mean = net(vec, objs, mask)[0][0]
    assert np.allclose(a["control"], torch.tanh(mean).numpy(), atol=1e-6)
    assert not np.allclose(a["control"], mean.clamp(-1.0, 1.0).numpy(), atol=1e-3)


def _ref_dists(net, vec, objs, mask):
    mean, log_std, logits = net(vec, objs, mask)
    return (mean, log_std, torch.distributions.Normal(mean, log_std.exp()),
            torch.distributions.Categorical(logits=logits))


def test_엔트로피_기본_모드는_가우시안_닫힌형이다(obs_batch):
    """`entropy_mode="gaussian"` 은 스쿼시에서도 사전-스쿼시 Normal 의 닫힌형을 쓴다.

    값으로 잠근다 — 옛 표본 1 개 추정(`-log_prob`)으로 되돌리면 깨진다.
    """
    assert PolicyConfig().entropy_mode == "gaussian"
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    vec, objs, mask = obs_batch(8)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(3))
    _lp, ent = net.evaluate_actions(vec, objs, mask, out["raw"], out["turn"])
    _mean, _ls, normal, cat = _ref_dists(net, vec, objs, mask)
    closed = normal.entropy().sum(dim=-1) + cat.entropy()
    assert torch.allclose(ent, closed, atol=1e-6)
    assert torch.allclose(out["entropy"], closed, atol=1e-6)
    old = -(normal.log_prob(out["raw"]).sum(dim=-1) - _tanh_log_det(out["raw"])) + cat.entropy()
    assert not torch.allclose(ent, old, atol=1e-3)      # 옛 구현과는 다르다


def test_엔트로피_squashed_모드는_새_재매개화_표본으로_잰다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True, entropy_mode="squashed"))
    vec, objs, mask = obs_batch(8)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(3))
    _lp, ent = net.evaluate_actions(vec, objs, mask, out["raw"], out["turn"],
                                    generator=torch.Generator().manual_seed(11))
    mean, log_std, normal, cat = _ref_dists(net, vec, objs, mask)
    xi = torch.randn(mean.shape, generator=torch.Generator().manual_seed(11))
    u = mean + xi * log_std.exp()
    ref = -(normal.log_prob(u).sum(dim=-1) - _tanh_log_det(u)) + cat.entropy()
    assert torch.allclose(ent, ref, atol=1e-6)
    # 저장된 raw 를 다시 쓰지 않는다(재매개화 경로가 끊긴 옛 구현과 다르다)
    stale = -(normal.log_prob(out["raw"]).sum(dim=-1) - _tanh_log_det(out["raw"])) + cat.entropy()
    assert not torch.allclose(ent, stale, atol=1e-3)
    # 닫힌형과도 다르다 — 야코비안 기대값이 들어 있다
    assert not torch.allclose(ent, normal.entropy().sum(dim=-1) + cat.entropy(), atol=1e-3)


def test_gaussian_모드_엔트로피는_평균에_기울기를_안_준다(obs_batch):
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    vec, objs, mask = obs_batch(4)
    _lp, ent = net.evaluate_actions(vec, objs, mask, torch.zeros(4, 2), torch.zeros(4, dtype=torch.long))
    net.zero_grad()
    ent.sum().backward()
    assert net.mean.bias.grad is None or torch.allclose(net.mean.bias.grad,
                                                        torch.zeros(2), atol=1e-9)


def test_squashed_모드_엔트로피는_포화를_벌한다(obs_batch):
    """포화할수록 `log(1-tanh^2)` 가 크게 음수 → 엔트로피 최대화가 평균을 상자 안으로 민다."""
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True, entropy_mode="squashed"))
    with torch.no_grad():
        net.mean.bias.copy_(torch.tensor([3.0, -3.0]))
    vec, objs, mask = obs_batch(16)
    _lp, ent = net.evaluate_actions(vec, objs, mask, torch.zeros(16, 2),
                                    torch.zeros(16, dtype=torch.long),
                                    generator=torch.Generator().manual_seed(2))
    net.zero_grad()
    ent.sum().backward()
    g = net.mean.bias.grad
    assert g is not None and g[0] < 0.0 and g[1] > 0.0     # 절댓값을 줄이는 방향


def test_비스쿼시는_entropy_mode와_무관하다(obs_batch):
    vec, objs, mask = obs_batch(8)
    torch.manual_seed(0); a = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    torch.manual_seed(0); b = DrivePolicy(PolicyConfig(trunk=(32, 32), entropy_mode="squashed"))
    oa = a.sample(vec, objs, mask, generator=torch.Generator().manual_seed(5))
    ob = b.sample(vec, objs, mask, generator=torch.Generator().manual_seed(5))
    assert torch.equal(oa["entropy"], ob["entropy"])       # 비트 단위로 같다
    assert torch.equal(oa["raw"], ob["raw"]) and torch.equal(oa["log_prob"], ob["log_prob"])
    _la, ea = a.evaluate_actions(vec, objs, mask, oa["raw"], oa["turn"])
    _lb, eb = b.evaluate_actions(vec, objs, mask, ob["raw"], ob["turn"])
    assert torch.equal(ea, eb)


def test_모르는_entropy_mode는_거부한다():
    with pytest.raises(ValueError):
        DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True, entropy_mode="아무거나"))


def test_스쿼시_기본값은_꺼짐이고_끄면_예전과_같다(obs_batch):
    assert PolicyConfig().squash is False
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    vec, objs, mask = obs_batch(8)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(4))
    assert torch.allclose(out["control"], out["raw"].clamp(-1.0, 1.0), atol=1e-6)
