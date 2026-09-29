import numpy as np
import torch

from vtd_rl.env.action import TURNS, action_space
from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.encode import flatten_obs, stack_obs, to_tensors
from vtd_rl.policy.net import DrivePolicy, PolicyConfig


def sample_obs(seed=0):
    space = observation_space(ObsConfig())
    space.seed(seed)
    return space.sample()


def test_출력_크기():
    net = DrivePolicy()
    vec, objs, mask = to_tensors(*stack_obs([sample_obs(i) for i in range(5)]), torch.device("cpu"))
    mean, log_std, logits = net(vec, objs, mask)
    assert mean.shape == (5, 2) and logits.shape == (5, len(TURNS))
    assert log_std.shape == (2,)


def test_행동은_행동공간_안에_있다():
    net, space = DrivePolicy(), action_space()
    for seed in range(5):
        a = net.act(sample_obs(seed))
        assert space.contains(a), a
    g = torch.Generator().manual_seed(0)
    sampled = net.act(sample_obs(0), deterministic=False, generator=g)
    assert space.contains(sampled)


def test_마스크된_물체는_결과를_바꾸지_않는다():
    net = DrivePolicy()
    obs = sample_obs(1)
    obs["object_mask"][:] = 0.0
    obs["objects"][:] = 0.0
    a = net.act(obs)
    obs2 = dict(obs)
    obs2["objects"] = obs["objects"].copy()
    obs2["objects"][3] = 0.9                      # 마스크가 0 인 자리를 채워도
    b = net.act(obs2)
    assert np.allclose(a["control"], b["control"], atol=1e-6) and a["turn"] == b["turn"]


def test_물체_순서를_바꿔도_결과가_같다():
    net = DrivePolicy()
    obs = sample_obs(2)
    obs["object_mask"][:] = 0.0
    obs["object_mask"][:3] = 1.0
    a = net.act(obs)
    obs2 = dict(obs)
    obs2["objects"] = obs["objects"].copy()
    obs2["objects"][[0, 2]] = obs2["objects"][[2, 0]]
    b = net.act(obs2)
    assert np.allclose(a["control"], b["control"], atol=1e-5) and a["turn"] == b["turn"]


def test_저장하고_불러오면_같은_행동(tmp_path):
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    path = tmp_path / "policy.pt"
    net.save(str(path))
    other = DrivePolicy.load(str(path))
    obs = sample_obs(4)
    a, b = net.act(obs), other.act(obs)
    assert np.allclose(a["control"], b["control"], atol=1e-6) and a["turn"] == b["turn"]
    assert other.cfg.trunk == (32, 32)


def test_log_std_는_바닥_아래로_안_내려간다():
    # forward 는 더 이상 자르지 않는다(M4 준비 — 기울기 보존). 범위는 최적화 한 걸음 뒤에
    # clamp_log_std() 가 지킨다. tests/rl/test_policy_fixes.py 에 기울기 보존 자체를 검증하는
    # 테스트가 따로 있다.
    net = DrivePolicy(PolicyConfig(log_std_min=-2.0))
    with torch.no_grad():
        net.log_std.copy_(torch.tensor([-5.0, -5.0]))    # 바닥보다 훨씬 아래로 강제
    net.clamp_log_std()
    vec, objs, mask = to_tensors(*flatten_obs(sample_obs(0)), net.device)
    _, log_std, _ = net(vec, objs, mask)
    assert torch.allclose(log_std, torch.full((2,), -2.0))


def test_log_std_min_은_저장하고_불러와도_남는다(tmp_path):
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_min=-3.5))
    path = tmp_path / "policy.pt"
    net.save(str(path))
    other = DrivePolicy.load(str(path))
    assert other.cfg.log_std_min == -3.5


def test_set_log_std가_값을_덮어쓰고_범위에_넣는다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std(-1.5)
    assert torch.allclose(net.log_std, torch.tensor([-1.5, -1.5]))
    net.set_log_std(-99.0)                      # 하한 아래
    assert torch.allclose(net.log_std, torch.full((2,), net.cfg.log_std_min))
    net.set_log_std(99.0)                       # 상한 위
    assert torch.allclose(net.log_std, torch.full((2,), net.cfg.log_std_max))


def test_set_log_std는_기울기를_안_남긴다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std(-1.5)
    assert net.log_std.grad is None
    assert net.log_std.requires_grad          # 파라미터인 것은 그대로다


def test_set_log_std가_축별_값을_받는다():
    """★ M4e 를 무효로 만든 자리 — 두 축을 한 스칼라로 덮어쓰면 낮은 축의 σ 가 폭증한다.

    시작 시점 `log_std = [-2.000, -0.175]` 에서 두 축을 `.max()` 하나로 채워 **조향 σ 가
    6.2 배 뛰었다**(`docs/reports/m4e-ppo-notes.md`). 축별로 받아야 한다.
    """
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std([-1.5, -0.5])
    assert torch.allclose(net.log_std, torch.tensor([-1.5, -0.5]))
    net.set_log_std(torch.tensor([-0.7, -1.9]))
    assert torch.allclose(net.log_std, torch.tensor([-0.7, -1.9]))


def test_set_log_std_스칼라는_예전처럼_두_축을_같게():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std(-1.25)
    assert torch.allclose(net.log_std, torch.tensor([-1.25, -1.25]))


def test_set_log_std_축별도_범위에_눌린다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std([-99.0, 99.0])
    assert torch.allclose(net.log_std,
                          torch.tensor([net.cfg.log_std_min, net.cfg.log_std_max]))


def test_완전히_마스크된_물체는_마스크_보호장치를_핀한다():
    """마스크 보호장치(torch.where)가 필요함을 직접 검증.

    완전히 마스크된 관측에서:
    - 보호장치 있음: pooled = zeros (정답)
    - 보호장치 없음: pooled = NEG_BIG fill values (오답)

    이 테스트는 보호장치가 제거되면 즉시 실패한다.
    """
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))

    # 완전히 마스크된 관측 (모든 물체 마스크가 0, 객체 값은 임의)
    obs = sample_obs(6)
    obs["object_mask"][:] = 0.0
    obs["objects"][:] = 12345.0  # 큰 쓰레기값 사용

    # 네트워크의 실제 답변
    vec, objs, mask = to_tensors(*flatten_obs(obs), net.device)
    mean_actual, log_std_actual, logits_actual = net(vec, objs, mask)

    # 예상 답변: pooled = zeros 로 계산
    z_expected = net.trunk(torch.cat([vec, torch.zeros(1, net.cfg.obj_out, device=net.device)], dim=-1))
    mean_expected = net.mean(z_expected)
    logits_expected = net.turn(z_expected)

    # 보호장치가 있을 때만 일치한다
    assert torch.allclose(mean_actual, mean_expected, atol=1e-6)
    assert torch.allclose(logits_actual, logits_expected, atol=1e-6)
