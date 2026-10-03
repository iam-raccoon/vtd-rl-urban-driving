import copy
import math

import numpy as np
import pytest
import torch

from vtd_rl.policy.dataset import DaggerDataset, Shard
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM
from vtd_rl.policy.train import TrainConfig, anchor_kl, make_reference
from vtd_rl.rl.buffer import RolloutBuffer
from vtd_rl.rl.ppo import (PPOConfig, anchor_loss, dagger_batches, imitation_coef, imitation_loss,
                           ppo_losses, update)


def filled_buffer(net, n_steps=4, n_envs=2):
    buf = RolloutBuffer(n_steps, n_envs, torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    for t in range(n_steps):
        vec = torch.randn(n_envs, VEC_DIM)
        objs = torch.randn(n_envs, OBJ_N, OBJ_DIM)
        mask = torch.ones(n_envs, OBJ_N)
        out = net.act(vec, objs, mask, generator=g)
        buf.add(vec=vec, objs=objs, mask=mask, raw=out["raw"], turn=out["turn"],
                log_prob=out["log_prob"], value=out["value"],
                reward=torch.randn(n_envs), done=torch.zeros(n_envs),
                valid=torch.ones(n_envs))
    buf.compute_gae(last_value=torch.zeros(n_envs))
    return buf


def toy_dagger(n=64):
    rng = np.random.default_rng(0)
    ds = DaggerDataset()
    ds.add(Shard(rng.standard_normal((n, VEC_DIM), dtype=np.float32),
                 np.zeros((n, OBJ_N, OBJ_DIM), np.float32), np.zeros((n, OBJ_N), np.float32),
                 rng.uniform(-1, 1, (n, 2)).astype(np.float32),
                 rng.integers(0, 3, n).astype(np.int64), {"board": "toy"}))
    return ds


class _StubNet:
    """검증용 가짜 net — `log_prob`·`entropy`·`value` 를 마음대로 통제해 손실 공식만 본다."""

    def __init__(self, log_prob=None, entropy=None, value=None, n=1):
        self.log_prob = log_prob if log_prob is not None else torch.zeros(n)
        self.entropy = entropy if entropy is not None else torch.zeros(n)
        self.value = value if value is not None else torch.zeros(n)

    def evaluate_actions(self, vec, objs, mask, raw, turn, generator=None):
        return self.log_prob, self.entropy, self.value


def _zero_batch(n, old_log_prob=None, adv=None, ret=None, old_value=None):
    """`ppo_losses` 가 `vec/objs/mask/raw`(모양만 필요) 를 무시하는 `_StubNet` 과 쓸 더미 배치."""
    z = torch.zeros(n, 1)
    turn = torch.zeros(n, dtype=torch.long)
    return (z, z, z, z, turn,
            old_log_prob if old_log_prob is not None else torch.zeros(n),
            adv if adv is not None else torch.zeros(n),
            ret if ret is not None else torch.zeros(n),
            old_value if old_value is not None else torch.zeros(n))


def test_처음_업데이트에서는_비율이_1이라_잘림이_없다(small_ac):
    net = small_ac()
    buf = filled_buffer(net)
    batch = next(iter(buf.batches(1000, generator=torch.Generator().manual_seed(0))))
    loss, parts = ppo_losses(net, batch, PPOConfig())
    assert abs(parts["approx_kl"]) < 1e-6 and parts["clip_frac"] == 0.0
    assert torch.isfinite(loss)


def test_모방_배치는_끝없이_돈다():
    it = dagger_batches(toy_dagger(8), 4, generator=torch.Generator().manual_seed(0))
    assert sum(1 for _ in zip(range(10), it)) == 10   # 두 배치짜리를 열 번 꺼내도 안 끊긴다


def test_모방_배치는_빈_데이터셋을_거부한다():
    empty = DaggerDataset()
    empty.add(Shard(np.zeros((0, VEC_DIM), np.float32), np.zeros((0, OBJ_N, OBJ_DIM), np.float32),
                     np.zeros((0, OBJ_N), np.float32), np.zeros((0, 2), np.float32),
                     np.zeros((0,), np.int64), {}))
    try:
        next(dagger_batches(empty, 4))
        assert False, "행이 없는 데이터셋은 즉시 거부해야 한다(안 그러면 while True 가 영원히 빈다)"
    except ValueError:
        pass


def test_모방_계수는_반감기로_준다():
    cfg = PPOConfig(imitation_coef0=1.0, imitation_half_life=100)
    assert imitation_coef(0, cfg) == 1.0
    assert abs(imitation_coef(100, cfg) - 0.5) < 1e-9
    assert abs(imitation_coef(200, cfg) - 0.25) < 1e-9


def test_정책_손실의_부호와_클리핑():
    """A=±1, r∈{0.7,1.3} 네 조합에서 dL/d(log_prob) 부호를 확인한다.

    비율이 클리핑 경계 밖으로 나가면서 '이미 충분히 움직인' 방향이면 기울기가 정확히 0 이어야
    한다 — 그게 PPO 가 막으려는 바로 그 두 칸이다(부호 반전·비율 역수·클리핑 제거·min↔max
    교체를 전부 잡아낸다).
    """
    cases = [(0.7, 1.0, -0.700), (0.7, -1.0, 0.0), (1.3, 1.0, 0.0), (1.3, -1.0, 1.300)]
    for r, adv_val, expected in cases:
        log_prob = torch.zeros(1, requires_grad=True)
        old_log_prob = torch.tensor([-math.log(r)])
        net = _StubNet(log_prob=log_prob)
        batch = _zero_batch(1, old_log_prob=old_log_prob, adv=torch.tensor([adv_val]))
        loss, _ = ppo_losses(net, batch, PPOConfig(value_coef=0.0, entropy_coef=0.0))
        loss.backward()
        assert abs(log_prob.grad.item() - expected) < 1e-3, (r, adv_val, log_prob.grad.item())


def test_가치_클리핑이_폭_밖에서_기울기를_막는다():
    cfg = PPOConfig(value_clip=0.2)

    def grad_at(value_val):
        value = torch.tensor([value_val], requires_grad=True)
        net = _StubNet(value=value)
        batch = _zero_batch(1, ret=torch.tensor([10.0]), old_value=torch.zeros(1))
        loss, _ = ppo_losses(net, batch, cfg)
        loss.backward()
        return value.grad.item()

    assert abs(grad_at(0.1)) > 1e-3        # 클리핑 폭(±0.2) 안 — 기울기가 살아 있다
    assert abs(grad_at(1.0)) < 1e-9        # 클리핑 폭 밖 — 기울기가 완전히 막힌다


def test_엔트로피_계수가_기울기_크기를_그대로_정한다():
    for coef in (0.0, 0.005, 0.5):
        for entropy_val in (3.0, -5.0):    # 엔트로피 값 자체와는 무관해야 한다
            entropy = torch.tensor([entropy_val], requires_grad=True)
            net = _StubNet(entropy=entropy)
            batch = _zero_batch(1)
            loss, _ = ppo_losses(net, batch, PPOConfig(entropy_coef=coef, value_coef=0.0))
            loss.backward()
            assert abs(entropy.grad.item() - (-coef)) < 1e-9


def test_approx_kl은_음수가_아니다():
    for logratio in torch.linspace(-2.0, 2.0, 21):
        old_log_prob = torch.tensor([-logratio.item()])
        net = _StubNet()   # log_prob 는 기본값 0 — old_log_prob 와의 차가 logratio
        batch = _zero_batch(1, old_log_prob=old_log_prob)
        _, parts = ppo_losses(net, batch, PPOConfig())
        assert parts["approx_kl"] >= -1e-9, (float(logratio), parts["approx_kl"])


def test_업데이트가_돌고_범위를_지킨다(small_ac):
    net = small_ac()
    with torch.no_grad():
        # 경계 밖(하한 -2.0 · 상한 0.5 보다 각각 0.5·0.4 벗어난 값)에서 시작한다 — 몇 걸음의
        # 작은 Adam 이동(~1e-3 대)으로는 절대 저절로 범위 안에 못 들어온다. clamp_log_std() 를
        # 지우면 이 단언은 반드시 깨져야 한다(아래 GREEN/RED 증거 참고).
        net.policy.log_std.copy_(torch.tensor([-2.5, 0.9]))
    buf = filled_buffer(net, n_steps=8, n_envs=4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    it = dagger_batches(toy_dagger(), 32, generator=torch.Generator().manual_seed(0))
    out = update(net, opt, buf, it, PPOConfig(epochs=2, minibatch=8), step=0,
                 generator=torch.Generator().manual_seed(0))
    assert set(out) >= {"policy", "value", "entropy", "approx_kl", "clip_frac", "imitation"}
    lo, hi = net.policy.cfg.log_std_min, net.policy.cfg.log_std_max
    assert torch.all(net.policy.log_std >= lo - 1e-6) and torch.all(net.policy.log_std <= hi + 1e-6)


def test_모방_손실이_실제로_loss에_합산된다(small_ac, monkeypatch):
    """`parts["imitation"]` 키가 있다는 것만으로는 부족하다 — 그 키는 미적용일 때도 0.0 으로
    항상 존재한다. 여기서는 `imitation_loss` 를 갈아 끼워 반환한 텐서가 실제로 역전파돼
    옵티마이저가 보는 파라미터에 기울기를 남기는지 직접 잰다.
    """
    net = small_ac()
    dummy = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD(list(net.parameters()) + [dummy], lr=0.0)   # lr=0: 이동 없이 grad 만 본다

    def fake_imitation_loss(_net, _batch, _cfg):
        return dummy * 3.0, {"total": 0.0, "control": 0.0, "turn": 0.0}

    monkeypatch.setattr("vtd_rl.rl.ppo.imitation_loss", fake_imitation_loss)

    buf = filled_buffer(net, n_steps=4, n_envs=2)
    it = dagger_batches(toy_dagger(), 8, generator=torch.Generator().manual_seed(0))
    update(net, opt, buf, it, PPOConfig(epochs=1, minibatch=1000), step=0,
           generator=torch.Generator().manual_seed(0))
    # step=0 이면 imitation_coef == imitation_coef0 == 1.0 이니 grad == 3.0 이어야 한다.
    assert dummy.grad is not None and abs(dummy.grad.item() - 3.0) < 1e-6


def test_가치_손실이_가치를_목표로_당긴다(small_ac):
    net = small_ac()
    buf = filled_buffer(net, n_steps=8, n_envs=4)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    before = None
    for k in range(5):
        batch = next(iter(buf.batches(1000, generator=torch.Generator().manual_seed(k))))
        loss, parts = ppo_losses(net, batch, PPOConfig(entropy_coef=0.0))
        before = parts["value"] if before is None else before
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    batch = next(iter(buf.batches(1000, generator=torch.Generator().manual_seed(0))))
    _loss, parts = ppo_losses(net, batch, PPOConfig(entropy_coef=0.0))
    assert parts["value"] < before


def test_모방_시그마_모드가_실제로_전달된다(small_ac):
    net = small_ac()
    batch = next(iter(toy_dagger(64).batches(32, generator=torch.Generator().manual_seed(0))))

    net.zero_grad()
    imitation_loss(net, batch, PPOConfig(imitation_sigma="learn"))[0].backward()
    assert torch.any(net.policy.log_std.grad != 0.0)

    net.zero_grad()
    imitation_loss(net, batch, PPOConfig(imitation_sigma="detach"))[0].backward()
    assert net.policy.log_std.grad is None or torch.all(net.policy.log_std.grad == 0.0)


def test_모방_시그마_기본값은_learn():
    assert PPOConfig().imitation_sigma == "learn"


def test_모방_시그마에_이상한_값을_주면_거부한다(small_ac):
    net = small_ac()
    batch = next(iter(toy_dagger(64).batches(32, generator=torch.Generator().manual_seed(0))))
    with pytest.raises(ValueError):
        imitation_loss(net, batch, PPOConfig(imitation_sigma="아무거나"))


def _squash_ac(squash=True, entropy_mode="gaussian"):
    from vtd_rl.policy.net import PolicyConfig
    from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig
    torch.manual_seed(0)
    cfg = ActorCriticConfig(policy=PolicyConfig(obj_hidden=16, obj_out=16, trunk=(32, 32),
                                                squash=squash, entropy_mode=entropy_mode),
                            value_hidden=(32, 32), obj_hidden=16, obj_out=16)
    return ActorCritic(cfg)


def test_모방손실은_정책의_스쿼시를_따라간다():
    """스쿼시 체크포인트를 `--init` 로 받아도 모방만 조용히 비스쿼시로 계산되면 안 된다.

    그러면 모방이 `mean -> control` 로 끌고 환경은 `tanh(mean)` 을 받아 어긋난다
    (실측 `|tanh(c)-c|` 평균 0.0428, 가속만 0.0813, 최대 0.2384).
    """
    from vtd_rl.policy.train import TrainConfig, policy_loss
    net = _squash_ac(squash=True)
    batch = next(iter(toy_dagger(64).batches(32, generator=torch.Generator().manual_seed(0))))
    _loss, parts = imitation_loss(net, batch, PPOConfig())
    _sref, sparts = policy_loss(net.policy, batch, TrainConfig(squash=True))
    _nref, nparts = policy_loss(net.policy, batch, TrainConfig(squash=False))
    assert parts["total"] == pytest.approx(sparts["total"], rel=1e-9)     # 스쿼시 손실과 등식
    assert parts["total"] != pytest.approx(nparts["total"], rel=1e-3)     # 비스쿼시와는 다르다
    # σ 모드와 함께 써도 유지된다
    _l2, p2 = imitation_loss(net, batch, PPOConfig(imitation_sigma="detach"))
    _r2, r2p = policy_loss(net.policy, batch, TrainConfig(squash=True, sigma_grad=False))
    assert p2["total"] == pytest.approx(r2p["total"], rel=1e-9)


def test_비스쿼시_정책의_모방손실은_예전과_같다():
    from vtd_rl.policy.train import TrainConfig, policy_loss
    net = _squash_ac(squash=False)
    batch = next(iter(toy_dagger(64).batches(32, generator=torch.Generator().manual_seed(0))))
    _loss, parts = imitation_loss(net, batch, PPOConfig())
    _ref, rparts = policy_loss(net.policy, batch, TrainConfig())
    assert parts["total"] == pytest.approx(rparts["total"], rel=1e-9)


def test_squashed_엔트로피는_generator를_받아_전역RNG와_무관하게_재현된다():
    """`entropy_mode="squashed"` 는 `_gauss_entropy()` 안에서 새 표본을 뽑는다 — `update()` 가

    거기까지 `generator` 를 안 넘기면 전역 RNG 를 타 같은 시드로도 결과가 달라진다(이번
    마일스톤에서 이 누락으로 실험 12 회를 통째로 버린 전례가 있다, `--init` 씨앗 문제와 같은
    종류). 두 실행을 완전히 같게(같은 초기 가중치·같은 버퍼·같은 미니배치 순서) 만든 뒤, 오직
    `update()` 호출 **직전** 전역 RNG 상태만 다르게 흔들어 본다 — `generator` 가 끝까지 먹히면
    전역 RNG 가 무엇이든 결과가 같아야 하고, 안 먹히면(옛 버그) 갈린다.
    """
    def run(perturb_global: bool):
        net = _squash_ac(squash=True, entropy_mode="squashed")
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        buf = filled_buffer(net, n_steps=8, n_envs=4)
        if perturb_global:
            torch.rand(37)   # 이 세션의 다른 코드가 전역 RNG 를 다르게 태웠다고 흉내
        it = dagger_batches(toy_dagger(), 32, generator=torch.Generator().manual_seed(0))
        update(net, opt, buf, it, PPOConfig(epochs=2, minibatch=8), step=0,
              generator=torch.Generator().manual_seed(123))
        return net.policy.log_std.detach().clone(), [p.detach().clone() for p in net.parameters()]

    ls_a, params_a = run(False)
    ls_b, params_b = run(True)
    assert torch.equal(ls_a, ls_b)
    assert len(params_a) == len(params_b)
    assert all(torch.equal(a, b) for a, b in zip(params_a, params_b))


def test_모방_계수에_바닥을_둘_수_있다():
    cfg = PPOConfig(imitation_half_life=1_000_000, imitation_floor=0.2)
    assert imitation_coef(0, cfg) == pytest.approx(1.0)
    assert imitation_coef(1_000_000, cfg) == pytest.approx(0.5)
    # 바닥 아래로 안 내려간다
    assert imitation_coef(10_000_000, cfg) == pytest.approx(0.2)
    assert imitation_coef(100_000_000, cfg) == pytest.approx(0.2)


def test_모방_바닥_기본값은_0이라_예전과_같다():
    cfg = PPOConfig(imitation_half_life=1_000_000)
    assert cfg.imitation_floor == 0.0
    assert imitation_coef(10_000_000, cfg) == pytest.approx(0.5 ** 10)


def _kl_to(ref, net, buf):
    """버퍼 상태 전체에서 `KL(ref ‖ net.policy)` — 테스트용 자."""
    flat = next(iter(buf.batches(10_000, generator=torch.Generator().manual_seed(0))))
    vec, objs, mask = flat[0], flat[1], flat[2]
    with torch.no_grad():
        return anchor_kl(net.policy(vec, objs, mask), ref(vec, objs, mask), TrainConfig()).item()


def test_앵커_계수_기본값은_0():
    assert PPOConfig().anchor_coef == 0.0


def test_앵커가_꺼져_있으면_ref를_줘도_예전과_비트단위로_같다(small_ac):
    torch.manual_seed(0)
    a = small_ac()
    b = copy.deepcopy(a)
    buf = filled_buffer(a, n_steps=8, n_envs=4)
    opt_a = torch.optim.Adam(a.parameters(), lr=1e-2)
    opt_b = torch.optim.Adam(b.parameters(), lr=1e-2)
    cfg = PPOConfig(epochs=2, minibatch=8)
    out_a = update(a, opt_a, buf, None, cfg, step=0, generator=torch.Generator().manual_seed(0))
    out_b = update(b, opt_b, buf, None, cfg, step=0, generator=torch.Generator().manual_seed(0),
                   ref=make_reference(b.policy))
    for (name, pa), (_n, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name
    assert out_a["anchor"] == 0.0 and out_b["anchor"] == 0.0


def test_앵커를_켰는데_ref가_없으면_거부한다(small_ac):
    net = small_ac()
    buf = filled_buffer(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="ref"):
        update(net, opt, buf, None, PPOConfig(anchor_coef=0.5), step=0,
               generator=torch.Generator().manual_seed(0))


def test_ref가_학생_정책과_같은_객체면_거부한다(small_ac):
    net = small_ac()
    buf = filled_buffer(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        update(net, opt, buf, None, PPOConfig(anchor_coef=0.5), step=0,
               generator=torch.Generator().manual_seed(0), ref=net.policy)


def test_앵커는_처음에_0이고_ref는_움직이지_않는다(small_ac):
    net = small_ac()
    ref = make_reference(net.policy)
    before = {k: v.clone() for k, v in ref.state_dict().items()}
    buf = filled_buffer(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    # 미니배치 하나·에폭 하나 — 첫(유일한) 미니배치는 학생 == 참조라 KL 이 0 이다.
    out = update(net, opt, buf, None, PPOConfig(anchor_coef=1.0, epochs=1, minibatch=1000), step=0,
                 generator=torch.Generator().manual_seed(0), ref=ref)
    assert abs(out["anchor"]) < 1e-6
    for k, v in ref.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert all(p.grad is None for p in ref.parameters())


def test_앵커_손실이_실제로_loss에_합산된다(small_ac, monkeypatch):
    net = small_ac()
    dummy = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD(list(net.parameters()) + [dummy], lr=0.0)   # lr=0: 이동 없이 grad 만 본다

    def fake_anchor_loss(_net, _batch, _ref):
        return dummy * 3.0

    monkeypatch.setattr("vtd_rl.rl.ppo.anchor_loss", fake_anchor_loss)
    buf = filled_buffer(net)
    update(net, opt, buf, None, PPOConfig(anchor_coef=0.5, epochs=1, minibatch=1000), step=0,
           generator=torch.Generator().manual_seed(0), ref=make_reference(net.policy))
    assert dummy.grad is not None and abs(dummy.grad.item() - 1.5) < 1e-6   # 3.0 × 계수 0.5


def test_앵커가_출발점_근처에_붙잡는다(small_ac):
    torch.manual_seed(0)
    start = small_ac()
    buf = filled_buffer(start, n_steps=16, n_envs=8)
    kls = {}
    for coef in (0.0, 10.0):
        net = copy.deepcopy(start)
        ref = make_reference(start.policy)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        cfg = PPOConfig(anchor_coef=coef, epochs=4, minibatch=32, target_kl=1e9)   # 조기 종료 없이
        for k in range(5):
            update(net, opt, buf, None, cfg, step=0, generator=torch.Generator().manual_seed(k),
                   ref=ref if coef > 0 else None)
        kls[coef] = _kl_to(ref, net, buf)
    assert kls[0.0] > 1e-3                 # 앵커가 없으면 실제로 멀어진다(공허한 테스트가 아님)
    assert kls[10.0] < 0.5 * kls[0.0]      # 앵커가 그 거리를 절반 아래로 묶는다


def test_anchor_loss는_anchor_kl과_같다(small_ac):
    net = small_ac()
    ref = make_reference(net.policy)
    with torch.no_grad():
        net.policy.mean.weight.add_(0.1)   # 학생만 조금 옮긴다
    buf = filled_buffer(net)
    batch = next(iter(buf.batches(1000, generator=torch.Generator().manual_seed(0))))
    got = anchor_loss(net, batch, ref).item()
    want = _kl_to(ref, net, buf)
    assert got > 0.0 and abs(got - want) < 1e-5
