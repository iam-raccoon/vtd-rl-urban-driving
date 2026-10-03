import copy

import numpy as np
import pytest
import torch

from vtd_rl.policy.dataset import DaggerDataset, Shard
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM
from vtd_rl.policy.net import DrivePolicy, PolicyConfig, _tanh_log_det
from vtd_rl.policy.train import (TrainConfig, evaluate_labels, make_reference, near_object_rows,
                                 policy_loss, train_epochs)


def toy_dataset(n=512, seed=0):
    """정답이 관측에서 결정되는 장난감 자료 — 학습이 실제로 되는지만 본다."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((n, VEC_DIM)).astype(np.float32)
    control = np.stack([np.tanh(vec[:, 0]), np.tanh(vec[:, 1])], axis=1).astype(np.float32)
    turn = (vec[:, 2] > 0).astype(np.int64) + (vec[:, 3] > 0.8).astype(np.int64)
    ds = DaggerDataset()
    ds.add(Shard(vec, np.zeros((n, OBJ_N, OBJ_DIM), np.float32), np.zeros((n, OBJ_N), np.float32),
                 control, turn, {"board": "toy"}))
    return ds


def test_학습하면_손실이_줄고_정확도가_오른다():
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(64, 64)))
    ds = toy_dataset()
    before = evaluate_labels(net, ds)
    # 운영 기본값(lr=3e-4, grad_clip=1.0)은 그대로 쓴다 — 12 에폭(48 스텝)으로는 이 씨앗에서
    # 기준 미달로 실패함을 확인했다(after_mae 0.425 vs 요구 0.341). 실제로 배우는지 보려면
    # 스텝 수가 더 필요해 에폭만 80(320 스텝)으로 늘렸다. 결과는 결정적이라 매번 같다.
    out = train_epochs(net, ds, TrainConfig(epochs=80, batch_size=128))
    after = evaluate_labels(net, ds)
    assert after["control_mae"] < before["control_mae"] * 0.6
    assert after["turn_acc"] > max(0.6, before["turn_acc"])
    assert out["epochs"] == 80 and out["samples"] == len(ds)
    assert np.isfinite(out["loss"])


def test_에폭마다_기록한다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    seen = []
    train_epochs(net, toy_dataset(128), TrainConfig(epochs=3, batch_size=64), log=seen.append)
    assert len(seen) == 3 and {"epoch", "loss", "control", "turn"} <= set(seen[0])
    assert [s["epoch"] for s in seen] == [0, 1, 2]


def test_손실은_두_머리의_합이다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    ds = toy_dataset(64)
    batch = next(iter(ds.batches(64, generator=torch.Generator().manual_seed(0))))
    from vtd_rl.policy.train import policy_loss
    loss, parts = policy_loss(net, batch, TrainConfig())
    assert torch.isfinite(loss)
    assert abs(parts["total"] - (parts["control"] + 0.5 * parts["turn"])) < 1e-5


def test_학습_뒤에도_log_std_범위를_지킨다():
    """forward() 는 더 이상 자르지 않으므로, train_epochs 가 매 스텝 뒤 clamp_log_std() 를
    불러야 σ 가 범위를 벗어나지 않는다. 안 그러면 M3 를 망가뜨렸던 σ_steer 붕괴가
    scripts/run_dagger.py 재실행 시 안전장치 없이 재발한다(코드 리뷰 지적)."""
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_min=-2.0, log_std_max=0.5))
    with torch.no_grad():
        net.log_std.copy_(torch.tensor([-5.0, 5.0]))     # 바닥 아래·천장 위로 동시에 강제
    train_epochs(net, toy_dataset(64), TrainConfig(epochs=2, batch_size=32))
    assert torch.all(net.log_std >= net.cfg.log_std_min - 1e-6)
    assert torch.all(net.log_std <= net.cfg.log_std_max + 1e-6)


def test_시그마를_분리하면_log_std에_기울기가_안_간다():
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    batch = next(iter(toy_dataset(64).batches(32, generator=torch.Generator().manual_seed(0))))

    net.zero_grad()
    policy_loss(net, batch, TrainConfig(sigma_grad=True))[0].backward()
    assert net.log_std.grad is not None and torch.any(net.log_std.grad != 0.0)
    learn_mean_grad = net.mean.weight.grad.clone()

    net.zero_grad()
    policy_loss(net, batch, TrainConfig(sigma_grad=False))[0].backward()
    assert net.log_std.grad is None or torch.all(net.log_std.grad == 0.0)
    # 평균 쪽 기울기는 살아 있어야 한다 — σ 만 떼는 것이지 모방을 끄는 게 아니다
    assert torch.any(net.mean.weight.grad != 0.0)
    # 그리고 σ 를 상수로 본 만큼 평균 기울기의 방향이 달라지지는 않는다(같은 var 로 나눈다)
    assert torch.allclose(net.mean.weight.grad, learn_mean_grad, atol=1e-6)


def test_시그마_분리_기본값은_학습이다():
    assert TrainConfig().sigma_grad is True


def test_스쿼시_모방손실은_선생님_행동을_atanh_로_옮긴다():
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    batch = next(iter(toy_dataset(64).batches(32, generator=torch.Generator().manual_seed(0))))
    loss, parts = policy_loss(net, batch, TrainConfig(squash=True))
    assert torch.isfinite(loss) and parts["total"] == parts["total"]   # NaN 아님
    # 스쿼시 손실을 줄이면 tanh(mean) 이 선생님 행동에 가까워진다
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(50):
        l, _p = policy_loss(net, batch, TrainConfig(squash=True))
        opt.zero_grad(set_to_none=True); l.backward(); opt.step(); net.clamp_log_std()
    vec, objs, mask, control, _turn = batch
    with torch.no_grad():
        err = (torch.tanh(net(vec, objs, mask)[0]) - control).abs().mean()
    assert float(err) < 0.3


class _FixedNet:
    """`policy_loss` 가 보는 최소 인터페이스만 흉내낸다 — 평균을 직접 쥐고 흔들기 위해."""

    def __init__(self, mean, log_std, logits):
        self.mean, self.log_std, self.logits = mean, log_std, logits

    def __call__(self, vec, objs, mask):
        return self.mean, self.log_std, self.logits


def _edge_batch():
    """±1 가까운 라벨을 섞은 작은 배치 — atanh 유무로 최소점이 크게 갈린다."""
    control = torch.tensor([[0.99, -0.99], [-0.98, 0.95], [0.5, -0.3], [0.0, 0.9]])
    n = control.shape[0]
    return (torch.zeros(n, VEC_DIM), torch.zeros(n, OBJ_N, OBJ_DIM), torch.zeros(n, OBJ_N),
            control, torch.zeros(n, dtype=torch.long))


def test_스쿼시_손실의_최소점은_atanh_라벨이다():
    """atanh 를 빼면 최소점이 `mean=control` 로 옮겨간다 — 기울기 0 지점으로 잠근다.

    `atanh_eps` 를 명시적으로 1e-6 으로 고정한다 — 이 테스트는 "거의 안 잘리는 atanh" 라는
    수학적 성질을 보는 것이지, `TrainConfig` 의 기본값이 무엇이든 그대로 통과해야 하는
    테스트가 아니다(기본값은 M4c Task 5 실측으로 정해진다).
    """
    vec, objs, mask, control, turn = _edge_batch()
    log_std = torch.zeros(2)
    at = torch.atanh(control.clamp(-1.0 + 1e-6, 1.0 - 1e-6))

    mean = at.clone().requires_grad_(True)
    policy_loss(_FixedNet(mean, log_std, torch.zeros(len(turn), 3)),
                (vec, objs, mask, control, turn),
                TrainConfig(squash=True, atanh_eps=1e-6))[0].backward()
    assert torch.allclose(mean.grad, torch.zeros_like(mean), atol=1e-5)   # atanh 지점이 최소

    mean2 = control.clone().requires_grad_(True)
    policy_loss(_FixedNet(mean2, log_std, torch.zeros(len(turn), 3)),
                (vec, objs, mask, control, turn),
                TrainConfig(squash=True, atanh_eps=1e-6))[0].backward()
    assert mean2.grad.abs().max() > 0.3          # 라벨 지점은 최소가 아니다


def test_atanh_eps가_클램프_경계를_정한다():
    """`atanh_eps` 가 커지면 atanh 이전에 자르는 경계가 넓어진다.

    선생님 라벨의 10.48% 가 정확히 ±1(가속은 20.77%)이라 이 경계 폭이 손실·기울기를 지배한다
    (M4c Task 5 실측) — 여기서는 경계가 실제로 `atanh_eps` 를 따라 움직이는지만 본다.
    """
    control = torch.tensor([[0.995, -0.995]])
    log_std = torch.zeros(2)
    vec = torch.zeros(1, VEC_DIM)
    objs = torch.zeros(1, OBJ_N, OBJ_DIM)
    mask = torch.zeros(1, OBJ_N)
    turn = torch.zeros(1, dtype=torch.long)

    # eps=1e-2 → 경계 0.99 보다 큰 0.995 는 잘려서 최소점이 atanh(0.99) 다.
    at_wide = torch.atanh(control.clamp(-0.99, 0.99))
    mean_wide = at_wide.clone().requires_grad_(True)
    policy_loss(_FixedNet(mean_wide, log_std, torch.zeros(1, 3)),
                (vec, objs, mask, control, turn),
                TrainConfig(squash=True, atanh_eps=1e-2))[0].backward()
    assert torch.allclose(mean_wide.grad, torch.zeros_like(mean_wide), atol=1e-5)

    # 같은 지점(atanh(0.99))이 eps=1e-6 에서는 최소가 아니다 — 0.995 가 거의 안 잘리기 때문.
    mean_narrow = at_wide.clone().requires_grad_(True)
    policy_loss(_FixedNet(mean_narrow, log_std, torch.zeros(1, 3)),
                (vec, objs, mask, control, turn),
                TrainConfig(squash=True, atanh_eps=1e-6))[0].backward()
    assert mean_narrow.grad.abs().max() > 1e-3


def test_스쿼시_모방손실은_행동의_음의로그가능도다():
    """야코비안(`+ jac`)이 있어야 손실이 행동 `a` 에 대한 `-log p(a)` 가 된다.

    빼면 `atanh(a)` 공간의 밀도라 비스쿼시 손실과 비교 자체가 범주 오류다.
    값으로 잠근다 — 기울기에는 안 보이기 때문(선생님 라벨만의 함수라 파라미터 기울기 0).
    """
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    batch = next(iter(toy_dataset(64).batches(32, generator=torch.Generator().manual_seed(0))))
    vec, objs, mask, control, _turn = batch
    # 기준식(ref)이 직접 하드코딩한 1e-6 클램프와 짝을 맞추려면 policy_loss 도 같은
    # atanh_eps 를 써야 한다 — 기본값(M4c Task 5 실측으로 정함)에 조용히 기대지 않는다.
    _loss, parts = policy_loss(net, batch, TrainConfig(squash=True, atanh_eps=1e-6))
    with torch.no_grad():
        mean, log_std, _logits = net(vec, objs, mask)
        squashed = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mean, log_std.exp()),
            torch.distributions.TanhTransform(cache_size=1))
        ref = float(-squashed.log_prob(control.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
                    .sum(dim=-1).mean())
        jac = float(_tanh_log_det(torch.atanh(control.clamp(-1.0 + 1e-6, 1.0 - 1e-6))).mean())
    assert parts["control"] == pytest.approx(ref, rel=1e-5)
    assert abs(jac) > 0.1                       # 빼면 이 만큼 어긋난다(무시할 수 없다)
    assert parts["control"] != pytest.approx(ref - jac, rel=1e-3)


def test_스쿼시면_control_mae는_tanh_평균으로_잰다():
    """사전-스쿼시 평균을 행동공간 라벨과 비교하면 MAE 보고가 틀린다."""
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    with torch.no_grad():
        net.mean.bias.copy_(torch.tensor([1.3, -1.0]))     # tanh 와 원값이 확실히 갈리게
    ds = toy_dataset(64)
    out = evaluate_labels(net, ds)
    vec, objs, mask, control, _turn = next(iter(ds.batches(len(ds))))
    with torch.no_grad():
        mean = net(vec, objs, mask)[0]
        want = float((torch.tanh(mean) - control).abs().mean())
        raw = float((mean - control).abs().mean())
    assert out["control_mae"] == pytest.approx(want, rel=1e-6)
    assert out["control_mae"] != pytest.approx(raw, rel=1e-3)
    # 정책이 스쿼시면 손실도 스쿼시로 — cfg 기본값(squash=False)에 조용히 어긋나지 않는다
    assert out["control"] == pytest.approx(
        policy_loss(net, (vec, objs, mask, control, _turn), TrainConfig(squash=True))[1]["control"],
        rel=1e-6)


def test_비스쿼시_control_mae는_그대로다():
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    with torch.no_grad():
        net.mean.bias.copy_(torch.tensor([1.3, -1.0]))
    ds = toy_dataset(64)
    out = evaluate_labels(net, ds)
    vec, objs, mask, control, _turn = next(iter(ds.batches(len(ds))))
    with torch.no_grad():
        mean = net(vec, objs, mask)[0]
    assert out["control_mae"] == pytest.approx(float((mean - control).abs().mean()), rel=1e-6)


def test_스쿼시_기본값은_꺼짐():
    assert TrainConfig().squash is False


# ---------------------------------------------------------------------------
# M4m — 참조 정책 KL 앵커("알던 것을 유지하라")
#
# M4l 이 확정한 것: BC 손상의 원인은 이동거리가 아니라 **방향**이다(같은 Δθ=0.54 라도 무작위
# 잡음은 단계 ① 83.3% 를 남기고 단계 ③ 기울기는 0% 로 만든다). 손실에 이전 정책과 묶어 두는
# 항이 없으니 옛 해를 버리는 쪽이 손실 기준으로 늘 이긴다 — 그 항을 여기서 잠근다.
# ---------------------------------------------------------------------------


def _anchor_net(seed=0, squash=False):
    """작은 그물 — 앵커 테스트는 전부 이 모양을 쓴다(학습이 아니라 손실의 성질만 본다)."""
    torch.manual_seed(seed)
    return DrivePolicy(PolicyConfig(trunk=(32, 32), squash=squash))


def _anchor_batch(n=64, seed=0):
    return next(iter(toy_dataset(n, seed=seed).batches(n, generator=torch.Generator().manual_seed(0))))


def _nudge(net, scale=0.1, seed=1):
    """학생을 참조에서 떼어 놓는다 — 앵커가 0 이 아니게 만들려는 것."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return net


def test_anchor_coef_기본값은_0이다():
    """기존 호출부(run_dagger·refit_m3·ppo)가 전부 이 기본값 위에 서 있다."""
    assert TrainConfig().anchor_coef == 0.0
    assert TrainConfig().anchor_lr is None


def test_anchor_coef_0은_예전과_완전히_같다():
    """기존 호출부·과거 성적표가 전부 이 보장 위에 있다 — 비트 단위로 같아야 한다."""
    net, b = _anchor_net(0), _anchor_batch()
    ref = copy.deepcopy(net)
    a, ap = policy_loss(net, b, TrainConfig(anchor_coef=0.0), ref=ref)
    c, cp = policy_loss(net, b, TrainConfig())           # ref 없음
    assert torch.equal(a, c)
    assert (ap["control"], ap["turn"], ap["total"]) == (cp["control"], cp["turn"], cp["total"])
    assert ap["anchor"] == 0.0 and cp["anchor"] == 0.0


def test_ref가_없으면_anchor_coef가_커도_앵커가_없다():
    """참조를 안 넘기는 호출부는 계수를 켜도 예전 그대로다(조용히 터지지 않는다)."""
    net, b = _anchor_net(0), _anchor_batch()
    a, ap = policy_loss(net, b, TrainConfig(anchor_coef=10.0))
    c, _cp = policy_loss(net, b, TrainConfig())
    assert torch.equal(a, c) and ap["anchor"] == 0.0


def test_참조와_같으면_KL이_0이다():
    net = _anchor_net(0)
    ref = copy.deepcopy(net)
    _, parts = policy_loss(net, _anchor_batch(), TrainConfig(anchor_coef=1.0), ref=ref)
    assert parts["anchor"] == pytest.approx(0.0, abs=1e-6)


def test_참조에서_멀어지면_KL이_커진다():
    net = _nudge(_anchor_net(0))
    ref = _anchor_net(0)                     # 같은 시드로 다시 지은 '움직이기 전' 그물
    _, parts = policy_loss(net, _anchor_batch(), TrainConfig(anchor_coef=1.0), ref=ref)
    assert parts["anchor"] > 0.01


def test_총손실은_앵커를_계수만큼_더한_값이다():
    """부호와 계수를 값으로 잠근다 — 부호를 뒤집으면 앵커가 '멀어져라' 가 된다."""
    net = _nudge(_anchor_net(0))
    ref = _anchor_net(0)
    b = _anchor_batch()
    base, _bp = policy_loss(net, b, TrainConfig(anchor_coef=0.0), ref=ref)
    unit = policy_loss(net, b, TrainConfig(anchor_coef=1.0), ref=ref)[1]["anchor"]
    assert unit > 0.01
    for coef in (0.1, 1.0, 10.0):
        _tot, parts = policy_loss(net, b, TrainConfig(anchor_coef=coef), ref=ref)
        # 보고되는 `anchor` 는 **계수를 안 곱한** KL 이다(계수를 바꿔도 같아야 한다)
        assert parts["anchor"] == pytest.approx(unit, rel=1e-6)
        assert parts["total"] == pytest.approx(
            parts["control"] + 0.5 * parts["turn"] + coef * parts["anchor"], rel=1e-6)
        assert parts["total"] > float(base.detach())   # 더하는 것이지 빼는 것이 아니다


def test_KL_방향은_참조에서_학생으로다():
    """`KL(ref ‖ net)` — 참조의 질량을 학생이 덮게 강제하는 방향(망각 방지).

    반대 방향(`KL(net ‖ ref)`)은 모드 추종이라 참조의 모드를 버려도 벌을 안 받는다 — 바로
    우리가 막으려는 것이다. 두 값이 실제로 다른 상황에서 어느 쪽인지 **값으로** 잠근다.
    """
    net = _nudge(_anchor_net(0))
    ref = _anchor_net(0)
    vec, objs, mask, control, turn = _anchor_batch()
    _, parts = policy_loss(net, (vec, objs, mask, control, turn),
                           TrainConfig(anchor_coef=1.0, turn_weight=0.0), ref=ref)
    with torch.no_grad():
        mu_n, ls_n, _lg_n = net(vec, objs, mask)
        mu_r, ls_r, _lg_r = ref(vec, objs, mask)
        var_n, var_r = (2 * ls_n).exp(), (2 * ls_r).exp()
        # 닫힌형을 손으로 적는다(구현은 torch.distributions 를 쓴다 — 서로 다른 길)
        fwd = float((((ls_n - ls_r) + (var_r + (mu_r - mu_n) ** 2) / (2 * var_n) - 0.5)
                     .sum(dim=-1)).mean())
        rev = float((((ls_r - ls_n) + (var_n + (mu_n - mu_r) ** 2) / (2 * var_r) - 0.5)
                     .sum(dim=-1)).mean())
    assert abs(fwd - rev) > 1e-3, "두 방향이 같은 상황이면 이 테스트가 아무것도 안 잠근다"
    assert parts["anchor"] == pytest.approx(fwd, rel=1e-5)
    assert parts["anchor"] != pytest.approx(rev, rel=1e-3)


def test_지시등_KL은_turn_weight로_섞인다():
    """앵커 안의 머리 균형은 데이터 항과 **같은** `turn_weight` 다 — 근거는 train.py 주석."""
    net = _nudge(_anchor_net(0))
    ref = _anchor_net(0)
    b = _anchor_batch()
    only_gauss = policy_loss(net, b, TrainConfig(anchor_coef=1.0, turn_weight=0.0),
                             ref=ref)[1]["anchor"]
    half = policy_loss(net, b, TrainConfig(anchor_coef=1.0, turn_weight=0.5), ref=ref)[1]["anchor"]
    full = policy_loss(net, b, TrainConfig(anchor_coef=1.0, turn_weight=1.0), ref=ref)[1]["anchor"]
    turn_kl = full - only_gauss
    assert turn_kl > 1e-4, "지시등 KL 이 0 이면 섞는 규칙을 잠글 수 없다"
    assert half == pytest.approx(only_gauss + 0.5 * turn_kl, rel=1e-6)


def test_앵커는_스쿼시_전에서_재도_같다_야코비안이_상쇄된다():
    """두 정책이 **같은** tanh 를 통과하므로 사전-스쿼시 가우시안 KL = 행동공간 KL 이다.

    KL 은 가역변환에 불변이다 — 두 밀도에 같은 `|det J|` 가 곱해져 로그 안에서 지워진다.
    말로만 두지 않고 몬테카를로로 확인한다: `TransformedDistribution`(torch 가 야코비안을
    따로 계산하는 길)으로 잰 행동공간 KL 이 우리가 쓰는 닫힌형과 같아야 한다. 틀리면
    앵커가 엉뚱한 분포를 묶게 된다.
    """
    net = _nudge(_anchor_net(0, squash=True), scale=0.2)
    ref = _anchor_net(0, squash=True)
    vec, objs, mask, control, turn = _anchor_batch(n=32)
    _, parts = policy_loss(net, (vec, objs, mask, control, turn),
                           TrainConfig(anchor_coef=1.0, turn_weight=0.0, squash=True), ref=ref)

    with torch.no_grad():
        mu_n, ls_n, _ = net(vec, objs, mask)
        mu_r, ls_r, _ = ref(vec, objs, mask)
        tanh = torch.distributions.TanhTransform(cache_size=1)
        p_a = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mu_r, ls_r.exp()), tanh)
        q_a = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mu_n, ls_n.exp()), tanh)
        g = torch.Generator().manual_seed(0)
        total, draws = 0.0, 400
        for _ in range(draws):
            u = mu_r + torch.randn(mu_r.shape, generator=g) * ls_r.exp()
            a = torch.tanh(u)
            total += float((p_a.log_prob(a) - q_a.log_prob(a)).sum(dim=-1).mean())
        mc = total / draws
    assert mc > 0.05, "KL 이 사실상 0 이면 이 비교가 아무것도 안 잠근다"
    assert parts["anchor"] == pytest.approx(mc, rel=0.05)


def test_참조가_학생과_파라미터를_공유하면_거부한다():
    """★ 참조가 학생과 같은 텐서를 보면 KL 이 항상 0 이다 — 조용히 아무것도 안 묶는다.

    `copy.deepcopy` 대신 같은 객체(또는 얕은 복사)를 넘기는 실수는 숫자가 멀쩡해 보여서
    실험 한 팔을 통째로 날린다. 손실에서 바로 거부한다.
    """
    net = _anchor_net(0)
    b = _anchor_batch()
    with pytest.raises(ValueError, match="참조"):
        policy_loss(net, b, TrainConfig(anchor_coef=1.0), ref=net)
    with pytest.raises(ValueError, match="참조"):
        policy_loss(net, b, TrainConfig(anchor_coef=1.0), ref=copy.copy(net))
    # 계수가 0 이면 앵커를 아예 안 쓰므로 막을 것도 없다(기존 경로를 안 건드린다)
    policy_loss(net, b, TrainConfig(anchor_coef=0.0), ref=net)


def test_make_reference는_얼린_사본이다():
    net = _anchor_net(0)
    ref = make_reference(net)
    assert ref is not net
    assert all(not p.requires_grad for p in ref.parameters())
    assert ref.training is False
    for p, q in zip(net.parameters(), ref.parameters()):
        assert p is not q and torch.equal(p, q)
    # 사본이다 — 학생을 움직여도 참조는 그대로여야 한다(얕은 복사면 같이 움직인다)
    _nudge(net)
    assert any(not torch.equal(p, q) for p, q in zip(net.parameters(), ref.parameters()))


def test_참조에는_기울기가_안_간다():
    """★ 참조 순전파를 `no_grad` 밖에서 하면 참조 파라미터에 `.grad` 가 쌓인다.

    옵티마이저가 학생 파라미터만 보므로 참조는 **움직이지 않는다** — 그래서 '참조가 안
    움직였다' 만 보는 테스트로는 이 실수를 못 잡는다. 여기서는 `.grad` 자체를 본다.
    참조를 `copy.deepcopy` 로 만드는 것이 핵심이다(`requires_grad=True` 가 그대로 따라온다
    — `make_reference` 로 바꾸면 기울기가 애초에 안 흘러 이 잠금이 사라진다).
    """
    net = _nudge(_anchor_net(0))
    ref = copy.deepcopy(_anchor_net(0))
    assert all(p.requires_grad for p in ref.parameters())
    net.zero_grad(set_to_none=True)
    policy_loss(net, _anchor_batch(), TrainConfig(anchor_coef=1.0), ref=ref)[0].backward()
    assert all(p.grad is None for p in ref.parameters()), "참조에 기울기가 흘렀다"
    assert any(p.grad is not None and torch.any(p.grad != 0.0) for p in net.parameters())


def test_앵커는_참조를_학습시키지_않는다():
    """★ 참조가 같이 움직이면 앵커가 아무것도 안 묶는다 — 조용한 무효화."""
    net = _anchor_net(0)
    ref = copy.deepcopy(net)
    before = [p.detach().clone() for p in ref.parameters()]
    cfg = TrainConfig(anchor_coef=1.0, epochs=1, lr=1e-2)
    train_epochs(net, toy_dataset(128), cfg, ref=ref)
    for p, q in zip(ref.parameters(), before):
        assert torch.equal(p, q), "참조가 움직였다"
    assert any(not torch.equal(p, q) for p, q in zip(net.parameters(), before)), "학생이 안 움직였다"
    assert all(p.grad is None for p in ref.parameters()), "참조에 기울기가 쌓였다"


def test_앵커가_가중치_이동을_실제로_줄인다():
    """같은 데이터·시드·에포크에서 anchor_coef 가 크면 Δθ 가 작아야 한다."""
    deltas = []
    for coef in (0.0, 1.0, 10.0):
        net = _anchor_net(0)
        ref = copy.deepcopy(net)
        start = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
        train_epochs(net, toy_dataset(256), TrainConfig(anchor_coef=coef, epochs=2, seed=0),
                     ref=ref)
        end = torch.cat([p.detach().flatten() for p in net.parameters()])
        deltas.append(float((end - start).norm()))
    assert deltas[0] > deltas[1] > deltas[2], deltas


def test_앵커_손실이_에폭_기록에_실린다():
    """계수를 키웠는데 앵커 값이 안 보이면 실험 결과를 해석할 수 없다."""
    net = _anchor_net(0)
    ref = copy.deepcopy(net)
    seen = []
    out = train_epochs(net, toy_dataset(128), TrainConfig(anchor_coef=1.0, epochs=2, lr=1e-2),
                       ref=ref, log=seen.append)
    assert "anchor" in seen[0] and "anchor" in out
    assert out["anchor"] > 0.0          # 학생이 움직였으니 참조와 벌어져 있다


def test_anchor_lr이_없으면_lr을_쓴다():
    assert TrainConfig(lr=3e-4).effective_lr() == 3e-4
    assert TrainConfig(lr=3e-4, anchor_lr=1e-5).effective_lr() == 1e-5
    # 0.0 은 "안 줬다" 가 아니다 — `or` 로 짜면 여기서 3e-4 가 나온다
    assert TrainConfig(lr=3e-4, anchor_lr=0.0).effective_lr() == 0.0


def test_anchor_lr이_진짜_학습률로_쓰인다():
    """★ `effective_lr()` 만 보는 테스트는 '필드는 있는데 배선이 끊긴' 돌연변이를 못 잡는다.

    `train_epochs` 가 실제로 그 값을 Adam 에 넘기는지 Δθ 로 확인한다.
    """
    deltas = []
    for kwargs in ({}, {"anchor_lr": 1e-5}, {"anchor_lr": 0.0}):
        net = _anchor_net(0)
        start = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
        train_epochs(net, toy_dataset(128), TrainConfig(lr=1e-2, epochs=1, seed=0, **kwargs))
        end = torch.cat([p.detach().flatten() for p in net.parameters()])
        deltas.append(float((end - start).norm()))
    assert deltas[0] > deltas[1] > deltas[2] == 0.0, deltas


# ---------------------------------------------------------------------------
# M4n — 회피 걸음에 표본별 가중치
#
# M4m 이 확정한 것: 회피는 데이터의 **2%** 다(r0 216,440 행 중 물체 30 m 안 전방 +
# |조향|>0.15 가 4,383 행). 손실을 98% 의 평범한 주행이 지배하니 Δθ 3.80 이 거기로 갔다.
# 그래서 **물체가 가까운 상태**의 손실을 올린다 — 라벨이 아니라 **관측**으로 고른다
# (물체 근처 걸음의 79.4% 는 큰 조향이 아니다. 라벨로 고르면 그 다수를 버린다).
# ---------------------------------------------------------------------------

#: 전방거리 역정규화 길이[m]. `ObsConfig.obj_x` 와 **따로** 적는다 — 구현이 쓰는 상수를
#: 테스트가 그대로 가져다 쓰면 그 상수를 바꾸는 돌연변이를 못 잡는다.
OBJ_X_M = 80.0


def _obj_batch(fx_m, mask=1.0, steer=0.0, n=8, slot=0):
    """슬롯 `slot` 에 물체 하나를 둔 배치 — 전방거리와 라벨을 따로 쥔다."""
    objs = torch.zeros(n, OBJ_N, OBJ_DIM)
    m = torch.zeros(n, OBJ_N)
    objs[:, slot, 0] = fx_m / OBJ_X_M            # observation.py:135 의 `_clip(fx, obj_x)`
    m[:, slot] = mask
    control = torch.zeros(n, 2)
    control[:, 0] = steer
    return (torch.zeros(n, VEC_DIM), objs, m, control, torch.zeros(n, dtype=torch.long))


def _cat_batch(a, b):
    return tuple(torch.cat([x, y], dim=0) for x, y in zip(a, b))


def _selected(batch, cfg):
    _vec, objs, mask, _control, _turn = batch
    return near_object_rows(objs, mask, cfg)


def test_near_m_기본값은_가중치_없음이다():
    assert TrainConfig().near_m == 0.0
    assert TrainConfig().near_weight == 1.0


def test_near_m_0은_예전과_완전히_같다():
    """기존 호출부·과거 성적표가 전부 이 보장 위에 있다 — 비트 단위로 같아야 한다."""
    net, b = _anchor_net(0), _anchor_batch()
    a, ap = policy_loss(net, b, TrainConfig(near_m=0.0, near_weight=7.0))
    c, cp = policy_loss(net, b, TrainConfig())
    assert torch.equal(a, c)
    assert (ap["control"], ap["turn"], ap["total"]) == (cp["control"], cp["turn"], cp["total"])


def test_near_weight_1은_가중치_없음과_같다():
    """실험의 **대조군**이다 — near_weight=1 칸이 '가중치 없음' 과 달라지면 비교축이 깨진다."""
    net = _anchor_net(0)
    b = _cat_batch(_obj_batch(fx_m=5.0), _obj_batch(fx_m=70.0))
    a, _ = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=1.0))
    c, _ = policy_loss(net, b, TrainConfig())
    assert torch.equal(a, c)


def test_가중치는_정규화된다_실효학습률을_안_키운다():
    """★ 핵심 함정 — 정규화를 빼면 near_weight 가 lr 을 같이 키워 효과를 못 가린다."""
    net = _anchor_net(0)
    b = _obj_batch(fx_m=5.0, n=16)               # 전부 '가까움' 인 배치
    a, ap = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=1.0))
    for w in (3.0, 100.0):
        c, cp = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=w))
        assert torch.allclose(a, c), f"전부 같은 가중치면 배율과 무관해야 한다(w={w})"
        assert cp["control"] == pytest.approx(ap["control"], rel=1e-6)
        assert cp["turn"] == pytest.approx(ap["turn"], rel=1e-6)
    # 기울기까지 같아야 한다 — 손실 값만 보면 '손실은 나눴는데 역전파 경로는 안 나눈' 꼴을 놓친다
    grads = []
    for w in (1.0, 100.0):
        net.zero_grad(set_to_none=True)
        policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=w))[0].backward()
        grads.append(net.mean.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_가까운_표본이_실제로_더_센다():
    """절반만 가까운 배치에서는 배율이 손실을 바꿔야 한다 — 안 바뀌면 가중이 배선 안 된 것."""
    net = _anchor_net(0)
    b = _cat_batch(_obj_batch(fx_m=5.0, steer=0.8), _obj_batch(fx_m=0.0, mask=0.0, steer=-0.8))
    base, _ = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=1.0))
    up, _ = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=10.0))
    assert not torch.allclose(base, up), "가중이 손실을 안 바꿨다"
    # 방향도 잠근다 — 가까운 쪽(라벨 0.8)만 쓴 손실로 수렴해야 한다('먼 쪽을 키우는' 부호 뒤집기)
    near_only, _ = policy_loss(net, _obj_batch(fx_m=5.0, steer=0.8), TrainConfig())
    huge, _ = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=1e6))
    assert float(huge.detach()) == pytest.approx(float(near_only.detach()), rel=1e-4)


def test_선택은_관측의_전방거리로_한다():
    """`objs[:,0,0]*80` 이 near_m 안이고 전방일 때만 고른다 — 라벨을 안 본다."""
    cfg = TrainConfig(near_m=30.0)
    assert _selected(_obj_batch(fx_m=10.0, steer=0.0), cfg).all()      # 가깝고 안 꺾음 → 뽑는다
    assert not _selected(_obj_batch(fx_m=70.0, steer=0.9), cfg).any()  # 멀고 크게 꺾음 → 안 뽑는다
    assert not _selected(_obj_batch(fx_m=-10.0, steer=0.9), cfg).any() # 뒤에 있다 → 안 뽑는다
    assert _selected(_obj_batch(fx_m=29.9), cfg).all()                 # 경계 안
    assert not _selected(_obj_batch(fx_m=30.1), cfg).any()             # 경계 밖
    # near_m 을 키우면 같은 배치가 뽑힌다 — 문턱이 진짜 cfg 를 따라간다
    assert _selected(_obj_batch(fx_m=70.0), TrainConfig(near_m=80.0)).all()
    # near_m=0(기본)은 아무도 안 고른다
    assert not _selected(_obj_batch(fx_m=10.0), TrainConfig()).any()


def test_마스크가_죽은_슬롯은_안_고른다():
    """object_mask 가 0 인 슬롯의 fx 는 쓰레기다 — 0 이면 '아주 가까움' 으로 오인된다."""
    cfg = TrainConfig(near_m=30.0)
    assert not _selected(_obj_batch(fx_m=0.0, mask=0.0), cfg).any()
    assert not _selected(_obj_batch(fx_m=10.0, mask=0.0), cfg).any()
    # 같은 fx 라도 **살아 있으면** 뽑힌다 — 둘이 갈려야 마스크가 진짜 문지기다
    assert _selected(_obj_batch(fx_m=0.0, mask=1.0), cfg).all()
    assert _selected(_obj_batch(fx_m=10.0, mask=1.0), cfg).all()


def test_손실도_관측으로_고른다_라벨로_안_고른다():
    """★ `near_object_rows` 만 잠그면 `policy_loss` 가 **그 함수를 안 쓰는** 돌연변이를 놓친다.

    실제로 당했다 — 선택을 `|조향|>0.15` 로 바꾸는 돌연변이가
    `test_선택은_관측의_전방거리로_한다`(헬퍼만 본다)를 통과했다. 그래서 손실 쪽에서도 잠근다.

    `_FixedNet` 으로 예측을 고정해 **물체가 그물 출력에 영향을 안 주게** 한 뒤, 라벨만
    맞바꾼 두 배치를 비교한다. 관측으로 고르면 "가까운 쪽 라벨" 이 무거워지고, 라벨로
    고르면 두 배치가 **같은 값**이 된다(둘 다 큰 조향 쪽을 무겁게 하므로).
    """
    half = 4
    mean = torch.zeros(2 * half, 2)
    fixed = _FixedNet(mean, torch.zeros(2), torch.zeros(2 * half, 3))

    def loss_of(near_steer, far_steer):
        b = _cat_batch(_obj_batch(fx_m=5.0, steer=near_steer, n=half),
                       _obj_batch(fx_m=70.0, steer=far_steer, n=half))
        return float(policy_loss(fixed, b, TrainConfig(near_m=30.0, near_weight=10.0))[0])

    near_small, near_big = loss_of(0.0, 0.9), loss_of(0.9, 0.0)
    assert near_small != pytest.approx(near_big, rel=1e-6), "라벨만 맞바꿔도 값이 같다"
    # 관측으로 고르면 **가까운 쪽**(라벨 0.0, 손실 작음)이 무거워져 총손실이 작아진다.
    # 라벨로 고르면 두 배치 다 '큰 조향' 쪽을 무겁게 해서 값이 같아진다.
    assert near_small < near_big


def test_선택은_최근접_슬롯만_본다():
    """슬롯 0 이 최근접이다(`observation.py:128`) — 뒤쪽 슬롯을 보면 정의가 달라진다."""
    cfg = TrainConfig(near_m=30.0)
    assert not _selected(_obj_batch(fx_m=10.0, slot=3), cfg).any()


def test_near_frac이_가중_비율을_보고한다():
    """실데이터에서 가중이 **몇 %에 걸렸나** — 0 이면 아무 일도 안 일어난 것이다."""
    net = _anchor_net(0)
    b = _cat_batch(_obj_batch(fx_m=5.0, n=4), _obj_batch(fx_m=70.0, n=12))
    assert policy_loss(net, b, TrainConfig(near_m=30.0))[1]["near_frac"] == pytest.approx(0.25)
    assert policy_loss(net, b, TrainConfig())[1]["near_frac"] == 0.0


def test_near_weight가_0이하면_거부한다():
    """가중치 합이 0 이면 손실이 NaN 이다 — 격자 한 팔이 조용히 죽는다."""
    net = _anchor_net(0)
    b = _obj_batch(fx_m=5.0)
    with pytest.raises(ValueError, match="near_weight"):
        policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=0.0))


def test_가중이_학습_경로까지_닿는다():
    """손실만 바뀌고 `train_epochs` 가 안 쓰면 격자 전체가 같은 실험이 된다."""
    deltas = []
    for weight in (1.0, 50.0):
        net = _anchor_net(0)
        ds = _near_dataset()
        start = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
        train_epochs(net, ds, TrainConfig(epochs=2, seed=0, lr=1e-2, batch_size=32,
                                          near_m=30.0, near_weight=weight))
        end = torch.cat([p.detach().flatten() for p in net.parameters()])
        deltas.append(float((end - start).norm()))
    assert deltas[0] != pytest.approx(deltas[1], rel=1e-6), deltas


def _near_dataset(n=128, seed=0):
    """절반은 물체가 5 m 앞, 절반은 물체 없음 — 두 쪽의 정답이 다르다(가중이 뜻을 갖게)."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((n, VEC_DIM)).astype(np.float32)
    objs = np.zeros((n, OBJ_N, OBJ_DIM), np.float32)
    mask = np.zeros((n, OBJ_N), np.float32)
    objs[: n // 2, 0, 0] = 5.0 / OBJ_X_M
    mask[: n // 2, 0] = 1.0
    control = np.stack([np.tanh(vec[:, 0]), np.tanh(vec[:, 1])], axis=1).astype(np.float32)
    control[: n // 2, 0] = 0.9                   # 가까운 쪽은 크게 꺾는다
    turn = (vec[:, 2] > 0).astype(np.int64)
    ds = DaggerDataset()
    ds.add(Shard(vec, objs, mask, control, turn, {"board": "near"}))
    return ds


def test_atanh_eps_기본값은_실측으로_고른_1e_2다():
    """M4c Task 5 실측(`runs/lab-main/2026-09-17-dagger-fix2/data`, 200,511 행, 진짜 크기

    정책, BC 400 스텝, 같은 시드·배치 순서): ε∈{1e-2,1e-3,1e-4,1e-6} 중 행동공간 MAE
    (`|tanh(mean)-control|` 평균)가 가장 낮은 것은 ε=1e-2(0.1802) 다 — 1e-3(0.1978),
    1e-4(0.2175), 1e-6(0.2483) 순으로 나빠진다(손실값 자체는 ε 마다 스케일이 달라 비교 기준이
    아니다). 라벨의 10.48%(가속 20.77%, 조향 0.19%)가 정확히 ±1 이라 ε=1e-6 이면
    기울기 노름 중앙값이 54.72(비스쿼시 기준 1.59 의 34배)까지 뛴다."""
    assert TrainConfig().atanh_eps == pytest.approx(1e-2)


# ── M4t: 갱신 횟수 상한 ────────────────────────────────────────────────────────────
# M4s 의 r4 하락이 "같은 8 에폭이 라운드마다 더 많은 갱신" 탓인지 가르려면 갱신 횟수를
# 에폭과 따로 정할 수 있어야 한다. 기본값(None)은 지금과 비트 동일해야 한다.


def _params(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _fresh(seed=0):
    torch.manual_seed(seed)
    return DrivePolicy(PolicyConfig(trunk=(32, 32)))


def test_max_updates_기본값은_None이다():
    assert TrainConfig().max_updates is None


def test_max_updates_None_은_아주_큰_상한과_비트_동일하다():
    ds = toy_dataset(300)                      # 300 / 128 → 배치 3 개(마지막 44 행)
    a, b = _fresh(), _fresh()
    out_a = train_epochs(a, ds, TrainConfig(epochs=2, batch_size=128, seed=7))
    out_b = train_epochs(b, ds, TrainConfig(epochs=2, batch_size=128, seed=7, max_updates=10**9))
    assert _same(_params(a), _params(b))
    assert out_a["loss"] == out_b["loss"]
    assert out_a["updates"] == out_b["updates"] == 2 * 3


def test_updates_는_실제_걸음_수다():
    ds = toy_dataset(512)                      # 512 / 128 → 배치 4 개
    out = train_epochs(_fresh(), ds, TrainConfig(epochs=3, batch_size=128))
    assert out["updates"] == 12 and out["epochs"] == 3


def test_max_updates_는_에폭_경계에서_멈춘_것과_같다():
    ds = toy_dataset(512)                      # 에폭당 4 걸음
    a, b = _fresh(), _fresh()
    out_a = train_epochs(a, ds, TrainConfig(epochs=1, batch_size=128, seed=3))
    out_b = train_epochs(b, ds, TrainConfig(epochs=5, batch_size=128, seed=3, max_updates=4))
    assert _same(_params(a), _params(b))
    assert out_b["updates"] == 4


def test_max_updates_는_뒤_에폭_수와_무관하다():
    # 같은 시드면 배치 순서가 같다 — 예산을 넘는 에폭을 몇으로 두든 결과가 같아야 한다.
    ds = toy_dataset(512)
    a, b = _fresh(), _fresh()
    train_epochs(a, ds, TrainConfig(epochs=2, batch_size=128, seed=5, max_updates=6))
    train_epochs(b, ds, TrainConfig(epochs=9, batch_size=128, seed=5, max_updates=6))
    assert _same(_params(a), _params(b))


def test_max_updates_가_에폭_중간에_끊으면_그_에폭을_한_번_기록한다():
    ds = toy_dataset(512)                      # 에폭당 4 걸음 → 6 걸음 = 에폭 0 전부 + 에폭 1 의 2 걸음
    seen = []
    out = train_epochs(_fresh(), ds, TrainConfig(epochs=5, batch_size=128, max_updates=6),
                       log=seen.append)
    assert [s["epoch"] for s in seen] == [0, 1]
    assert out["updates"] == 6
    assert np.isfinite(seen[-1]["loss"])


def test_예산이_에폭보다_크면_에폭에서_멈추고_실제_걸음을_보고한다():
    ds = toy_dataset(512)
    out = train_epochs(_fresh(), ds, TrainConfig(epochs=2, batch_size=128, max_updates=100))
    assert out["updates"] == 8                 # 상한이 아니라 실제 걸음 — 부르는 쪽이 확인한다


@pytest.mark.parametrize("bad", [0, -1])
def test_max_updates_가_0이하면_학습_전에_거부한다(bad):
    net = _fresh()
    before = _params(net)
    with pytest.raises(ValueError, match="max_updates"):
        train_epochs(net, toy_dataset(64), TrainConfig(epochs=1, batch_size=32, max_updates=bad))
    assert _same(before, _params(net))
