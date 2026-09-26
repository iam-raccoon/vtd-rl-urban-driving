import numpy as np
import torch

from vtd_rl.policy.dataset import DaggerDataset, Shard
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.policy.train import TrainConfig, evaluate_labels, policy_loss, train_epochs


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


def test_스쿼시_기본값은_꺼짐():
    assert TrainConfig().squash is False
