"""`scripts/refit_round.py` — M4s 데이터로 라운드 K 를 다시 학습한다(M4t).

판을 하나도 안 짓는다. 평가는 `evaluate_policy`·`load_window` 를 가짜로 바꾸고, 학습은
아주 작은 그물·장난감 조각으로 몇 걸음만 돈다.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from vtd_rl.policy.dataset import DaggerDataset, Shard, load_shard, save_shard
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.policy.train import TrainConfig, make_reference, squash_aligned, train_epochs

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "refit_round.py")
PY = os.path.join(REPO, ".venv", "bin", "python")


@pytest.fixture(scope="module")
def rr():
    spec = importlib.util.spec_from_file_location("refit_round_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shard(n, seed):
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((n, VEC_DIM)).astype(np.float32)
    control = np.stack([np.tanh(vec[:, 0]), np.tanh(vec[:, 1])], axis=1).astype(np.float32)
    turn = (vec[:, 2] > 0).astype(np.int64)
    return Shard(vec, np.zeros((n, OBJ_N, OBJ_DIM), np.float32),
                 np.zeros((n, OBJ_N), np.float32), control, turn, {"board": "toy"})


def _fake_run(tmp_path, rounds=3, per_round=2, n=40):
    """`<run>/data/r{K}-course_X@v{i}-s0.npz` 꼴 조각 — M4s 폴더와 같은 이름 규칙."""
    data = tmp_path / "run" / "data"
    for k in range(rounds):
        for i in range(per_round):
            save_shard(_shard(n, 100 * k + i), str(data / f"r{k}-course_X@v{i}-s0.npz"))
    return tmp_path / "run"


def _init_ckpt(tmp_path):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    path = str(tmp_path / "init.pt")
    net.save(path)
    return path


def _params(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ── 조각 고르기 ──────────────────────────────────────────────────────────────

def test_round_of_는_이름_앞의_라운드_번호다(rr):
    assert rr.round_of("r0-course_A@v0-s0.npz") == 0
    assert rr.round_of("/x/y/r12-course_B@v3-s1.npz") == 12


def test_round_of_는_모르는_이름을_거부한다(rr):
    with pytest.raises(ValueError, match="라운드"):
        rr.round_of("course_A@v0-s0.npz")


def test_shard_paths_는_K_이하만_이름순으로(rr, tmp_path):
    run = _fake_run(tmp_path, rounds=3)
    got = rr.shard_paths(str(run / "data"), 1)
    names = [os.path.basename(p) for p in got]
    assert names == sorted(names)
    assert {rr.round_of(p) for p in got} == {0, 1}
    assert len(got) == 4


def test_라운드가_비면_거부한다(rr, tmp_path):
    run = _fake_run(tmp_path, rounds=2)
    with pytest.raises(ValueError, match="라운드 2"):
        rr.shard_paths(str(run / "data"), 2)
    os.remove(run / "data" / "r0-course_X@v0-s0.npz")
    os.remove(run / "data" / "r0-course_X@v1-s0.npz")
    with pytest.raises(ValueError, match="라운드 0"):
        rr.shard_paths(str(run / "data"), 1)


def test_load_upto_는_그때_폴더를_load_dir_로_읽은_것과_같다(rr, tmp_path):
    from vtd_rl.policy.dataset import load_dir
    run = _fake_run(tmp_path, rounds=3)
    then = tmp_path / "then"
    then.mkdir()
    for p in rr.shard_paths(str(run / "data"), 1):
        shutil.copy(p, then / os.path.basename(p))
    a = rr.load_upto(str(run / "data"), 1).arrays()
    b = load_dir(str(then)).arrays()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# ── 예산 산수 ────────────────────────────────────────────────────────────────

def test_예산_산수(rr):
    assert rr.batches_per_epoch(1000, 256) == 4
    assert rr.batches_per_epoch(1024, 256) == 4
    assert rr.budget(1000, 8, 256) == 32
    assert rr.effective_epochs(8, None, 1000, 256) == 8
    assert rr.effective_epochs(8, 20, 1000, 256) == 8      # 예산이 8 에폭 안이면 8 그대로
    assert rr.effective_epochs(8, 40, 1000, 256) == 10     # 40 / 4 = 10 에폭이 있어야 채운다
    assert rr.effective_epochs(8, 41, 1000, 256) == 11


# ── 창 ───────────────────────────────────────────────────────────────────────

def test_평가_창이_수집_보고_창과_겹치면_거부한다(rr):
    rr.check_window(4, 4, 8)                                # v8~v11 — 통과
    with pytest.raises(ValueError, match="보고"):
        rr.check_window(4, 4, 4)
    with pytest.raises(ValueError, match="수집"):
        rr.check_window(4, 4, 0)
    with pytest.raises(ValueError, match="보고"):
        rr.check_window(4, 2, 7)                            # v7~v8 — 한 변종만 겹쳐도


# ── 학습 순서 ────────────────────────────────────────────────────────────────

def test_refit_은_run_dagger_와_같은_순서로_학습한다(rr, tmp_path):
    run = _fake_run(tmp_path, rounds=3)
    init = _init_ckpt(tmp_path)
    ds = rr.load_upto(str(run / "data"), 2)
    net, train = rr.refit(init, ds, upto=2, batch_seed=5, epochs=2, max_updates=None,
                          anchor_coef=3.0, init_seed=0, dev="cpu")

    # run_dagger.py:952-987 을 손으로 옮긴 것
    ref = make_reference(rr.start_net(init, "cpu"))
    torch.manual_seed(0 + 2)
    want = rr.start_net(init, "cpu")
    tcfg = squash_aligned(TrainConfig(epochs=2, seed=5, anchor_coef=3.0), want)
    want_train = train_epochs(want, rr.load_upto(str(run / "data"), 2), tcfg, device="cpu", ref=ref)

    assert _same(_params(net), _params(want))
    assert train["updates"] == want_train["updates"]
    assert train["anchor"] == want_train["anchor"] and train["anchor"] > 0.0


def test_refit_의_배치_시드는_더하지_않고_그대로다(rr, tmp_path):
    run = _fake_run(tmp_path, rounds=2)
    init = _init_ckpt(tmp_path)
    a, _ = rr.refit(init, rr.load_upto(str(run / "data"), 1), upto=1, batch_seed=4, epochs=1,
                    max_updates=None, anchor_coef=0.0, init_seed=0, dev="cpu")
    b, _ = rr.refit(init, rr.load_upto(str(run / "data"), 1), upto=1, batch_seed=5, epochs=1,
                    max_updates=None, anchor_coef=0.0, init_seed=0, dev="cpu")
    c, _ = rr.refit(init, rr.load_upto(str(run / "data"), 1), upto=1, batch_seed=4, epochs=1,
                    max_updates=None, anchor_coef=0.0, init_seed=0, dev="cpu")
    assert not _same(_params(a), _params(b))
    assert _same(_params(a), _params(c))


def test_same_weights(rr, tmp_path):
    init = _init_ckpt(tmp_path)
    net = rr.start_net(init, "cpu")
    assert rr.same_weights(net, init)
    with torch.no_grad():
        next(net.parameters()).add_(1e-6)
    assert not rr.same_weights(net, init)


# ── CLI 끝까지 ────────────────────────────────────────────────────────────────

def _fake_eval(monkeypatch, rr, calls):
    def fake_window(path, variants, offset):
        varied = "stage3" in os.path.basename(path)
        boards = [f"{os.path.basename(path)}@v{offset + i}" for i in range(variants if varied else 1)]
        return boards, varied

    def fake_eval(net, boards, seeds=(0, 1, 2), **kw):
        calls.append((list(boards), tuple(seeds)))
        eps = [types.SimpleNamespace(steps=10) for _ in range(len(boards) * len(seeds))]
        return {"goal_rate": 0.5, "mean_score": 50.0, "mean_score_completed": 99.0,
                "mean_reward": 1.0, "episodes": eps}

    monkeypatch.setattr(rr, "load_window", fake_window)
    monkeypatch.setattr(rr, "evaluate_policy", fake_eval)


def _main(rr, argv):
    return rr.main(argv)


def test_main_이_row_json_과_체크포인트를_쓴다(rr, tmp_path, monkeypatch):
    run = _fake_run(tmp_path, rounds=3)
    init = _init_ckpt(tmp_path)
    calls = []
    _fake_eval(monkeypatch, rr, calls)
    out = tmp_path / "out"
    rc = _main(rr, ["--run", str(run), "--upto", "2", "--init", init, "--anchor-coef", "3",
                    "--batch-seed", "5", "--epochs", "2", "--out", str(out), "--device", "cpu",
                    "--eval-stage", "stage1", "--eval-stage", "stage3b"])
    assert rc == 0
    row = json.loads((out / "row.json").read_text(encoding="utf-8"))
    assert os.path.exists(out / "policy.pt")
    assert row["upto"] == 2 and row["batch_seed"] == 5 and row["anchor_coef"] == 3.0
    assert row["samples"] == 3 * 2 * 40
    assert row["train"]["updates"] == 2 * 1          # 240 행 / 256 → 에폭당 1 배치
    assert row["max_updates"] is None and row["epochs"] == 2
    assert row["window"] == "v8~v11"
    assert row["eval"]["stage1"]["window"] == "original"
    assert row["eval"]["stage3b"]["window"] == "v8~v11"
    assert row["eval"]["stage3b"]["episodes"] == 4 * 3
    assert row["compare"] is None


def test_예산이_8에폭보다_크면_에폭을_늘려_예산을_채운다(rr, tmp_path, monkeypatch):
    run = _fake_run(tmp_path, rounds=3, n=300)       # 라운드당 600 행
    init = _init_ckpt(tmp_path)
    _fake_eval(monkeypatch, rr, [])
    out = tmp_path / "out"
    # D 조건 모양: D1(라운드 0~1, 1200 행 → 에폭당 5 배치) 을 D2(1800 행 → 8 에폭 = 64 걸음) 예산까지
    rc = _main(rr, ["--run", str(run), "--upto", "1", "--init", init, "--anchor-coef", "0",
                    "--batch-seed", "1", "--updates-like", "2", "--out", str(out),
                    "--device", "cpu", "--no-eval"])
    assert rc == 0
    row = json.loads((out / "row.json").read_text(encoding="utf-8"))
    assert row["max_updates"] == 8 * 8 and row["updates_like"] == 2
    assert row["epochs"] == 13                         # ceil(64 / 5)
    assert row["train"]["updates"] == 64
    assert row["eval"] is None


def test_updates_like_가_더_작으면_중간에_멈춘다(rr, tmp_path, monkeypatch):
    run = _fake_run(tmp_path, rounds=3, n=300)
    init = _init_ckpt(tmp_path)
    out = tmp_path / "out"
    # C 조건 모양: D2(1800 행, 에폭당 8 배치) 를 D1 예산(8 × 5 = 40 걸음)에서 멈춘다
    rc = _main(rr, ["--run", str(run), "--upto", "2", "--init", init, "--anchor-coef", "0",
                    "--batch-seed", "1", "--updates-like", "1", "--out", str(out),
                    "--device", "cpu", "--no-eval"])
    assert rc == 0
    row = json.loads((out / "row.json").read_text(encoding="utf-8"))
    assert row["max_updates"] == 40 and row["epochs"] == 8 and row["train"]["updates"] == 40


def test_compare_는_같은_조리법이면_같다고_말한다(rr, tmp_path):
    run = _fake_run(tmp_path, rounds=2)
    init = _init_ckpt(tmp_path)

    def args(seed, out):
        return ["--run", str(run), "--upto", "1", "--init", init, "--anchor-coef", "3",
                "--batch-seed", str(seed), "--epochs", "1", "--device", "cpu", "--no-eval",
                "--out", str(tmp_path / out)]

    ref = str(tmp_path / "a" / "policy.pt")
    assert _main(rr, args(2, "a")) == 0
    assert _main(rr, args(2, "b") + ["--compare", ref]) == 0
    row = json.loads((tmp_path / "b" / "row.json").read_text(encoding="utf-8"))
    assert row["compare"] == {"ckpt": ref, "identical": True}
    assert _main(rr, args(3, "c") + ["--compare", ref]) == 0
    row = json.loads((tmp_path / "c" / "row.json").read_text(encoding="utf-8"))
    assert row["compare"]["identical"] is False


def test_out_에_row_json_이_있으면_거부한다(rr, tmp_path):
    run = _fake_run(tmp_path, rounds=2)
    init = _init_ckpt(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "row.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _main(rr, ["--run", str(run), "--upto", "1", "--init", init, "--anchor-coef", "0",
                   "--batch-seed", "0", "--out", str(out), "--device", "cpu", "--no-eval"])
    assert e.value.code == 2
    assert (out / "row.json").read_text(encoding="utf-8") == "{}"


def test_max_updates_와_updates_like_는_같이_못_준다(rr, tmp_path):
    with pytest.raises(SystemExit) as e:
        _main(rr, ["--run", str(tmp_path), "--upto", "1", "--init", "x", "--anchor-coef", "0",
                   "--batch-seed", "0", "--out", str(tmp_path / "o"), "--max-updates", "5",
                   "--updates-like", "0"])
    assert e.value.code == 2


def test_dry_run_은_아무것도_안_쓴다(rr, tmp_path, capsys):
    run = _fake_run(tmp_path, rounds=3)
    init = _init_ckpt(tmp_path)
    out = tmp_path / "out"
    rc = _main(rr, ["--run", str(run), "--upto", "2", "--init", init, "--anchor-coef", "3",
                    "--batch-seed", "5", "--updates-like", "1", "--out", str(out),
                    "--device", "cpu", "--dry-run"])
    assert rc == 0 and not out.exists()
    text = capsys.readouterr().out
    assert "라운드 0~2" in text and "v8~v11" in text and "예산" in text


def test_스크립트를_명령줄로_부를_수_있다(tmp_path):
    # 임포트 경로·argv 처리가 실제 실행에서도 되는지(가짜 없이 --dry-run 만)
    run = _fake_run(tmp_path, rounds=2)
    init = _init_ckpt(tmp_path)
    p = subprocess.run([PY, SCRIPT, "--run", str(run), "--upto", "1", "--init", init,
                        "--anchor-coef", "0", "--batch-seed", "0", "--out", str(tmp_path / "o"),
                        "--dry-run"], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert p.returncode == 0, p.stderr
