import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(REPO, "runs", "lab-main", "2026-09-17-dagger-fix2", "data")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_재적합이_스쿼시_체크포인트를_만든다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "refit_m3.py"),
                          "--data", DATA, "--out", str(tmp_path / "policy.pt"),
                          "--epochs", "1", "--seed", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["epochs"] == 1 and summary["samples"] > 0

    from vtd_rl.policy.net import DrivePolicy
    net = DrivePolicy.load(str(tmp_path / "policy.pt"))
    assert net.cfg.squash is True          # 스쿼시 매개화로 저장됐다

    # 결정적 행동이 항상 상자 안이다 — 이게 이 체크포인트의 존재 이유다
    import torch
    from vtd_rl.env.observation import ObsConfig, observation_space
    from vtd_rl.policy.encode import stack_obs, to_tensors
    sp = observation_space(ObsConfig()); sp.seed(0)
    vec, objs, mask = to_tensors(*stack_obs([sp.sample() for _ in range(64)]),
                                 torch.device("cpu"))
    with torch.no_grad():
        action = torch.tanh(net(vec, objs, mask)[0])
    assert torch.all(action.abs() <= 1.0)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_같은_폴더에_두_번_쓰지_않는다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    args = [os.path.join(REPO, ".venv", "bin", "python"),
            os.path.join(REPO, "scripts", "refit_m3.py"),
            "--data", DATA, "--out", str(tmp_path / "policy.pt"), "--epochs", "1"]
    assert subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO,
                          timeout=1800).returncode == 0
    second = subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO, timeout=300)
    assert second.returncode != 0 and "이미" in (second.stdout + second.stderr)
