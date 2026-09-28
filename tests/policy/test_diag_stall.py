import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


@pytest.mark.slow
def test_정체_진단이_판별_요약을_낸다(tmp_path):
    """**체크포인트를 그 자리에서 만들어** 쓴다 — `runs/` 에 있는 것을 게이트로 걸지 않는다.

    2026-09-28: 원래 이 테스트는 `runs/omen/2026-09-26-m3-squash-warm/policy.pt` 가 있어야
    돌게 돼 있었는데 그 파일은 **학습 머신(OMEN)에만** 있다. 개발 머신에서는 `skipif` 가
    항상 참이라 **구현 전후 모두 SKIP** 이었다 — 한 번도 초록을 본 적이 없는 테스트였다.
    무작위 초기화 정책이라 주행은 엉망이지만, 이 테스트가 보는 것은 **요약의 모양**이다.
    """
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig

    ck = str(tmp_path / "policy.pt")
    DrivePolicy(PolicyConfig(squash=True)).save(ck)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "diag_stall.py"),
                          "--checkpoint", ck, "--stage", "stage1", "--seeds", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["checkpoint"] == ck
    assert summary["episodes"] >= 1
    # 무작위 정책이라 완주하든 정체하든 상관없다 — 요약의 **모양**만 본다
    assert "stalled" in summary and "accel" in summary
    assert set(summary["accel"]) >= {"policy_mean", "teacher_mean", "sample_mean"}
