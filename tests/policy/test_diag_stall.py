import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
CK = os.path.join(REPO, "runs", "omen", "2026-09-26-m3-squash-warm", "policy.pt")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(CK), reason="스쿼시 재적합 체크포인트가 있어야 한다")
def test_정체_진단이_판별_요약을_낸다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "diag_stall.py"),
                          "--checkpoint", CK, "--stage", "stage1", "--seeds", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["checkpoint"] == CK
    assert summary["episodes"] >= 1
    # 이 체크포인트는 정체가 없다(M4d 실측: 18/18 완주) — 그래도 필드는 있어야 한다
    assert "stalled" in summary and "accel" in summary
    assert set(summary["accel"]) >= {"policy_mean", "teacher_mean", "sample_mean"}
