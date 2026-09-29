import importlib.util
import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_diag_stall():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다

    (`tests/rl/test_train_ppo.py::_load_train_ppo_module` 과 같은 패턴).
    """
    path = os.path.join(REPO, "scripts", "diag_stall.py")
    spec = importlib.util.spec_from_file_location("diag_stall_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tail이_뒤쪽_n개를_가져온다():
    """'정체 직전 100 걸음' 이 **뒤쪽** 이어야 한다 — 앞쪽을 보면 진단이 뒤집힌다.

    M4e 진단("정체 직전 가속 −0.81~−0.87")은 이 함수가 판의 **끝** 을 본다는 데 전부를 건다.
    최종 리뷰가 `rows[:n]` 으로 뒤집었을 때 기존 테스트는 아무것도 못 잡았다.
    """
    m = _load_diag_stall()
    rows = [{"a": i} for i in range(10)]
    assert m._tail(rows, "a", 3) == [7, 8, 9]
    assert m._tail(rows, "a", 100) == list(range(10))   # n 이 크면 전부
    assert m._tail([], "a", 3) == []


def test_정체_판만_고른다():
    """`stalled` 판만 골라야 한다 — 필터가 반전되면 '완주한 판의 가속' 을 정체라 부르게 된다."""
    m = _load_diag_stall()
    eps = [{"outcome": "goal", "rows": [1]}, {"outcome": "stalled", "rows": [2]},
           {"outcome": "timeout", "rows": [3]}, {"outcome": "stalled", "rows": [4]}]
    got = m._stalled_only(eps)
    assert [e["rows"][0] for e in got] == [2, 4]
    assert m._stalled_only([]) == []
    assert m._stalled_only([{"outcome": "goal", "rows": [1]}]) == []


def test_행동_축_인덱스가_조향0_가속1이다():
    """`ActionConfig` 의 순서와 같아야 한다 — 뒤집히면 진단이 통째로 거짓말한다.

    `vtd_rl/env/action.py::to_command` 가 `control[0]` 을 조향, `control[1]` 을 가속으로
    쓴다. 상수만 보지 않고 `_action_row` 가 **실제로 어느 칸을 읽는지** 까지 본다 —
    상수와 사용처 어느 쪽이 뒤집혀도 잡히게.
    """
    m = _load_diag_stall()
    assert (m.STEER_IDX, m.ACCEL_IDX) == (0, 1)
    row = m._action_row({"control": [0.3, -0.7]}, {"control": [0.4, -0.6]}, speed=0.5)
    assert row["steer_det"] == pytest.approx(0.3)
    assert row["accel_det"] == pytest.approx(-0.7)
    assert row["steer_sample"] == pytest.approx(0.4)
    assert row["accel_sample"] == pytest.approx(-0.6)
    assert row["speed_norm"] == pytest.approx(0.5)


def test_행동_축_인덱스가_action_to_command와_같다():
    """상수를 실제 환경 변환기(`to_command`)에 **먹여서** 확인한다.

    위 테스트는 0·1 이라는 숫자를 리터럴로 박아 둔 것뿐이라, 환경 쪽 축 순서가 바뀌면
    같이 틀린다. 여기서는 `STEER_IDX` 칸만 1 로 세운 행동을 `to_command` 에 넣어
    **조향만 움직이는지** 를 본다 — 진단과 환경이 같은 축을 가리킨다는 유일한 증거다.
    """
    from vtd_rl.env.action import ActionConfig, to_command

    m = _load_diag_stall()
    cfg = ActionConfig()
    control = [0.0, 0.0]
    control[m.STEER_IDX] = 1.0
    steer, accel, _turn = to_command({"control": control, "turn": 0}, cfg)
    assert steer == pytest.approx(cfg.max_steer)
    assert accel == pytest.approx(0.0)

    control = [0.0, 0.0]
    control[m.ACCEL_IDX] = 1.0
    steer, accel, _turn = to_command({"control": control, "turn": 0}, cfg)
    assert steer == pytest.approx(0.0)
    assert accel == pytest.approx(cfg.accel_max)


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
