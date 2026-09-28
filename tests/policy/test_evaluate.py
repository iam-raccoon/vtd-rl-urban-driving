import pytest
import torch

from vtd_rl.policy.evaluate import (EpisodeOutcome, _summary, evaluate_policy, evaluate_teacher,
                                    run_policy_episode)
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.env.drive_env import VtdDriveEnv
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def short_board():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_학습_전_학생도_판을_끝낸다():
    # 학습 전 신경망 초기값은 무작위다 — 이 시드는 재현성만을 위한 것이고, 테스트 자체는
    # 학습 전 정책이 어떤 결과(goal/offroad/collision/timeout/stalled)에 이르든 통과해야
    # 한다. 세계는 60 초(10 Hz 기준 ~600 스텝) 진행이 없으면 stalled 로 끝나므로, 1000
    # 스텝이면 초기화가 무엇이든 반드시 어떤 결과로든 끝난다(구조적 종료 — 운 좋은 시드에
    # 기대지 않는다. seed 0~5 에서 모두 확인함, task-7-report.md 참고).
    torch.manual_seed(0)
    board = short_board()
    env = VtdDriveEnv([board])
    out = run_policy_episode(env, DrivePolicy(PolicyConfig(trunk=(32, 32))), board.name, 0,
                             max_steps=1000)
    env.close()
    assert out.outcome in ("goal", "offroad", "collision", "timeout", "stalled")
    assert out.steps > 0 and len(out.sheet) == 5


def test_평가_요약():
    torch.manual_seed(0)      # 학습 전 신경망 초기값은 무작위다 — 결과 재현을 위해 고정한다
    boards = [short_board()]
    res = evaluate_policy(DrivePolicy(PolicyConfig(trunk=(32, 32))), boards, seeds=(0, 1))
    assert 0.0 <= res["goal_rate"] <= 1.0 and len(res["episodes"]) == 2
    assert res["mean_score"] <= 100.0


@pytest.mark.slow
def test_선생님_기준():
    res = evaluate_teacher([short_board()], seeds=(0,))
    assert res["goal_rate"] == 1.0 and res["mean_score"] > 80.0


def _fake_episode(score: float, outcome: str = "goal") -> EpisodeOutcome:
    return EpisodeOutcome("판", 0, outcome, 10, 1.0, score, [])


def test_mean_score_completed은_완주_판만의_평균이다():
    """M4d 최종 리뷰 Critical 1: `mean_score_raw` 는 미완주 판도 그 판 점수 그대로 섞는데,

    세계가 안 밟은 구간까지 5구간 전부 채점하고 미방문 구간은 100점을 줘서(`evaluate.py:24`
    가 이미 적어 둔 함정) 일찍 멈춘 정책일수록 그 값이 오히려 높아진다. `mean_score_completed`
    는 `outcome == "goal"` 인 판만의 평균이어야 한다.
    """
    episodes = [_fake_episode(90.0, "goal"), _fake_episode(100.0, "stalled"),
               _fake_episode(80.0, "goal")]
    summary = _summary(episodes)
    assert summary["mean_score_completed"] == pytest.approx((90.0 + 80.0) / 2)
    # 기존 키는 이번 변경으로 안 바뀐다 — `mean_score_raw` 는 미완주 판도 그대로, `mean_score`
    # 는 미완주 판을 0점으로 친다(둘 다 기존 동작).
    assert summary["mean_score_raw"] == pytest.approx((90.0 + 100.0 + 80.0) / 3)
    assert summary["mean_score"] == pytest.approx((90.0 + 0.0 + 80.0) / 3)


def test_완주_판이_없으면_mean_score_completed는_None이다():
    episodes = [_fake_episode(100.0, "stalled"), _fake_episode(90.0, "offroad")]
    summary = _summary(episodes)
    assert summary["mean_score_completed"] is None
    # 미완주 판만 있어도 기존 키는 그대로 계산된다.
    assert summary["mean_score_raw"] == pytest.approx(95.0)
    assert summary["mean_score"] == pytest.approx(0.0)
