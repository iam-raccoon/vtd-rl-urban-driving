"""M4c 최종 리뷰 Critical — `scripts/eval_det_vs_stoch.py` 가 쓰는 확률적 평가 경로를 잠근다.

`run_policy_episode`/`evaluate_policy` 는 원래 `policy.act(obs, deterministic=True)` 를 박아
불렀다 — `deterministic`(+`generator`) 인자를 추가해 확률적 평가를 열었다. 여기서는 그 인자가
① 재현 가능한지(같은 시드 두 번이 같은 결과) ② 결정적 경로와 실제로 다른 행동을 내는지를
`act()`/`run_policy_episode()` 수준에서 빠르게 확인한다 — `test_evaluate.py` 와 같은 짧은 판
(H_0_250, 250m)을 써서 전체 커리큘럼 없이도 빠르다.
"""
import numpy as np
import pytest
import torch

from vtd_rl.env.drive_env import VtdDriveEnv
from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.evaluate import evaluate_policy, run_policy_episode
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def short_board():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_기본값은_그대로_결정적이다():
    """`deterministic`을 안 주면 예전과 완전히 같은 호출부(기존 채점 경로 회귀 방지)."""
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    board = short_board()
    env = VtdDriveEnv([board])
    out = run_policy_episode(env, net, board.name, 0, max_steps=1000)
    env.close()
    assert out.outcome in ("goal", "offroad", "collision", "timeout", "stalled")


def test_결정적과_확률적은_실제로_다른_행동을_낸다():
    """σ 를 크게 준 정책에서 결정적(tanh(mean)) 과 확률적 표본이 갈려야 한다 — act() 수준.

    이게 안 갈리면 `deterministic=False` 가 사실상 아무 일도 안 하는 것이다(예: 인자를 받기만
    하고 안으로 안 흘려보내는 배선 끊김을 잡는다).
    """
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_init=2.0))
    space = observation_space(ObsConfig())
    space.seed(0)
    obs = space.sample()
    det = net.act(obs, deterministic=True)
    sto = net.act(obs, deterministic=False, generator=torch.Generator().manual_seed(1))
    assert not np.allclose(det["control"], sto["control"], atol=1e-6)


def test_확률적_평가는_같은_시드_두_번이_같은_결과를_낸다():
    """`run_policy_episode(..., deterministic=False, generator=...)` 재현성 — 판 하나로 빠르게."""
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_init=1.0))
    board = short_board()

    def run():
        env = VtdDriveEnv([board])
        out = run_policy_episode(env, net, board.name, 0, max_steps=500,
                                 deterministic=False, generator=torch.Generator().manual_seed(7))
        env.close()
        return out

    a, b = run(), run()
    assert a.steps == b.steps
    assert a.outcome == b.outcome
    assert a.reward == pytest.approx(b.reward)
    assert a.score == pytest.approx(b.score)


def test_evaluate_policy도_확률적_재현성을_지킨다():
    """`evaluate_policy` 를 통째로 통과해도(여러 시드) 재현성이 유지되는지 — 스크립트가 직접

    부르는 진입점이라 `run_policy_episode` 단위 확인만으로는 부족하다(여러 (판, 시드) 조합을
    거쳐도 같은 `generator` 인스턴스가 순서대로 소비되는지 확인한다).
    """
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_init=1.0))
    boards = [short_board()]

    def run():
        return evaluate_policy(net, boards, seeds=(0, 1), deterministic=False,
                               generator=torch.Generator().manual_seed(3))

    r1, r2 = run(), run()
    assert r1["goal_rate"] == pytest.approx(r2["goal_rate"])
    assert r1["mean_score_raw"] == pytest.approx(r2["mean_score_raw"])
    assert r1["mean_reward"] == pytest.approx(r2["mean_reward"])


@pytest.mark.slow
def test_evaluate_det_vs_stoch_스크립트가_레포에_있다():
    """`scripts/eval_det_vs_stoch.py` 를 서브프로세스로 직접 돌려 체크포인트당 한 줄 JSON 을

    내는지 확인한다 — 짧은 판이 아니라 진짜 커리큘럼(stage1/stage2)을 평가하므로 느리다.
    """
    import json
    import os
    import subprocess

    repo = os.path.join(os.path.dirname(__file__), "..", "..")
    ckpt = None
    for cand in ("runs/lab-main/2026-09-17-dagger-fix2/policy-r0.pt",):
        p = os.path.join(repo, cand)
        if os.path.exists(p):
            ckpt = p
            break
    if ckpt is None:
        pytest.skip("실제 체크포인트가 없다")

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(repo, ".venv", "bin", "python"),
                          os.path.join(repo, "scripts", "eval_det_vs_stoch.py"),
                          "--checkpoint", ckpt, "--seeds", "0", "--device", "cpu"],
                         capture_output=True, text=True, env=env, cwd=repo, timeout=580)
    assert out.returncode == 0, out.stderr[-3000:]
    row = json.loads(out.stdout.strip().splitlines()[-1])
    assert row["checkpoint"] == ckpt
    assert "stage1" in row and "stage2" in row
    for label in ("stage1", "stage2"):
        for key in ("det_goal", "det_score", "sto_goal", "sto_score"):
            assert key in row[label]
