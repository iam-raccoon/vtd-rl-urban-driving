"""M4c 최종 리뷰 Critical — `scripts/eval_det_vs_stoch.py` 가 쓰는 확률적 평가 경로를 잠근다.

`run_policy_episode`/`evaluate_policy` 는 원래 `policy.act(obs, deterministic=True)` 를 박아
불렀다 — `deterministic`(+`generator`) 인자를 추가해 확률적 평가를 열었다. 여기서는 그 인자가
① 재현 가능한지(같은 시드 두 번이 같은 결과) ② 결정적 경로와 실제로 다른 행동을 내는지를
`act()`/`run_policy_episode()` 수준에서 빠르게 확인한다 — `test_evaluate.py` 와 같은 짧은 판
(H_0_250, 250m)을 써서 전체 커리큘럼 없이도 빠르다.

M4i 에서 `--stage`(반복 가능)를 더했다 — 아래 "단계를 고르는 길" 절이 그 배선을 잠근다.
잠그는 것 셋: **기본값이 예전 그대로 stage1+stage2** 인가(과거 성적표와의 비교가 여기 걸려
있다), `--stage stage3` 가 **진짜 그 커리큘럼**으로 풀리는가, 모르는 단계 이름이 **조용히
아무것도 안 재는 대신 터지는가**. 셋 다 틀려도 스크립트는 오류 없이 그럴듯한 JSON 한 줄을
내므로 사람이 읽어서는 못 잡는다(`tests/test_measure_teacher.py` 와 같은 이유).
"""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from vtd_rl.env.drive_env import VtdDriveEnv
from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.evaluate import evaluate_policy, run_policy_episode
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def short_board():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


class _SpyPolicy:
    """`act()` 가 어떤 `deterministic` 으로 불렸는지 받아 적는다."""

    def __init__(self):
        self.seen = []

    def act(self, obs, deterministic=True, generator=None):
        self.seen.append(deterministic)
        return {"control": np.zeros(2, np.float32), "turn": 0}


class _FakeEnv:
    """`run_policy_episode` 가 쓰는 만큼만 흉내 낸다 — VTD 월드를 안 띄워 빠르다."""

    def __init__(self, steps=3):
        self.steps, self.n = steps, 0

    def reset(self, seed=None, options=None):
        self.n = 0
        return {}, {}

    def step(self, action):
        self.n += 1
        done = self.n >= self.steps
        info = {"outcome": "goal", "result": {"score": [100.0], "sheet": []}} if done else {}
        return {}, 0.0, done, False, info

    def close(self):
        pass


def test_기본값은_그대로_결정적이다():
    """**기본값이 `deterministic=True` 로 정책까지 닿는지**를 값으로 잠근다.

    2026-09-27 재리뷰: 원래 이 테스트는 `outcome` 이 올바른 문자열인지만 봤다 —
    `run_policy_episode`·`evaluate_policy` 의 기본값을 **둘 다 `False` 로 뒤집어도 스위트
    457 개가 전부 통과**했다. 이름이 약속하는 것을 안 지키는 테스트는 없는 것보다 나쁘다.
    이 경로는 M1 부터의 모든 성적표가 지나는 채점 경로라 기본값이 조용히 바뀌면 과거 성적표와의
    비교가 통째로 무너진다.
    """
    spy = _SpyPolicy()
    out = run_policy_episode(_FakeEnv(), spy, "판", 0, max_steps=10)
    assert spy.seen and all(d is True for d in spy.seen), spy.seen
    assert out.outcome == "goal"


def test_evaluate_policy_도_기본값이_결정적이다(monkeypatch):
    """`evaluate_policy` 가 그 기본값을 `run_policy_episode` 로 그대로 넘기는지 잠근다."""
    import vtd_rl.policy.evaluate as ev

    seen = []

    def _spy_episode(env, policy, board_name, seed, max_steps=20000,
                     deterministic=True, generator=None):
        seen.append((deterministic, generator))
        return ev.EpisodeOutcome(board_name, seed, "goal", 1, 0.0, 100.0, [])

    monkeypatch.setattr(ev, "run_policy_episode", _spy_episode)
    monkeypatch.setattr(ev, "VtdDriveEnv", lambda *a, **k: _FakeEnv())
    ev.evaluate_policy(_SpyPolicy(), [short_board()], seeds=(0,))
    assert seen == [(True, None)], seen


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


# ------------------------------------------------------------------ 단계를 고르는 길(M4i)
def _script():
    """스크립트를 모듈로 불러온다(`tests/test_measure_teacher.py::_load` 와 같은 패턴)."""
    path = os.path.join(REPO, "scripts", "eval_det_vs_stoch.py")
    spec = importlib.util.spec_from_file_location("eval_det_vs_stoch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ep:
    """`policy.evaluate.EpisodeOutcome` 중 이 스크립트가 읽는 칸만."""

    def __init__(self, outcome="goal"):
        self.outcome = outcome


class _Net:
    """`load_policy` 가 주는 것 중 이 스크립트가 읽는 칸만."""

    log_std = torch.zeros(2)

    def eval(self):
        return self


def _stub_run(monkeypatch, m):
    """`main()` 에서 실제 주행만 들어낸다 — 남는 것이 '어느 단계를 어떻게 넘기는가' 배선이다.

    돌려주는 리스트에는 `evaluate_policy` 호출마다 (판 개수, 판 이름들) 이 쌓인다.
    """
    seen = []

    def fake_evaluate(policy, boards, seeds=(0, 1, 2), config=None,
                      deterministic=True, generator=None):
        seen.append((len(boards), tuple(b.name for b in boards)))
        return {"goal_rate": 1.0, "mean_score_raw": 90.0, "mean_score_completed": 90.0,
                "episodes": [_Ep()]}

    monkeypatch.setattr(m, "evaluate_policy", fake_evaluate)
    monkeypatch.setattr(m, "load_policy", lambda path, dev: _Net())
    return seen


def _row(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_기본은_예전_그대로_stage1과_stage2다(monkeypatch, capsys):
    """★ `--stage` 를 더하면서 **기본 동작이 바뀌지 않았음**을 잠근다.

    M1 부터의 모든 성적표(JSONL 키 `stage1`·`stage2`)가 이 기본값으로 나왔다. 기본이
    한 단계로 줄거나 stage3 가 끼어들면 과거 숫자와의 비교가 통째로 무너진다.
    """
    m = _script()
    _stub_run(monkeypatch, m)
    assert m.main(["--checkpoint", "없는.pt"]) == 0
    row = _row(capsys)
    assert [k for k in row if k not in ("checkpoint", "log_std")] == ["stage1", "stage2"]


def test_stage3를_주면_진짜_그_단계를_잰다(monkeypatch, capsys):
    """★ `--stage` 가 조용히 무시되면(언제나 stage1) 여기서 걸린다.

    판 **이름**은 세 단계가 똑같으므로 이름으로는 못 가른다 — JSONL 키(이름표)로 가른다.
    """
    m = _script()
    _stub_run(monkeypatch, m)
    assert m.main(["--checkpoint", "없는.pt", "--stage", "stage3"]) == 0
    row = _row(capsys)
    assert [k for k in row if k not in ("checkpoint", "log_std")] == ["stage3"]


def test_단계를_여러_번_줄_수_있고_준_순서대로_나온다(monkeypatch, capsys):
    """`action="append"` 의 기본값 함정도 같이 잠근다 — 기본 리스트에 덧붙으면 4 개가 된다."""
    m = _script()
    _stub_run(monkeypatch, m)
    assert m.main(["--checkpoint", "없는.pt", "--stage", "stage3", "--stage", "stage1"]) == 0
    row = _row(capsys)
    assert [k for k in row if k not in ("checkpoint", "log_std")] == ["stage3", "stage1"]


def test_단계_이름은_그_단계의_커리큘럼으로_풀린다():
    """이름표만으로는 부족하다 — 실제로 그 파일의 판이 왔는지 액터·신호로 확인한다.

    (`tests/test_measure_teacher.py::test_단계를_이름으로_부르면_그_단계의_판이_온다` 와 같은
    방법. 세 단계는 판 이름이 같고 신호·액터만 다르다.)
    """
    m = _script()
    assert [label for label, _ in m._stages()] == ["stage1", "stage2"]
    assert all(sum(len(b.scenario.actors) for b in boards) == 0 for _, boards in m._stages())
    (label, boards), = m._stages(["stage3"])
    assert label == "stage3"
    assert sum(len(b.scenario.actors) for b in boards) >= 18
    assert {b.signals for b in boards} == {"cycle"}
    # 경로를 직접 줘도 된다(임시 커리큘럼으로 배치를 시험할 때 필요하다)
    direct = os.path.join(REPO, "curricula", "stage3.json")
    assert [label for label, _ in m._stages([direct])] == ["stage3"]


def test_모르는_단계_이름은_터진다(monkeypatch):
    """★ 조용히 아무것도 안 재는 것이 최악이다 — 빈 성적표는 "완주 0/0" 이라 읽을 수가 없다.

    `_stages()` 수준과 `main()` 수준을 **둘 다** 본다. `main()` 이 예외를 삼키면
    체크포인트만 적힌 한 줄이 나오고, 그게 진짜 측정처럼 보인다.
    """
    m = _script()
    with pytest.raises(ValueError, match="stage없음"):
        m._stages(["stage없음"])
    seen = _stub_run(monkeypatch, m)
    with pytest.raises(ValueError, match="stage없음"):
        m.main(["--checkpoint", "없는.pt", "--stage", "stage없음"])
    assert seen == []       # 터지기 전에 아무것도 재면 안 된다


def test_단계는_따로_평가한다_판을_합치지_않는다(monkeypatch):
    """★ 세계 캐시 보호 — 판 이름이 단계끼리 같아서 합쳐 넘기면 `VtdDriveEnv` 가 죽는다.

    (`drive_env.py:89-96` 의 `assert`). 한 번에 12 판을 넘기는 순간 그 단언이 터지거나,
    더 나쁘게는 다른 단계의 세계로 잰 숫자가 나온다. 단계마다 6 판씩 따로 불러야 한다.
    """
    m = _script()
    seen = _stub_run(monkeypatch, m)
    assert m.main(["--checkpoint", "없는.pt", "--stage", "stage1", "--stage", "stage3"]) == 0
    assert len(seen) == 4                       # (단계 2) × (결정적·확률적)
    assert [n for n, _ in seen] == [6, 6, 6, 6], seen
    assert len({names for _, names in seen}) == 1   # 단계끼리 판 이름은 같다(그래서 위험하다)


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
