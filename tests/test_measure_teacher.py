"""`scripts/measure_teacher.py` 를 잠근다 — 단계 ③ 성적표의 숫자가 전부 여기서 나온다.

빠른 테스트는 **단계를 고르는 길**과 **완주를 세는 길** 둘을 본다. 그 둘이 이 스크립트에서
조용히 틀릴 수 있는 지점이다 — 엉뚱한 커리큘럼을 읽으면 단계 ③ 성적표에 단계 ① 숫자가
찍히고, 완주를 느슨하게 세면 못 달리는 배치가 100% 로 보인다. 둘 다 오류 없이 **그럴듯한
숫자**를 내므로 사람이 읽어서는 못 잡는다.

느린 테스트는 판을 실제로 몬다 — `@pytest.mark.slow`.
"""
import importlib.util
import json
import os

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")


def _load():
    """스크립트를 모듈로 불러온다(`tests/policy/test_diag_stall.py` 와 같은 패턴)."""
    path = os.path.join(REPO, "scripts", "measure_teacher.py")
    spec = importlib.util.spec_from_file_location("measure_teacher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ep:
    """`policy.evaluate.EpisodeOutcome` 중 이 스크립트가 읽는 칸만."""

    def __init__(self, board, outcome, steps=100, score=90.0):
        self.board, self.outcome, self.steps, self.score = board, outcome, steps, score
        self.seed, self.reward, self.sheet = 0, 0.0, []


# ------------------------------------------------------------------ 단계를 고르는 길
def test_단계_이름이_그_단계의_커리큘럼_파일로_풀린다():
    """★ 여기가 틀리면 단계 ③ 성적표에 단계 ① 숫자가 찍힌다 — 오류 없이, 그럴듯하게."""
    m = _load()
    for name in ("stage1", "stage2", "stage3"):
        assert os.path.basename(m.stage_path(name)) == f"{name}.json"
        assert os.path.exists(m.stage_path(name)), name
    # 경로를 직접 주면 그대로 쓴다(임시 커리큘럼으로 배치를 시험할 때 필요하다)
    direct = os.path.join(REPO, "curricula", "stage2.json")
    assert m.stage_path(direct) == direct


def test_단계를_이름으로_부르면_그_단계의_판이_온다():
    """`stage_path` 만으로는 부족하다 — `load_stage` 가 그 경로를 실제로 쓰는지 본다.

    세 단계는 판 이름이 **같고** 신호·액터만 다르다. 그래서 판 이름이 아니라
    신호 운용과 액터 수로 확인해야 어느 파일을 읽었는지 갈린다.
    """
    m = _load()
    _, s1 = m.load_stage("stage1")
    _, s2 = m.load_stage("stage2")
    _, s3 = m.load_stage("stage3")
    assert [b.signals for b in s1] == ["always_green"] * 6
    assert [b.signals for b in s2] == ["cycle"] * 6
    assert [b.signals for b in s3] == ["cycle"] * 6
    assert sum(len(b.scenario.actors) for b in s1) == 0
    assert sum(len(b.scenario.actors) for b in s2) == 0
    assert sum(len(b.scenario.actors) for b in s3) >= 18


def test_판_이름_오타는_터진다():
    """`--boards` 오타로 판이 0 개면 완주율 0/0 짜리 빈 성적표가 나온다."""
    m = _load()
    with pytest.raises(ValueError, match="course_없음"):
        m.load_stage("stage3", only=["course_없음"])
    _, only = m.load_stage("stage3", only=["course_H"])
    assert [b.name for b in only] == ["course_H"]


# ------------------------------------------------------------------ 완주를 세는 길
def test_완주는_goal만_센다():
    """★ `collision` 은 단계 ③ 에서 **처음** 열리는 종료 사유다(`drive_env.py:142-143`).

    "goal 이 아닌 것" 을 느슨하게 세면 규칙 스택이 라바콘을 밟고 끝낸 판까지 완주로
    올라가고, 그러면 "선생님이 못 푸는 판" 이라는 신호가 통째로 사라진다.
    """
    m = _load()
    ev = {"episodes": [_Ep("course_A", "goal"), _Ep("course_B", "collision"),
                       _Ep("course_D", "timeout"), _Ep("course_E", "offroad"),
                       _Ep("course_G", "stalled"), _Ep("course_H", "goal")]}
    assert m.completion(ev) == (2, 6)
    assert m.completion({"episodes": []}) == (0, 0)
    assert m.completion({"episodes": [_Ep("course_A", "goal")] * 3}) == (3, 3)


def test_판별_줄은_판마다_따로_센다():
    """평균만 보면 어느 판이 못 달리는지 안 보인다 — 판마다 완주 수가 따로 나와야 한다."""
    m = _load()
    ev = {"episodes": [_Ep("course_A", "goal", 10, 100.0), _Ep("course_A", "collision", 20, 50.0),
                       _Ep("course_H", "goal", 30, 80.0)]}
    rows = {r["board"]: r for r in m.board_rows(ev)}
    assert rows["course_A"]["goal"] == 1 and rows["course_A"]["n"] == 2
    assert rows["course_A"]["mean_score"] == pytest.approx(75.0)
    assert rows["course_H"]["goal"] == 1 and rows["course_H"]["n"] == 1
    assert "collision(20보)" in rows["course_A"]["outcomes"]


def test_시드가_0이면_터진다():
    m = _load()
    with pytest.raises(ValueError, match="seeds"):
        m.measure("stage1", seeds=0)


# ------------------------------------------------------------------ 성적표 본문
def _fake_result(stage, label, completed, episodes, mean, violations):
    return {"stage": stage, "label": label, "seeds": 1,
            "lengths": {"course_H": 2807.0}, "actors": {"course_H": 4},
            "signals": {"course_H": "cycle"},
            "ev": {"mean_score": mean, "mean_score_completed": mean, "episodes": []},
            "completed": completed, "episodes": episodes,
            "rows": [{"board": "course_H", "goal": completed, "n": episodes,
                      "mean_score": mean, "outcomes": "goal(10보)"}],
            "violations": violations}


def test_성적표에_단계별_요약과_항목표가_들어간다():
    m = _load()
    text = "\n".join(m.report_lines([
        _fake_result("stage1", "빈 경로", 1, 1, 99.6, {}),
        _fake_result("stage3", "사물·정지차", 1, 1, 96.8, {11: {"minor": 2, "major": 0},
                                                          13: {"minor": 7, "major": 0}}),
    ]))
    assert "`stage1`" in text and "`stage3`" in text
    assert "99.6" in text and "96.8" in text
    assert "⑪ 장애물 충돌" in text and "⑬ 차로변경 지시등" in text
    # 단계 ① 에 없던 항목도 0 으로 자리를 잡아야 두 단계를 나란히 읽을 수 있다
    assert text.count("| ⑪ 장애물 충돌 | 0 | 0 | 2 | 0 |") == 1


# ------------------------------------------------------------------ 실제로 몰아 본다
@pytest.mark.slow
def test_선생님_측정이_임의_단계를_받는다():
    """단계 ① 전용이던 것을 일반화한 것이므로 단계 ① 에서 기존 값과 맞아야 한다.

    기존 값 = `docs/reports/m1-stage1-teacher.md`(완주 6/6) 와 `docs/reports/m4h-*` 가
    인용하는 선생님 점수 99.6. 점수는 판정이 바뀌면 움직일 수 있으니 폭을 준다 —
    이 테스트가 잠그는 것은 **완주 6/6** 과 **일반화된 경로가 단계 ① 을 제대로 읽는다**는
    것이다.
    """
    m = _load()
    r = m.measure("stage1", seeds=1)
    assert (r["completed"], r["episodes"]) == (6, 6), r["rows"]
    assert sum(r["actors"].values()) == 0
    assert 99.0 <= r["ev"]["mean_score"] <= 100.0, r["ev"]["mean_score"]


@pytest.mark.slow
def test_선생님이_단계3_판을_완주한다():
    """★ 규칙 스택이 못 푸는 판을 학생에게 주지 않는다는 보증(계획 Task 2 Step 5).

    선생님이 **감점을 받는 것은 정상이고 오히려 목표다**(그게 RL 이 이길 여지다).
    여기서 단언하는 것은 완주뿐이다.
    """
    m = _load()
    r = m.measure("stage3", seeds=1)
    # 액터가 실제로 실린 판을 쟀다는 것부터 확인한다 — 단계 ① 을 읽었으면 여기서 걸린다
    assert sum(r["actors"].values()) >= 18, r["actors"]
    assert set(r["signals"].values()) == {"cycle"}
    assert (r["completed"], r["episodes"]) == (6, 6), r["rows"]


@pytest.mark.slow
def test_못_달리는_배치는_완주로_안_센다(tmp_path):
    """★ 위 두 테스트가 "언제나 100%" 를 찍는 저울로 잰 것이 아님을 보인다.

    편도 1차선(코스 A s=1620, l=r=1.45) **한가운데** 에 라바콘을 놓는다 — 규칙 스택은
    비킬 차로가 없으면 비키지 않으므로(`drive.py:1838-1840`) 반드시 접촉하고,
    항목⑪ 은 `env/reward.py:17` 의 COLLISION_ITEMS 라 판이 그 자리에서 끝난다.
    """
    m = _load()
    bad = {"name": "막힌 판", "signals": "cycle", "boards": [
        {"name": "course_A", "route": "routes/HL_FMA_NEW_A.json",
         "lane": "routes/HL_FMA_NEW_A_lane.json",
         "actors": [{"id": 1, "kind": "static", "type": "obstacle", "s": 1620.0,
                     "lateral": 0.0, "size": [0.15, 0.46, 0.61]}]}]}
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    r = m.measure(str(path), seeds=1)
    assert r["completed"] == 0, r["rows"]
    assert r["rows"][0]["outcomes"].startswith("collision")
    assert m.main(["--stage", str(path), "--seeds", "1"]) == 1
