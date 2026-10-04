"""단계 ③c(학습용 남은 장면, M6g) — `tests/test_stage3.py` 의 "달릴 수 있는 배치" 규칙을 그대로 먹인다.

③c 는 M6c~M6f 그물이 ③ 전체에서 계속 부딪히던 장면만 모았다 — 정지차 바로 뒤(70 m) 라바콘(좌·우)과
오른쪽 라바콘. **학습에만 쓰고 평가에는 안 쓴다.** 그래서 ③ 전체(와 ③a·③b) 액터의 원래 자리에서
150 m 이상 떨어뜨렸다 — 같은 자리에 같은 장면을 놓으면 ③ 전체 평가가 본 장면을 재게 된다.
"""
import bisect
import json

from vtd_rl.world.board import load_curriculum

PATH = "curricula/stage3c.json"
EGO_HALF_W = 0.943      # drive.py:141 HALF_WIDTH — 자차 반폭[m]
NUDGE_CLEAR = 0.40      # drive.py:147 — 비켜 갈 때 남기는 여유[m]
NUDGE_MAX_DIM = 1.2     # drive.py:144 — 이보다 크면 nudge 가 아니라 추월 FSM 담당
AFTER_VEHICLE_M = 60.0  # tests/test_stage3.py 와 같다 — 정지차 추월 뒤 복귀 거리
MIN_GAP_M = 10.0
FAR_FROM_EVAL_M = 150.0
TRAIN_VARIANTS = 4      # train_ppo --train-variants 4 가 쓰는 v0~v3


def _need(plan, size):
    big = max(size[0], size[1])
    if big < NUDGE_MAX_DIM:
        return EGO_HALF_W + big / 2.0 + NUDGE_CLEAR + EGO_HALF_W
    return plan["w"] + EGO_HALF_W


def _s_of(board, actor):
    x, y = actor.motion["pos"]
    return board.route.project(x, y).s


def test_단계3c_는_여섯_코스를_덮고_정지물만_쓴다():
    with open(PATH, encoding="utf-8") as f:
        d = json.load(f)
    assert d["signals"] == "cycle"
    assert sorted(e["name"] for e in d["boards"]) == [
        "course_A", "course_B", "course_D", "course_E", "course_G", "course_H"]
    kinds = {(a["kind"], a["type"]) for e in d["boards"] for a in e["actors"]}
    assert kinds == {("static", "vehicle"), ("static", "obstacle")}
    assert all(e["actors"] for e in d["boards"])


def test_단계3c_는_평가_액터_자리에서_150m_떨어져_있다():
    """같은 코스의 ③·③a·③b 액터 원래 자리(s)에서 150 m 안에 ③c 액터가 없다(흔들기 ±40 m 전)."""
    with open(PATH, encoding="utf-8") as f:
        mine = {e["name"]: [a["s"] for a in e["actors"]] for e in json.load(f)["boards"]}
    for other in ("curricula/stage3.json", "curricula/stage3a.json", "curricula/stage3b.json"):
        with open(other, encoding="utf-8") as f:
            for e in json.load(f)["boards"]:
                for a in e["actors"]:
                    for s in mine[e["name"]]:
                        assert abs(s - a["s"]) >= FAR_FROM_EVAL_M, (other, e["name"], a["s"], s)


def test_단계3c_변종도_달릴_수_있는_배치다():
    """경로 앞뒤 100 m · 비킬 차로 · 정지차 뒤 60 m 를 학습 변종 v0~v3 전부에 먹인다."""
    _, vs = load_curriculum(PATH, variants=TRAIN_VARIANTS)
    assert len(vs) == 6 * TRAIN_VARIANTS
    for b in vs:
        acts = sorted(b.scenario.actors, key=lambda a: _s_of(b, a))
        for a in acts:
            s = _s_of(b, a)
            assert 100.0 <= s <= b.route.total - 100.0, (b.name, a.id, s)
            i = max(0, bisect.bisect_right(b.route.cum, s) - 1)
            plan = b.lane_plan[min(i, len(b.lane_plan) - 1)]
            need = _need(plan, a.size)
            assert max(plan["l"], plan["r"]) >= need, (
                f"{b.name} 액터{a.id} s={s:.0f}: l={plan['l']:.2f} r={plan['r']:.2f} — {need:.2f} m 가 필요하다")
        for prev, nxt in zip(acts, acts[1:]):
            gap = _s_of(b, nxt) - _s_of(b, prev)
            need = AFTER_VEHICLE_M if prev.size[0] > 2.2 else MIN_GAP_M
            assert gap >= need - 1e-6, f"{b.name}: 액터{prev.id} 뒤 {gap:.0f} m 에 액터{nxt.id}"


def test_단계3c_는_평가_기본_단계에_없다():
    """③c 는 학습 전용 — 평가 스크립트의 기본 단계에 끼면 '안 쓴 판' 성적이 본 장면을 잰다."""
    import importlib.util
    import os
    repo = os.path.join(os.path.dirname(__file__), "..")
    for script, names in (("eval_unseen.py", ("DEFAULT_STAGES",)),
                          ("run_dagger.py", ("DEFAULT_COLLECT_STAGES", "DEFAULT_EVAL_STAGES")),
                          ("refit_round.py", ("DEFAULT_STAGES",))):
        spec = importlib.util.spec_from_file_location(f"chk_{script[:-3]}", os.path.join(repo, "scripts", script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for n in names:
            assert "stage3c" not in getattr(mod, n), (script, n)
