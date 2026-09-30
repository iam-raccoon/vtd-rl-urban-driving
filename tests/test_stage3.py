import bisect
import json
from collections import defaultdict

from vtd_rl.world.board import load_curriculum


def test_단계3이_여섯_코스를_모두_덮는다():
    _, boards = load_curriculum("curricula/stage3.json")
    assert sorted(b.name for b in boards) == [
        "course_A", "course_B", "course_D", "course_E", "course_G", "course_H"]


def test_단계3의_모든_판에_액터가_있다():
    """액터 0 개인 판이 하나라도 있으면 그 판은 단계 ② 와 같다 — 조용한 오염."""
    _, boards = load_curriculum("curricula/stage3.json")
    for b in boards:
        assert len(b.scenario.actors) >= 3, b.name


def test_단계3은_신호_주기를_쓴다():
    """단계 ③ 은 ② 를 포함한다 — 신호를 끄면 뒷걸음질이다."""
    _, boards = load_curriculum("curricula/stage3.json")
    assert all(b.signals == "cycle" for b in boards)


def test_단계3에_장애물과_차량이_둘_다_있다():
    """관측의 분류 채널(observation.py:60-71)이 둘 다 쓰이게 한다."""
    _, boards = load_curriculum("curricula/stage3.json")
    for b in boards:
        lens = [a.size[0] for a in b.scenario.actors]
        assert any(l <= 2.2 for l in lens), f"{b.name}: 장애물 없음"
        assert any(l > 2.2 for l in lens), f"{b.name}: 차량 없음"


def test_단계3은_정지물만_쓴다():
    """움직이는 교통·보행자는 단계 ④ 다."""
    _, boards = load_curriculum("curricula/stage3.json")
    for b in boards:
        for a in b.scenario.actors:
            assert a.motion["kind"] in ("static", "constant"), (b.name, a.motion)


# ---------------------------------------------------------------- 배치가 달릴 수 있는가
# 위 다섯은 **개수와 종류만** 본다 — 코스를 완전히 막은 배치도 전부 통과한다(계획 Task 2
# Step 5 의 경고). 아래 셋이 "달릴 수 있는 배치" 쪽을 잠근다. 진짜 보증은
# `tests/test_measure_teacher.py::test_선생님이_단계3_판을_완주한다`(느림)이지만, 그건 몇 분이
# 걸리므로 빠른 쪽에서 배치의 기하학적 전제를 먼저 붙잡는다.
#
# ★ 아래 문턱은 규칙 스택이 실제로 쓰는 값이다(읽기 전용, 2026-09-30 확인).
#   `drive.py:1838` 이 작은 물체를 비켜 갈 때 `_side_for(plan, need + HALF_WIDTH)` 를 부르고,
#   `_side_for`(`drive.py:1430-1433`)는 **`plan["l"]` 이나 `plan["r"]` 이 그만큼 안 되면 0** 을
#   돌려준다 = "합법적으로 비킬 데가 없다" → **비키지 않고 그대로 밟고 지나간다.**
#   실측(2026-09-30 첫 배치): l=r=1.45 인 편도 1차선에 라바콘을 놓았더니 코스 A·D·G 가
#   전부 항목⑪ 접촉으로 조기 종료했다(`env/reward.py:17` COLLISION_ITEMS 에 ⑪ 이 있다).
#   차량은 `NUDGE_MAX_DIM`(1.2) 을 넘어 추월 FSM 담당이고, 그쪽은 `drive.py:1846` 의
#   "차로 하나를 통째로 쓸 수 있으면" 조건(`room >= lane_w + HALF_WIDTH`)이 기준이다.

EGO_HALF_W = 0.943      # drive.py:141 HALF_WIDTH — 자차 반폭[m]
NUDGE_CLEAR = 0.40      # drive.py:147 — 비켜 갈 때 남기는 여유[m]
NUDGE_MAX_DIM = 1.2     # drive.py:144 — 이보다 크면 nudge 가 아니라 추월 FSM 담당


def _actor_specs():
    with open("curricula/stage3.json", encoding="utf-8") as f:
        d = json.load(f)
    return [(e["name"], a) for e in d["boards"] for a in e.get("actors", [])]


def test_액터는_경로_처음과_끝_100m_를_비운다():
    """출발 직후·목표 직전에 물체를 놓으면 판정이 흐려진다 — ③ 은 출발 8 m 를 빼고

    (`judges/lane_geometry.py:20`), 목표 판정은 경로 끝 10 m 안에서 난다. 그 언저리에
    장애물을 두면 "완주 못 한 이유" 가 배치인지 종료 판정인지 구분이 안 된다.
    """
    _, boards = load_curriculum("curricula/stage3.json")
    by_name = {b.name: b for b in boards}
    for name, spec in _actor_specs():
        total = by_name[name].route.total
        assert 100.0 <= spec["s"] <= total - 100.0, (name, spec["id"], spec["s"], total)


def test_액터_자리에_규칙스택이_비킬_차로가_있다():
    """★★ 이 파일에서 유일하게 "달릴 수 있는 배치" 를 잠그는 빠른 테스트.

    규칙 스택은 **비킬 차로가 없으면 안 비키고 그대로 밟는다**(`drive.py:1838-1840`).
    그래서 편도 1차선(`l`·`r` 이 둘 다 좁은 곳)에 놓인 물체는 아무리 차로 가장자리에
    붙여도 유령이 아니라 **진짜 접촉**이 되고, 항목⑪ 은 판을 그 자리에서 끝낸다.
    배치를 고칠 때 이 테스트가 먼저 잡아 주지 않으면 코스 여섯 판을 다 몰아 본 뒤에야 안다.
    """
    _, boards = load_curriculum("curricula/stage3.json")
    by_name = {b.name: b for b in boards}
    for name, spec in _actor_specs():
        board = by_name[name]
        i = max(0, bisect.bisect_right(board.route.cum, spec["s"]) - 1)
        plan = board.lane_plan[i]
        length, width = spec["size"][0], spec["size"][1]
        if max(length, width) < NUDGE_MAX_DIM:
            # 작은 사물 — `drive.py:1838` 의 nudge 게이트를 그대로 옮긴 식
            need = EGO_HALF_W + max(length, width) / 2.0 + NUDGE_CLEAR + EGO_HALF_W
            what = "라바콘류(nudge)"
        else:
            # 차량 — `drive.py:1846` 의 "차로 하나를 통째로" 조건
            need = plan["w"] + EGO_HALF_W
            what = "차량(추월 FSM)"
        assert max(plan["l"], plan["r"]) >= need, (
            f"{name} 액터{spec['id']} s={spec['s']} {what}: "
            f"l={plan['l']:.2f} r={plan['r']:.2f} — {need:.2f} m 가 필요하다")


# 정지 차량을 추월한 자차가 차로로 되돌아온 **직후** 에 다음 물체를 만나면, 복귀 중이라
# nudge 가 다시 안 걸려 그대로 밟는다. 실측 2026-09-30: 코스 A 에 정지차 s=560 · 라바콘
# s=580(20 m 뒤)을 놓았더니 자차가 횡 −3.0 m 로 추월했다가 s=575.9 에 횡 −0.17 m 까지
# 되돌아온 뒤 라바콘에 접촉해 조기 종료했다. 같은 코스 G 의 정지차 s=570 · 라바콘 s=640
# (70 m)은 멀쩡했다. 그 사이를 60 m 로 끊는다.
AFTER_VEHICLE_M = 60.0
MIN_GAP_M = 10.0


def test_정지차_뒤에는_충분한_복귀_거리를_둔다():
    """★ 액터 둘이 앞뒤로 붙어 서면 한쪽을 피한 자차가 곧장 다른 쪽에 부딪힌다."""
    by_board = defaultdict(list)
    for name, spec in _actor_specs():
        by_board[name].append(spec)
    for name, specs in by_board.items():
        ordered = sorted(specs, key=lambda a: a["s"])
        for prev, nxt in zip(ordered, ordered[1:]):
            gap = nxt["s"] - prev["s"]
            need = AFTER_VEHICLE_M if prev["size"][0] > 2.2 else MIN_GAP_M
            assert gap >= need, (
                f"{name}: 액터{prev['id']}(s={prev['s']}) 뒤 {gap:.0f} m 에 "
                f"액터{nxt['id']} — {need:.0f} m 는 떨어져야 한다")
