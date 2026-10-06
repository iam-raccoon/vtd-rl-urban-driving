import os

import pytest

from vtd_rl.world.board import Board, load_board, load_curriculum, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
STAGE1 = os.path.join(os.path.dirname(__file__), "..", "curricula", "stage1.json")


def test_코스_H_불러오기():
    b = load_board(H)
    assert len(b.route.pts) == len(b.lane_plan) == 2015     # 원본 ego_route 와 차로계획이 짝
    assert b.route.total == pytest.approx(2807, abs=2)
    x, y, h = b.start_pose
    assert (x, y) == pytest.approx(b.route.pts[0])
    assert h == pytest.approx(b.route.heading_at(0))
    assert b.signals == "always_green"


def test_짝이_안_맞으면_거부():
    b = load_board(H)
    with pytest.raises(ValueError):
        Board("bad", b.scenario, b.lane_plan[:-1])


def test_자르기():
    b = load_board(H)
    s = slice_board(b, 0.0, 250.0, "H_0_250")
    assert s.name == "H_0_250"
    assert 245.0 <= s.route.total <= 251.0
    assert len(s.route.pts) == len(s.lane_plan)
    assert s.scenario.actors == [] and s.scenario.lights == []
    assert s.goal == pytest.approx(s.route.pts[-1])
    assert s.scenario.duration >= 60.0


def test_단계1_목록():
    name, boards = load_curriculum(STAGE1)
    assert name.startswith("빈 경로")
    assert [b.name for b in boards] == ["course_A", "course_B", "course_D", "course_E", "course_G", "course_H"]
    assert all(b.signals == "always_green" for b in boards)
    assert all(len(b.route.pts) == len(b.lane_plan) for b in boards)


def test_차로변경_표식_보존과_자르기():
    b = load_board(H)
    assert len(b.ego_lanes) == len(b.route.pts)
    s = slice_board(b, 0.0, 250.0, "H_0_250")
    assert len(s.ego_lanes) == len(s.route.pts)
    assert s.ego_lanes == b.ego_lanes[:len(s.ego_lanes)]
    assert s.signals == b.signals
    red = slice_board(b, 0.0, 250.0, "H_red", signals="always_red")
    assert red.signals == "always_red"
    with pytest.raises(ValueError):
        slice_board(b, 0.0, 250.0, "H_bad", signals="purple")


def _g_entry(**kw):
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "curricula", "stage3.json")
    with open(path, encoding="utf-8") as f:
        e = dict(json.load(f)["boards"][4])     # course_G
    assert e["name"] == "course_G"
    e.update(kw)
    return e


def test_slice_는_구간만_자르고_구간_안_액터만_남긴다():
    full = load_board(_g_entry())
    cut = load_board(_g_entry(name="course_G_s2665", slice=[2515.0, 2785.0]))
    assert abs(cut.route.total - 270.0) < 5.0
    assert len(cut.scenario.actors) == 1                       # 4 번째 정지차만
    car = cut.scenario.actors[0]
    full_car = [a for a in full.scenario.actors if a.id == car.id][0]
    assert car.motion == full_car.motion                       # 세계 좌표 그대로
    x0, y0, _h = full.route.point_at(2515.0)
    assert abs(cut.scenario.ego_start[0] - x0) < 2.0 and abs(cut.scenario.ego_start[1] - y0) < 2.0


def test_slice_변종은_흔든_뒤_구간_안_액터를_남긴다():
    for v in range(4):
        cut = load_board(_g_entry(name="course_G_s2665", slice=[2515.0, 2785.0]), variant=v)
        assert cut.name == f"course_G_s2665@v{v}"
        assert len(cut.scenario.actors) == 1


def test_slice_가_없으면_예전과_같다():
    a, b = load_board(_g_entry()), load_board(_g_entry())
    assert a.route.total == b.route.total and len(a.scenario.actors) == len(b.scenario.actors) == 4
    assert a.route.total > 2500.0                             # 자르지 않은 코스 전체


@pytest.mark.parametrize("bad", [[2515.0], [2785.0, 2515.0], [2515.0, 2515.5], "x", [None, 5.0]])
def test_잘못된_slice_는_거부한다(bad):
    with pytest.raises(ValueError):
        load_board(_g_entry(name="bad", slice=bad))
