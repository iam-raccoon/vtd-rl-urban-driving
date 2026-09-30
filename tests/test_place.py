import math

import pytest

from vtd_rl.world.board import load_board
from vtd_rl.world.place import actor_from_spec, route_point

BOARD = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json",
         "lane": "routes/HL_FMA_NEW_H_lane.json"}


def test_route_point의_횡_오프셋_공식():
    """`referee/scenarios.py:56-58` 의 공식 그대로여야 한다.

    ⚠ `sc._route_point` 와 대조하지 마라 — Task 1 이 그것을 `place.route_point` 로
    위임시키므로 **동어반복이 된다.** 공식을 직접 확인한다: 왼쪽(+lateral)이
    `(-sin h, +cos h)` 방향이고, 이동 거리가 |lateral| 이어야 한다.
    """
    b = load_board(BOARD, "always_green")
    for s in (0.0, 50.0, 150.0, 400.0):
        x0, y0, h = route_point(b, s, 0.0)
        assert (x0, y0, h) == b.route.point_at(s)        # lateral 0 이면 경로 점 그대로
        for lat in (-3.0, 2.5):
            x, y, h2 = route_point(b, s, lat)
            assert h2 == h
            assert x == pytest.approx(x0 - lat * math.sin(h))
            assert y == pytest.approx(y0 + lat * math.cos(h))
            assert math.hypot(x - x0, y - y0) == pytest.approx(abs(lat))


def test_static_액터는_그_지점의_경로_방향을_받는다():
    """★ `mock_vtd.py:46-50` 이 경고하는 버그 — hd 가 0 으로 고정되면 굽은 코스에서
    큰 차의 바운딩박스가 도로를 가로질러 유령 충돌(-50)을 만든다."""
    b = load_board(BOARD, "always_green")
    a = actor_from_spec(b, {"id": 1, "kind": "static", "type": "vehicle",
                            "s": 400.0, "lateral": 0.0, "size": [4.5, 1.8, 1.5]})
    _, _, h = route_point(b, 400.0, 0.0)
    assert a.motion["hd"] == h
    assert abs(h) > 1e-6, "굽은 지점을 골라야 이 테스트가 의미 있다 — s 를 바꿔라"


def test_액터_기술은_경로_상대_좌표를_세계_좌표로_바꾼다():
    b = load_board(BOARD, "always_green")
    a = actor_from_spec(b, {"id": 2, "kind": "static", "type": "obstacle",
                            "s": 150.0, "lateral": -1.5, "size": [0.15, 0.46, 0.61]})
    x, y, _ = route_point(b, 150.0, -1.5)
    assert a.motion["pos"] == [x, y]
    assert a.size == [0.15, 0.46, 0.61]


def test_모르는_운동_kind는_터진다():
    """`mock_vtd.py:63` 은 모르는 kind 를 **조용히** 정지 물체로 만든다. 여기서 막는다."""
    b = load_board(BOARD, "always_green")
    with pytest.raises(ValueError, match="운동"):
        actor_from_spec(b, {"id": 3, "kind": "teleport", "s": 10.0, "size": [1, 1, 1]})
