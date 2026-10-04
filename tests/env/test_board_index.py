import math

import pytest

from vtd_rl.env.board_index import board_index, clear_board_index_cache
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
G = {"name": "course_G", "route": "routes/HL_FMA_NEW_G.json", "lane": "routes/HL_FMA_NEW_G_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def g_zone_slice():
    return slice_board(load_board(G), 640.0, 780.0, "G_640_780")


def test_코스_H_앞쪽_거리():
    bi = board_index(h_slice())
    assert bi.signal_s and abs(bi.signal_s[0] - 96.3) < 2.0
    assert [s.s for s in bi.signals] and min(s.s for s in bi.signals) == bi.signal_s[0]
    assert any(abs(s - 104.5) < 2.0 for s in bi.crosswalk_s)
    assert bi.ahead(0.0, "signal") == bi.signal_s[0]
    assert bi.ahead(0.0, "crosswalk") > bi.ahead(110.0, "crosswalk")
    assert bi.ahead(9e9, "signal") == math.inf
    assert bi.plan_at(100).get("lim") == 13.889      # 코스 H 제한 50 km/h
    assert bi.plan_at(10 ** 9) == {}
    # 도색 정지선은 진행 방향이 같은 것만: s≈135.35의 반대 차선 정지선 쌍은 제외되어야 함
    assert not any(abs(s - 135.35) < 1.0 for s in bi.stopline_s)
    # 신호 근처의 정지선은 남아 있어야 함 (필터가 모든 것을 버리지 않음)
    assert any(abs(s - 96.3) < 3.0 for s in bi.stopline_s)


def test_교차로는_묶음마다_하나():
    """`j`/`jx` 가 이어지는 묶음의 **첫 점**만 남긴다 — 안 그러면 교차로 안에서 거리가 0 에 붙는다."""
    bi = board_index(h_slice())
    assert len(bi.junction_s) == 1                   # 코스 H 앞 250 m 에는 교차로 하나(96.5~135.2 m)
    assert abs(bi.junction_s[0] - 96.5) < 1.0
    assert bi.ahead(0.0, "junction") == pytest.approx(bi.junction_s[0])
    assert bi.ahead(100.0, "junction") == math.inf   # 묶음 안에서는 '다음 교차로'가 없다


def test_모르는_종류는_명확한_오류():
    bi = board_index(h_slice())
    with pytest.raises(ValueError) as e:
        bi.ahead(0.0, "없는것")
    assert "junction" in str(e.value) and "signal" in str(e.value)


def test_캐시는_같은_색인을_준다():
    clear_board_index_cache()
    b = h_slice()
    assert board_index(b) is board_index(b)
    clear_board_index_cache()
    assert board_index(b) is not None


def test_코스_G_보호구역():
    """코스 G의 슬라이스 640~780 m 구간에 30 km/h 보호구역이 있음. 경로점 색인이 구간을 따라 False→True 변함."""
    bi = board_index(g_zone_slice())
    # 슬라이스 시작 부근 (경로점 인덱스 낮음): 보호구역 아님
    assert bi.zone[0] is False
    # 슬라이스 끝 부근 (경로점 인덱스 높음): 보호구역임 (30 km/h ≤ 20 km/h 한계)
    # 보호구역이 슬라이스 끝까지 이어짐
    assert bi.zone[-1] is True
    assert any(bi.zone)  # 슬라이스 안에 보호구역이 있음


H_ENTRY = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def test_cache_key_가_같으면_색인을_같이_쓴다():
    a = slice_board(load_board(H_ENTRY), 0.0, 250.0, "H_ck_a")
    b = slice_board(load_board(H_ENTRY), 0.0, 250.0, "H_ck_b")
    clear_board_index_cache()
    assert board_index(a) is not board_index(b)          # 열쇠가 없으면 예전처럼 이름마다 따로
    a.cache_key = b.cache_key = "H|0|250"
    clear_board_index_cache()
    assert board_index(a) is board_index(b)
    clear_board_index_cache()
