import math
import os
import subprocess
import sys

import pytest

from vtd_rl.world.board import load_board
from vtd_rl.world.place import (DEFAULT_JITTER, JITTER_DEADZONE_M, actor_from_spec, check_jitter,
                                half_lane_at, jitter_specs, route_point, s_offset)

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

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


# ---------------------------------------------------------------- 액터 흔들기(jitter)

SPECS = [{"id": 1, "kind": "static", "type": "obstacle", "s": 200.0, "lateral": 1.05,
          "size": [0.15, 0.46, 0.61]},
         {"id": 2, "kind": "static", "type": "vehicle", "s": 1140.0, "lateral": 0.90,
          "size": [4.5, 1.8, 1.5]},
         {"id": 3, "kind": "static", "type": "obstacle", "s": 1340.0, "lateral": -1.10,
          "size": [0.15, 0.46, 0.61]}]


def test_jitter는_액터_사이_간격을_한_치도_안_바꾼다():
    """★★ 흔들기 설계의 핵심 불변식.

    "달릴 수 있는 배치" 의 두 조건(정지차 뒤 60 m 복귀 거리, 그 밖 10 m)은 **간격**의
    조건이다(`tests/test_stage3.py`). 그래서 `s` 는 액터마다 따로가 아니라 판마다 공통
    오프셋 하나로 민다. 액터마다 따로 흔들면 이 테스트가 곧바로 깨진다.
    """
    b = load_board(BOARD, "always_green")
    base = [b["s"] for b in SPECS]
    for k in range(1, 8):
        got = [s["s"] for s in jitter_specs(b, SPECS, DEFAULT_JITTER, "course_H", k)]
        assert [y - x for x, y in zip(got, got[1:])] == \
               pytest.approx([y - x for x, y in zip(base, base[1:])])


def test_jitter는_원본_기술을_안_건드린다():
    """호출부가 같은 목록을 변종마다 다시 쓴다 — 제자리에서 바꾸면 흔들기가 누적된다."""
    b = load_board(BOARD, "always_green")
    before = [dict(s) for s in SPECS]
    jitter_specs(b, SPECS, DEFAULT_JITTER, "course_H", 1)
    assert SPECS == before


def test_횡_흔들기는_차로_반폭을_안_넘고_부호를_지킨다():
    """물체가 차로 밖으로 나가면 그 변종은 아무것도 안 시험한다 — 조용한 무효."""
    b = load_board(BOARD, "always_green")
    seen = set()
    for k in range(1, 12):
        for spec in jitter_specs(b, SPECS, {"s": 40.0, "lateral": 0.9}, "course_H", k):
            base = next(x["lateral"] for x in SPECS if x["id"] == spec["id"])
            assert math.copysign(1.0, spec["lateral"]) == math.copysign(1.0, base), spec
            assert abs(spec["lateral"]) <= half_lane_at(b, spec["s"]) + 1e-9, spec
            seen.add(round(spec["lateral"], 6))
    assert len(seen) > 10, "횡 흔들기가 사실상 안 걸렸다"


def test_jitter가_0이면_배치가_그대로다():
    b = load_board(BOARD, "always_green")
    got = jitter_specs(b, SPECS, {"s": 0.0, "lateral": 0.0}, "course_H", 1)
    assert got == SPECS


def test_흔들기_폭이_커도_경로_앞뒤_여유는_지킨다():
    """폭을 경로 길이보다 크게 줘도 액터가 끝점에 뭉치면 안 된다."""
    b = load_board(BOARD, "always_green")
    for k in range(1, 20):
        for spec in jitter_specs(b, SPECS, {"s": 9999.0, "lateral": 0.0}, "course_H", k):
            assert 100.0 <= spec["s"] <= b.route.total - 100.0, spec


def test_변종_오프셋은_서로_충분히_떨어진다():
    """★ 균등 난수로 변종마다 따로 뽑으면 **우연히 겹친다** — 실측(2026-09-30): 판마다

    독립 균등 추출이었을 때 `course_B` 의 변종 4·7 이 둘 다 +22.2 m 로 사실상 같은 판이
    됐다. 겹친 변종은 원본의 사본이라 수집 시간만 먹고 다양성은 안 는다. 0(변종 0 = 원본)
    까지 포함해서 재고, 변종 수 N 을 2~8 어느 것으로 잡아도 지켜져야 한다.
    """
    for name in ("course_A", "course_B", "course_D", "course_E", "course_G", "course_H"):
        for n in range(2, 9):
            offs = sorted([0.0] + [s_offset(name, k, 40.0) for k in range(1, n)])
            gaps = [y - x for x, y in zip(offs, offs[1:])]
            assert min(gaps) >= JITTER_DEADZONE_M, (name, n, [round(o, 1) for o in offs])


def test_변종_오프셋은_흔들기_폭_안에_있다():
    for name in ("course_A", "course_H"):
        for k in range(0, 20):
            off = s_offset(name, k, 40.0)
            assert JITTER_DEADZONE_M <= abs(off) <= 40.0, (name, k, off)
    assert s_offset("course_A", 3, 0.0) == 0.0


def test_jitter_모르는_축은_터진다():
    """`"lat": 0.6` 오타 하나면 횡 흔들기가 통째로 안 걸린 채 멀쩡해 보인다."""
    with pytest.raises(ValueError, match="lat"):
        check_jitter({"s": 40.0, "lat": 0.6})


def test_jitter_음수_폭은_터진다():
    with pytest.raises(ValueError, match="0 이상"):
        check_jitter({"s": -40.0})


def test_jitter를_안_적으면_기본값이고_기본값은_0이_아니다():
    """변종을 켰는데 배치가 똑같은 것이야말로 이 장치가 막으려는 실패다."""
    assert check_jitter(None) == DEFAULT_JITTER
    assert DEFAULT_JITTER["s"] > 0.0 and DEFAULT_JITTER["lateral"] > 0.0
    assert check_jitter({"s": 5.0}) == {"s": 5.0, "lateral": DEFAULT_JITTER["lateral"]}


_PROBE = """
import json, sys
sys.path.insert(0, %r)
from vtd_rl.world.board import load_curriculum
_, bs = load_curriculum(%r, variants=3)
print(json.dumps([[b.name, [a.motion["pos"] for a in b.scenario.actors]] for b in bs]))
"""


def test_흔들기는_프로세스가_달라도_같다():
    """★ 내장 `hash()` 로 시드를 만들면 프로세스마다 다른 판이 나온다 —

    수집은 `mp.get_context("spawn").Pool` 로 별도 프로세스에서 돈다(`scripts/run_dagger.py`).
    문자열 해시는 PYTHONHASHSEED 로 소금이 바뀌므로 일꾼마다 판이 달라지고, 그러면 라운드
    사이에 무엇이 달라졌는지 못 가린다. 한 프로세스 안에서만 보는 재현성 테스트는 이걸 못 잡는다.
    """
    code = _PROBE % (REPO, os.path.join(REPO, "curricula", "stage3.json"))
    outs = []
    for salt in ("1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        env.pop("PYTHONPATH", None)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=env, cwd=REPO, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], "PYTHONHASHSEED 가 바뀌자 배치가 달라졌다"
