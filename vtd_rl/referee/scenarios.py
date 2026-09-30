"""일치 검증용 대본 판. 각 판은 채점기에서 expect 항목을 일으키고 forbid 항목은 일으키지 않아야 한다.

위치 근거(2026-09-15 측정):
- 코스 H 0~250 m: 제한 50, 중앙선까지 l≈1.5, 교차로 106~135 m, 신호151 정지선 DB 점 s=96.3,
  신호 횡단보도 중심 s=104.5(횡 +3.14)·s=130.7(횡 +2.99).
- 코스 G 640~780 m: 683 m 부터 보호구역 30.
- 코스 G 2575~2841 m: 교차로 없는 구간, 왼쪽에 같은 방향 차로(l≈4.5).
대본 수치는 기대 항목이 일어나도록 조정해도 된다. 채점기·심판·월드 문턱은 바꾸지 않는다.
"""
import copy
import functools
import math
from dataclasses import dataclass

from vtd_rl import rule_stack as rs
from vtd_rl.drivers.scripted import (Cruise, LaneShift, Offset, ScriptedDriver, SignalWindow,
                                     StopAt, TeacherDriver)
from vtd_rl.rollout import run_episode
from vtd_rl.world.board import load_board, slice_board
from vtd_rl.world.place import actor_from_spec
from vtd_rl.world.place import route_point as _route_point

KMH = 1 / 3.6
FRONT = rs.score_fma.FRONT
H_TL151 = 96.3          # 신호151 정지선 DB 점의 경로 s
H_CW = 130.7            # 신호 횡단보도 중심의 경로 s(횡 +2.99)


@dataclass
class Scenario:
    name: str
    board: object
    make_driver: object
    expect: frozenset
    forbid: frozenset
    note: str


@functools.lru_cache(maxsize=None)
def _course(letter):
    return load_board({"name": f"course_{letter}", "route": f"routes/HL_FMA_NEW_{letter}.json",
                       "lane": f"routes/HL_FMA_NEW_{letter}_lane.json"})


def _slice(letter, s0, s1, signals="always_green"):
    return slice_board(_course(letter), s0, s1, f"{letter}_{s0}_{s1}", signals=signals)


def _script(speed, offset=None, signal=None):
    # StopAt 은 상태를 갖는 one-shot 이다(_since/_done) — make_driver 를 두 번 부르면
    # (episode_rows 캐시된 Scenario 에 대해 진단하려고 직접 다시 돌릴 때 등) 같은 인스턴스를
    # 재사용해 이미 소진된 상태로 시작한다. 호출마다 깊은 복사로 새 일정을 준다.
    return lambda b: ScriptedDriver(b, copy.deepcopy(speed),
                                    copy.deepcopy(offset) or Offset(0.0),
                                    copy.deepcopy(signal) or SignalWindow(0, 0, 0))


def _static(board, aid, kind, s, size):
    # 좌표 변환과 `hd`(방위) 주입은 `world/place.py` 가 한다 — 커리큘럼과 같은 길을 쓴다.
    # 예전에는 `hd` 를 안 넣어 방위가 0 으로 고정됐다(`mock_vtd.py:46-52` 의 잠복 버그).
    return actor_from_spec(board, {"id": aid, "kind": "static", "type": kind,
                                   "s": s, "lateral": 0.0, "size": list(size)})


def _scenario(name, board, make_driver, expect, note, forbid=()):
    return Scenario(name, board, make_driver, frozenset(expect), frozenset(forbid), note)


def speeder():
    return _scenario("speeder", _slice("G", 2575, 2841), _script(Cruise(80 * KMH)), {1},
                     "제한 50 에서 80 km/h")


def zone_speeder():
    return _scenario("zone_speeder", _slice("G", 640, 780), _script(Cruise(45 * KMH)), {2},
                     "보호구역 30 에서 45 km/h")


def red_runner():
    return _scenario("red_runner", _slice("H", 0, 250, "always_red"), _script(Cruise(40 * KMH)), {7},
                     "적색 무정차 통과(중대)")


def red_far_stopper():
    return _scenario("red_far_stopper", _slice("H", 0, 250, "always_red"),
                     _script(StopAt(H_TL151 - FRONT - 5.0, 3.0, 30 * KMH)), {7},
                     "앞범퍼가 정지선 5 m 앞에 섰다가 통과(경미)")


def flash_runner():
    return _scenario("flash_runner", _slice("H", 0, 250, "always_flash"), _script(Cruise(40 * KMH)), {9},
                     "적색점멸 무정차 통과")


def green_idler():
    return _scenario("green_idler", _slice("H", 0, 250), _script(StopAt(H_TL151 - 18.0, 12.0, 30 * KMH)),
                     {8}, "녹색에 정지선 18 m 앞 12 초 정차")


def crosswalk_stopper():
    return _scenario("crosswalk_stopper", _slice("H", 0, 250),
                     _script(StopAt(H_CW - FRONT / 2, 5.0, 20 * KMH), LaneShift(110.0, 1.0, 10.0)),
                     {12}, "횡단보도 위 5 초 정지(횡 +1 m 로 붙어서)")


def obstacle_rammer():
    b = _slice("H", 0, 250)
    b.scenario.actors = [_static(b, 901, "obstacle", 150.0, (0.15, 0.46, 0.61))]
    return _scenario("obstacle_rammer", b, _script(Cruise(25 * KMH)), {11}, "라바콘 들이받기")


def vehicle_rammer():
    b = _slice("H", 0, 250)
    b.scenario.actors = [_static(b, 902, "vehicle", 150.0, (4.5, 1.8, 1.5))]
    return _scenario("vehicle_rammer", b, _script(Cruise(25 * KMH)), {14}, "서 있는 차 들이받기")


def ped_ignorer():
    b = _slice("H", 0, 250)
    x0, y0, h = _route_point(b, 180.0, -4.0)
    cx, cy, _ = _route_point(b, 180.0, 0.0)
    b.scenario.actors = [rs.Actor(id=903, type="pedestrian", size=[0.6, 0.7, 1.8],
                                  spawn={"ego_within": 30.0, "of": [cx, cy]},
                                  motion={"kind": "crossing", "pos": [x0, y0],
                                          "vel": [-1.3 * math.sin(h), 1.3 * math.cos(h)]})]
    return _scenario("ped_ignorer", b, _script(Cruise(25 * KMH)), {10}, "건너는 사람 앞을 안 서고 통과")


def centerline_crosser():
    return _scenario("centerline_crosser", _slice("H", 0, 250),
                     _script(Cruise(20 * KMH), LaneShift(150.0, 1.8, 15.0)), {4},
                     "중앙선 너머로 1.8 m 붙어 달리기")


def sidewalk_rider():
    return _scenario("sidewalk_rider", _slice("H", 0, 250),
                     _script(Cruise(15 * KMH), LaneShift(150.0, -4.8, 25.0)), {5},
                     "오른쪽 보도 쪽으로 4.8 m 붙어 달리기")


def unsignaled_lane_change():
    return _scenario("unsignaled_lane_change", _slice("G", 2575, 2841),
                     _script(Cruise(30 * KMH), LaneShift(100.0, 3.3, 40.0)), {13},
                     "지시등 없이 왼쪽 차로로")


def signaled_lane_change():
    return _scenario("signaled_lane_change", _slice("G", 2575, 2841),
                     _script(Cruise(30 * KMH), LaneShift(100.0, 3.3, 40.0), SignalWindow(55.0, 150.0, 1)),
                     set(), "5 초 앞서 좌측 지시등을 켜고 왼쪽 차로로", forbid={13})


def teacher_H_0_250():
    return _scenario("teacher_H_0_250", _slice("H", 0, 250), lambda b: TeacherDriver(b), set(),
                     "선생님 — 기대 항목 없이 일치만 본다")


SCENARIOS = {f.__name__: f for f in (
    speeder, zone_speeder, red_runner, red_far_stopper, flash_runner, green_idler,
    crosswalk_stopper, obstacle_rammer, vehicle_rammer, ped_ignorer, centerline_crosser,
    sidewalk_rider, unsignaled_lane_change, signaled_lane_change, teacher_H_0_250)}


@functools.lru_cache(maxsize=None)
def episode_rows(name):
    sc = SCENARIOS[name]()
    run = run_episode(sc.board, sc.make_driver(sc.board))
    return sc, tuple(run.rows)
