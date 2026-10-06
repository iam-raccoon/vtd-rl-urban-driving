import math
from types import SimpleNamespace

import numpy as np
import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.env.board_index import board_index
from vtd_rl.env.drive_env import VtdDriveEnv
from vtd_rl.env.observation import (ObsConfig, _object_class, anticipated_limit, build_observation,
                                    observation_space)
from vtd_rl.env.teacher_policy import TeacherPolicy
from vtd_rl.referee.judges.contact import PERSON_MIN_H
from vtd_rl.world.board import load_board, slice_board
from vtd_rl.world.world import World

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
G = {"name": "course_G", "route": "routes/HL_FMA_NEW_G.json", "lane": "routes/HL_FMA_NEW_G_lane.json"}


def h_world(signals="always_green"):
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250", signals=signals)
    return World(b)


def test_관측은_공간_안에_들어간다():
    w = h_world()
    state = w.reset()
    info = None
    space = observation_space()
    for k in range(40):
        obs = build_observation(w, state, info, (0.1, 0.5, rs.TS_LEFT))
        assert space.contains(obs), {k: (v.dtype, float(v.min()), float(v.max()))
                                     for k, v in obs.items()}
        state, info = w.step(0.0, 1.0, rs.TS_LEFT)


def test_자차_묶음():
    w = h_world()
    state = w.reset()
    obs = build_observation(w, state, None, (0.0, 0.0, rs.TS_OFF))
    assert obs["ego"][0] == pytest.approx(0.0)              # 정지 상태
    assert list(obs["ego"][4:7]) == [1.0, 0.0, 0.0]         # 지시등 끔
    for _ in range(40):
        state, info = w.step(0.2, 1.5, rs.TS_RIGHT)
    obs = build_observation(w, state, info, (0.2 / math.radians(35), 0.75, rs.TS_RIGHT))
    cfg = ObsConfig()
    assert obs["ego"][0] == pytest.approx(w.ego.v / cfg.v_max, abs=1e-3)
    assert obs["ego"][1] == pytest.approx(0.2 / math.radians(35), abs=1e-3)
    assert list(obs["ego"][4:7]) == [0.0, 0.0, 1.0]         # 우측 지시등
    assert obs["ego"][3] != 0.0                             # 요레이트


def test_경로는_자차_기준이고_앞을_본다():
    w = h_world()
    state = w.reset()
    for _ in range(60):
        state, info = w.step(0.0, 1.0, 0)
    obs = build_observation(w, state, info, (0.0, 0.5, 0))
    cfg = ObsConfig()
    pts = obs["route"] * cfg.route_ahead
    assert np.all(np.diff(pts[:, 0]) > -1.0)                # 앞으로 간다
    assert abs(pts[0, 1]) < 3.0                             # 바로 앞은 옆으로 많이 안 떨어져 있다
    assert abs(obs["nav"][0]) < 1.0                         # 경로에 붙어 있다
    assert 0.0 < obs["nav"][2] <= 1.0                       # 남은 거리


def test_신호와_물체():
    b = slice_board(load_board(H), 0.0, 250.0, "H_red", signals="always_red")
    x, y, h = b.route.point_at(20.0)  # 60 스텝 뒤 자차는 s≈4m — 세계 물체 범위(80m) 안에 두려고 150→20
    b.scenario.actors = [rs.Actor(id=7, type="vehicle", size=[4.5, 1.8, 1.5],
                                  spawn={"at_time": 0.0}, motion={"kind": "static", "pos": [x, y]})]
    w = World(b)
    state = w.reset()
    for _ in range(60):
        state, info = w.step(0.0, 1.0, 0)
    obs = build_observation(w, state, info, (0.0, 0.5, 0))
    assert obs["signal"][rs.TL_RED] == 1.0
    assert 0.0 < obs["signal"][7] < 1.0                     # 정지선까지 거리
    assert obs["object_mask"][0] == 1.0 and obs["object_mask"][-1] == 0.0
    assert obs["objects"][0, 0] > 0.0                       # 앞에 있다
    assert list(obs["objects"][0, 9:12]) == [1.0, 0.0, 0.0]  # 차량
    assert np.all(obs["objects"][1:] == 0.0)                # 빈 자리는 0


def test_물체는_가까운_순_16개까지():
    b = slice_board(load_board(H), 0.0, 250.0, "H_obj")
    acts = []
    for k in range(20):
        x, y, _h = b.route.point_at(20.0 + 3.0 * k)     # 전부 세계의 물체 범위(80 m) 안
        acts.append(rs.Actor(id=100 + k, type="obstacle", size=[0.5, 0.5, 0.8],
                             spawn={"at_time": 0.0}, motion={"kind": "static", "pos": [x, y + 6.0]}))
    b.scenario.actors = acts
    w = World(b)
    state = w.reset()
    obs = build_observation(w, state, None, (0.0, 0.0, 0))
    assert obs["object_mask"].sum() == 16
    d = np.hypot(obs["objects"][:, 0], obs["objects"][:, 1])
    assert np.all(np.diff(d) >= -1e-6)                      # 가까운 순


def test_남은_거리는_그_판의_경로_길이로_잰다():
    """nav[2] 는 판 길이로 나눈 남은 비율이다 — 긴 판에서도 1.0 에 붙어 죽지 않는다.

    고정 1000 m 로 나누던 때는 단계 ① 경로(2152~5242 m)에서 판의 절반~8할 동안 값이 1.0 이었다.
    """
    b = slice_board(load_board(H), 0.0, 1200.0, "H_0_1200")     # 1000 m 보다 긴 판이어야 회귀가 잡힌다
    assert b.route.total > 1000.0
    env = VtdDriveEnv([b])
    obs, _info = env.reset(seed=0)
    assert obs["nav"][2] == pytest.approx(1.0)                  # 출발선
    policy = TeacherPolicy(env)
    policy.reset()
    seen = []
    for k in range(4000):
        obs, _r, term, trunc, _i = env.step(policy.act())
        if k % 100 == 0:
            seen.append(float(obs["nav"][2]))
        if term or trunc:
            break
    env.close()
    assert len(seen) >= 8
    assert all(a > b_ for a, b_ in zip(seen, seen[1:])), seen    # 한 번도 안 멈추고 줄어든다
    assert seen[1] < 1.0 and seen[-1] < 0.2, seen               # 1.0 에 붙어 있지 않다


def test_묶음_값_교차로_거리와_보호구역():
    """관측 묶음의 값 자체를 못 박는다 — 교차로까지 거리(plan[9])와 보호구역 깃발(signal[10])."""
    cfg = ObsConfig()
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250_plan")
    idx = board_index(b)
    assert idx.junction_s and 96.0 < idx.junction_s[0] < 97.0   # 코스 H 첫 교차로 96.5~135.2 m
    w = World(b)
    state = w.reset()
    info = None
    seen = []
    for _ in range(1400):                                       # 교차로 입구를 지날 때까지
        obs = build_observation(w, state, info, (0.0, 0.4, 0))
        seen.append((info.s if info else 0.0, float(obs["plan"][9])))
        state, info = w.step(0.0, 0.4, 0)
        if info.done or info.s > 90.0:
            break
    before = [(s, d) for s, d in seen if s < 90.0]
    assert before[0][1] == pytest.approx(idx.junction_s[0] / cfg.dist_max, abs=1e-3)
    assert all(a[1] >= b_[1] for a, b_ in zip(before, before[1:]))      # 다가갈수록 준다
    assert before[-1][1] < before[0][1] - 0.5                           # 90 m 가까이 줄었다
    assert float(obs["plan"][9]) != float(obs["signal"][7])             # 신호 거리와 다른 값이다

    gb = slice_board(load_board(G), 640.0, 780.0, "G_640_780_zone")
    gw = World(gb)
    gstate = gw.reset()
    ginfo = None
    flags = []
    for _ in range(1200):
        obs = build_observation(gw, gstate, ginfo, (0.0, 0.4, 0))
        flags.append(float(obs["signal"][10]))
        gstate, ginfo = gw.step(0.0, 0.4, 0)
        if ginfo.done:
            break
    assert flags[0] == 0.0 and 1.0 in flags                     # 보호구역에 들어가면 깃발이 선다


def test_물체_종류는_접촉_판정기와_같은_기준():
    """관측의 사람·차량·사물 분류가 보상을 내는 심판(judges/contact.py)의 ⑪⑭ 판정과 같아야 한다.

    치수는 VTD 카탈로그 실측값. 기대값을 여기 그대로 적어 둬서, 나중에 문턱(PED_L·PED_W·
    PERSON_MIN_H)이 바뀌면 이 테스트가 조용히 따라가지 않고 소리 내어 실패하게 한다.
    """
    cases = [
        ("승용차", 4.5, 1.8, 1.5, [1.0, 0.0, 0.0]),           # 차량 (length > PED_L)
        ("보행자", 0.6, 0.7, 1.8, [0.0, 1.0, 0.0]),           # 사람
        ("휠체어", 1.01, 0.62, 0.92, [0.0, 0.0, 1.0]),        # 키 0.92 < 1.2m → 사물(사람 아님)
        ("입식 자전거", 1.90, 0.65, 1.10, [0.0, 0.0, 1.0]),    # 키 1.10 < 1.2m → 사물(사람 아님)
        ("라바콘", 0.15, 0.46, 0.61, [0.0, 0.0, 1.0]),         # 사물
    ]
    for name, length, width, height, expected in cases:
        o = rs.Obj(id=1, x=0.0, y=0.0, z=0.0, heading=0.0, speed=0.0,
                   length=length, width=width, height=height)
        got = _object_class(o)
        assert got == expected, name

        # judges/contact.py 의 사람·차량 판정식과 정확히 일치해야 한다
        person = length <= rs.score_fma.PED_L and width <= rs.score_fma.PED_W and height >= PERSON_MIN_H
        vehicle = length > rs.score_fma.PED_L
        assert (got == [0.0, 1.0, 0.0]) == person, name
        assert (got == [1.0, 0.0, 0.0]) == vehicle, name


def _idx(drops):
    return SimpleNamespace(lim_drop_s=[d[0] for d in drops], lim_drop_v=[d[1] for d in drops])


@pytest.mark.parametrize("s, lim, want", [
    (0.0, 14.0, 14.0),                                   # sqrt(64 + 3·95) ≈ 18.7 > 14
    (80.0, 14.0, (64.0 + 3.0 * 15.0) ** 0.5),            # ds 20 − margin 5 = 15
    (97.0, 14.0, 8.0),                                   # margin 안: 새 제한속도
    (101.0, 14.0, 14.0),                                 # 이미 지났다
    (0.0, 6.0, 6.0),                                     # 지금 제한이 더 낮다
])
def test_anticipated_limit(s, lim, want):
    cfg = ObsConfig(lim_anticipate=True)
    assert anticipated_limit(_idx([(100.0, 8.0)]), s, lim, cfg) == pytest.approx(want)


def test_anticipated_limit_은_lookahead_밖을_안_본다():
    cfg = ObsConfig(lim_anticipate=True, lim_lookahead=50.0)
    assert anticipated_limit(_idx([(100.0, 1.0)]), 0.0, 14.0, cfg) == pytest.approx(14.0)


def test_lim_anticipate_기본값은_꺼짐():
    assert ObsConfig().lim_anticipate is False


def test_lim_anticipate_를_끄면_관측이_예전과_같고_켜면_lim_이_같거나_낮다():
    """코스 A 의 첫 내려감(s≈1435 m, 50 → 30 km/h) 앞뒤를 자른 판을 선생님이 몬다 — 일정 조향으로는

    출발 직후 길을 벗어나 내려감에 닿지 못한다. 앞당김이 실제로 낮춘 걸음이 있어야 뜻이 있다."""
    import json
    from vtd_rl.env.drive_env import EnvConfig
    e = json.load(open("curricula/stage2.json"))["boards"][0]
    b = slice_board(load_board(e), 1300.0, 1500.0, "A_1300_1500")
    off, on = VtdDriveEnv([b]), VtdDriveEnv([b], EnvConfig(obs=ObsConfig(lim_anticipate=True)))
    t_off, t_on = TeacherPolicy(off), TeacherPolicy(on)
    off.reset(seed=0)
    on.reset(seed=0)
    t_off.reset()
    t_on.reset()
    lower = 0
    for _ in range(3000):
        o1, _r1, term1, trunc1, _i1 = off.step(t_off.act())
        o2, _r2, term2, trunc2, _i2 = on.step(t_on.act())
        for k in o1:
            if k != "ego":
                assert np.array_equal(o1[k], o2[k]), k
        assert o2["ego"][-2] <= o1["ego"][-2] + 1e-6      # 앞당긴 lim(정규화) ≤ 지금 lim
        lower += int(o2["ego"][-2] < o1["ego"][-2] - 1e-6)
        if term1 or trunc1 or term2 or trunc2:
            break
    off.close()
    on.close()
    assert lower > 0                                      # 내려감 앞에서 실제로 미리 낮아졌다
