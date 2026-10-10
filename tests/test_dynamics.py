import math

import pytest

from vtd_rl.world import dynamics as dyn
from vtd_rl.world.dynamics import DynamicsParams, EgoState, step

DT = 0.05


def run(st, steer, accel, seconds, p):
    for _ in range(int(round(seconds / DT))):
        st = step(st, steer, accel, DT, p)
    return st


def test_지연_없는_직선_가속은_규칙_스택_오프라인_시뮬과_같다():
    p = DynamicsParams(accel_tau=0.0, max_steer_rate=100.0)
    st = run(EgoState(0.0, 0.0, 0.0), 0.0, 2.0, 1.0, p)
    assert st.v == pytest.approx(2.0)
    assert st.x == pytest.approx(1.05)       # sum_{k=1..20} (0.1k) * 0.05
    assert st.y == pytest.approx(0.0)


def test_가속_1차_지연():
    p = DynamicsParams(accel_tau=0.25)
    st = run(EgoState(0.0, 0.0, 0.0), 0.0, 2.0, 0.25, p)
    assert st.accel == pytest.approx(2.0 * (1.0 - math.exp(-1.0)), rel=1e-6)


def test_조향_속도_제한():
    p = DynamicsParams(max_steer_rate=1.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=5.0), 0.5, 0.0, 0.1, p)
    assert st.steer == pytest.approx(0.1)


def test_조향_한계():
    p = DynamicsParams(max_steer_rate=100.0)
    st = step(EgoState(0.0, 0.0, 0.0), 2.0, 0.0, DT, p)
    assert st.steer == pytest.approx(math.radians(35.0))


def test_가속_범위와_후진_없음():
    p = DynamicsParams(accel_tau=0.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=1.0), 0.0, -9.0, 1.0, p)
    assert st.accel == pytest.approx(-5.0)
    assert st.v == 0.0 and st.x < 0.2


def test_좌회전은_왼쪽으로_간다():
    p = DynamicsParams(max_steer_rate=100.0, accel_tau=0.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=5.0), 0.3, 0.0, 1.0, p)
    assert st.heading > 0.3 and st.y > 0.5


def _turn(p, v=10.0, steer=0.3, n=20):
    st = dyn.EgoState(0.0, 0.0, 0.0, v=v, accel=0.0, steer=steer)
    for _ in range(n):
        st = dyn.step(st, steer, 0.0, 0.05, p)
    return st


def test_언더스티어와_측가속_한계는_기본_꺼짐이면_예전과_같다():
    base = dyn.DynamicsParams(accel_tau=0.0)
    off = dyn.DynamicsParams(accel_tau=0.0, understeer=0.0, lat_accel_max=0.0)
    assert _turn(base) == _turn(off)


def test_언더스티어는_빠를수록_덜_돈다():
    p0 = dyn.DynamicsParams(accel_tau=0.0)
    pu = dyn.DynamicsParams(accel_tau=0.0, understeer=0.016)
    for v in (5.0, 12.0):
        k = _turn(pu, v=v).heading / _turn(p0, v=v).heading
        assert k == pytest.approx(2.95 / (2.95 + 0.016 * v * v), rel=1e-6)


def test_측가속_한계를_넘으면_요레이트가_막힌다():
    p = dyn.DynamicsParams(accel_tau=0.0, lat_accel_max=6.0)
    v = 11.0                                   # 40 km/h, 조향 0.4 rad → 모형 요레이트 1.57 rad/s(측가속 17 m/s²)
    st = _turn(p, v=v, steer=0.4, n=10)
    assert st.heading == pytest.approx(6.0 / v * 0.5, rel=1e-6)      # 0.5 초 동안 a_max / v
    slow = _turn(p, v=4.0, steer=0.1, n=10)                          # 한계 아래는 그대로
    assert slow.heading == pytest.approx(4.0 * math.tan(0.1) / 2.95 * 0.5, rel=1e-6)
