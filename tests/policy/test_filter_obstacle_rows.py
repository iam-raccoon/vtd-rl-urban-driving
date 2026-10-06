"""M7k — DAgger 조각에서 장애물 앞 장면 행 고르기."""
import importlib.util
import os

import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _mod():
    spec = importlib.util.spec_from_file_location("filter_obstacle_rows", os.path.join(REPO, "scripts", "filter_obstacle_rows.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _obj(fx, fy, speed=0.0, length=4.5, width=1.8, height=1.5, person=False):
    """관측과 같은 정규화로 물체 특징 12 칸을 짓는다."""
    o = np.zeros(12, dtype=np.float32)
    o[0], o[1], o[4] = fx / 80.0, fy / 20.0, speed / 25.0
    o[5], o[6], o[7] = length / 12.0, width / 12.0, height / 12.0
    o[9 + (1 if person else 0)] = 1.0
    return o


def test_장애물_앞_장면만_고른다():
    m = _mod()
    rows = [
        [_obj(20.0, -1.0)],                       # 통로 안 멈춘 차 20 m — 고름
        [_obj(20.0, -5.0)],                       # 옆 차로 — 버림
        [_obj(20.0, -1.0, speed=3.0)],            # 움직이는 차 — 버림
        [_obj(10.0, 0.0, length=0.5, width=0.5, height=1.7, person=True)],   # 사람 — 버림
        [_obj(70.0, 0.0)],                        # 40 m 밖 — 버림
        [_obj(-5.0, 0.0)],                        # 뒤 — 버림
        [_obj(15.0, 0.5, length=0.4, width=0.4, height=0.7)],                 # 라바콘 — 고름
    ]
    objs = np.zeros((len(rows), 16, 12), dtype=np.float32)
    mask = np.zeros((len(rows), 16), dtype=np.float32)
    for i, r in enumerate(rows):
        objs[i, 0] = r[0]
        mask[i, 0] = 1.0
    keep = m.obstacle_rows(objs, mask, 40.0, 0.3)
    assert keep.tolist() == [True, False, False, False, False, False, True]


def test_마스크가_꺼진_물체는_안_본다():
    m = _mod()
    objs = np.zeros((1, 16, 12), dtype=np.float32)
    objs[0, 0] = _obj(20.0, 0.0)
    mask = np.zeros((1, 16), dtype=np.float32)
    assert m.obstacle_rows(objs, mask).tolist() == [False]
