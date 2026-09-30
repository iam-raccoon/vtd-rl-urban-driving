"""`scripts/diag_obs.py` 잠그기 — 이 진단이 거짓말하면 (A)/(B) 판정이 통째로 뒤집힌다.

이 스크립트가 내는 답은 "관측에 물체가 들어오는가" 다. 진단 쪽이 **엉뚱한 칸을 읽어서**
빈 값을 내면 그 결과는 진짜 파이프라인 버그(B)와 **구별이 안 된다** — 그래서 아래 테스트는
열 번호·축척·마스크 출처·행동 축을 하나씩 못박고, 마지막에는 **진짜 단계 ③ 판**에서
`observation.py` 가 실제로 채운 값과 맞춰 본다.
"""
import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_diag_obs():
    """스크립트를 모듈로 불러온다(`tests/policy/test_diag_stall.py` 와 같은 패턴)."""
    path = os.path.join(REPO, "scripts", "diag_obs.py")
    spec = importlib.util.spec_from_file_location("diag_obs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs(objs, mask, ego=None):
    """관측 사전 흉내 — 물체 행렬·마스크·자차 묶음만 채운다."""
    return {"objects": np.asarray(objs, dtype=np.float32),
            "object_mask": np.asarray(mask, dtype=np.float32),
            "ego": np.asarray(ego if ego is not None else [0.0] * 9, dtype=np.float32)}


def _slot(fx=0.0, fy=0.0, speed=0.0, length=0.0, cls=(0.0, 0.0, 0.0)):
    """정규화된 물체 한 줄(12 칸) — 인자는 전부 **정규화 값**이다."""
    row = [0.0] * 12
    row[0], row[1], row[4], row[5] = fx, fy, speed, length
    row[9], row[10], row[11] = cls
    return row


# ─────────────────────────────────────────────────────────── 열·축척·마스크 출처

def test_가장_가까운_슬롯을_고른다():
    """`_nearest_live` 는 **최소** 거리여야 한다 — 최대로 뒤집히면 진단이 늘 엉뚱한 물체를 본다.

    관측은 가까운 순으로 슬롯을 채우지만(`observation.py:128`) 여기서는 일부러 **뒤집힌
    순서**로 넣는다. 슬롯 0 을 그냥 믿는 구현이면 이 테스트가 잡는다.
    """
    m = _load_diag_obs()
    from vtd_rl.env.observation import ObsConfig
    cfg = ObsConfig()
    objs = [_slot(fx=60.0 / cfg.obj_x), _slot(fx=10.0 / cfg.obj_x), _slot(fx=30.0 / cfg.obj_x)]
    assert m._nearest_live(objs, [1.0, 1.0, 1.0], cfg) == 1      # 10 m 짜리
    assert m._nearest_live(objs, [1.0, 0.0, 1.0], cfg) == 2      # 10 m 는 죽어 있다 -> 30 m
    assert m._nearest_live(objs, [0.0, 0.0, 0.0], cfg) is None
    assert m._nearest_live([], [], cfg) is None


def test_거리는_역정규화한_미터로_잰다():
    """fx 는 80 m, fy 는 20 m 로 나뉘어 있다(`ObsConfig.obj_x/obj_y`).

    정규화된 채로 `hypot` 을 재면 좌우가 4 배 무겁게 세어져 **옆 물체가 앞 물체보다
    가깝다**고 나온다. 아래 두 슬롯은 정규화 좌표로는 0 번(0.30)이 1 번(0.35)보다
    가깝지만, 미터로는 0 번이 24 m, 1 번이 8 m 라 답이 뒤집힌다.
    """
    m = _load_diag_obs()
    from vtd_rl.env.observation import ObsConfig
    cfg = ObsConfig()
    objs = [_slot(fx=0.30, fy=0.0),        # 24 m 앞
            _slot(fx=0.00, fy=0.40)]       #  8 m 옆
    assert m._nearest_live(objs, [1.0, 1.0], cfg) == 1
    got = m._obs_object(_obs(objs, [1.0, 1.0]), cfg)
    assert got["fy"] == pytest.approx(8.0)
    assert got["range"] == pytest.approx(8.0)
    # 0 번만 살려 두면 fx 축척(80 m)이 그대로 드러난다 — 20 을 곱하면 6 m 가 나온다
    got0 = m._obs_object(_obs(objs, [1.0, 0.0]), cfg)
    assert got0["fx"] == pytest.approx(24.0)
    assert got0["fy"] == pytest.approx(0.0)


def test_좌우_포화를_표시한다():
    """`fy` 는 ±`obj_y`(20 m)에서 포화한다(`observation.py:51`).

    포화하면 75 m 옆 물체도 20.00 m 로 찍혀 `range` 가 진짜 거리보다 작게 나온다 —
    "몇 m 앞에서 보였나" 를 그 값으로 답하면 조용히 틀린다. 그래서 걸음마다 표시하고
    요약은 세계 쪽 거리를 함께 낸다.
    """
    m = _load_diag_obs()
    from vtd_rl.env.observation import ObsConfig
    cfg = ObsConfig()
    sat = m._obs_object(_obs([_slot(fx=0.2, fy=-1.0)], [1.0]), cfg)
    assert sat["fy_clipped"] is True and sat["fy"] == pytest.approx(-cfg.obj_y)
    ok = m._obs_object(_obs([_slot(fx=0.2, fy=-0.9)], [1.0]), cfg)
    assert ok["fy_clipped"] is False
    assert m._obs_object(_obs([_slot()], [0.0]), cfg)["fy_clipped"] is False   # 물체 없음

    rows = _rows([(1, 1.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0)])
    rows[0]["fy_clipped"] = rows[2]["fy_clipped"] = True
    rows[1]["fy_clipped"] = False
    rows[0]["world_dist"] = 79.0
    s = m.summarize(rows, "b", 0, "collision", "policy")
    assert s["fy_clipped_steps"] == 2
    assert s["first_seen_world_dist"] == pytest.approx(79.0)   # 포화 안 먹는 진짜 거리


def test_마스크는_마스크_배열에서만_온다():
    """산 슬롯 수는 `object_mask` 가 정한다 — 물체 행렬의 내용으로 추측하면 안 된다.

    죽은 슬롯에 쓰레기가 남아 있어도(관측은 0 으로 채우지만 그건 `observation.py` 의
    약속일 뿐이다) 진단은 마스크만 봐야 한다. 반대로 값이 전부 0 인 **산** 슬롯
    (자차와 같은 자리)도 세어야 한다 — 값으로 추측하면 이 둘이 서로 바뀐다.

    ⚠ 2026-09-30 돌연변이 M3: "0 이 아닌 줄을 센다" 로 바꿨더니 **살아남았다**.
    산 1 + 죽은 1 짜리 예제는 두 세는 법이 우연히 같은 1 을 내기 때문이다. 그래서
    아래처럼 **두 답이 갈리는** 두 판(2 대 0, 0 대 2)을 함께 둔다.
    """
    m = _load_diag_obs()
    from vtd_rl.env.observation import ObsConfig
    cfg = ObsConfig()
    objs = [_slot(fx=0.0, fy=0.0),                 # 산 슬롯인데 값은 전부 0
            _slot(fx=0.5, fy=0.5, length=0.9)]     # 죽은 슬롯인데 값이 남아 있다
    got = m._obs_object(_obs(objs, [1.0, 0.0]), cfg)
    assert got["mask_sum"] == 1.0
    assert got["slot"] == 0
    assert got["range"] == pytest.approx(0.0)

    # 값이 전부 0 인 산 슬롯 둘 — 마스크는 2, 내용으로 추측하면 0
    allzero = m._obs_object(_obs([_slot(), _slot()], [1.0, 1.0]), cfg)
    assert allzero["mask_sum"] == 2.0
    assert allzero["slot"] == 0                    # 그래도 '가장 가까운 것' 은 고른다
    # 값이 남은 죽은 슬롯 둘 — 마스크는 0, 내용으로 추측하면 2
    allghost = m._obs_object(_obs([_slot(fx=0.5), _slot(fy=0.5)], [0.0, 0.0]), cfg)
    assert allghost["mask_sum"] == 0.0
    assert allghost["slot"] is None and allghost["fx"] is None


def test_열_번호가_observation_레이아웃과_같다():
    """`objects` 12 칸의 뜻(`observation.py:135-138`)을 값으로 못박는다."""
    m = _load_diag_obs()
    from vtd_rl.env.observation import ObsConfig
    cfg = ObsConfig()
    assert (m.FX_COL, m.FY_COL) == (0, 1)
    assert (m.SPEED_COL, m.LENGTH_COL) == (4, 5)
    assert m.CLASS_COLS == (9, 10, 11)
    got = m._obs_object(_obs([_slot(fx=0.25, fy=-0.5, speed=0.4, length=0.5,
                                    cls=(0.0, 1.0, 0.0))], [1.0]), cfg)
    assert got["fx"] == pytest.approx(0.25 * cfg.obj_x)       # 20 m
    assert got["fy"] == pytest.approx(-0.5 * cfg.obj_y)       # -10 m
    assert got["obj_speed"] == pytest.approx(0.4 * cfg.v_max)  # 10 m/s
    assert got["length"] == pytest.approx(0.5 * cfg.obj_size)  # 6 m
    assert got["cls"] == "person"
    assert got["cls_onehot"] == [0.0, 1.0, 0.0]


def test_클래스_원핫_순서는_차량_사람_사물이다():
    m = _load_diag_obs()
    assert m.CLASS_NAMES == ("vehicle", "person", "obstacle")
    assert m._class_name([1.0, 0.0, 0.0]) == "vehicle"
    assert m._class_name([0.0, 1.0, 0.0]) == "person"
    assert m._class_name([0.0, 0.0, 1.0]) == "obstacle"
    assert m._class_name([0.0, 0.0, 0.0]) == "?"


def test_자차_속도는_ego_0번_칸이다():
    """`ego` 1·2 번은 **직전 행동**(조향·가속)이다(`observation.py:86`).

    한 칸만 밀려도 '속도' 자리에 직전 조향이 들어앉아, 서 있는 차가 달리는 것처럼 보인다.
    """
    m = _load_diag_obs()
    from vtd_rl.env.observation import ObsConfig
    cfg = ObsConfig()
    assert m.EGO_SPEED_IDX == 0
    ego = [0.4, -0.9, 0.8] + [0.0] * 6      # 속도 0.4, 직전 조향 -0.9, 직전 가속 0.8
    row = m._row(7, _obs([_slot()], [0.0], ego=ego), _FakeState([]), _FakeEgo(),
                 {"control": [0.1, -0.2], "turn": 2}, cfg)
    assert row["speed"] == pytest.approx(0.4 * cfg.v_max)     # 10 m/s


def test_행동_축이_to_command와_같다():
    """`STEER_IDX`·`ACCEL_IDX` 를 **실제 환경 변환기에 먹여** 확인한다.

    숫자 0·1 을 리터럴로 비교만 하면 환경 쪽 축이 바뀔 때 같이 틀린다
    (`tests/policy/test_diag_stall.py` 가 같은 이유로 두 겹을 둔다).
    """
    from vtd_rl.env.action import ActionConfig, to_command
    from vtd_rl.env.observation import ObsConfig

    m = _load_diag_obs()
    cfg = ActionConfig()
    assert (m.STEER_IDX, m.ACCEL_IDX) == (0, 1)

    control = [0.0, 0.0]
    control[m.STEER_IDX] = 1.0
    steer, accel, _turn = to_command({"control": control, "turn": 0}, cfg)
    assert steer == pytest.approx(cfg.max_steer) and accel == pytest.approx(0.0)

    control = [0.0, 0.0]
    control[m.ACCEL_IDX] = 1.0
    steer, accel, _turn = to_command({"control": control, "turn": 0}, cfg)
    assert steer == pytest.approx(0.0) and accel == pytest.approx(cfg.accel_max)

    row = m._row(0, _obs([_slot()], [0.0]), _FakeState([]), _FakeEgo(),
                 {"control": [0.3, -0.7], "turn": 1}, ObsConfig())
    assert row["steer"] == pytest.approx(0.3)
    assert row["accel"] == pytest.approx(-0.7)
    assert row["turn"] == 1


# ─────────────────────────────────────────────────────────── 세계 쪽 · 요약

class _FakeEgo:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y


class _FakeObj:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeState:
    def __init__(self, objs):
        self.objects = objs


def test_세계_쪽은_State_objects_를_센다():
    """(B) 판별의 반쪽 — 세계가 낸 물체 수와 최단 수평 거리."""
    m = _load_diag_obs()
    assert m._world_object(_FakeState([]), _FakeEgo()) == {"world_n": 0, "world_dist": None}
    got = m._world_object(_FakeState([_FakeObj(30.0, 40.0), _FakeObj(3.0, 4.0)]), _FakeEgo())
    assert got["world_n"] == 2
    assert got["world_dist"] == pytest.approx(5.0)


def _rows(spec):
    """(world_n, mask_sum, accel, steer) 네 칸만 다른 걸음들."""
    return [{"step": i, "world_n": w, "mask_sum": ms, "accel": a, "steer": s,
             "turn": 0, "range": None, "fx": None, "cls": None, "world_dist": None,
             "speed": 10.0}
            for i, (w, ms, a, s) in enumerate(spec)]


def test_blind_steps가_세계와_관측의_어긋남을_센다():
    """**이 한 칸이 (A)/(B) 를 가른다.**

    세계엔 물체가 있는데(`world_n > 0`) 관측 마스크가 0 인 걸음 = 관측이 안 찬 걸음.
    반대 방향(`ghost_steps`)도 따로 센다 — 둘을 한 칸으로 묶으면 어느 쪽이 끊겼는지 모른다.

    ⚠ 2026-09-30 돌연변이 M7: 조건을 `mask_sum > 0`(어긋남 대신 **일치**)로 뒤집었을 때
    '어긋남 1, 일치 1' 짜리 예제는 **양쪽 다 1 을 내서** 아무 말을 안 했다. 그래서
    일치 걸음을 하나 더 둬 두 답이 1 대 2 로 갈리게 한다.
    """
    m = _load_diag_obs()
    rows = _rows([(0, 0.0, 1.0, 0.0),    # 세계에도 관측에도 없다 — 어긋남 아님
                  (1, 0.0, 1.0, 0.0),    # 세계엔 있는데 관측이 비었다 -> blind (1 개뿐)
                  (1, 1.0, 1.0, 0.0),    # 둘 다 있다 — 정상 (2 개)
                  (1, 1.0, 1.0, 0.0),
                  (0, 1.0, 1.0, 0.0)])   # 관측에만 있다 -> ghost
    s = m.summarize(rows, "course_H", 0, "collision", "policy")
    assert s["blind_steps"] == 1         # 일치를 세는 구현이면 2 가 나온다
    assert s["ghost_steps"] == 1
    assert s["world_seen_steps"] == 3
    assert s["obs_seen_steps"] == 3


def test_요약이_물체_전후_행동을_가른다():
    """"보고도 반응이 없나" 는 물체가 보이기 **전/후** 행동을 갈라야 답할 수 있다.

    두 묶음이 섞이면(예: 전부를 한 평균으로) 접근 중 감속이 있었는지 없었는지가 지워진다.
    """
    m = _load_diag_obs()
    rows = _rows([(0, 0.0, 1.0, 0.1), (0, 0.0, 0.6, -0.1),      # 안 보일 때: 가속 1.0, 0.6
                  (1, 1.0, 0.4, 0.5), (1, 1.0, -0.2, -0.3)])    # 보일 때: 0.4, -0.2
    s = m.summarize(rows, "course_H", 0, "collision", "policy")
    assert s["accel_before_mean"] == pytest.approx(0.8)
    assert s["accel_seen_mean"] == pytest.approx(0.1)
    assert s["accel_seen_min"] == pytest.approx(-0.2)
    assert s["steer_before_absmean"] == pytest.approx(0.1)
    assert s["steer_seen_absmean"] == pytest.approx(0.4)
    assert s["steer_seen_absmax"] == pytest.approx(0.5)
    assert s["first_seen_step"] == 2
    assert s["steps"] == 4 and s["outcome"] == "collision" and s["final"]["step"] == 3


def test_직전_창은_보인_걸음과_같은_길이다():
    """`accel_before_mean` 은 **출발 가속**이 섞여 늘 크게 나온다 — 그래서 못 쓴다.

    아래 판은 출발 4 걸음이 전력 가속(1.0)이고 그 뒤 순항 2 걸음(0.1)에서 물체가 보인다.
    `accel_before_mean` 은 0.7 이라 "보고 나서 가속이 줄었다" 가 저절로 참이 되지만,
    같은 길이(2 걸음) 직전 창은 0.1 이라 **실제로는 아무것도 안 달라졌다**고 말한다.
    창이 전체(`before`)로 새면 이 구별이 사라진다.
    """
    m = _load_diag_obs()
    rows = _rows([(0, 0.0, 1.0, 0.0), (0, 0.0, 1.0, 0.0),        # 출발 전력 가속
                  (0, 0.0, 0.1, 0.0), (0, 0.0, 0.1, 0.0),        # 순항(직전 창)
                  (1, 1.0, 0.1, 0.0), (1, 1.0, 0.1, 0.0)])       # 물체가 보인다
    s = m.summarize(rows, "b", 0, "collision", "policy")
    assert s["accel_before_mean"] == pytest.approx(0.55)   # 출발이 섞인 값 — 오해를 부른다
    assert s["accel_pre_mean"] == pytest.approx(0.1)       # 같은 길이 직전 창(2 걸음)
    assert s["accel_seen_mean"] == pytest.approx(0.1)      # 달라진 게 없다
    assert s["speed_pre_mean"] == pytest.approx(10.0)
    # 물체가 한 번도 안 보인 판에서는 직전 창도 정의되지 않는다
    none_seen = m.summarize(_rows([(0, 0.0, 0.3, 0.0)]), "b", 0, "goal", "policy")
    assert none_seen["accel_pre_mean"] is None and none_seen["accel_seen_mean"] is None


def test_tail은_뒤쪽_n걸음이다():
    """'종료 직전' 은 **뒤쪽**이다 — 앞쪽을 보면 출발 직후를 충돌 직전이라 부르게 된다."""
    m = _load_diag_obs()
    rows = [{"step": i} for i in range(10)]
    assert [r["step"] for r in m._tail(rows, 3)] == [7, 8, 9]
    assert len(m._tail(rows, 100)) == 10
    assert m._tail([], 3) == []
    assert m._tail(rows, 0) == []


def test_표는_걸음마다_한_줄이다():
    m = _load_diag_obs()
    rows = _rows([(1, 1.0, 0.5, -0.2), (1, 1.0, 0.4, -0.1)])
    table = m.markdown_table(rows).splitlines()
    assert table[0].startswith("| step |") and "mask_sum" in table[0] and "world_n" in table[0]
    assert len(table) == 2 + len(rows)        # 머리글 + 구분선 + 줄마다 하나
    assert "—" in table[2]                    # None 은 빈칸이 아니라 대시로 보인다


# ─────────────────────────────────────────────────────────── 진짜 판에서 맞춰 보기

@pytest.mark.parametrize("ego_s, want_cls, want_len", [(150.0, "obstacle", 0.15),
                                                       (1090.0, "vehicle", 4.5)])
def test_진짜_단계3_판에서_관측_값과_세계_값이_맞는다(ego_s, want_cls, want_len):
    """**이 진단이 `observation.py` 와 같은 칸을 읽는다는 유일한 증거.**

    `course_H` 는 s=200 에 라바콘(0.15×0.46×0.61), s=1140 에 정지 차량(4.5×1.8×1.5)이
    있다(`curricula/stage3.json`). 자차를 그 50 m 앞으로 옮겨 한 걸음 밟고, 진단이 낸
    `fx`·`range`·종류가 **세계가 실제로 쥔 물체**와 맞는지 본다.

    - fx 와 fy 가 바뀌면 `fx≈50` 이 `fx≈1` 로 떨어진다.
    - 축척을 헷갈리면(fx 에 obj_y=20) `range` 가 `world_dist` 의 1/4 로 어긋난다.
    - 클래스 열이 밀리면 라바콘이 차량으로 둔갑한다(치수 기준은 `observation.py:60-71`).
    """
    from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
    from vtd_rl.world import dynamics as dyn
    from vtd_rl.world.place import route_point

    m = _load_diag_obs()
    board = next(b for b in m._stage_boards("stage3") if b.name == "course_H")
    env = VtdDriveEnv([board], EnvConfig())
    try:
        env.reset(seed=0, options={"board": board.name})
        x, y, h = route_point(board, ego_s, 0.0)
        env.world.ego = dyn.EgoState(x, y, h)          # 물체 50 m 앞으로 순간이동
        act = {"control": np.zeros(2, dtype=np.float32), "turn": 0}
        obs, _r, _t, _tr, _info = env.step(act)
        row = m._row(0, obs, env.state, env.world.ego, act, env.cfg.obs)
    finally:
        env.close()

    assert row["world_n"] == 1                          # 세계는 그 물체 하나를 쥐고 있다
    assert row["mask_sum"] == 1.0                       # 관측에도 정확히 하나 들어왔다
    assert row["fx"] == pytest.approx(50.0, abs=3.0)    # 전후로 ~50 m — 좌우가 아니다
    assert abs(row["fy"]) < 5.0
    assert row["range"] == pytest.approx(row["world_dist"], abs=0.5)   # 축척이 맞다
    assert row["cls"] == want_cls
    assert row["length"] == pytest.approx(want_len, abs=0.05)
    assert row["obj_speed"] == pytest.approx(0.0, abs=0.01)            # 정지 액터


def test_스크립트가_요약_한_줄을_낸다(tmp_path):
    """CLI 가 돌고 요약의 **모양**이 나오는지 — 체크포인트는 그 자리에서 만든다.

    `runs/` 의 학습 체크포인트를 게이트로 걸지 않는다(그건 OMEN 에만 있어서 개발
    머신에서는 영원히 SKIP 이다 — `test_diag_stall.py` 가 그렇게 한 번 데었다).
    무작위 정책이라 주행은 엉망이고 걸음도 짧게 끊지만, 여기서 보는 건 요약의 모양이다.
    """
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig

    ck = str(tmp_path / "policy.pt")
    DrivePolicy(PolicyConfig(squash=True)).save(ck)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "diag_obs.py"),
                          "--checkpoint", ck, "--stage", "stage3", "--boards", "course_H",
                          "--seeds", "0", "--max-steps", "30", "--teacher-seeds",
                          "--device", "cpu", "--table"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = out.stdout.strip().splitlines()
    assert any(line.startswith("| step |") for line in lines)     # --table 이 표를 냈다
    total = json.loads(lines[-1])
    assert total["episodes"] == 1 and total["stage"] == "stage3"
    assert total["teacher_seeds"] == []                            # 빈 목록이면 선생님은 안 돈다
    assert set(total) >= {"blind_steps_total", "ghost_steps_total", "verdict", "boards"}
    s = total["boards"][0]
    assert s["board"] == "course_H" and s["kind"] == "policy" and s["steps"] <= 30
    assert set(s) >= {"blind_steps", "first_seen_range", "accel_before_mean", "accel_seen_mean"}


def test_판정문구는_blind_steps가_정한다(tmp_path):
    """`verdict` 는 장식이 아니다 — `blind_steps_total` 이 0 이면 A, 0 보다 크면 B 다.

    스크립트를 두 번 돌릴 수는 없으니 판정 규칙만 여기서 못박는다(같은 식이 `main()` 에
    있다). 규칙이 반대로 뒤집히면 (A)/(B) 결론이 통째로 뒤바뀐다.
    """
    m = _load_diag_obs()
    src = open(os.path.join(REPO, "scripts", "diag_obs.py"), encoding="utf-8").read()
    assert '"B: ' in src and '"A: ' in src
    # 어긋남이 있는 판 / 없는 판을 각각 요약해 blind_steps 가 실제로 갈리는지 본다
    bad = m.summarize(_rows([(1, 0.0, 0.0, 0.0)]), "b", 0, "collision", "policy")
    good = m.summarize(_rows([(1, 1.0, 0.0, 0.0)]), "b", 0, "collision", "policy")
    assert bad["blind_steps"] == 1 and good["blind_steps"] == 0
