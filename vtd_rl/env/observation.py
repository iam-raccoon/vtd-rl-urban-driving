"""관측 — 스펙 §4.1. 자차 기준 좌표, 정규화, float32.

2부에서 영상 인코더로 갈아 끼울 수 있게 관측 생성은 이 모듈 하나에 모은다.
물체는 (개수 고정 + 마스크)로 내보내고, DeepSets 합치기는 정책이 한다.
"""
import bisect
import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from vtd_rl import rule_stack as rs
from vtd_rl.env.board_index import board_index
from vtd_rl.referee.judges.contact import PERSON_MIN_H

TL_STATES = 7             # vtd_io: UNSET·RED·YELLOW·GREEN·LEFT·GREEN_LEFT·FLASH
PED_L, PED_W = rs.score_fma.PED_L, rs.score_fma.PED_W     # 사람·이륜차 판별(채점기와 같은 치수)
PED_MIN_H = PERSON_MIN_H  # judges/contact.py 의 ⑪⑭ 판정과 같은 키 문턱(score_fma.PED_MIN_H=0.85 아님)


@dataclass(frozen=True)
class ObsConfig:
    objects: int = 16
    route_points: int = 20
    route_ahead: float = 50.0        # 경로점을 이만큼 앞까지 고르게 뽑는다[m]
    v_max: float = 25.0              # 속도 정규화[m/s] (90 km/h)
    yaw_max: float = 1.0             # 요레이트 정규화[rad/s]
    lat_max: float = 5.0             # 횡오프셋 정규화[m]
    room_max: float = 8.0            # 차로 여유 정규화[m]
    dist_max: float = 100.0          # 앞쪽 거리 정규화[m]
    obj_x: float = 80.0              # 물체 전후 정규화[m] — World.object_range(9910 수평 범위)와 맞춘다
    obj_y: float = 20.0              # 물체 좌우 정규화[m]
    obj_size: float = 12.0           # 물체 치수 정규화[m]
    # M7h — 앞당긴 제한속도. True 면 `ego` 묶음의 lim·v−lim 에, 앞에 낮은 제한속도가 있을 때 거기까지
    # lim_decel 로 줄이는 곡선을 따라 미리 낮아진 값을 쓴다(관측 차원은 그대로). 기본 False 면 예전과 같다.
    lim_anticipate: bool = False
    lim_decel: float = 1.5           # [m/s²]
    lim_margin: float = 5.0          # [m] — 내려가는 자리보다 이만큼 앞에서 새 제한속도에 닿는다
    lim_lookahead: float = 150.0     # [m]
    # VTD 검증 — VTD 의 녹색은 대부분 '녹색 + 좌회전 화살표'(TL_GREEN_LEFT = 5) 인데 오프라인 세계는 녹색(3)만 낸다.
    # True 면 관측의 신호 상태 5 를 3 으로 읽는다. 채점기(item_traffic_light)와 규칙 스택(drive.py)도 5 를 녹색으로 본다.
    # 오프라인 세계는 5 를 내지 않으므로 오프라인 결과는 이 값과 무관하다.
    tl_green_left_as_green: bool = False


def observation_space(cfg: ObsConfig = ObsConfig()) -> spaces.Dict:
    def box(shape, lo=-1.0, hi=1.0):
        return spaces.Box(low=lo, high=hi, shape=shape, dtype=np.float32)
    return spaces.Dict({
        "ego": box((9,)),
        "route": box((cfg.route_points, 2)),
        "nav": box((3,)),
        "plan": box((10,)),
        "signal": box((TL_STATES + 4,)),
        "objects": box((cfg.objects, 12)),
        "object_mask": box((cfg.objects,), 0.0, 1.0),
    })


def _clip(value, scale):
    return float(np.clip(value / scale, -1.0, 1.0))


def anticipated_limit(idx, s: float, lim: float, cfg: ObsConfig) -> float:
    """앞당긴 제한속도[m/s] — `s` 앞 lim_lookahead 안의 내려감마다 등감속 곡선 값을 구해 가장 작은 것(M7h)."""
    i = bisect.bisect_right(idx.lim_drop_s, s)
    for k in range(i, len(idx.lim_drop_s)):
        ds = idx.lim_drop_s[k] - s
        if ds > cfg.lim_lookahead:
            break
        lim = min(lim, math.sqrt(idx.lim_drop_v[k] ** 2 + 2.0 * cfg.lim_decel * max(ds - cfg.lim_margin, 0.0)))
    return lim


def _turn_onehot(turn):
    out = [0.0, 0.0, 0.0]
    out[int(turn) if int(turn) in (0, 1, 2) else 0] = 1.0
    return out


def _object_class(o):
    """치수로 종류를 나눈다 — `referee/judges/contact.py` 의 ⑪⑭ 판정과 같은 기준(차량·사람류·사물).

    관측의 분류가 보상을 내는 심판의 분류와 어긋나면 정책이 틀린 신호를 배운다 —
    그래서 PED_MIN_H 는 score_fma.PED_MIN_H(0.85) 가 아니라 judges/contact.py 의
    PERSON_MIN_H(1.2, item_contact 안의 숫자를 그대로 옮김)를 쓴다.
    """
    if o.length > PED_L:
        return [1.0, 0.0, 0.0]
    if o.length <= PED_L and o.width <= PED_W and o.height >= PED_MIN_H:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def build_observation(world, state, info, prev_action, cfg: ObsConfig = ObsConfig()) -> dict:
    board = world.board
    idx = board_index(board)
    route = board.route
    ego = world.ego
    s = info.s if info is not None else 0.0
    lateral = info.lateral if info is not None else 0.0
    index = info.index if info is not None else 0
    plan = idx.plan_at(index)
    lim = plan.get("lim") or 0.0
    if cfg.lim_anticipate and lim > 0.0:
        lim = anticipated_limit(idx, s, lim, cfg)
    yaw_rate = ego.v * math.tan(ego.steer) / world.cfg.dynamics.wheelbase

    ego_vec = [_clip(ego.v, cfg.v_max), float(np.clip(prev_action[0], -1.0, 1.0)),
               float(np.clip(prev_action[1], -1.0, 1.0)), _clip(yaw_rate, cfg.yaw_max)]
    ego_vec += _turn_onehot(prev_action[2])
    ego_vec += [_clip(lim, cfg.v_max), _clip(ego.v - lim, cfg.v_max)]

    cos_h, sin_h = math.cos(-ego.heading), math.sin(-ego.heading)
    pts = np.zeros((cfg.route_points, 2), dtype=np.float32)
    step = cfg.route_ahead / cfg.route_points
    for k in range(cfg.route_points):
        px, py, _ph = route.point_at(min(s + step * (k + 1), route.total))
        dx, dy = px - ego.x, py - ego.y
        pts[k, 0] = _clip(dx * cos_h - dy * sin_h, cfg.route_ahead)
        pts[k, 1] = _clip(dx * sin_h + dy * cos_h, cfg.route_ahead)

    heading_err = (route.heading_at(index) - ego.heading + math.pi) % (2.0 * math.pi) - math.pi
    # 남은 거리는 **그 판의 경로 길이**로 나눈다(1 = 출발, 0 = 끝). 고정 1000 m 로 나누면
    # 단계 ① 경로(2152~5242 m)에서는 판의 절반~8할 동안 1.0 에 붙어 죽은 값이 된다.
    nav = [_clip(lateral, cfg.lat_max), _clip(heading_err, math.pi),
           _clip(route.total - s, max(route.total, 1.0))]

    sig_dir = plan.get("sig") or 0
    plan_vec = [_clip(plan.get("w") or 0.0, cfg.room_max),
                _clip(plan.get("l") or 0.0, cfg.room_max),
                _clip(plan.get("r") or 0.0, cfg.room_max),
                _clip(plan.get("xl") or 0.0, cfg.room_max),
                _clip(plan.get("xr") or 0.0, cfg.room_max),
                _clip(plan.get("need") or 0.0, cfg.lat_max),
                1.0 if plan.get("j") else 0.0,
                1.0 if sig_dir > 0 else 0.0,
                1.0 if sig_dir < 0 else 0.0,
                _clip(idx.ahead(s, "junction"), cfg.dist_max)]   # 다음 교차로까지(신호 거리는 signal 묶음에)

    signal = [0.0] * TL_STATES
    tl_state = int(state.tl_state) if 0 <= int(state.tl_state) < TL_STATES else 0
    if cfg.tl_green_left_as_green and tl_state == rs.TL_GREEN_LEFT:
        tl_state = rs.TL_GREEN
    signal[tl_state if state.tl_id > 0 else 0] = 1.0
    signal += [_clip(idx.ahead(s, "signal"), cfg.dist_max),
               _clip(idx.ahead(s, "stopline"), cfg.dist_max),
               _clip(idx.ahead(s, "crosswalk"), cfg.dist_max),
               1.0 if (index < len(idx.zone) and idx.zone[index]) else 0.0]

    objs = np.zeros((cfg.objects, 12), dtype=np.float32)
    mask = np.zeros((cfg.objects,), dtype=np.float32)
    near = sorted(state.objects, key=lambda o: math.hypot(o.x - ego.x, o.y - ego.y))
    for k, o in enumerate(near[:cfg.objects]):
        dx, dy = o.x - ego.x, o.y - ego.y
        fx = dx * cos_h - dy * sin_h
        fy = dx * sin_h + dy * cos_h
        dh = (o.heading - ego.heading + math.pi) % (2.0 * math.pi) - math.pi
        p = route.project(o.x, o.y, hint=index)
        objs[k] = [_clip(fx, cfg.obj_x), _clip(fy, cfg.obj_y), math.cos(dh), math.sin(dh),
                   _clip(o.speed, cfg.v_max), _clip(o.length, cfg.obj_size),
                   _clip(o.width, cfg.obj_size), _clip(o.height, cfg.obj_size),
                   _clip(p.lateral, cfg.lat_max)] + _object_class(o)
        mask[k] = 1.0

    return {
        "ego": np.asarray(ego_vec, dtype=np.float32),
        "route": pts,
        "nav": np.asarray(nav, dtype=np.float32),
        "plan": np.asarray(plan_vec, dtype=np.float32),
        "signal": np.asarray(signal, dtype=np.float32),
        "objects": objs,
        "object_mask": mask,
    }
