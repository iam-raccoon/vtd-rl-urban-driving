"""경로-상대 좌표로 액터를 놓는 순수 함수.

커리큘럼 JSON 은 세계 좌표를 모른다 — 코스마다 지도 좌표가 다르고 사람이 쓸 수 없다.
그래서 판 기술은 **(호 길이 `s`, 횡 오프셋 `lateral`)** 로 적고, 여기서 세계 좌표로 바꾼다.
`lateral` 은 경로 진행 방향 **왼쪽이 +** 다(`world/route.py` 의 `Projection.lateral` 과 같은 부호).

⚠ 이 모듈은 `vtd_rl.world.board` 를 **임포트하지 않는다**(순환 임포트: board 가 이 모듈을
   쓴다). 판은 인자로만 받고, 쓰는 것은 `board.route` 뿐이다.

⚠ 허용하는 운동 `kind` 는 **읽기 전용 규칙 스택이 실제로 구현한 것만**이다
   (`third_party/rule_stack/eval/mock_vtd.py:44-63`). 거기서 모르는 kind 는 오류가 아니라
   **조용히 정지 물체**가 된다(`:63`) — 판 하나가 통째로 의도와 달라져도 아무도 모른다.
   그래서 여기서 `ValueError` 로 막는다.
"""
import math

from vtd_rl import rule_stack as rs

#: `mock_vtd.init_actor_state` 가 실제로 분기하는 운동 kind.
#:   static(:50-53) / lane(:54-56) / constant·crossing(:57-59) / waypoints(:60-62)
MOTION_KINDS = ("static", "lane", "constant", "crossing", "waypoints")

#: 움직이는 kind — `speed` 를 반드시 받는다(빠뜨리면 조용히 0 m/s 인 유령 차가 된다).
_MOVING_KINDS = ("lane", "constant", "crossing", "waypoints")

#: 액터 기술이 받는 필드. 오타는 조용한 오배치가 되므로 벗어나면 터뜨린다.
ACTOR_KEYS = {"id", "kind", "type", "size", "s", "lateral", "speed", "pts", "spawn", "events"}

#: 등장 트리거 형태(`mock_vtd.spawned`, `:108-117`). 셋 중 하나는 있어야 한다.
_SPAWN_KEYS = ("at_time", "ego_x", "ego_within")


def route_point(board, s, lateral=0.0):
    """(호 길이 `s`, 횡 오프셋 `lateral`) → 세계 `(x, y, heading)`.

    `referee/scenarios.py:56-58` 의 `_route_point` 를 그대로 옮긴 것이다(동작 동일).
    """
    x, y, h = board.route.point_at(s)
    return x - lateral * math.sin(h), y + lateral * math.cos(h), h


def _need(spec, key):
    if key not in spec:
        raise ValueError(f"액터 기술에 '{key}' 가 없다: {spec}")
    return spec[key]


def _check_spawn(spawn, spec):
    if not isinstance(spawn, dict) or not any(k in spawn for k in _SPAWN_KEYS):
        # 셋 다 없으면 `mock_vtd.spawned` 가 영원히 False 를 돌려준다 — 액터가
        # 판에 있는데 한 번도 안 나타나고, 그 판은 액터 0 개인 판과 구별이 안 된다.
        raise ValueError(f"액터 등장 트리거는 {list(_SPAWN_KEYS)} 중 하나를 써야 한다: {spawn}")
    if "ego_within" in spawn and "of" not in spawn:
        raise ValueError(f"'ego_within' 등장 트리거는 'of': [x, y] 가 필요하다: {spawn}")
    unknown = sorted(set(spawn) - set(_SPAWN_KEYS) - {"of"})
    if unknown:
        raise ValueError(f"액터 {spec.get('id', '?')} 의 등장 트리거에 모르는 키: {', '.join(unknown)}")
    return spawn


def actor_from_spec(board, spec: dict):
    """커리큘럼의 액터 기술 한 개를 `rs.Actor`(세계 좌표)로 바꾼다.

    받는 필드: `id`·`kind`·`size`·`s`(필수), `type`(기본 "vehicle"), `lateral`(기본 0.0),
    `speed`(움직이는 kind 는 필수), `pts`(waypoints 용 경로-상대 `[[s, lateral], ...]`),
    `spawn`(기본 `{"at_time": 0.0}`), `events`(그대로 넘긴다).
    """
    unknown = sorted(set(spec) - ACTOR_KEYS)
    if unknown:
        raise ValueError(f"액터 기술에 모르는 키가 있다: {', '.join(unknown)} "
                         f"(허용: {', '.join(sorted(ACTOR_KEYS))})")
    kind = _need(spec, "kind")
    if kind not in MOTION_KINDS:
        raise ValueError(f"모르는 운동 kind '{kind}' — 규칙 스택이 구현한 것은 "
                         f"{', '.join(MOTION_KINDS)} 뿐이다(mock_vtd.py:44-63)")
    aid = _need(spec, "id")
    size = list(_need(spec, "size"))
    if len(size) != 3:
        raise ValueError(f"액터 {aid} 의 size 는 [길이, 폭, 높이] 세 개다: {size}")
    s = float(_need(spec, "s"))
    lateral = float(spec.get("lateral", 0.0))
    x, y, h = route_point(board, s, lateral)

    if kind in _MOVING_KINDS and "speed" not in spec:
        raise ValueError(f"액터 {aid}({kind}) 는 'speed' 가 필요하다")
    speed = float(spec.get("speed", 0.0))

    if kind == "static":
        # ★ `hd` 를 반드시 넣는다. 빠뜨리면 `mock_vtd.py:52` 가 방위를 0 으로 고정하고,
        #   채점기가 상자 대 상자(SAT)로 재므로 굽은 코스에서 긴 차의 상자가 도로를
        #   가로질러 눕는다 — 없는 충돌(-50). `mock_vtd.py:46-50` 의 실측 경고.
        motion = {"kind": "static", "pos": [x, y], "hd": h}
    elif kind == "crossing":
        # 횡단(보행자) — 경로를 **가로지른다**. `referee/scenarios.py:122-127` 의 패턴:
        # 경로 왼쪽 방향이 (-sin h, +cos h) 이므로 그 방향으로 speed 를 준다.
        motion = {"kind": "crossing", "pos": [x, y],
                  "vel": [-speed * math.sin(h), speed * math.cos(h)]}
    elif kind == "constant":
        # 등속 — 단계 ③ 에서 쓰는 "아주 느린 정체 차"다. 가로지르는 보행자와 달리
        # **경로 진행 방향**으로 간다(그래서 crossing 과 속도 방향이 다르다).
        motion = {"kind": "constant", "pos": [x, y],
                  "vel": [speed * math.cos(h), speed * math.sin(h)]}
    elif kind == "lane":
        # ⚠ `mock_vtd.py:54-56` 의 lane 은 방위 0 고정(+x)으로만 달린다 — 합성 직선
        #   도로용이다. 실제 지도 경로에서는 경로 방향이 0 에 가까운 데서만 뜻이 있다.
        motion = {"kind": "lane", "x0": x, "y0": y, "speed": speed}
    else:   # waypoints — 경로-상대 [[s, lateral], ...] 를 세계 좌표로 편다
        pts = _need(spec, "pts")
        if len(pts) < 2:
            raise ValueError(f"액터 {aid} 의 waypoints 는 점이 2 개 이상이어야 한다: {pts}")
        motion = {"kind": "waypoints", "speed": speed,
                  "pts": [list(route_point(board, float(ps), float(pl))[:2]) for ps, pl in pts]}

    spawn = _check_spawn(dict(spec.get("spawn") or {"at_time": 0.0}), spec)
    return rs.Actor(id=aid, type=spec.get("type", "vehicle"), size=size,
                    spawn=spawn, motion=motion, events=list(spec.get("events") or []))
