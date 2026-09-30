"""경로-상대 좌표로 액터를 놓는 순수 함수.

커리큘럼 JSON 은 세계 좌표를 모른다 — 코스마다 지도 좌표가 다르고 사람이 쓸 수 없다.
그래서 판 기술은 **(호 길이 `s`, 횡 오프셋 `lateral`)** 로 적고, 여기서 세계 좌표로 바꾼다.
`lateral` 은 경로 진행 방향 **왼쪽이 +** 다(`world/route.py` 의 `Projection.lateral` 과 같은 부호).

⚠ 이 모듈은 `vtd_rl.world.board` 를 **임포트하지 않는다**(순환 임포트: board 가 이 모듈을
   쓴다). 판은 인자로만 받고, 쓰는 것은 `board.route` 와 `board.lane_plan` 뿐이다.

⚠ 허용하는 운동 `kind` 는 **읽기 전용 규칙 스택이 실제로 구현한 것만**이다
   (`third_party/rule_stack/eval/mock_vtd.py:44-63`). 거기서 모르는 kind 는 오류가 아니라
   **조용히 정지 물체**가 된다(`:63`) — 판 하나가 통째로 의도와 달라져도 아무도 모른다.
   그래서 여기서 `ValueError` 로 막는다.
"""
import bisect
import math
import random

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

#: 판 항목의 `jitter` 가 받는 축. 축마다 균등분포 ± 최대 폭[m] 이다.
JITTER_KEYS = {"s", "lateral"}

#: `jitter` 를 안 적은 판의 기본 흔들기 폭. **0 이 아니다** — 변종을 켰는데 배치가 똑같은
#: 것이야말로 이 장치가 막으려는 조용한 실패이기 때문이다. 끄려면 `{"s": 0, "lateral": 0}`
#: 을 명시해라. 40 m 는 여섯 코스 전부에서 "규칙 스택이 비킬 차로가 있는" 구간 안에 머무는
#: 폭이다(2026-09-30 전 액터 ±40 m 를 1 m 간격으로 전수 확인, ±60 m 부터 깨진다).
DEFAULT_JITTER = {"s": 40.0, "lateral": 0.6}

#: 흔든 뒤에도 비워 둘 경로 앞뒤 여유[m]. `tests/test_stage3.py::test_액터는_경로_처음과_
#: 끝_100m_를_비운다` 와 같은 값이다 — 출발 직후·목표 직전의 물체는 "완주 못 한 이유" 를
#: 배치인지 종료 판정인지 못 가리게 만든다.
JITTER_MARGIN_M = 100.0

#: 변종의 호 길이 오프셋이 피해 가는 0 언저리[m]. 변종 0 이 안 흔든 원본이라, 오프셋이
#: 0 에 붙은 변종은 원본의 사본이 되어 수집 시간만 먹는다.
JITTER_DEADZONE_M = 5.0


def check_jitter(jitter):
    """판 항목의 `jitter` 를 검사해 돌려준다(`None` 이면 기본값).

    모르는 축은 **조용히 무시하지 않는다** — `"lat": 0.6` 같은 오타 하나면 횡 흔들기가
    통째로 안 걸리고, 그 변종 묶음은 s 만 민 판이 되어 겉보기로는 멀쩡하다.
    """
    if jitter is None:
        return dict(DEFAULT_JITTER)
    if not isinstance(jitter, dict):
        raise ValueError(f"jitter 는 {{'s': ..., 'lateral': ...}} 꼴이어야 한다: {jitter!r}")
    unknown = sorted(set(jitter) - JITTER_KEYS)
    if unknown:
        raise ValueError(f"jitter 에 모르는 축이 있다: {', '.join(unknown)} "
                         f"(허용: {', '.join(sorted(JITTER_KEYS))})")
    out = dict(DEFAULT_JITTER)
    for key in JITTER_KEYS:
        if key in jitter:
            amp = float(jitter[key])
            if amp < 0.0:
                raise ValueError(f"jitter['{key}'] 는 0 이상이어야 한다: {amp}")
            out[key] = amp
    return out


def half_lane_at(board, s):
    """호 길이 `s` 지점의 **차로 반폭**[m]. 횡 흔들기를 여기서 자른다."""
    i = bisect.bisect_right(board.route.cum, s) - 1
    i = max(0, min(i, len(board.lane_plan) - 1))
    return float(board.lane_plan[i]["w"]) / 2.0


#: 황금비의 소수부. 변종 번호를 **저불일치(low-discrepancy) 수열**로 펴는 데 쓴다 —
#: 균등분포로 변종마다 따로 뽑으면 우연히 겹친다(실측: `course_B` 의 변종 4·7 이 둘 다
#: +22.2 m 로 사실상 같은 판이 됐다). `frac(위상 + k·φ)` 는 k 가 몇이든 고르게 퍼지고,
#: **변종 번호만으로 정해지므로 변종 수 N 을 바꿔도 앞의 변종이 안 움직인다.**
_GOLDEN = 0.6180339887498949


def jitter_rng(key: str) -> random.Random:
    """`(판 이름, 변종 번호)` 문자열 하나로 결정되는 rng.

    ⚠ 내장 `hash()` 를 쓰면 안 된다 — 문자열 해시는 프로세스마다 소금이 달라서(PYTHONHASHSEED)
      수집 일꾼 프로세스마다 다른 판이 나온다. `random.Random(str)` 은 문자열을 sha512 로
      섞으므로 프로세스·실행이 달라도 같다.
    """
    return random.Random(key)


def s_offset(name: str, variant: int, amp: float) -> float:
    """판 `name` 의 변종 `variant` 가 액터 전체를 밀 호 길이[m].

    판마다 다른 위상에서 시작해 변종 번호를 황금비로 편다 — 같은 판의 변종끼리 서로
    충분히 떨어지면서(균등 난수는 겹친다) 변종 수를 바꿔도 앞의 변종은 안 움직인다.
    0 언저리는 비워 둔다(`JITTER_DEADZONE_M`) — 변종 0 이 안 흔든 원본이라, 거기에 겹치는
    변종은 원본의 사본일 뿐이다.
    """
    if amp <= 0.0:
        return 0.0
    dead = min(JITTER_DEADZONE_M, amp)
    u = (jitter_rng(name).random() + variant * _GOLDEN) % 1.0
    v = 2.0 * u - 1.0
    return math.copysign(dead + abs(v) * (amp - dead), v or 1.0)


def jitter_specs(board, specs, jitter, name: str, variant: int) -> list:
    """액터 기술 목록을 `(판, 변종)` 마다 결정적으로 흔든 **새 목록**으로 돌려준다.

    ★ `s` 는 액터마다 따로 흔들지 않고 **판 하나에 공통 오프셋 하나**를 민다.
      액터 사이 간격이 그대로 남기 때문이다 — `tests/test_stage3.py` 가 잠근 두 가지
      "달릴 수 있는 배치" 조건(정지차 뒤 60 m 복귀 거리, 그 밖 10 m)은 **간격**의 조건이고,
      액터마다 따로 흔들면 코스 A 의 라바콘 줄(15 m 간격)과 코스 G 의 정지차→라바콘
      70 m 가 곧바로 깨진다(±40 m 면 음수 간격까지 간다). 공통 오프셋은 간격을 한 치도
      안 바꾸면서 "물체를 어디서 만나는가"(곡률·신호·속도)를 전부 바꾼다.

    ★ `lateral` 은 액터마다 따로 흔든다. 관측의 `obj_y` 채널과 필요한 회피량을 직접
      바꾸는 축이라 여기서 다양성이 나온다. 다만 차로 반폭으로 잘라 **물체가 차로 밖으로
      나가 아무 일도 안 일어나는 변종**이 생기지 않게 한다(부호는 유지 — 반대편 차로로
      넘어가면 배치의 뜻이 달라진다).
    """
    out = [dict(spec) for spec in specs]
    if not out:
        return out
    rng = jitter_rng(f"{name}@v{variant}")
    amp_s = float(jitter.get("s", 0.0))
    amp_lat = float(jitter.get("lateral", 0.0))

    off = s_offset(name, variant, amp_s)
    # 경로 밖으로 밀면 `route_point` 가 끝점에서 포화해 액터가 통째로 한 점에 뭉친다.
    lo = max(JITTER_MARGIN_M - float(spec["s"]) for spec in out)
    hi = min(board.route.total - JITTER_MARGIN_M - float(spec["s"]) for spec in out)
    off = 0.0 if lo > hi else max(lo, min(hi, off))

    for spec in out:
        spec["s"] = float(spec["s"]) + off
        base = float(spec.get("lateral", 0.0))
        if amp_lat > 0.0:
            lat = base + rng.uniform(-amp_lat, amp_lat)
            half = half_lane_at(board, spec["s"])
            spec["lateral"] = math.copysign(min(abs(lat), half), base or lat)
    return out


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
