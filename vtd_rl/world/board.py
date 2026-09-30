"""판 = 규칙 스택 Scenario + 경로점별 차로계획 + 신호 운용 방식."""
import bisect
import json
from dataclasses import dataclass, field

from vtd_rl import rule_stack as rs
from vtd_rl.world.place import actor_from_spec, check_jitter, jitter_specs
from vtd_rl.world.route import RouteIndex
from vtd_rl.world.signals import SIGNAL_MODES

#: 커리큘럼 최상위에서 허용하는 키. `stage` 는 기존 파일이 갖고 있어 받아만 주고 안 읽는다.
CURRICULUM_KEYS = {"name", "signals", "boards", "stage"}

#: 판 항목에서 허용하는 키. `jitter` 는 변종이 액터를 얼마나 흔들지다(`world/place.py`).
BOARD_KEYS = {"name", "route", "lane", "actors", "jitter"}

#: 변종 이름에 붙는 꼬리. **이름이 달라야** 한다 — `VtdDriveEnv._worlds` 는 판을 **이름으로**
#: 캐시하고(`env/drive_env.py:65,89-96`) `World` 는 액터를 생성 때 한 번만 읽으므로
#: (`world/world.py`), 같은 이름으로 액터만 바꿔 끼우면 관측·심판은 이 판을, 세계는 저 판을
#: 본다(`drive_env.py:96` assert 가 그때 죽는다). `rl/vec_env.py:35` 의 `#{stage}` 와 같은 전례다.
VARIANT_TAG = "@v"


@dataclass
class Board:
    name: str
    scenario: object
    lane_plan: list
    signals: str = "always_green"
    ego_lanes: list = field(default_factory=list)   # 경로 JSON ego_lanes — [i][4] 가 계획기 차로변경 표식
    route: RouteIndex = field(init=False)

    def __post_init__(self):
        if self.signals not in SIGNAL_MODES:
            raise ValueError(f"신호 운용은 {SIGNAL_MODES} 중 하나: {self.signals}")
        pts = self.scenario.ego_route or self.scenario.route_points()
        if len(pts) != len(self.lane_plan):
            # 차로계획은 원본 ego_route 와 짝이다. route_points() 로 촘촘하게 나누면
            # 코스 H 는 2021 점이 되어 6점 어긋난다(2026-09-15 확인).
            raise ValueError(f"{self.name}: 경로점 {len(pts)} != 차로계획 {len(self.lane_plan)}")
        if self.ego_lanes and len(self.ego_lanes) != len(pts):
            raise ValueError(f"{self.name}: 경로점 {len(pts)} != ego_lanes {len(self.ego_lanes)}")
        self.route = RouteIndex(pts)

    @property
    def start_pose(self):
        x, y = self.route.pts[0]
        return x, y, self.route.heading_at(0)

    @property
    def goal(self):
        return self.route.pts[-1]


def _load_json(rel):
    with open(rs.path(rel), encoding="utf-8") as f:
        return json.load(f)


def _check_keys(d: dict, allowed: set, what: str):
    """모르는 키를 **조용히 버리지 않는다**.

    예전에는 `"actors"` 를 넣어도 오류 없이 무시됐다. 커리큘럼에 액터를 싣는 지금은
    오타 하나가 액터 0 개인 판을 만들고, 그 판은 단계 ①② 와 구별이 안 된다 —
    실험 한 팔이 통째로 무의미해진다.
    """
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ValueError(f"{what}에 모르는 키가 있다: {', '.join(unknown)} "
                         f"(허용: {', '.join(sorted(allowed))})")


def load_board(entry: dict, signals: str = "always_green", variant: int | None = None) -> Board:
    """판 항목 하나를 읽는다. `variant` 가 정수면 **변종** 이다.

    - 이름이 `<원래이름>@v<k>` 가 된다(`VARIANT_TAG` 의 설명 참고).
    - `k >= 1` 이면 액터를 `(원래이름, k)` 로 결정되는 rng 로 흔든다. **`k == 0` 은 손대지
      않은 원본 배치**다 — 손으로 맞춰 검증한 배치가 항상 한 벌은 섞이게 해 둔다.
    - `variant=None`(기본)은 예전과 완전히 같다: 이름도 그대로, 흔들지도 않는다.
    """
    _check_keys(entry, BOARD_KEYS, f"판 항목 '{entry.get('name', '?')}'")
    # 변종을 안 쓰는 호출에서도 `jitter` 오타는 여기서 터뜨린다 — 나중에 변종을 켠 사람이
    # "왜 배치가 안 흔들리지" 를 디버깅하게 두지 않는다.
    jitter = check_jitter(entry.get("jitter"))
    sc = rs.Scenario.load(rs.path(entry["route"]))
    lane = _load_json(entry["lane"])["pts"]
    ego_lanes = _load_json(entry["route"]).get("ego_lanes") or []
    name = entry["name"] if variant is None else f"{entry['name']}{VARIANT_TAG}{int(variant)}"
    board = Board(name, sc, lane, signals, ego_lanes)
    # 액터는 판을 먼저 지은 뒤에 만든다 — 경로-상대 좌표를 풀려면 `board.route` 가 필요하다.
    specs = list(entry.get("actors") or [])
    if specs and variant is not None and int(variant) > 0:
        specs = jitter_specs(board, specs, jitter, entry["name"], int(variant))
    # 흔든 뒤에도 `actor_from_spec` 을 다시 통과시킨다 — 세계 좌표와 `hd`(경로 방위)가
    # **새 `s` 에서** 다시 계산되어야 한다. 이미 만든 액터의 `pos` 만 밀면 `hd` 가 옛 자리의
    # 방위로 남아 굽은 코스에서 긴 차의 상자가 도로를 가로질러 눕는다(`mock_vtd.py:46-50`).
    actors = [actor_from_spec(board, spec) for spec in specs]
    if actors:
        # 경로 JSON 은 액터를 갖지 않는다(2026-09-30 확인). 그래도 있으면 덮지 않고 더한다.
        sc.actors = list(sc.actors) + actors
    return board


def slice_board(board: Board, s_from: float, s_to: float, name: str, signals: str | None = None) -> Board:
    if signals is not None and signals not in SIGNAL_MODES:
        raise ValueError(f"신호 운용은 {SIGNAL_MODES} 중 하나: {signals}")
    cum = board.route.cum
    i0 = bisect.bisect_left(cum, s_from)
    i1 = bisect.bisect_right(cum, s_to) - 1
    if i1 - i0 < 2:
        raise ValueError(f"자른 구간이 너무 짧다: {s_from}~{s_to}")
    pts = [list(p) for p in board.route.pts[i0:i1 + 1]]
    length = cum[i1] - cum[i0]
    sc0 = board.scenario
    sc = rs.Scenario(
        name=name,
        ego_start=[pts[0][0], pts[0][1], board.route.heading_at(i0)],
        ego_goal=pts[-1],
        speed_limit=sc0.speed_limit,
        duration=max(60.0, length / 5.0 + 30.0),
        actors=[], lights=[], zones=[],
        ego_route=pts, respawns=[], tl_stops={},
    )
    return Board(name, sc, board.lane_plan[i0:i1 + 1], signals or board.signals,
                 board.ego_lanes[i0:i1 + 1])


def load_curriculum(path: str, variants: int = 1):
    """커리큘럼을 읽는다. `variants > 1` 이면 판마다 **변종 N 개**를 낸다.

    `variants == 1` 은 예전과 완전히 같다(이름도 그대로, 흔들지도 않는다) — 기존 호출부와
    옛 성적표가 그 이름을 그대로 쓴다.
    """
    if variants < 1:
        raise ValueError(f"variants 는 1 이상이어야 한다: {variants}")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    _check_keys(d, CURRICULUM_KEYS, f"커리큘럼 '{path}'")
    if variants == 1:
        return d["name"], [load_board(e, d["signals"]) for e in d["boards"]]
    return d["name"], [load_board(e, d["signals"], variant=k)
                       for e in d["boards"] for k in range(variants)]
