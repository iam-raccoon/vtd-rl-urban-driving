"""온라인 심판 뼈대.

- SpanTracker: score_fma.spans() 를 한 행씩. 닫힐 때 내거나(close), 길이 조건에 닿는 행에서 낸다(reach).
- Context: score_fma.main() 과 같은 보조 데이터(구간·제한속도·차로변경 표식·정지선·횡단보도).
- Referee: 행마다 (지도 판정) -> 판정기들 -> Sheet 반영. 판정기는 미래 행을 보지 않고, 필요하면 늦게 낸다.
"""
from dataclasses import dataclass, field

from vtd_rl import rule_stack as rs
from vtd_rl.referee.oracle import make_lc_at, make_lim_at, match_inputs
from vtd_rl.referee.sections import FastSections
from vtd_rl.world.board import cache_key

sf = rs.score_fma


@dataclass
class Hit:
    t: float
    sec: int
    item: int
    level: str
    why: str = ""


class SpanTracker:
    """score_fma.spans() 를 한 행씩. 돌려주는 구간은 (t0, t1, 첫 행, 구간 행들).

    emit="close": 구간이 닫힐 때(거짓 행·finish) 길이가 min_sec 이상이면 낸다. t1 = 마지막 참 행.
    emit="reach": 길이 조건(spans 와 같은 식)이 처음 참이 되는 행에서 한 번 낸다. t1 = 그 행.
      닫힐 때는 내지 않고, min_sec 에 못 미치고 닫힌 구간도 내지 않는다.
      `reached` 는 지금 열린 구간을 이미 냈는지 — 그 뒤 행은 모으지 않는다.
    """

    def __init__(self, pred, min_sec, emit="close"):
        if emit not in ("close", "reach"):
            raise ValueError(f"emit={emit!r}")
        self.pred, self.min_sec, self.emit = pred, min_sec, emit
        self._rows = []
        self.reached = False

    def _long(self, rows):
        return bool(rows) and rows[-1]["t"] - rows[0]["t"] >= self.min_sec - 1e-9

    def _close(self):
        rows, self._rows, self.reached = self._rows, [], False
        if self.emit == "close" and self._long(rows):
            return [(rows[0]["t"], rows[-1]["t"], rows[0], rows)]
        return []

    def update(self, row):
        if not self.pred(row):
            return self._close()
        if self.reached:
            return []
        self._rows.append(row)
        if self.emit == "reach" and self._long(self._rows):
            self.reached = True
            return [(self._rows[0]["t"], row["t"], self._rows[0], self._rows)]
        return []

    def finish(self):
        return self._close()


@dataclass
class Context:
    board: object
    sections: int
    secs: object
    lim_at: object
    lc_at: object
    tl_stops: dict
    cws: list
    notes: list = field(default_factory=list)

    @classmethod
    def build(cls, board, sections=5):
        m = match_inputs(board)
        secs = FastSections(m.route, sections)
        return cls(board, sections, secs, make_lim_at(secs, m.lims), make_lc_at(secs, m.lc_flag),
                   m.tl_stops, m.cws)


_CONTEXT_CACHE: dict = {}


def clear_context_cache():
    _CONTEXT_CACHE.clear()


def context_for(board, sections=5) -> Context:
    """판·구간 수마다 한 번만 비싼 입력을 만들고, notes 만 새로 담은 Context 를 준다.

    match_inputs 는 횡단보도를 경로 전체와 비교하고 FastSections 는 격자를 짓는다(판마다 40~99 ms).
    환경은 리셋마다 심판을 새로 만들기 때문에 그대로 두면 그 비용을 매 판 낸다.

    열쇠는 `cache_key(board)` 다(M6f, 없으면 판 이름 — 예전과 같다). 열쇠를 같이 쓰는 판(같은
    경로의 변종)은 비싼 입력(구간·제한속도·차로변경·신호)을 같이 쓰되, `Context.board` 는
    **지금 판**이어야 한다. 열쇠가 이름인 예전 판은 캐시된 판이 곧 이 판이라 그대로 둔다.
    """
    key = (cache_key(board), sections, len(board.route.pts))
    cached = _CONTEXT_CACHE.get(key)
    if cached is None:
        cached = _CONTEXT_CACHE[key] = Context.build(board, sections)
    owner = board if getattr(board, "cache_key", None) else cached.board
    return Context(owner, cached.sections, cached.secs, cached.lim_at, cached.lc_at,
                   cached.tl_stops, cached.cws)


def default_judges(use_map):
    from vtd_rl.referee.judges.contact import ContactJudge, CrosswalkStopJudge
    from vtd_rl.referee.judges.lane_geometry import LaneGeometryJudge
    from vtd_rl.referee.judges.pedestrian import PedestrianJudge
    from vtd_rl.referee.judges.speed import SpeedJudge
    from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
    from vtd_rl.referee.judges.turn_signal import RespawnJudge, TurnSignalJudge
    judges = [SpeedJudge]
    if use_map:
        judges.append(LaneGeometryJudge)
    return judges + [TrafficLightJudge, ContactJudge, PedestrianJudge, CrosswalkStopJudge,
                     TurnSignalJudge, RespawnJudge]


class Referee:
    def __init__(self, board, sections=5, use_map=True, judges=None):
        self.ctx = context_for(board, sections)
        self.map = rs.load_map() if use_map else None
        classes = default_judges(use_map) if judges is None else judges
        self.judges = [cls(self.ctx) for cls in classes]
        self.sheet = sf.Sheet(self.ctx.secs.n)
        self.hits = []
        self.respawns = {}

    def _apply(self, hits):
        for h in hits:
            if h.item == 15:
                self.respawns.setdefault(h.sec, []).append(h.t)
            else:
                self.sheet.hit(h.sec, h.item, h.level, h.why)
            self.hits.append(h)
        return hits

    def step(self, row):
        """한 프레임을 판정하고 이번에 낸 Hit 목록을 돌려준다.

        row 사전은 심판이 가져간다: 지도 판정(classify)이 제자리에서 키를 붙이고, 판정기들은 늦게
        내려고 행을 참조로 들고 있다. 부르는 쪽은 프레임마다 새 사전을 넘기고 넘긴 뒤 고치지 않는다.
        """
        if self.map is not None:
            rs.check_lanes.classify([row], self.map)
        out = []
        for j in self.judges:
            out += j.step(row)
        return self._apply(out)

    def finish(self):
        out = []
        for j in self.judges:
            out += j.finish()
        return self._apply(out)
