"""판마다 한 번 만드는 색인 — 관측이 쓰는 '앞에 무엇이 얼마나 남았나'.

리셋마다 다시 계산하면 판당 수십 ms 가 그냥 나간다(횡단보도를 경로 전체와 비교한다).
"""
import bisect
import math

from vtd_rl import rule_stack as rs
from vtd_rl.world.board import cache_key
from vtd_rl.world.signals import route_signals, LINE_MIN_COS

NEAR_ROUTE = 8.0          # 경로에서 이 안에 있는 것만 경로 위의 것으로 본다[m]
ZONE_LIM = rs.score_fma.ZONE_LIM


class BoardIndex:
    def __init__(self, board):
        self.board = board
        db = rs.map_db()
        route = board.route
        self.signals = route_signals(route, db["tl_map"], db["stoplines_all"])
        self.signal_s = sorted(sig.s for sig in self.signals)
        self.crosswalk_s = self._project(route, [(c["x"], c["y"]) for c in db["crosswalks"]])
        self.stopline_s = self._project_stoplines(route, db["stoplines_all"])
        self.zone = [bool(p and (p.get("lim") or 99.0) <= ZONE_LIM) for p in board.lane_plan]
        self.junction_s = self._junctions(route, board.lane_plan)
        self._kinds = {"signal": self.signal_s, "crosswalk": self.crosswalk_s,
                       "stopline": self.stopline_s, "junction": self.junction_s}

    @staticmethod
    def _junctions(route, lane_plan):
        """교차로 **구간이 시작하는** s [m] — 차로계획의 `j`/`jx` 가 이어지는 묶음마다 하나.

        교차로는 경로점 수십 개에 걸쳐 켜져 있으므로 점마다 넣으면 '다음 교차로까지'가
        구간 안에서 0 에 붙어 버린다. 묶음의 첫 점만 남긴다.
        """
        out, prev = [], False
        for i, p in enumerate(lane_plan):
            cur = bool(p and (p.get("j") or p.get("jx")))
            if cur and not prev:
                out.append(route.cum[i])
            prev = cur
        return out

    @staticmethod
    def _project(route, points):
        out = []
        for x, y in points:
            p = route.project(x, y)
            if abs(p.lateral) <= NEAR_ROUTE and 0.0 < p.s < route.total:
                out.append(p.s)
        return sorted(out)

    @staticmethod
    def _project_stoplines(route, stoplines_with_heading):
        """도색 정지선을 경로에 투영하되, 진행 방향이 같은 것만 (신호.py 와 같은 논리).

        신호 정지선은 경로의 다른 차선에도 있고, 반대 차선 정지선을 '앞에 온다'로 보면 잘못된 판단을 한다.
        """
        out = []
        for x, y, heading in stoplines_with_heading:
            p = route.project(x, y)
            if abs(p.lateral) <= NEAR_ROUTE and 0.0 < p.s < route.total:
                # 경로의 이 지점 방향과 정지선 방향이 같은지 확인
                route_heading = route.heading_at(p.index)
                heading_diff = heading - route_heading
                # 각도 차이를 [-π, π] 범위로 정규화
                heading_diff = (heading_diff + math.pi) % (2.0 * math.pi) - math.pi
                if math.cos(heading_diff) >= LINE_MIN_COS:
                    out.append(p.s)
        return sorted(out)

    def ahead(self, s: float, kind: str) -> float:
        xs = self._kinds.get(kind)
        if xs is None:
            raise ValueError(f"모르는 종류: {kind!r} (있는 것: {sorted(self._kinds)})")
        i = bisect.bisect_left(xs, s)
        return xs[i] - s if i < len(xs) else math.inf

    def plan_at(self, index: int) -> dict:
        plan = self.board.lane_plan[index] if 0 <= index < len(self.board.lane_plan) else None
        return plan or {}


_CACHE: dict = {}


def clear_board_index_cache():
    _CACHE.clear()


def board_index(board) -> BoardIndex:
    # 열쇠는 `cache_key(board)`(없으면 판 이름 — 예전과 같다). 같은 경로의 변종끼리 색인을 같이
    # 쓴다(M6f) — 색인은 경로·차로계획만 보고, `plan_at` 이 읽는 차로계획도 같은 경로면 같다.
    key = (cache_key(board), len(board.route.pts))
    idx = _CACHE.get(key)
    if idx is None:
        idx = _CACHE[key] = BoardIndex(board)
    return idx
