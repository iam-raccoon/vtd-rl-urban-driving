"""③ 차로 유지 · ④ 중앙선 · ⑤ 보도 · ⑥ 실선 차로변경 — score_fma.item_lane_geometry 의 한 행 판.

행에는 Referee.step 이 check_lanes.classify 결과를 붙여 준다.
- 출발 직후 제외: 첫 행에서 8 m 넘게 가고 |d_ego| < 0.5 인 첫 행부터 본다(원본의 live).
- ⑥ 사건은 lane_change_events 를 [직전 행, 이 행] 에 불러 얻는다. 원본도 직전 행과만 비교한다.
- ③ 은 행마다 '차로변경 사건 ±2.5 초 안인가'를 봐야 해서 행을 2.5 초 붙잡아 두었다가 구간 추적에 넣는다.
- 판정 시점: ⑥ 은 사건 프레임, ④⑤ 는 침범이 최소 시간(CENTER_S·WALK_S)에 닿는 프레임에 낸다(끝을 기다리지 않는다).
  ③ 은 풀려난 물림이 LANE_EDGE_S 에 닿는 행에서 그때까지 지나온 채점 구간마다 한 번씩(처음 본 순서) 내고,
  같은 물림이 이어지는 동안 새 채점 구간에 처음 들어가면 그 행에서 한 번 더 낸다. 채점기도 물림 하나를
  채점 구간마다 한 번씩 세므로 구간별 횟수와 등급이 같다.
"""
import math
from collections import deque

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit, SpanTracker

sf = rs.score_fma
cl = rs.check_lanes
LC_WIN = 2.5            # item_lane_geometry 안의 지역 상수 — 차로변경 사건 앞뒤 이 시간은 '변경 중'[s]
SPAWN_M = 8.0           # item_lane_geometry 안의 숫자 — 출발점에서 이만큼은 뺀다[m]
ATTACHED_D = 0.5        # item_lane_geometry 안의 숫자 — 경로에 붙었다고 볼 |d_ego|[m]


def _over(key, limit):
    return lambda r: r.get(key) is not None and r[key] + 1e-9 >= limit


class LaneGeometryJudge:
    items = (3, 4, 5, 6)

    def __init__(self, ctx):
        self.ctx = ctx
        self.start = None
        self.live = False
        self.prev = None                 # lane_change_events 의 직전 행
        self.changes = deque()           # 차로변경 사건 시각
        self.held = deque()              # ③ 판정을 기다리는 행
        self.edge = SpanTracker(self._edge_pred, sf.LANE_EDGE_S, emit="reach")
        self.center = SpanTracker(_over("center_intrusion", sf.CENTER_M), sf.CENTER_S, emit="reach")
        self.walk = SpanTracker(_over("sidewalk_intrusion", sf.WALK_M), sf.WALK_S, emit="reach")
        self.edge_t0, self.edge_secs = None, set()    # 낸 ③ 물림의 시작 t · 그 물림에서 이미 낸 채점 구간
        self.n_edge, self.n_solid = {}, {}
        # M6x — 보상용(판정에는 안 쓴다). `_edge_pred` 가 참인 행(차로변경·변경 구역 면책을 뺀 0.1 m 이상
        # 침범 — 0.3 초 미만이라 ③ 이 안 나는 짧은 물림도 포함)마다 0.1 m 를 넘는 깊이를 더한다.
        # 행은 2.5 초 붙잡혔다 풀릴 때 판정되므로 차로변경 면책이 이미 들어 있다.
        self.edge_excess = 0.0

    def _edge_pred(self, r):
        # 부수효과(M6x `edge_excess`)가 있다 — 행마다 한 번(SpanTracker.update 전용) 불려야 한다.
        if not _over("lane_intrusion", sf.LANE_EDGE_M)(r):
            return False
        if self.ctx.lc_at(r["x"], r["y"]):
            return False
        if any(abs(r["t"] - ct) < LC_WIN for ct in self.changes):
            return False
        self.edge_excess += r["lane_intrusion"] - sf.LANE_EDGE_M
        return True

    def step(self, r):
        if self.start is None:
            self.start = (r["x"], r["y"])
        if not self.live:
            if not (math.hypot(r["x"] - self.start[0], r["y"] - self.start[1]) > SPAWN_M
                    and abs(r.get("d_ego") or 0.0) < ATTACHED_D):
                return []
            self.live = True
        out = self._lane_changes(r)
        self.held.append(r)
        out += self._release(lambda h: r["t"] >= h["t"] + LC_WIN)
        out += self._major_hits(4, self.center.update(r))
        out += self._major_hits(5, self.walk.update(r))
        return out

    def finish(self):
        if not self.live:
            return []
        out = self._release(lambda h: True)
        out += self._edge_spans(self.edge.finish())
        out += self._major_hits(4, self.center.finish())
        out += self._major_hits(5, self.walk.finish())
        return out

    def _lane_changes(self, r):
        events = cl.lane_change_events([self.prev, r] if self.prev is not None else [r])
        self.prev = r if r.get("lane") is not None else None
        out = []
        for row, p, mk in events:
            self.changes.append(row["t"])
            if p.get("off") or row.get("off"):
                continue
            if "solid" not in str(mk or ""):
                continue
            sec = self.ctx.secs.of(row["x"], row["y"])
            self.n_solid[sec] = self.n_solid.get(sec, 0) + 1
            out.append(Hit(row["t"], sec, 6, "major" if self.n_solid[sec] >= 2 else "minor",
                           f"t={row['t']:.1f} 실선({mk}) 넘어 차로변경"))
        return out

    def _release(self, ready):
        out = []
        while self.held and ready(self.held[0]):
            out += self._edge_step(self.held.popleft())
        if self.held:
            oldest = self.held[0]["t"]
            while self.changes and self.changes[0] <= oldest - LC_WIN:
                self.changes.popleft()
        return out

    def _edge_step(self, r):
        spans = self.edge.update(r)
        if not spans and self.edge.reached:                   # 이미 낸 물림이 이어진다
            return self._edge_hit(r)
        return self._edge_spans(spans)

    def _edge_spans(self, spans):
        out = []
        for t0, _t1, _r0, span_rows in spans:
            self.edge_t0, self.edge_secs = t0, set()
            for r in span_rows:
                out += self._edge_hit(r)
        return out

    def _edge_hit(self, r):
        sec = self.ctx.secs.of(r["x"], r["y"])
        if sec in self.edge_secs:
            return []
        self.edge_secs.add(sec)
        self.n_edge[sec] = self.n_edge.get(sec, 0) + 1
        return [Hit(self.edge_t0, sec, 3, "major" if self.n_edge[sec] >= 2 else "minor",
                    f"t={self.edge_t0:.1f}~{r['t']:.1f} 차로 경계 물림")]

    def _major_hits(self, item, spans):
        return [Hit(t0, self.ctx.secs.of(r0["x"], r0["y"]), item, "major", f"t={t0:.1f}~{t1:.1f}")
                for t0, t1, r0, _ in spans]
