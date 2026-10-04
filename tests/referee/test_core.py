import collections
import copy
import random
from types import SimpleNamespace

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit, Referee, SpanTracker
from vtd_rl.referee.judges.speed import SpeedJudge
from vtd_rl.referee.oracle import score_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_온라인_구간은_spans_와_같다():
    rng = random.Random(3)
    for _ in range(300):
        rows = [dict(t=round(k * 0.05, 2), on=rng.random() < 0.6) for k in range(rng.randint(0, 120))]
        pred = lambda r: r["on"]
        want = rs.score_fma.spans(rows, pred, 0.3)
        tr = SpanTracker(pred, 0.3)
        got = []
        for r in rows:
            got += tr.update(r)
        got += tr.finish()
        assert [(t0, t1, r0) for t0, t1, r0, _ in got] == want
        for t0, t1, _, span_rows in got:
            assert span_rows == [r for r in rows if t0 <= r["t"] <= t1]


def test_reach_모드는_길이에_처음_닿는_행에서_한_번만_낸다():
    rng = random.Random(4)
    for _ in range(300):
        min_sec = rng.choice([0.0, 0.3, 0.6, 1.0])
        rows = [dict(t=round(k * 0.05, 2), on=rng.random() < 0.75) for k in range(rng.randint(0, 120))]
        pred = lambda r: r["on"]
        want = rs.score_fma.spans(rows, pred, min_sec)
        close, reach = SpanTracker(pred, min_sec), SpanTracker(pred, min_sec, emit="reach")
        closed, reached = [], []
        for k, r in enumerate(rows):
            closed += [k for _ in close.update(r)]
            reached += [(k, s) for s in reach.update(r)]
        closed += [len(rows) for _ in close.finish()]
        assert reach.finish() == []
        assert [(t0, r0) for _, (t0, _t1, r0, _) in reached] == [(t0, r0) for t0, _t1, r0 in want]
        assert len(closed) == len(reached)
        for k_close, (k, (t0, t1, _r0, span_rows)) in zip(closed, reached):
            assert k < k_close
            assert t1 == rows[k]["t"] and span_rows == [r for r in rows[:k + 1] if r["t"] >= t0]
            assert t1 - t0 >= min_sec - 1e-9
            assert len(span_rows) == 1 or span_rows[-2]["t"] - t0 < min_sec - 1e-9


def test_심판은_항목15를_Sheet_대신_따로_모은다():
    class Fake:
        items = (1, 15)

        def __init__(self, ctx):
            self.ctx = ctx

        def step(self, r):
            return [Hit(r["t"], 0, 1, "minor"), Hit(r["t"], 1, 15, "major")]

        def finish(self):
            return [Hit(9.0, 0, 1, "major")]

    b = h_slice()
    x, y, h = b.route.point_at(10.0)
    ref = Referee(b, use_map=False, judges=[Fake])
    ref.step(dict(t=1.0, x=x, y=y, h=h, v=1.0, tl=0, tid=0, reason="", sig=0, clr=99.9,
                  d_ego=0.0, objs="", cap_by=""))
    ref.finish()
    assert ref.sheet.state[0] == {1: "major"} and ref.respawns == {1: [1.0]}
    assert [hit.item for hit in ref.hits] == [1, 15, 1]


def test_속도_판정은_합성_행에서_채점기와_같다():
    b = h_slice()
    rows, s = [], 0.0
    for k in range(300):
        v = [10.0, 14.2, 20.0][k // 100]            # 규정 · 경미(+1.1 km/h) · 중대(+22 km/h)
        x, y, h = b.route.point_at(s)
        rows.append(dict(t=round(k * 0.05, 2), x=x, y=y, h=h, v=v, tl=0, tid=0, reason="", sig=0,
                         clr=99.9, d_ego=0.0, objs="", cap_by=""))
        s += v * 0.05
    want = collections.Counter(score_episode(rows, b, use_map=False).hits)
    ref = Referee(b, use_map=False, judges=[SpeedJudge])
    for r in rows:
        ref.step(copy.deepcopy(r))
    ref.finish()
    assert collections.Counter((hit.sec, hit.item, hit.level) for hit in ref.hits) == want
    assert {lv for _, _, lv in want} == {"minor", "major"}


def test_take_edge_excess_는_합을_돌려주고_비운다():
    from vtd_rl.referee.core import Referee
    ref = Referee.__new__(Referee)
    ref.judges = [SimpleNamespace(edge_excess=1.5), SimpleNamespace(), SimpleNamespace(edge_excess=0.25)]
    assert ref.take_edge_excess() == 1.75
    assert ref.judges[0].edge_excess == 0.0 and ref.judges[2].edge_excess == 0.0
    assert ref.take_edge_excess() == 0.0
