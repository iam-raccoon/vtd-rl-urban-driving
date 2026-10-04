"""LaneGeometryJudge 대 score_fma.item_lane_geometry — 지도 없이, 손으로 만든 행으로 빠르게.

Task 9 리뷰(수정 라운드 1)에서 지적된 두 '어려운 부분'을 직접 겨냥한다.
- 대본 판(test_parity_lane_geometry.py)에는 ⑥ 실선 차로변경 사건이 하나도 없었다.
- 대본 판에는 '차로경계 물림 구간이 끝난 뒤에 생기는 차로변경'이 그 구간을 면책하는
  경우가 없었다 — ③ 을 2.5 초 붙잡아 두는 이유 그 자체를 아무 것도 검증하지 못했다.

여기서는 `check_lanes.classify` 가 붙이는 키(lane, road, off,
classification_hold_reason, _bnd, obb_t_min, obb_t_max, lat, lane_intrusion,
center_intrusion, sidewalk_intrusion, d_ego, x, y, t)를 손으로 채운 행을 만들어
지도 로딩 없이(빠르게) `score_fma.item_lane_geometry` 와 `LaneGeometryJudge` 를 같은
행에 돌려 (sec, item, level) 다중집합을 비교한다.
"""
import collections
import copy
import random
from types import SimpleNamespace

from vtd_rl import rule_stack as rs
from vtd_rl.referee.judges.lane_geometry import LaneGeometryJudge

sf = rs.score_fma
DT = 0.05


class _Secs:
    """x 를 20 m 폭 구간 5 개로 나누는 가짜 Sections — 실제 FastSections 대신 빠르게."""

    def of(self, x, y):
        return min(int(x // 20), 4)


class _RecordingSheet(sf.Sheet):
    def __init__(self, n):
        super().__init__(n)
        self.calls = []

    def hit(self, sec, item, level, why):
        self.calls.append((sec, item, level))
        super().hit(sec, item, level, why)


def _no_lc(x, y):
    return False


def _score(rows, lc_at):
    sheet = _RecordingSheet(5)
    sf.item_lane_geometry(copy.deepcopy(rows), sheet, _Secs(), lc_at)
    return collections.Counter(sheet.calls)


def _judge(rows, lc_at):
    ctx = SimpleNamespace(secs=_Secs(), lc_at=lc_at)
    j = LaneGeometryJudge(ctx)
    hits = []
    for r in copy.deepcopy(rows):
        hits += j.step(r)
    hits += j.finish()
    return collections.Counter((h.sec, h.item, h.level) for h in hits)


def _row(t, x, lane, obb, lat, mark, intrusion):
    return dict(t=round(t, 2), x=x, y=0.0, d_ego=0.0, lane=lane, road=1, off=False,
                classification_hold_reason=None, obb_t_min=obb[0], obb_t_max=obb[1],
                lat=lat, _bnd=[(0.0, mark)], lane_intrusion=intrusion,
                center_intrusion=0.0, sidewalk_intrusion=0.0)


# ------------------------------------------------------------------ 결정적 시나리오
def _solid_crossing_rows():
    """같은 구간에서 solid 경계를 두 번 넘는다 — 1 번째는 경미, 2 번째는 중대."""
    LEFT, RIGHT = (-1.9, -0.1), (0.1, 1.9)
    rows = [dict(t=0.0, x=0.0, y=0.0, d_ego=0.0)]                      # 스폰 기준점(판정 제외)
    rows.append(_row(0.05, 9.0, -1, LEFT, -1.0, "solid", 0.0))         # live 시작, 경계 왼쪽
    rows.append(_row(0.10, 9.0, 1, RIGHT, 1.0, "solid", 0.0))          # 진입 1 (경미)
    rows.append(_row(0.15, 9.0, 1, RIGHT, 1.0, "solid", 0.0))          # 유지
    rows.append(_row(0.20, 9.0, -1, LEFT, -1.0, "solid", 0.0))         # 진입 2 (중대)
    rows.append(_row(0.25, 9.0, -1, LEFT, -1.0, "solid", 0.0))         # 유지
    return rows


def _edge_then_change_rows(gap):
    """차로경계 물림 0.35 초 구간이 끝난 `gap` 초 뒤에 (실선 아닌) 경계를 넘는다.

    gap 이 2.5 초보다 훨씬 작으면 채점기가 물림 구간 전체를 '변경 중'으로 보고
    ③ 을 면책한다. gap 이 2.5 초보다 훨씬 크면 면책되지 않고 ③(경미)이 뜬다.
    두 결과의 차이 자체가 '나중에 생기는 사건이 이미 지난 위반을 면책한다'는
    사실이고, 심판이 행을 2.5 초 붙잡아 둬야 하는 바로 그 이유다.
    """
    LEFT, RIGHT = (-1.9, -0.1), (0.1, 1.9)
    rows = [dict(t=0.0, x=0.0, y=0.0, d_ego=0.0)]                      # 스폰 기준점
    t = DT
    for _ in range(8):                                                # 0.35s(>=LANE_EDGE_S) 물림
        rows.append(_row(t, 9.0, -1, LEFT, -1.0, "broken", 0.5))
        t += DT
    span_end = round(t - DT, 2)                                       # 마지막 물림 행의 t
    change_t = round(span_end + gap, 2)
    prev_t = round(change_t - DT, 2)
    rows.append(_row(prev_t, 9.0, -1, LEFT, -1.0, "broken", 0.0))      # 물림 해제, 경계 왼쪽
    rows.append(_row(change_t, 9.0, 1, RIGHT, 1.0, "broken", 0.0))     # 경계를 넘는다
    rows.append(_row(round(change_t + DT, 2), 9.0, 1, RIGHT, 1.0, "broken", 0.0))
    return rows


def test_실선_두번_넘으면_경미_다음_중대():
    rows = _solid_crossing_rows()
    want = _score(rows, _no_lc)
    assert want[(0, 6, "minor")] == 1
    assert want[(0, 6, "major")] == 1
    assert _judge(rows, _no_lc) == want


def test_경계물림이_나중의_차로변경으로_면책된다():
    near = _edge_then_change_rows(gap=1.0)     # 사건까지 1.0s < LC_WIN — 전부 면책
    far = _edge_then_change_rows(gap=3.0)      # 사건까지 3.0s > LC_WIN — 면책 없음
    want_near, want_far = _score(near, _no_lc), _score(far, _no_lc)
    assert want_near[(0, 3, "minor")] == 0, "채점기 기준 자체가 면책돼야 하는데 안 됐다"
    assert want_far[(0, 3, "minor")] == 1, "채점기 기준 자체가 위반을 내야 하는데 안 났다"
    # 이 두 assert 가 진짜 관문이다: ③ 을 2.5 초 붙잡아 두지 않으면(붙잡지 않고 그
    # 자리에서 판정하면) near 의 물림 구간은 아직 안 생긴 미래의 차로변경을 몰라서
    # 잘못 (0,3,"minor") 를 낸다 — want_near 에는 그게 없으므로 바로 불일치한다.
    assert _judge(near, _no_lc) == want_near
    assert _judge(far, _no_lc) == want_far


# ------------------------------------------------------------------ 무작위 합성 판
def _gen(rng):
    """빠른 무작위 합성 행 — solid/broken 경계 넘기, lane=None, off, 차로변경 표식
    구간(lc_at), 차로경계 물림/중앙선/보도 침범을 뒤섞는다(리뷰어 fuzz2.py 변형).
    """
    n = rng.randint(50, 500)
    rows = []
    lat = 0.0
    lc_lo = rng.uniform(0, 100)
    lc_hi = lc_lo + rng.uniform(0, 30)
    for i in range(n):
        t = round(i * DT, 2)
        x = 10.0 + i * 0.25
        lat += rng.choice([-0.3] + [0.0] * 30 + [0.3])
        lat = max(-3.0, min(3.0, lat))
        r = dict(t=t, x=x, y=0.0, d_ego=rng.choice([0.0, 0.0, 0.0, 0.8]))
        if rng.random() < 0.05:
            r["lane"] = None
        else:
            r["lane"] = rng.choice([1, 1, 1, 2])
            r["road"] = rng.choice([7, 7, 7, 7, 8])
            r["off"] = rng.random() < 0.05
            r["classification_hold_reason"] = "x" if rng.random() < 0.02 else None
            r["obb_t_min"], r["obb_t_max"] = lat - 0.9, lat + 0.9
            r["_bnd"] = [(0.0, rng.choice(["solid", "broken", "solid solid"])),
                        (rng.choice([-1.5, 3.0]), "broken")]
            r["lat"] = lat
            r["lane_intrusion"] = rng.choice([None, 0.0, 0.3, 0.3, 0.3, 0.3])
            r["center_intrusion"] = rng.choice([None, 0.0, 0.7])
            r["sidewalk_intrusion"] = rng.choice([None, 0.0, 0.6])
        rows.append(r)
    return rows, (lc_lo, lc_hi)


def test_무작위_합성_행_300판_일치():
    n_seeds = 300
    item6 = collections.Counter()
    for seed in range(n_seeds):
        rng = random.Random(seed)
        rows, (lo, hi) = _gen(rng)

        def lc_at(x, y, lo=lo, hi=hi):
            return lo <= x <= hi

        want = _score(rows, lc_at)
        got = _judge(rows, lc_at)
        assert got == want, f"seed={seed} 채점기에만 {dict(want-got)} 심판에만 {dict(got-want)}"
        for (_sec, item, level), cnt in want.items():
            if item == 6:
                item6[level] += cnt
    assert sum(item6.values()) > 0, "300 판 동안 ⑥ 실선 차로변경이 한 번도 안 울렸다"
    assert item6["minor"] > 0 and item6["major"] > 0, f"⑥ 경미/중대 둘 다 나와야 하는데 {item6}"


def _excess(rows, lc_at):
    ctx = SimpleNamespace(secs=_Secs(), lc_at=lc_at)
    j = LaneGeometryJudge(ctx)
    for r in copy.deepcopy(rows):
        j.step(r)
    j.finish()
    return j.edge_excess


def test_edge_excess_는_채점기가_세는_물림만_쌓는다():
    # 물림 8 행(깊이 0.5 m) — 2.5 초 안에 차로변경이 오면 면책돼 0, 멀면 8 × (0.5 − 0.1).
    assert _excess(_edge_then_change_rows(gap=1.0), _no_lc) == 0.0
    assert _excess(_edge_then_change_rows(gap=3.0), lambda x, y: True) == 0.0   # 차로변경 구역 면책
    far = _excess(_edge_then_change_rows(gap=3.0), _no_lc)
    assert abs(far - 8 * (0.5 - sf.LANE_EDGE_M)) < 1e-9


def test_edge_excess_를_쌓아도_판정은_그대로다():
    rng = random.Random(11)
    for _ in range(50):
        rows, _lc = _gen(rng)
        assert _judge(rows, _no_lc) == _score(rows, _no_lc)
