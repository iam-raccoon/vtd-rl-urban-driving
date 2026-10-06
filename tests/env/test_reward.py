import random

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.env.reward import COLLISION_ITEMS, RewardConfig, RewardShaper, ViolationTracker, lat_excess, obs_excess, red_excess
from vtd_rl.referee.core import Hit
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_board():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_같은_항목은_1초에_한_번만_센다():
    tr = ViolationTracker(1.0)
    hits = [Hit(t=k * 0.05, sec=0, item=1, level="minor") for k in range(60)]   # 3 초 연속 과속
    counted = [h for k in range(60) for h in tr.count([hits[k]])]
    assert [round(h.t, 2) for h in counted] == [0.0, 1.0, 2.0]


def test_항목이_다르면_따로_센다():
    tr = ViolationTracker(1.0)
    counted = tr.count([Hit(0.0, 0, 1, "minor"), Hit(0.0, 0, 3, "minor"), Hit(0.05, 0, 1, "minor")])
    assert [(h.item, h.t) for h in counted] == [(1, 0.0), (3, 0.0)]


def test_진행과_완주와_시간():
    b = h_board()
    cfg = RewardConfig()
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], b.route.total / 10.0, zero, zero, "running")
    assert out.terms["progress"] == pytest.approx(cfg.progress_total / 10.0)
    assert out.terms["time"] == pytest.approx(cfg.time_cost * -1.0)
    assert out.terms["goal"] == 0.0
    done = sh.step([], 0.0, zero, zero, "goal")
    assert done.terms["goal"] == pytest.approx(cfg.goal_bonus)


def test_위반과_충돌과_이탈():
    b = h_board()
    cfg = RewardConfig(rule_scale=2.0)
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([Hit(1.0, 0, 1, "minor"), Hit(1.0, 0, 4, "major")], 0.0, zero, zero, "running")
    assert out.terms["violation"] == pytest.approx((cfg.minor + cfg.major) * cfg.rule_scale)
    assert out.collision is False
    hit_out = sh.step([Hit(2.0, 0, 14, "major")], 0.0, zero, zero, "running")
    assert hit_out.terms["collision"] == pytest.approx(cfg.collision)
    assert hit_out.terms["violation"] == 0.0 and hit_out.collision is True and hit_out.counted == 1
    off = sh.step([], 0.0, zero, zero, "offroad")
    assert off.terms["offroad"] == pytest.approx(cfg.offroad)


def test_승차감은_변화량에_붙는다():
    b = h_board()
    cfg = RewardConfig()
    sh = RewardShaper(b, cfg)
    sh.reset()
    prev = {"control": [0.0, 0.0], "turn": 0}
    now = {"control": [0.5, -0.4], "turn": 0}
    out = sh.step([], 0.0, now, prev, "running")
    assert out.terms["comfort"] == pytest.approx(cfg.comfort_steer * 0.5 + cfg.comfort_accel * 0.4)
    same = sh.step([], 0.0, now, now, "running")
    assert same.terms["comfort"] == 0.0


def test_총합은_항의_합이다():
    b = h_board()
    sh = RewardShaper(b)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([Hit(1.0, 0, 2, "major")], 5.0, zero, zero, "running")
    assert out.total == pytest.approx(sum(out.terms.values()))


def test_같은_항목_같은_시각_다른_구간():
    """한 침범이 구간 경계를 걸치면(같은 t, 다른 sec) 각 구간마다 센다."""
    tr = ViolationTracker(1.0)
    hits = [Hit(5.0, 0, 3, "minor"), Hit(5.0, 1, 3, "minor")]
    counted = tr.count(hits)
    assert len(counted) == 2
    assert [(h.item, h.sec, h.t) for h in counted] == [(3, 0, 5.0), (3, 1, 5.0)]


def test_같은_항목_같은_구간_반복_압축():
    """같은 (항목, 구간) 반복은 repeat_gap에 한 번만 센다."""
    tr = ViolationTracker(1.0)
    hits = [Hit(t=k * 0.05, sec=0, item=2, level="major") for k in range(30)]  # 1.5 초 연속
    counted = [h for k in range(30) for h in tr.count([hits[k]])]
    assert [round(h.t, 2) for h in counted] == [0.0, 1.0]


def test_항목15는_충돌이_아니라_위반():
    """오프라인 세계에 없는 항목 15는 major 위반으로 센다."""
    b = h_board()
    cfg = RewardConfig()
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([Hit(1.0, 0, 15, "major")], 0.0, zero, zero, "running")
    assert out.terms["violation"] == pytest.approx(cfg.major)
    assert out.terms["collision"] == 0.0
    assert out.collision is False
    assert out.counted == 1


def test_의도로_재면_표본_잡음이_승차감에_안_잡힌다():
    sh = RewardShaper(h_board(), RewardConfig(comfort_on_intent=True))
    sh.reset()
    intent = (0.20, 0.10)                              # 의도는 두 걸음 내내 같다
    a1 = {"control": [0.20, 0.10], "turn": 0}
    a2 = {"control": [0.60, 0.50], "turn": 0}          # 표본이 크게 튀었다
    sh.step([], 0.0, a1, a1, "running", intent=intent)
    out = sh.step([], 0.0, a2, a1, "running", intent=intent)
    assert out.terms["comfort"] == 0.0                 # 의도가 안 변했으니 0


def test_의도가_변하면_그_변화량으로_잰다():
    cfg = RewardConfig(comfort_on_intent=True)
    sh = RewardShaper(h_board(), cfg)
    sh.reset()
    a = {"control": [0.0, 0.0], "turn": 0}             # 실행 행동은 내내 같다
    sh.step([], 0.0, a, a, "running", intent=(0.0, 0.0))
    out = sh.step([], 0.0, a, a, "running", intent=(0.5, -0.4))
    assert out.terms["comfort"] == pytest.approx(cfg.comfort_steer * 0.5 + cfg.comfort_accel * 0.4)


def test_의도를_안_주면_예전처럼_실행_행동으로_잰다():
    cfg = RewardConfig()
    sh = RewardShaper(h_board(), cfg)
    sh.reset()
    prev = {"control": [0.0, 0.0], "turn": 0}
    now = {"control": [0.5, -0.4], "turn": 0}
    out = sh.step([], 0.0, now, prev, "running")
    assert out.terms["comfort"] == pytest.approx(cfg.comfort_steer * 0.5 + cfg.comfort_accel * 0.4)


def test_승차감_계수를_바꿀_수_있다():
    sh = RewardShaper(h_board(), RewardConfig(comfort_steer=-0.01, comfort_accel=-0.005))
    sh.reset()
    prev = {"control": [0.0, 0.0], "turn": 0}
    now = {"control": [1.0, 1.0], "turn": 0}
    out = sh.step([], 0.0, now, prev, "running")
    assert out.terms["comfort"] == pytest.approx(-0.015)


def test_comfort_on_intent_기본값은_꺼짐():
    assert RewardConfig().comfort_on_intent is False


def test_구간당_한_번_모드는_같은_항목을_한_번만_센다():
    tr = ViolationTracker(1.0, violation_mode="once_per_section")
    hits = [Hit(t=k * 0.05, sec=0, item=1, level="minor") for k in range(200)]   # 10 초 연속
    counted = [h for k in range(200) for h in tr.count([hits[k]])]
    assert [round(h.t, 2) for h in counted] == [0.0]      # repeat 모드였다면 10 개다


def test_구간이_바뀌면_다시_센다():
    tr = ViolationTracker(1.0, violation_mode="once_per_section")
    counted = tr.count([Hit(0.0, 0, 1, "minor"), Hit(0.1, 1, 1, "minor"), Hit(0.2, 0, 1, "minor")])
    assert [(h.sec, h.t) for h in counted] == [(0, 0.0), (1, 0.1)]


def test_심화되면_차액만_깎는다():
    """경미를 깎은 뒤 중대가 오면 채점기처럼 차액만 더 깎는다(score_fma.Sheet 와 같은 규칙)."""
    cfg = RewardConfig(violation_mode="once_per_section")
    tr = ViolationTracker(cfg.repeat_gap, violation_mode=cfg.violation_mode)
    _c1, p1 = tr.charge([Hit(0.0, 0, 1, "minor")], cfg.minor, cfg.major)
    _c2, p2 = tr.charge([Hit(1.0, 0, 1, "major")], cfg.minor, cfg.major)
    _c3, p3 = tr.charge([Hit(2.0, 0, 1, "minor")], cfg.minor, cfg.major)   # 중대 뒤 경미는 무시
    assert p1 == pytest.approx(cfg.minor)
    assert p2 == pytest.approx(cfg.major - cfg.minor)
    assert p3 == pytest.approx(0.0)
    assert p1 + p2 == pytest.approx(cfg.major)        # 합치면 중대 한 번과 같다


def test_같은_슬롯에_major가_다시_와도_또_안_깎는다():
    """M4d 최종 리뷰 Important #4: 심화 때(`minor`->`major`) `self._level[key] = h.level` 갱신을

    `if cur is None:` 아래로 옮겨도(즉 첫 발생에만 기록하고 심화 때는 안 갱신해도) 기존
    테스트 전부가 통과했다 — 그러면 슬롯이 계속 'minor' 로 보여 그 뒤에 오는 major 가 차액
    (major-minor)을 매번 다시 문다. `test_심화되면_차액만_깎는다` 는 `minor`->`major` 까지만
    보고 그 뒤에 오는 `major`->`major` 재청구는 안 본다.
    """
    cfg = RewardConfig(violation_mode="once_per_section")
    tr = ViolationTracker(cfg.repeat_gap, violation_mode=cfg.violation_mode)
    _c1, p1 = tr.charge([Hit(0.0, 0, 1, "minor")], cfg.minor, cfg.major)
    _c2, p2 = tr.charge([Hit(1.0, 0, 1, "major")], cfg.minor, cfg.major)
    _c3, p3 = tr.charge([Hit(2.0, 0, 1, "major")], cfg.minor, cfg.major)   # 다시 major
    assert p3 == pytest.approx(0.0), "이미 major 인 슬롯에 또 major 가 와서 차액을 또 물었다"
    assert p1 + p2 + p3 == pytest.approx(cfg.major)   # 합치면 여전히 중대 한 번과 같다


def test_repeat_모드가_기본값이고_예전과_같다():
    assert RewardConfig().violation_mode == "repeat"
    tr = ViolationTracker(1.0)
    hits = [Hit(t=k * 0.05, sec=0, item=1, level="minor") for k in range(60)]
    counted = [h for k in range(60) for h in tr.count([hits[k]])]
    assert [round(h.t, 2) for h in counted] == [0.0, 1.0, 2.0]     # 기존 테스트와 같은 값


def test_구간당_한_번_모드의_판당_위반이_훨씬_작다():
    """같은 히트 열을 두 모드에 먹여 크기 차이를 값으로 잠근다 — 이 마일스톤의 전제다."""
    hits = [Hit(t=k * 0.05, sec=0, item=1, level="minor") for k in range(200)]
    zero = {"control": [0.0, 0.0], "turn": 0}
    base = RewardConfig()
    totals = {}
    for mode in ("repeat", "once_per_section"):
        sh = RewardShaper(h_board(), RewardConfig(violation_mode=mode))
        sh.reset()
        totals[mode] = sum(sh.step([h], 0.0, zero, zero, "running").terms["violation"]
                           for h in hits)
    assert totals["once_per_section"] == pytest.approx(base.minor)
    assert totals["repeat"] < totals["once_per_section"]        # 더 많이(음수로 크게) 깎는다
    assert totals["repeat"] == pytest.approx(base.minor * 10)   # 10 초 / repeat_gap 1 초


def test_잘못된_violation_mode는_거부한다():
    with pytest.raises(ValueError):
        ViolationTracker(1.0, violation_mode="nonsense")


def test_reset하면_구간_기록이_지워진다():
    tr = ViolationTracker(1.0, violation_mode="once_per_section")
    assert tr.count([Hit(0.0, 0, 1, "minor")])
    assert not tr.count([Hit(1.0, 0, 1, "minor")])
    tr.reset()
    assert tr.count([Hit(2.0, 0, 1, "minor")])      # 새 판이면 다시 센다


def test_콜드스타트_판_첫_걸음은_변화량이_0이다():
    """`comfort_on_intent=True` 인데 직전 의도가 없는 판 첫 걸음 — 변화량을 0 으로 봐야 한다.

    M4c 최종 리뷰 Important: 이 자리(`prev_intent = ... if ... is not None else intent`)를
    `(0, 0)` 으로 대체하는 돌연변이가 살아남았다 — 값으로는 -0.0 대 -0.11 이라 동치가 아닌데도
    구조적 검사(`== 0.0`이 부동소수 -0.0 과도 같게 비교돼)만으로는 못 갈렸다. 여기서는 의도가
    (0.8, 0.6) 처럼 원점에서 뚜렷이 떨어진 값일 때를 골라 `(0, 0)` 폴백과 확실히 갈리게 한다
    — `(0,0)` 이면 d_steer=0.8·d_accel=0.6 이 되어 comfort = -0.10*0.8 + -0.05*0.6 = -0.11.
    """
    sh = RewardShaper(h_board(), RewardConfig(comfort_on_intent=True))
    sh.reset()
    a = {"control": [0.8, 0.6], "turn": 0}     # 실행 행동도 크게 줘 실행-행동 폴백과도 갈린다
    out = sh.step([], 0.0, a, a, "running", intent=(0.8, 0.6))
    assert out.terms["comfort"] == pytest.approx(0.0)


def test_구간당_한_번_모드가_채점기와_같은_감점을_낸다_속성():
    """M4d 최종 리뷰 Important #5: "합성 히트 4000 순열 불일치 0" 을 리뷰가 일회용 스크립트로

    확인했지만 레포에 회귀 테스트로 안 남아 있었다. 여기서는 고정 시드로 무작위 히트 열
    수백 개를 만들어 채점기 `Sheet`(`vtd_rl.rule_stack.score_fma`, third_party 는 이 어댑터를
    통해서만 읽는다)와 `ViolationTracker(once_per_section)` 의 총 감점을 대조한다.

    항목 15(리스폰)는 뺀다 — 채점기가 `Sheet.hit` 밖에서 따로 세는 예외이고, `_charge_once`
    독스트링(`vtd_rl/env/reward.py`)이 그 어긋남을 이미 문서화하고 있다.
    """
    sf = rs.score_fma
    cfg = RewardConfig(violation_mode="once_per_section")
    items = [i for i in sf.ITEMS if i != 15]
    n_sections = 4
    rng = random.Random(20260928)

    for _ in range(300):
        n_hits = rng.randint(0, 40)
        hits = [Hit(t=float(k), sec=rng.randrange(n_sections), item=rng.choice(items),
                    level=rng.choice(("minor", "major")))
               for k in range(n_hits)]
        rng.shuffle(hits)   # 순서를 섞어도 같은 규칙이 나오는지까지 본다

        sheet = sf.Sheet(n_sections)
        for h in hits:
            sheet.hit(h.sec, h.item, h.level, "속성 테스트")
        sheet_total = sum(100 - sheet.score(sec) for sec in range(n_sections))

        tr = ViolationTracker(cfg.repeat_gap, violation_mode=cfg.violation_mode)
        _counted, total = tr.charge(hits, cfg.minor, cfg.major)
        assert -total == pytest.approx(sheet_total), (n_hits, hits)


def test_item_scale_기본값은_비어_있고_예전과_같다():
    assert RewardConfig().item_scale == ()
    assert RewardConfig(item_scale=()) == RewardConfig()


def test_item_scale_은_그_항목의_벌만_키운다():
    b = h_board()
    cfg = RewardConfig(item_scale=((7, 5.0),), rule_scale=2.0)
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([Hit(1.0, 0, 7, "major"), Hit(1.0, 0, 3, "major"), Hit(1.0, 0, 1, "minor")],
                  0.0, zero, zero, "running")
    assert out.terms["violation"] == pytest.approx((cfg.major * 5.0 + cfg.major + cfg.minor) * 2.0)
    assert out.counted == 3


def test_item_scale_은_구간당_한_번_심화_차액에도_곱한다():
    b = h_board()
    cfg = RewardConfig(item_scale=((7, 5.0),), violation_mode="once_per_section")
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    p1 = sh.step([Hit(0.0, 0, 7, "minor")], 0.0, zero, zero, "running").terms["violation"]
    p2 = sh.step([Hit(1.0, 0, 7, "major")], 0.0, zero, zero, "running").terms["violation"]
    p3 = sh.step([Hit(2.0, 0, 7, "major")], 0.0, zero, zero, "running").terms["violation"]
    assert p1 == pytest.approx(cfg.minor * 5.0)
    assert p2 == pytest.approx((cfg.major - cfg.minor) * 5.0)
    assert p3 == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["repeat", "once_per_section"])
def test_배율_1_은_배율이_없는_것과_같다_속성(mode):
    """항목끼리 나눠 세도 `(항목, 구간)` 열쇠라 결과가 같아야 한다 — 무작위 히트 흐름 300 개."""
    b = h_board()
    rng = random.Random(7)
    zero = {"control": [0.0, 0.0], "turn": 0}
    for _ in range(300):
        stream = [[Hit(t * 0.05, rng.randrange(3), rng.choice((1, 3, 7, 13, 14)),
                       rng.choice(("minor", "major"))) for _k in range(rng.randrange(3))]
                  for t in range(40)]
        plain = RewardShaper(b, RewardConfig(violation_mode=mode))
        ones = RewardShaper(b, RewardConfig(violation_mode=mode, item_scale=((7, 1.0), (3, 1.0))))
        plain.reset()
        ones.reset()
        for hits in stream:
            a = plain.step(hits, 0.0, zero, zero, "running")
            c = ones.step(hits, 0.0, zero, zero, "running")
            assert c.terms["violation"] == pytest.approx(a.terms["violation"])
            assert c.counted == a.counted and c.collision == a.collision


def test_item_scale_목록을_줘도_튜플로_바뀌고_해시된다():
    cfg = RewardConfig(item_scale=[[7, 5]])
    assert cfg.item_scale == ((7, 5.0),)
    assert isinstance(cfg.item_scale[0][1], float)
    hash(cfg)


@pytest.mark.parametrize("bad", [((11, 5.0),), ((14, 2.0),), ((0, 2.0),), ((16, 2.0),),
                                 ((7, -1.0),), ((7, 2.0), (7, 3.0)), ((7,),),
                                 ((7, float("nan")),), ((7, float("inf")),)])
def test_잘못된_item_scale_은_거부한다(bad):
    with pytest.raises(ValueError):
        RewardConfig(item_scale=bad)


@pytest.mark.parametrize("red, want", [
    (None, 0.0),
    ((50.0, 10.0), 0.0),                     # v_ok = sqrt(4·49) = 14 > 10
    ((10.0, 10.0), 10.0 - 6.0),              # v_ok = sqrt(4·9) = 6
    ((0.5, 0.0), 0.0),                       # 정지선 앞에 서 있다
    ((0.5, 3.0), 3.0),                       # 1 m 안쪽: v_ok = 0
    ((-0.5, 3.0), 3.0),                      # 앞범퍼가 선을 조금 넘었지만 아직 '넘었다' 가 아니다
    ((-2.0, 10.0), 0.0),                     # 이미 넘었다 — ⑦ 이 맡는다
])
def test_red_excess(red, want):
    assert red_excess(red, 2.0) == pytest.approx(want)


def test_red_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().red_profile == 0.0
    b = h_board()
    sh = RewardShaper(b, RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", red=(10.0, 10.0))
    assert "red" not in out.terms


def test_red_profile_은_넘은_속도에_비례해_깎는다():
    b = h_board()
    cfg = RewardConfig(red_profile=0.1)
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", red=(10.0, 10.0))
    assert out.terms["red"] == pytest.approx(-0.1 * 4.0)
    assert out.total == pytest.approx(sum(out.terms.values()))
    calm = sh.step([], 0.0, zero, zero, "running", red=None)
    assert calm.terms["red"] == 0.0


@pytest.mark.parametrize("kw", [{"red_profile": -0.1}, {"red_profile": float("nan")},
                                {"red_decel": 0.0}, {"red_decel": -1.0}, {"red_decel": float("inf")}])
def test_잘못된_red_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)


def test_lane_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().lane_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "lane" not in sh.step([], 0.0, zero, zero, "running", lane=2.0).terms


def test_lane_profile_은_침범_깊이_시간에_비례해_깎는다():
    sh = RewardShaper(h_board(), RewardConfig(lane_profile=5.0))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", lane=0.3)
    assert out.terms["lane"] == pytest.approx(-1.5)
    assert out.total == pytest.approx(sum(out.terms.values()))
    calm = sh.step([], 0.0, zero, zero, "running", lane=None)
    assert calm.terms["lane"] == 0.0 and str(calm.terms["lane"]) == "0.0"   # −0.0 이 아니다


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_잘못된_lane_profile_은_거부한다(bad):
    with pytest.raises(ValueError):
        RewardConfig(lane_profile=bad)


@pytest.mark.parametrize("lat, want", [
    (None, 0.0),
    ((0.3, 999.0), 0.0),          # 문턱 안
    ((0.9, 999.0), 0.5),          # 0.9 − 0.4
    ((0.9, 30.0), 0.0),           # 40 m 안에 물체 — 끈다
    ((0.9, float("inf")), 0.5),   # 물체 없음
    ((0.9, 40.0), 0.0),           # 경계: 딱 40 m 면 끈다
    ((0.4, 999.0), 0.0),          # 경계: 문턱과 같으면 0
])
def test_lat_excess(lat, want):
    assert lat_excess(lat, 0.4, 40.0) == pytest.approx(want)


def test_lat_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().lat_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "lat" not in sh.step([], 0.0, zero, zero, "running", lat=(1.0, 999.0)).terms


def test_lat_profile_은_문턱_넘은_횡오차에_비례해_깎는다():
    sh = RewardShaper(h_board(), RewardConfig(lat_profile=0.5))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", lat=(1.0, 999.0))
    assert out.terms["lat"] == pytest.approx(-0.5 * 0.6)
    assert out.total == pytest.approx(sum(out.terms.values()))
    calm = sh.step([], 0.0, zero, zero, "running", lat=(0.1, 999.0))
    assert calm.terms["lat"] == 0.0 and str(calm.terms["lat"]) == "0.0"


@pytest.mark.parametrize("kw", [{"lat_profile": -0.1}, {"lat_profile": float("nan")},
                                {"lat_deadband": -0.1}, {"lat_deadband": float("inf")},
                                {"lat_free_range": -1.0}, {"lat_free_range": float("nan")}])
def test_잘못된_lat_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)


def test_stall_기본값은_0이고_항이_없다():
    assert RewardConfig().stall == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "stall" not in sh.step([], 0.0, zero, zero, "stalled").terms


def test_stall_을_주면_정체한_걸음에만_깎는다():
    sh = RewardShaper(h_board(), RewardConfig(stall=-200.0))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    run = sh.step([], 0.0, zero, zero, "running")
    assert run.terms["stall"] == 0.0 and str(run.terms["stall"]) == "0.0"
    out = sh.step([], 0.0, zero, zero, "stalled")
    assert out.terms["stall"] == -200.0
    assert out.total == pytest.approx(sum(out.terms.values()))


@pytest.mark.parametrize("kw", [{"stall": 1.0}, {"stall": float("nan")}, {"collision": 10.0},
                                {"collision": float("-inf")}, {"offroad": 5.0}, {"offroad": float("nan")}])
def test_실패_벌은_유한한_0_이하여야_한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)


def test_정체와_충돌이_같은_걸음이면_둘_다_깎는다():
    cfg = RewardConfig(stall=-200.0)
    sh = RewardShaper(h_board(), cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([Hit(1.0, 0, 11, "major")], 0.0, zero, zero, "stalled")
    assert out.terms["collision"] == cfg.collision and out.terms["stall"] == -200.0
    assert out.total == pytest.approx(sum(out.terms.values()))


@pytest.mark.parametrize("obs, want", [
    (None, 0.0),
    ((60.0, 13.0), 0.0),                      # v_ok = sqrt(4·57) ≈ 15.1
    ((12.0, 10.0), 10.0 - 6.0),               # v_ok = sqrt(4·9) = 6
    ((2.0, 3.0), 3.0),                        # buffer 안: v_ok = 0
    ((2.0, 0.0), 0.0),                        # 서 있으면 벌 없음
    ((12.0, -2.0), 0.0),                      # 멀어지는 중
])
def test_obs_excess(obs, want):
    assert obs_excess(obs, 2.0, 3.0) == pytest.approx(want)


def test_obs_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().obs_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "obs" not in sh.step([], 0.0, zero, zero, "running", obs=(12.0, 10.0)).terms


def test_obs_profile_은_넘은_속도에_비례해_깎는다():
    sh = RewardShaper(h_board(), RewardConfig(obs_profile=0.5))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", obs=(12.0, 10.0))
    assert out.terms["obs"] == pytest.approx(-0.5 * 4.0)
    assert out.total == pytest.approx(sum(out.terms.values()))
    calm = sh.step([], 0.0, zero, zero, "running", obs=None)
    assert calm.terms["obs"] == 0.0 and str(calm.terms["obs"]) == "0.0"


@pytest.mark.parametrize("kw", [{"obs_profile": -0.1}, {"obs_profile": float("nan")}, {"obs_decel": 0.0},
                                {"obs_decel": float("inf")}, {"obs_buffer": -1.0}, {"obs_margin": -0.1},
                                {"obs_margin": float("nan")}])
def test_잘못된_obs_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)


def test_block_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().block_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "block" not in sh.step([], 0.0, zero, zero, "running", block=5.0).terms


@pytest.mark.parametrize("block, want", [(None, 0.0), (20.0, 0.0), (15.0, 0.0), (14.9, -0.5), (0.0, -0.5)])
def test_block_profile_은_가까운_멈춘_장애물_뒤에서만_깎는다(block, want):
    sh = RewardShaper(h_board(), RewardConfig(block_profile=0.5))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", block=block)
    assert out.terms["block"] == pytest.approx(want)
    assert str(out.terms["block"]) != "-0.0"
    assert out.total == pytest.approx(sum(out.terms.values()))


@pytest.mark.parametrize("kw", [{"block_profile": -0.1}, {"block_profile": float("nan")},
                                {"block_range": -1.0}, {"block_range": float("inf")}])
def test_잘못된_block_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)
