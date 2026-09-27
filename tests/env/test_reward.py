import pytest

from vtd_rl.env.reward import COLLISION_ITEMS, RewardConfig, RewardShaper, ViolationTracker
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
