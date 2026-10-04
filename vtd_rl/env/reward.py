"""보상 — 스펙 §5. 심판이 낸 감점과 진행 거리로 매 걸음 보상을 만든다.

위반을 세는 규칙은 `RewardConfig.violation_mode` 로 고른다 — 두 가지가 있고 기본값은
"repeat"(예전과 같은 동작)다.

- "repeat"(기본값): 대회는 같은 항목을 구간당 한 번만 깎지만 학습 보상은 위반마다 깎는다.
  다만 심판은 채점기와 같은 호출을 다 내므로(과속 한 판에 198번, 침범이 구간을 걸치면
  구간마다) 같은 (항목, 구간)이 이어지면 repeat_gap 초에 한 번만 센다 — '한 번 깎였으니
  계속 어겨도 된다' 와 '프레임마다 깎여서 다른 항이 묻힌다' 사이를 가른다. 구간 경계 지날
  때는 새 구간이 시작되어 새로 센다(채점기 방식 따름).
- "once_per_section": 대회 채점기와 같은 규칙 — (항목, 구간)마다 최종 등급 한 번만 세고,
  경미→중대로 심화되면 차액만 문다(`ViolationTracker._charge_once`). 이 모드에서는
  `repeat_gap` 을 안 쓴다.
"""
import math
from dataclasses import dataclass, field

COLLISION_ITEMS = (11, 14)       # score_fma ⑪ 장애물 충돌 · ⑭ 차량·보행자 접촉
RED_STOP_TARGET = 1.0            # M6v — 적색 감속 곡선이 겨누는 정지 위치: 앞범퍼가 정지선 앞 1 m
                                 # (채점기 ⑦ 정지 인정 구간 [0, 2 m) 의 가운데)


def red_excess(red, decel: float) -> float:
    """적색 접근 `(gap, v)` → 감속 곡선을 넘은 속도[m/s]. `red` 가 None(적색 아님)이면 0.

    `gap` 은 앞범퍼에서 정지선까지[m](앞이 +). 허용 속도 `v_ok = sqrt(2·decel·max(gap−1, 0))`
    는 정지선 앞 1 m 에 서는 등감속 곡선이다. 채점기가 '넘었다' 고 보는 −1 m 너머는 ⑦ 이 맡으므로 0.
    """
    if red is None:
        return 0.0
    gap, v = red
    if gap < -1.0:
        return 0.0
    v_ok = math.sqrt(2.0 * decel * max(gap - RED_STOP_TARGET, 0.0))
    return max(0.0, float(v) - v_ok)


@dataclass(frozen=True)
class RewardConfig:
    progress_total: float = 100.0    # 완주까지 진행 항의 합
    goal_bonus: float = 50.0
    time_cost: float = 0.01          # 판단 한 걸음마다
    minor: float = -3.0
    major: float = -6.0
    rule_scale: float = 1.0          # 커리큘럼 단계마다 키운다(스펙 §5)
    collision: float = -50.0
    offroad: float = -50.0
    comfort_steer: float = -0.10
    comfort_accel: float = -0.05
    # True 면 승차감을 실행 행동이 아니라 정책의 의도(결정적 평균)의 변화량으로 잰다 — PPO 가
    # 매 걸음 독립 표본을 뽑으면 그 변화량의 대부분이 탐색 잡음이고, 정책은 그걸 줄이려고
    # 탐색을 죽인다(M4b 실측: 판당 -108.67). 의도로 재면 탐색은 공짜가 되고 진짜 급조작만 벌한다.
    comfort_on_intent: bool = False
    repeat_gap: float = 1.0          # 같은 항목을 다시 세기까지[s]. once_per_section 모드에서는
                                     # 안 쓰인다(그 모드는 (항목,구간)마다 딱 한 번만 센다).
    violation_mode: str = "repeat"   # "repeat" = 같은 (항목,구간)을 repeat_gap 마다 다시 센다.
                                     # "once_per_section" = 대회 채점기와 같은 규칙 — (항목,구간)
                                     # 마다 한 번, 경미→중대 심화는 차액만. M4c 실측: repeat 는
                                     # 대회 기준 94.2/100 인 학생에게 판당 −115 를 물려 PPO 가
                                     # 점수를 올릴 이유를 없앤다(docs/reports/m4c-ppo-notes.md).
    # M6u — 항목별 벌 배율 `((항목, 배율), ...)`. 그 항목의 히트는 minor·major 에 배율을 곱해 센다.
    # 기본 `()` 이면 예전과 같다. 충돌 항목(⑪⑭)은 `collision` 항이 따로 물리므로 받지 않는다.
    item_scale: tuple = ()
    # M6v — 적색 감속 곡선. 적색 신호 앞에서 허용 속도(`red_excess`)를 넘으면 걸음마다
    # 넘은 속도[m/s] × red_profile 을 깎는다(항 `red`). 기본 0 이면 꺼짐(항도 없다, 예전과 같다).
    red_profile: float = 0.0
    red_decel: float = 2.0           # 곡선의 감속[m/s²]
    # M6x — 차로 침범 깊이 벌. 채점기가 ③ 물림으로 세는 행의 0.1 m 넘는 깊이×시간[m·s] × lane_profile 을
    # 걸음마다 깎는다(항 `lane`, 행이 풀리는 2.5 초 뒤에 온다). 기본 0 이면 꺼짐(항도 없다).
    lane_profile: float = 0.0

    def __post_init__(self):
        pairs = []
        for pair in (self.item_scale or ()):   # None(`--item-scale` 을 안 준 실행의 hparams)도 ()
            if len(pair) != 2:
                raise ValueError(f"item_scale 은 (항목, 배율) 짝이어야 한다: {pair!r}")
            item, scale = int(pair[0]), float(pair[1])
            if not 1 <= item <= 15 or item in COLLISION_ITEMS:
                raise ValueError(f"item_scale 항목은 1~15 이고 충돌 항목 {COLLISION_ITEMS} 가 아니어야 한다: {item}")
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError(f"item_scale 배율은 유한한 0 이상이어야 한다: {scale}")
            pairs.append((item, scale))
        if len({item for item, _ in pairs}) != len(pairs):
            raise ValueError(f"item_scale 에 같은 항목이 두 번 있다: {pairs}")
        object.__setattr__(self, "item_scale", tuple(pairs))   # frozen — 목록으로 줘도 튜플로 둔다(해시)
        if not math.isfinite(self.red_profile) or self.red_profile < 0.0:
            raise ValueError(f"red_profile 은 유한한 0 이상이어야 한다: {self.red_profile}")
        if not math.isfinite(self.red_decel) or self.red_decel <= 0.0:
            raise ValueError(f"red_decel 은 유한한 양수여야 한다: {self.red_decel}")
        if not math.isfinite(self.lane_profile) or self.lane_profile < 0.0:
            raise ValueError(f"lane_profile 은 유한한 0 이상이어야 한다: {self.lane_profile}")


@dataclass
class RewardStep:
    total: float
    terms: dict
    collision: bool
    counted: int


class ViolationTracker:
    def __init__(self, repeat_gap: float, violation_mode: str = "repeat"):
        if violation_mode not in ("repeat", "once_per_section"):
            raise ValueError(f"violation_mode 는 repeat|once_per_section 이어야 한다: {violation_mode!r}")
        self.repeat_gap = repeat_gap
        self.violation_mode = violation_mode
        self._last: dict = {}     # repeat 모드: (item, sec) -> 마지막으로 센 시각
        self._level: dict = {}    # once_per_section 모드: (item, sec) -> 이미 깎은 등급

    def reset(self):
        self._last.clear()
        self._level.clear()

    def count(self, hits):
        """센 히트만 돌려준다 — 감점 크기는 `charge()` 가 함께 낸다."""
        return self.charge(hits, 0.0, 0.0)[0]

    def charge(self, hits, minor: float, major: float):
        """(센 히트, 그 히트들이 깎는 총점)."""
        if self.violation_mode == "once_per_section":
            return self._charge_once(hits, minor, major)
        out, total = [], 0.0
        for h in hits:
            key = (h.item, h.sec)
            last = self._last.get(key)
            if last is None or h.t - last >= self.repeat_gap - 1e-9:
                self._last[key] = h.t
                out.append(h)
                total += major if h.level == "major" else minor
        return out, total

    def _charge_once(self, hits, minor: float, major: float):
        """대회 채점기 `score_fma.Sheet` 와 같은 규칙 — (항목,구간)마다 최종 등급 한 번만.

        항목 1~14 는 무작위 순열 4000 개로 채점기 원본(`Sheet.hit`)과 완전 일치를 확인했다.

        **예외: 항목 15(리스폰).** 채점기(`third_party/rule_stack/eval/score_fma.py`의
        `item_respawn`, 그 결과를 쓰는 `main` 의 `pen_r` 계산)는 15 를 `Sheet.hit` 으로 처리하지
        않고 따로 센다 — 구간당 1 회는 무료, 그 뒤로는 `MAJOR * (횟수 - 1)`. 여기 `_charge_once`
        는 15 도 다른 항목과 똑같이 첫 발생에서 곧장 등급을 매기므로, 한 구간에 리스폰이 1 회뿐이면
        채점기는 0 점을 매기는데 이 함수는 major 를 문다. 기존 동작이고(`repeat` 모드도 이미
        어긋나 있었다) 스펙 §5 가 "항목 15 는 오프라인 세계에 없다" 고 적어 뒀으므로 고치지
        않는다 — 다만 이 독스트링이 "채점기와 같은 규칙" 이라고 주장하니 예외를 이름으로 적는다.
        """
        out, total = [], 0.0
        for h in hits:
            key = (h.item, h.sec)
            cur = self._level.get(key)
            if cur == "major" or cur == h.level:
                continue                       # 이미 중대거나 같은 등급이면 아무 일도 없다
            self._level[key] = h.level
            out.append(h)
            total += (major - minor) if cur == "minor" else (
                major if h.level == "major" else minor)
        return out, total


@dataclass
class RewardShaper:
    board: object
    cfg: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        self.tracker = ViolationTracker(self.cfg.repeat_gap, self.cfg.violation_mode)
        self._per_m = self.cfg.progress_total / max(self.board.route.total, 1.0)
        self._prev_intent = None

    def reset(self):
        self.tracker.reset()
        self._prev_intent = None

    def step(self, hits, ds: float, action, prev_action, outcome: str, intent=None, red=None,
             lane=None) -> RewardStep:
        cfg = self.cfg
        # 충돌 항목(⑪⑭)은 `collision` 항이 따로 −50 을 물리므로 위반 합계에서 뺀다(기존 규칙).
        # `charge` 가 감점까지 내므로, 충돌 히트를 **처음부터 갈라서** 두 번 부른다 —
        # 그래야 "감점에서 충돌을 뺀다" 를 뺄셈으로 다시 구하지 않아도 된다.
        rule_hits = [h for h in hits if h.item not in COLLISION_ITEMS]
        col_hits = [h for h in hits if h.item in COLLISION_ITEMS]
        # M6u — 배율을 건 항목은 따로 센다. 추적기 상태가 (항목, 구간) 열쇠라 나눠 불러도 결과가 같다.
        scaled = {item for item, _ in cfg.item_scale}
        counted, penalty = self.tracker.charge(
            [h for h in rule_hits if h.item not in scaled], cfg.minor, cfg.major)
        for item, scale in cfg.item_scale:
            c, p = self.tracker.charge([h for h in rule_hits if h.item == item],
                                       cfg.minor * scale, cfg.major * scale)
            counted = counted + c
            penalty += p
        col_counted, _col_penalty = self.tracker.charge(col_hits, cfg.minor, cfg.major)
        collision = bool(col_counted)
        violation = penalty * cfg.rule_scale
        if cfg.comfort_on_intent and intent is not None:
            # 콜드스타트(이 판의 첫 걸음, `_prev_intent`가 아직 없다)는 "직전 의도 = 지금 의도"로
            # 본다(diff=0) — 실행 행동으로 대체 계산하면(당시 `_prev_action`은 0으로 시작한다)
            # 의도 모드를 켠 첫 걸음만 탐색 잡음이 아니라 진짜 행동으로 벌점을 매기게 되어,
            # comfort_on_intent 가 없애려던 바로 그 잡음이 첫 걸음에서 새어 든다.
            prev_intent = self._prev_intent if self._prev_intent is not None else intent
            d_steer = abs(float(intent[0]) - float(prev_intent[0]))
            d_accel = abs(float(intent[1]) - float(prev_intent[1]))
        else:
            d_steer = abs(float(action["control"][0]) - float(prev_action["control"][0]))
            d_accel = abs(float(action["control"][1]) - float(prev_action["control"][1]))
        if intent is not None:
            self._prev_intent = (float(intent[0]), float(intent[1]))
        terms = {
            "progress": self._per_m * ds,
            "time": -cfg.time_cost,
            "violation": violation,
            "collision": cfg.collision if collision else 0.0,
            "offroad": cfg.offroad if outcome == "offroad" else 0.0,
            "goal": cfg.goal_bonus if outcome == "goal" else 0.0,
            "comfort": cfg.comfort_steer * d_steer + cfg.comfort_accel * d_accel,
        }
        if cfg.red_profile > 0.0:     # 끈 실행은 항 자체가 없다(합·로그가 예전과 같다)
            terms["red"] = 0.0 - cfg.red_profile * red_excess(red, cfg.red_decel)   # 0.0− : 적색 아닌 걸음이 −0.0 이 안 되게
        if cfg.lane_profile > 0.0:    # 끈 실행은 항 자체가 없다
            terms["lane"] = 0.0 - cfg.lane_profile * (lane or 0.0)   # 0.0− : 침범 없는 걸음이 −0.0 이 안 되게
        return RewardStep(sum(terms.values()), terms, collision, len(counted) + len(col_counted))
