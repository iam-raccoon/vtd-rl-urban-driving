"""보상 — 스펙 §5. 심판이 낸 감점과 진행 거리로 매 걸음 보상을 만든다.

대회는 같은 항목을 구간당 한 번만 깎지만 학습 보상은 위반마다 깎는다. 다만 심판은 채점기와 같은
호출을 다 내므로(과속 한 판에 198번, 침범이 구간을 걸치면 구간마다) 같은 (항목, 구간)이
이어지면 repeat_gap 초에 한 번만 센다 — '한 번 깎였으니 계속 어겨도 된다' 와
'프레임마다 깎여서 다른 항이 묻힌다' 사이를 가른다. 구간 경계 지날 때는 새 구간이 시작되어
새로 센다(채점기 방식 따름).
"""
from dataclasses import dataclass, field

COLLISION_ITEMS = (11, 14)       # score_fma ⑪ 장애물 충돌 · ⑭ 차량·보행자 접촉


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
    repeat_gap: float = 1.0          # 같은 항목을 다시 세기까지[s]
    violation_mode: str = "repeat"   # "repeat" = 같은 (항목,구간)을 repeat_gap 마다 다시 센다.
                                     # "once_per_section" = 대회 채점기와 같은 규칙 — (항목,구간)
                                     # 마다 한 번, 경미→중대 심화는 차액만. M4c 실측: repeat 는
                                     # 대회 기준 94.2/100 인 학생에게 판당 −115 를 물려 PPO 가
                                     # 점수를 올릴 이유를 없앤다(docs/reports/m4c-ppo-notes.md).


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
        """대회 채점기 `score_fma.Sheet` 와 같은 규칙 — (항목,구간)마다 최종 등급 한 번만."""
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

    def step(self, hits, ds: float, action, prev_action, outcome: str, intent=None) -> RewardStep:
        cfg = self.cfg
        # 충돌 항목(⑪⑭)은 `collision` 항이 따로 −50 을 물리므로 위반 합계에서 뺀다(기존 규칙).
        # `charge` 가 감점까지 내므로, 충돌 히트를 **처음부터 갈라서** 두 번 부른다 —
        # 그래야 "감점에서 충돌을 뺀다" 를 뺄셈으로 다시 구하지 않아도 된다.
        rule_hits = [h for h in hits if h.item not in COLLISION_ITEMS]
        col_hits = [h for h in hits if h.item in COLLISION_ITEMS]
        counted, penalty = self.tracker.charge(rule_hits, cfg.minor, cfg.major)
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
        return RewardStep(sum(terms.values()), terms, collision, len(counted) + len(col_counted))
