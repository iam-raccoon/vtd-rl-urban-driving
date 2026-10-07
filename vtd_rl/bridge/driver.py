"""VTD 주행기 — 9910 상태를 한 프레임씩 받아 정책(또는 선생님)의 명령을 낸다.

오프라인 환경(`env/drive_env.py`)과 **같은 관측·행동·심판 코드**를 그대로 쓴다. 그래서 이 주행기를
오프라인 세계에 물리면 환경과 같은 판을 낸다(`tests/bridge/test_driver.py` 가 그것을 잠근다).
VTD 에서 다른 점은 입력이 어디서 오느냐뿐이다:

- 자차 상태는 9910 패킷에서 온다. 속도는 패킷에 없어 `VTDLink` 의 위치 미분 추정값을 쓴다.
- 실제 조향각은 모른다. 관측의 요레이트는 오프라인 세계와 같은 조향 속도 제한 모형(`dynamics.step`)으로 낸다.
- 판단은 평균 0.1 초마다(오프라인과 같은 10 Hz, 고정 위상), 명령은 프레임마다 다시 보낸다.
- 리스폰(순간이동)이 오면 진행 방향이 맞는 경로점 중 가장 가까운 곳에서 투영을 다시 시작한다.
- VTD 는 자차를 우리 경로의 출발 차로가 아닌 옆 차로에 놓기도 한다(실측: 코스 A·B·D 모두 3.0~3.3 m 왼쪽,
  시나리오 PathRef StartLane=1). 그래서 첫 투영은 출발점 근처에서 찾고, 처음 경로 차로 안에 들어올 때까지는
  도로 이탈 판정을 미룬다. 오프라인 세계는 출발점 위에서 시작하므로 둘 다 결과를 바꾸지 않는다.
- 종료는 오프라인 세계와 같은 규칙(완주·도로 이탈·시간초과·정체) + 심판이 낸 충돌이다.
"""
from dataclasses import dataclass

import math

import numpy as np

from vtd_rl import rule_stack as rs
from vtd_rl.env.action import ActionConfig, from_command, to_command
from vtd_rl.env.observation import ObsConfig, build_observation
from vtd_rl.env.reward import COLLISION_ITEMS
from vtd_rl.env.tags import RL, world_cap_by
from vtd_rl.referee.core import Referee
from vtd_rl.referee.rows import RowRecorder
from vtd_rl.world.world import WorldConfig, _room

RUNNING = "running"
RELOCATE_HEADING = math.radians(60.0)   # 리스폰 뒤 경로점을 다시 찾을 때 허용하는 방위 차 — 겹치거나 나란한 길의 반대 가지를 거른다


@dataclass
class _Ego:
    """`build_observation` 이 읽는 자차 칸만."""
    x: float
    y: float
    heading: float
    v: float = 0.0
    steer: float = 0.0


class _WorldView:
    """`build_observation` 이 읽는 세계 모양(board·ego·cfg.dynamics)."""

    def __init__(self, board, cfg: WorldConfig):
        self.board = board
        self.cfg = cfg
        self.ego = _Ego(*board.start_pose)


@dataclass
class _Info:
    s: float
    lateral: float
    index: int


class VtdDriver:
    """한 판을 몬다. `step(state, t)` 를 프레임마다 부르고 `done` 이 참이 되면 `finish()` 로 성적을 받는다.

    `policy` 를 주면 학생(결정적 평균 행동), `teacher=True` 면 규칙 스택(오프라인 환경의 그림자 선생님과 같은 구성).
    `t` 는 판 시작부터 흐른 **시뮬** 시간[s] 이다(VTD 에서는 `scripts/drive_vtd.py` 가 벽시계 차에 `sim_scale` 을 곱해 쌓는다).

    선생님은 오프라인 환경의 `TeacherPolicy` 와 같은 순서로 밟는다: 판단에는 **직전 프레임**까지 나온 명령을 쓰고,
    규칙 스택은 프레임마다 한 번(첫 프레임만 두 번 — 환경이 첫 명령을 만들려고 한 번 더 밟는다) 돈다.
    """

    def __init__(self, board, policy=None, teacher: bool = False, obs_cfg: ObsConfig = ObsConfig(),
                 action_cfg: ActionConfig = ActionConfig(), world_cfg: WorldConfig | None = None,
                 sections: int = 5, csv_path: str | None = None):
        if (policy is None) == (not teacher):
            raise ValueError("policy 와 teacher=True 중 정확히 하나를 준다")
        self.board = board
        self.policy = policy
        self.obs_cfg = obs_cfg
        self.action_cfg = action_cfg
        self.world_cfg = world_cfg or WorldConfig()
        self.view = _WorldView(board, self.world_cfg)
        self.referee = Referee(board, sections, True)
        self.rows = RowRecorder(csv_path)
        self.csv_path = csv_path
        self.teacher = None
        if teacher:
            from vtd_rl.teacher.shadow import ShadowTeacher
            self.teacher = ShadowTeacher(board)
        self.decision_dt = 1.0 / action_cfg.decision_hz
        self.time_limit = board.scenario.duration * self.world_cfg.time_limit_scale
        self.outcome = RUNNING
        self.info = None
        self.respawns = 0
        self.decisions = 0
        self._hint = None
        self._t_prev = None
        self._t_next = None               # 다음 판단 시각 — 고정 위상(평균 10 Hz, 프레임율과 무관)
        self._best = None                 # (s, t) — 정체 판정
        self.merged_at = None             # 처음 경로 차로 안에 들어온 시각 — 그 전에는 도로 이탈 판정을 미룬다
        self._collided = False            # 이번 판단 구간에 충돌 판정이 났다 — 구간 끝에서 끝낸다(환경과 같다)
        self._teacher_cmd = None
        self._cmd = (0.0, 0.0, rs.TS_OFF)
        self._prev = {"control": np.zeros(2, dtype=np.float32), "turn": 0}
        self._result = None

    @property
    def done(self) -> bool:
        return self.outcome != RUNNING

    # ------------------------------------------------------------------ 프레임
    def step(self, state, t: float):
        """이 프레임의 상태 → 보낼 명령 `(조향[rad], 가속[m/s²], 지시등)`. 판이 끝났으면 `done` 이 참이다."""
        if self.done:
            return self._cmd
        if state.respawned:
            self.respawns += 1
            self._hint = self._relocate(state)    # 순간이동 — 창 안 추적이 아니라 경로 전체에서 다시 찾는다
            self.view.ego.steer = 0.0
        first = self._t_prev is None
        # 첫 프레임은 출발점 근처에서 찾는다 — 경로가 뒤에서 출발점 곁을 다시 지나면 전역 최근접이 그쪽에 붙는다(코스 B)
        p = self.board.route.project(state.x, state.y, hint=0 if first else self._hint)
        self._hint = p.index
        dt = 0.0 if first else max(0.0, t - self._t_prev)
        self._t_prev = t
        self._advance_ego(state, dt)
        info = _Info(p.s, p.lateral, p.index)
        self.info = info

        if first:
            self._best = (p.s, t)
        else:
            out = self._outcome(p, t)
            if out != RUNNING:
                return self._end(out)

        # 판단 시각은 고정 위상으로 잡는다 — 오프라인의 0.05 초 프레임에서는 정확히 두 프레임마다, VTD 의 25 Hz 에서도 평균 10 Hz
        decide = first or t >= self._t_next - 1e-6
        if decide and self._collided:
            return self._end("collision")
        now = self.world_cfg.clock_origin + t
        if self.teacher is not None and first:
            self._teacher_cmd = self.teacher.act(state, now)      # 환경의 TeacherPolicy.act 가 첫 명령을 만들 때
        if decide:
            self._decide(state, None if first else info)
            self._schedule(t, first)
        if self.teacher is not None:
            self._teacher_cmd = self.teacher.act(state, now)      # 환경의 frame_hook — 이 프레임의 행 태그와 다음 판단에 쓴다

        cmd = rs.Command(steer=self._cmd[0], accel=self._cmd[1], turn=self._cmd[2], reason=RL, cap_by=RL)
        cmd.d_ego = 0.0 if first else p.lateral
        cmd.reason, cmd.cap_by = self._tags(state)
        state.speed = self.view.ego.v
        hits = self.referee.step(self.rows.record(t, state, cmd))
        if any(h.item in COLLISION_ITEMS for h in hits):
            self._collided = True
        return self._cmd

    def finish(self) -> dict:
        """판을 닫고 성적을 낸다(환경 `_finish` 와 같은 모양). 두 번 불러도 같은 것을 돌려준다."""
        if self._result is None:
            if not self.done:
                self.outcome = "aborted"
            self.referee.finish()
            self.rows.close()
            sheet = self.referee.sheet
            self._result = {"outcome": self.outcome,
                            "s": self.info.s if self.info else 0.0,
                            "route_total": self.board.route.total,
                            "sim_time": self._t_prev or 0.0,
                            "score": [sheet.score(k) for k in range(sheet.n)],
                            "sheet": [dict(s) for s in sheet.state],
                            "respawns": dict(self.referee.respawns),
                            "respawns_seen": self.respawns,
                            "merged_at": self.merged_at,
                            "decisions": self.decisions,
                            "notes": list(self.referee.ctx.notes),
                            "rows_csv": self.csv_path}
        return self._result

    # ------------------------------------------------------------------ 안쪽
    def _schedule(self, t, first):
        if first:
            self._t_next = t + self.decision_dt
            return
        self._t_next += self.decision_dt
        if t >= self._t_next - 1e-6:          # 한 주기 넘게 밀렸다(프레임이 뭉치거나 끊겼다) — 지금부터 다시 센다
            self._t_next = t + self.decision_dt

    def _relocate(self, state):
        """진행 방향이 `RELOCATE_HEADING` 안인 경로점 중 가장 가까운 색인. 없으면 None(전역 검색)."""
        route = self.board.route
        best, best_d = None, math.inf
        for j, (px, py) in enumerate(route.pts):
            dh = (route.heading_at(j) - state.heading + math.pi) % (2.0 * math.pi) - math.pi
            if abs(dh) > RELOCATE_HEADING:
                continue
            d = (px - state.x) ** 2 + (py - state.y) ** 2
            if d < best_d:
                best, best_d = j, d
        return best

    def _advance_ego(self, state, dt):
        ego = self.view.ego
        ego.x, ego.y, ego.heading, ego.v = state.x, state.y, state.heading, float(state.speed)
        # 조향 실제값 추정 — 오프라인 세계의 `dynamics.step` 과 같은 한계·속도 제한
        dyn = self.world_cfg.dynamics
        target = min(max(self._cmd[0], -dyn.max_steer), dyn.max_steer)
        dmax = dyn.max_steer_rate * dt
        ego.steer += min(max(target - ego.steer, -dmax), dmax)

    def _decide(self, state, info):
        if self.teacher is not None:
            c = self._teacher_cmd
            action = from_command(c.steer, c.accel, c.turn, self.action_cfg)
        else:
            prev = (float(self._prev["control"][0]), float(self._prev["control"][1]), int(self._prev["turn"]))
            obs = build_observation(self.view, state, info, prev, self.obs_cfg)
            action = self.policy.act(obs, deterministic=True)
        self._cmd = to_command(action, self.action_cfg)
        self._prev = {"control": np.asarray(action["control"], dtype=np.float32).copy(),
                      "turn": int(action["turn"])}
        self.decisions += 1

    def _tags(self, state):
        if self.teacher is not None:
            c = self._teacher_cmd
            return str(c.reason), str(c.cap_by)
        return RL, world_cap_by(state)

    def _outcome(self, p, t) -> str:
        """오프라인 세계 `World._outcome` 과 같은 순서·문턱."""
        cfg = self.world_cfg
        if self.board.route.total - p.s <= cfg.goal_radius:
            return "goal"
        plan = self.board.lane_plan[p.index] or {}
        if self.merged_at is None and -_room(plan, "r", "pr") <= p.lateral <= _room(plan, "l", "pl"):
            self.merged_at = t
        if self.merged_at is not None and not (plan.get("j") or plan.get("jx")):
            left = _room(plan, "l", "pl") + cfg.offroad_margin
            right = _room(plan, "r", "pr") + cfg.offroad_margin
            if p.lateral > left or p.lateral < -right:
                return "offroad"
        if t >= self.time_limit - 1e-9:
            return "timeout"
        best_s, best_t = self._best
        if p.s > best_s + cfg.stall_progress:
            self._best = (p.s, t)
        elif t - best_t >= cfg.stall_seconds - 1e-9:
            return "stalled"
        return RUNNING

    def _end(self, outcome):
        self.outcome = outcome
        self._cmd = (0.0, self.action_cfg.accel_min * 0.8, rs.TS_OFF)
        return self._cmd


def summarize(result: dict) -> dict:
    """성적 한 줄 — 완주 여부, 점수(구간 평균, 미완주 0), 진행률."""
    sc = result["score"] or [0.0]
    score = sum(sc) / len(sc)
    goal = result["outcome"] == "goal"
    return {"outcome": result["outcome"], "goal": goal, "score": score,
            "score_incl": score if goal else 0.0,
            "progress": result["s"] / max(result["route_total"], 1e-9),
            "respawns": sum(len(v) for v in result["respawns"].values())}
