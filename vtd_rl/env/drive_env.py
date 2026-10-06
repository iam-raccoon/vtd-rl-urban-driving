"""Gymnasium 환경 — 세계(M1) + 온라인 심판(M2a) + 관측·행동·보상.

한 걸음(기본 10 Hz)은 시뮬 두 프레임이다. 심판과 행 기록은 **프레임마다** 돈다 —
채점기와 같은 판정을 내려면 20 Hz 행이 필요하고, 그림자 선생님도 프레임마다 같은 입력을 받아야 한다.

판이 끝나면(`terminated`/`truncated`) 그 걸음의 `info["result"]`에 성적(`score`·`sheet`·`respawns`·
`notes`·`rows_csv`)이 실린다 — `info["episode"]`가 아니다. Gymnasium 의 `RecordEpisodeStatistics`,
SB3 의 `Monitor` 래퍼가 `info["episode"]`를 덮어써서 성적표가 조용히 사라지기 때문이다.
그 뒤로 `step()`이 또 오면(정상적으로는 오면 안 되지만) 세계·심판·행 기록기를 더 건드리지
않고 마지막 상태를 그대로 돌려준다 — `finish()`는 판마다 정확히 한 번만 불러야 한다.
"""
import math
import os
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from vtd_rl import rule_stack as rs
from vtd_rl.env.action import ActionConfig, action_space, frames_per_step, to_command
from vtd_rl.env.board_index import board_index
from vtd_rl.env.observation import ObsConfig, build_observation, observation_space
from vtd_rl.env.reward import COLLISION_ITEMS, RewardConfig, RewardShaper
from vtd_rl.env.tags import RL, world_cap_by
from vtd_rl.referee.core import Referee
from vtd_rl.referee.rows import RowRecorder
from vtd_rl.world.signals import PASSED_MARGIN
from vtd_rl.world.world import World, WorldConfig

RUNNING = "running"
EGO_HALF_W = 0.943               # 내 차 반폭[m] — 채점기 run_logger.EGO_HALF_W 와 같다(M7d 장애물 통로)
ROAD_SURFACE_H = 0.25            # 이보다 낮은 물체는 노면 물체 — vtd_io.classify_object 의 경계(M7d)
BLOCK_STILL_V = 0.5              # 이보다 느린 물체를 '멈춘 장애물' 로 본다[m/s](M7g)


@dataclass
class EnvConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sections: int = 5
    use_map: bool = True
    log_dir: str | None = None


class VtdDriveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, boards, config: EnvConfig | None = None, render_mode=None):
        if not boards:
            raise ValueError("판이 하나는 있어야 한다")
        self.boards = list(boards)
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.observation_space = observation_space(self.cfg.obs)
        self.action_space = action_space(self.cfg.action)
        self.frames = frames_per_step(self.cfg.action, self.cfg.world.dt)
        self.frame_hook = None
        # 프레임마다 (reason, cap_by) 를 주는 갈고리. 규칙 스택이 운전할 때는 스택이 내는 진짜
        # 태그를 그대로 넘겨야 채점기 ⑧ 면책이 살아난다(TeacherPolicy 가 건다). 안 걸려 있으면
        # 세계의 물체에서 짓는다(env/tags.py) — 태그는 채점기가 어차피 면책할 정차만 면책한다.
        self.command_tags = None
        # `command_tags` 와 같은 패턴 — 프레임 대신 걸음마다 정책의 의도(조향, 가속)를 준다.
        # 값은 (조향, 가속) 튜플이거나 None(안 걸려 있으면 RewardShaper 는 실행 행동을 쓴다).
        # 콜러블이 아니라 평범한 속성인 이유: 학습은 AsyncVectorEnv(별도 프로세스)를 쓰고,
        # gymnasium 의 `set_attr()`은 값은 프로세스 경계를 넘겨주지만 람다는 피클이 안 된다.
        self.intent = None
        # M6f — 판 고르기를 넘겨받는 콜러블 `rng -> (판, 새로_지은_판인가)`. 안 걸려 있으면(기본)
        # 예전처럼 `self.boards` 에서 고른다. 이름을 준 리셋(`options={"board": ...}`)은 늘 목록에서
        # 찾는다. 새로 지은 판은 이름이 리셋마다 달라 세계를 캐시하지 않는다(캐시가 끝없이 자란다).
        self.board_sampler = None
        self.board = self.world = self.referee = self.state = None
        self._worlds: dict = {}          # 판마다 세계를 다시 짓지 않는다(신호 찾기가 판당 수십 ms)
        self._episode = 0
        self.episode = 0                 # 리셋마다 오르는 번호 — 판 이름만으로는 판 바뀜을 못 본다
        self._recorder = None
        self._info = None
        self._prev_action = self._zero_action()
        self._shaper = None
        self._done = False       # 판이 끝난 뒤 다시 step() 이 와도 세계·심판을 더 밟지 않는 걸쇠
        self._terminal = None    # 걸쇠가 걸릴 때의 (outcome, terminated, truncated) — 그대로 되돌려준다

    # ------------------------------------------------------------------ Gymnasium
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_recorder()
        name = (options or {}).get("board")
        fresh = False
        if name is not None:
            try:
                self.board = next(b for b in self.boards if b.name == name)
            except StopIteration:
                names = [b.name for b in self.boards]
                raise ValueError(f"판을 찾을 수 없다: {name!r} (있는 판: {names})") from None
        elif self.board_sampler is not None:
            self.board, fresh = self.board_sampler(self.np_random)
        else:
            self.board = self.boards[int(self.np_random.integers(len(self.boards)))]
        world_seed = int(self.np_random.integers(2 ** 31 - 1))
        self.world = None if fresh else self._worlds.get(self.board.name)
        if self.world is None:
            idx = board_index(self.board)
            self.world = World(self.board, self.cfg.world, signals=idx.signals, seed=world_seed)
            if not fresh:
                self._worlds[self.board.name] = self.world
        # 세계 캐시의 열쇠는 판 **이름**이다 — 이름만 같고 다른 판이 섞이면 관측·심판은 이 판을,
        # 세계는 저 판을 보게 된다. 조용히 갈라지지 않게 여기서 못을 박는다.
        assert self.world.board is self.board, f"세계 캐시가 다른 판을 준다: {self.board.name}"
        self.state = self.world.reset(world_seed)
        self.referee = Referee(self.board, self.cfg.sections, self.cfg.use_map)
        self._shaper = RewardShaper(self.board, self.cfg.reward)
        self._shaper.reset()
        self._info = None
        self._prev_action = self._zero_action()
        self._episode += 1
        self.episode = self._episode
        self._recorder = None            # 행 CSV 는 첫 record 에서 연다(밟지 않은 리셋은 파일도 안 남긴다)
        self._done = False
        self._terminal = None
        obs = build_observation(self.world, self.state, self._info, self._prev_pair(), self.cfg.obs)
        return obs, {"board": self.board.name, "outcome": RUNNING, "s": 0.0, "sim_time": 0.0,
                     "frames": 0, "hits": [], "counted": 0, "reward_terms": {}}

    def step(self, action):
        if self.world is None:
            raise RuntimeError("reset() 을 먼저 불러야 한다")
        if self._done:
            return self._frozen_step()
        steer, accel, turn = to_command(action, self.cfg.action)
        cmd = rs.Command(steer=steer, accel=accel, turn=turn, reason=RL, cap_by=RL)
        s0 = self._info.s if self._info is not None else 0.0
        hits, frames, outcome = [], 0, RUNNING
        for _ in range(self.frames):
            if self.frame_hook is not None:
                self.frame_hook(self.state, self.world.clock)
            cmd.d_ego = self._info.lateral if self._info is not None else 0.0
            cmd.reason, cmd.cap_by = self._tags()        # 프레임마다 — 갈고리는 그 프레임의 답을 준다
            self.state.speed = self.world.ego.v          # 로그 속도의 주인은 세계다
            hits += self.referee.step(self._rows().record(self.world.t, self.state, cmd))
            self.state, self._info = self.world.step(steer, accel, turn)
            frames += 1
            if self._info.done:
                outcome = self._info.outcome
                break
        if outcome != RUNNING or any(h.item in COLLISION_ITEMS for h in hits):
            hits += self.referee.finish()          # 남은 대기 판정까지 이 걸음의 보상에 넣는다
        red = self._red_gap() if self.cfg.reward.red_profile > 0.0 else None
        lane = (self.referee.take_edge_excess() * self.cfg.world.dt
                if self.cfg.reward.lane_profile > 0.0 else None)       # [m·행] × 프레임 dt = [m·s]
        lat = self._lat_gap() if self.cfg.reward.lat_profile > 0.0 else None
        obs = self._obs_gap() if self.cfg.reward.obs_profile > 0.0 else None
        block = self._block_gap() if self.cfg.reward.block_profile > 0.0 else None
        shaped = self._shaper.step(hits, max(0.0, self._info.s - s0), action, self._prev_action,
                                   outcome, intent=self.intent, red=red, lane=lane, lat=lat, obs=obs,
                                   block=block)
        self._prev_action = {"control": np.asarray(action["control"], dtype=np.float32).copy(),
                             "turn": int(action["turn"])}
        # M7a — 정체 벌을 주면 정체는 실패로 끝난다(기본 0 이면 예전처럼 아래에서 잘리기만 한다).
        stall_fail = outcome == "stalled" and self.cfg.reward.stall != 0.0
        terminated = outcome in ("goal", "offroad") or shaped.collision or stall_fail
        # 둘이 함께 참이면 안 된다(Gymnasium 규약) — 충돌과 시간초과가 같은 걸음에 올 수 있다
        truncated = outcome in ("timeout", "stalled") and not terminated
        if outcome == RUNNING and shaped.collision:
            outcome = "collision"        # 세계는 충돌을 모른다 — 심판이 낸 충돌 항목이 종료 사유다
        info = {"board": self.board.name, "outcome": outcome, "s": self._info.s,
                "sim_time": self._info.t, "frames": frames,
                "hits": [(h.t, h.sec, h.item, h.level) for h in hits],
                "counted": shaped.counted, "reward_terms": shaped.terms}
        if terminated or truncated:
            self._done = True
            self._terminal = (outcome, bool(terminated), bool(truncated))
            info["result"] = self._finish()
        obs = build_observation(self.world, self.state, self._info, self._prev_pair(), self.cfg.obs)
        return obs, float(shaped.total), bool(terminated), bool(truncated), info

    def _red_gap(self):
        """신호가 적색이면 `(앞범퍼~그 신호 정지선 거리[m], 속도[m/s])`, 아니면 None — M6v 감속 곡선용.

        정지선 s 는 세계가 보고하는 신호(`SignalReporter` 와 같은 규칙: 같은 tl_id, 뒷축이 선을
        PASSED_MARGIN 넘기 전)에서 찾고, 앞범퍼 오프셋은 채점기와 같은 `score_fma.FRONT` 다.
        """
        st = self.state
        if self._info is None or st.tl_id <= 0 or st.tl_state != rs.TL_RED:
            return None
        s = self._info.s
        for sig in self.world.reporter.signals:          # s 순서
            if sig.tl_id == st.tl_id and sig.s + PASSED_MARGIN >= s:
                return sig.s - s - rs.score_fma.FRONT, float(self.world.ego.v)
        return None

    def _lat_gap(self):
        """`(|경로 기준 횡오차|[m], 가장 가까운 물체까지 거리[m])` — M6y 횡오차 벌용. 물체가 없으면 거리 inf."""
        if self._info is None:
            return None
        ego = self.world.ego
        near = min((math.hypot(o.x - ego.x, o.y - ego.y) for o in self.state.objects), default=math.inf)
        return abs(float(self._info.lateral)), float(near)

    def _obs_gap(self):
        """차 앞 통로 안 가장 가까운 물체의 `(앞범퍼~물체 뒤끝 거리[m], 다가가는 속도[m/s])`, 없으면 None(M7d).

        내 차 기준(뒷축, 진행 방향 x)으로 물체를 돌려, 앞(fx > 0)에 있고 옆 거리가
        `물체 반폭 + 내 차 반폭 + obs_margin` 안인 것만 본다. 노면 물체(높이 < 0.25 m)는 뺀다.
        """
        if self._info is None:
            return None
        ego = self.world.ego
        ch, sh = math.cos(-ego.heading), math.sin(-ego.heading)
        margin = self.cfg.reward.obs_margin
        best = None
        for o in self.state.objects:
            if o.height < ROAD_SURFACE_H:
                continue
            dx, dy = o.x - ego.x, o.y - ego.y
            fx, fy = dx * ch - dy * sh, dx * sh + dy * ch
            if fx <= 0.0 or abs(fy) >= o.width / 2.0 + EGO_HALF_W + margin:
                continue
            gap = max(0.0, fx - rs.score_fma.FRONT - o.length / 2.0)
            if best is None or gap < best[0]:
                v_close = float(ego.v) - max(0.0, float(o.speed) * math.cos(o.heading - ego.heading))
                best = (gap, v_close)
        return best

    def _block_gap(self):
        """통로 안 가장 가까운 '멈춘 장애물'(속도 < 0.5 m/s, 사람·자전거 아님)까지 앞범퍼 거리[m], 없으면 None(M7g).

        통로·거리 계산은 `_obs_gap` 과 같다. 사람·자전거는 `vtd_io.classify_object` 로 가른다 — 보행자 앞에서는
        비키지 말고 기다려야 하므로 머무는 벌에서 뺀다.
        """
        if self._info is None:
            return None
        import vtd_io                                   # rule_stack 이 경로를 잡아 둔다
        ego = self.world.ego
        ch, sh = math.cos(-ego.heading), math.sin(-ego.heading)
        margin = self.cfg.reward.obs_margin
        best = None
        for o in self.state.objects:
            if o.height < ROAD_SURFACE_H or abs(float(o.speed)) >= BLOCK_STILL_V:
                continue
            if vtd_io.classify_object(o.length, o.width, o.height) == vtd_io.ObjectClass.VRU:
                continue
            dx, dy = o.x - ego.x, o.y - ego.y
            fx, fy = dx * ch - dy * sh, dx * sh + dy * ch
            if fx <= 0.0 or abs(fy) >= o.width / 2.0 + EGO_HALF_W + margin:
                continue
            gap = max(0.0, fx - rs.score_fma.FRONT - o.length / 2.0)
            if best is None or gap < best:
                best = gap
        return best

    def _frozen_step(self):
        """판이 이미 끝난 뒤 온 step() — 세계·심판·행 기록기를 더 밟지 않고 마지막 상태를 그대로 준다.

        `_finish()`가 행 기록기를 닫아 두기도 했고(다시 쓰면 예외), 세계는 충돌 뒤에도
        `outcome="running"`을 낼 수 있어 그대로 다시 밟으면 `finish()`가 또 불려 같은 판정을
        성적표에 다시 쓸 수 있다 — 그래서 이 걸음은 아무것도 하지 않는다.
        """
        outcome, terminated, truncated = self._terminal
        obs = build_observation(self.world, self.state, self._info, self._prev_pair(), self.cfg.obs)
        info = {"board": self.board.name, "outcome": outcome, "s": self._info.s,
                "sim_time": self._info.t, "frames": 0, "hits": [], "counted": 0, "reward_terms": {}}
        return obs, 0.0, terminated, truncated, info

    def close(self):
        self._close_recorder()

    # ------------------------------------------------------------------ 안쪽
    def _zero_action(self):
        return {"control": np.zeros(2, dtype=np.float32), "turn": 0}

    def _prev_pair(self):
        return (float(self._prev_action["control"][0]), float(self._prev_action["control"][1]),
                int(self._prev_action["turn"]))

    def _tags(self):
        """이 프레임의 (reason, cap_by) — 채점기 ⑧ 면책 판정이 읽는 두 칸."""
        if self.command_tags is not None:
            reason, cap_by = self.command_tags()
            return str(reason), str(cap_by)
        return RL, world_cap_by(self.state)

    def _rows(self):
        """행 기록기 — 첫 record 에서 연다. 밟지 않은 리셋이 머리글만 든 CSV 를 남기지 않게."""
        if self._recorder is None:
            self._recorder = RowRecorder(self._csv_path())
        return self._recorder

    def _csv_path(self):
        if not self.cfg.log_dir:
            return None
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        return os.path.join(self.cfg.log_dir, f"{self.board.name}-{self._episode:04d}.csv")

    def _close_recorder(self):
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    def _finish(self):
        self._close_recorder()
        sheet = self.referee.sheet
        return {"score": [sheet.score(k) for k in range(sheet.n)],
                "sheet": [dict(s) for s in sheet.state],
                "respawns": dict(self.referee.respawns),
                "notes": list(self.referee.ctx.notes),
                "rows_csv": self._csv_path()}
