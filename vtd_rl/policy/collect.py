"""한 판을 몰며 **선생님 정답**을 모은다(스펙 §6.2).

DAgger 의 핵심은 '학생이 간 상태에서의 선생님 답'이다. 그래서 실행 행동은 β 로 섞되,
라벨은 언제나 그 프레임의 선생님 행동이다. 선생님은 env.frame_hook 으로 매 시뮬 프레임 돌아
내부 상태를 이어 간다(스펙 §6.1) — 학생이 몰아도 마찬가지다.

라벨은 **기록하는 관측과 같은 상태**에서 나온 선생님 명령이어야 한다(수정 라운드 1). 한 판단
스텝(`env.step()`)은 여러 시뮬 프레임을 밟는데, `teacher.act()` 가 돌려주는 `self.command` 는
그 스텝의 **마지막** frame_hook 호출 결과 — 즉 관측을 만든 마지막 `world.step()` 보다 한 프레임
전의 상태에서 나온 값이다. 그래서 선생님이 건 진짜 갈고리를 얇게 감싸, **이번 `env.step()` 의
첫 frame_hook 호출**(아직 세계가 한 프레임도 더 안 나간 시점 — 정확히 지금 관측의 상태)에서
나온 명령만 따로 받아 라벨로 쓴다. 몰기(행동 선택)는 그대로 `teacher.act()`(선생님이 몰 때) —
`run_teacher_in_env` 와 같은 값이라 완주 여부가 달라지지 않는다.
"""
import numpy as np

from vtd_rl.env.action import from_command
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.env.teacher_policy import TeacherPolicy
from vtd_rl.policy.dataset import Shard
from vtd_rl.policy.encode import flatten_obs

#: β 를 섞는 단위. "step" 은 걸음마다(예전 그대로), "segment" 는 `mix_len` 걸음마다 한 번,
#: "episode" 는 판마다 한 번 동전을 던진다(M4z).
MIX_MODES = ("step", "segment", "episode")


class DriverSchedule:
    """걸음마다 **누가 모나**(학생이면 True)를 정한다 — 난수는 필요할 때만 뽑는다.

    ★ "step" 은 예전 `rng.random() >= beta` 를 걸음마다 한 번 부르는 것과 **같은 난수 흐름**이다.
    그래야 같은 시드로 지난 실행의 조각을 재현한다. 정책이 없으면(선생님만) 난수를 아예 안
    쓴다 — 예전에도 `policy is not None and ...` 이라 안 썼다.

    "segment" 는 `step % mix_len == 0` 인 걸음에서만 새로 뽑고 그 구간 내내 같은 운전자를 쓴다.
    "episode" 는 첫 걸음에서 한 번 뽑는다. M4w·M4x 가설: 걸음마다 섞으면 학생이 다르게 움직일
    때 선생님 계획기(차로변경 결정 같은 상태)가 자기 계획대로 차가 안 움직이는 걸 겪고, 그
    상태의 라벨이 흔들린다.

    아직 한 번도 안 뽑았으면(`_on is None`) 걸음 번호와 상관없이 먼저 뽑는다 — 0 걸음이 아닌
    곳에서 처음 불려도 '뽑지 않은 채 선생님이 모는' 값을 돌려주지 않게. 0 걸음부터 부르는
    수집에서는 난수 흐름이 달라지지 않는다.
    """

    def __init__(self, beta, mix, mix_len, rng, has_policy):
        if mix not in MIX_MODES:
            raise ValueError(f"모르는 mix {mix!r} — {MIX_MODES} 중 하나")
        if int(mix_len) < 1:
            raise ValueError(f"mix_len 은 1 이상이어야 한다(mix={mix!r}) — {mix_len!r}")
        self.beta, self.mix, self.mix_len = float(beta), mix, int(mix_len)
        self.rng, self.has_policy = rng, has_policy
        self.draws, self._on = 0, None     # _on: 지금 운전자(None=아직 안 뽑음)

    def _draw(self) -> bool:
        self.draws += 1
        return self.rng.random() >= self.beta

    def student(self, step: int) -> bool:
        if not self.has_policy:
            return False
        if self.mix == "step":
            return self._draw()
        if self.mix == "segment":
            if self._on is None or step % self.mix_len == 0:
                self._on = self._draw()
            return self._on
        if self._on is None:
            self._on = self._draw()
        return self._on


def collect_episode(board, policy=None, beta: float = 1.0, seed: int = 0,
                    config: EnvConfig | None = None, max_steps: int = 20000,
                    rng=None, mix: str = "step", mix_len: int = 30) -> Shard:
    rng = rng or np.random.default_rng(seed)
    schedule = DriverSchedule(beta, mix, mix_len, rng, policy is not None)   # 잘못된 값은 여기서
    env = VtdDriveEnv([board], config or EnvConfig())
    teacher = TeacherPolicy(env)
    teacher_hook = env.frame_hook          # 선생님이 건 진짜 갈고리(TeacherPolicy.__init__ 이 배선)
    holder = {"cmd": None}

    def _label_hook(state, clock):
        teacher_hook(state, clock)         # 선생님 계산은 그대로 — 매 프레임, 학생이 몰아도
        if holder["cmd"] is None:          # 이번 env.step() 의 첫 프레임만 잡는다 — 지금 관측의 상태다
            holder["cmd"] = teacher.command

    env.frame_hook = _label_hook

    vecs, objs, masks, controls, turns = [], [], [], [], []
    student_steps, total_reward = 0, 0.0
    try:
        obs, info = env.reset(seed=seed, options={"board": board.name})
        teacher.reset()
        for step in range(max_steps):
            vec, obj, mask = flatten_obs(obs)
            if schedule.student(step):
                action = policy.act(obs, deterministic=True)
                student_steps += 1
            else:
                action = teacher.act()     # 선생님이 몬다 — run_teacher_in_env 와 같은 행동 선택
            holder["cmd"] = None           # 이번 env.step() 의 첫 프레임을 기다린다
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if holder["cmd"] is None:      # 있어선 안 되는 경우 — 실행 행동을 정답으로 속여 쌓지 않는다
                raise RuntimeError(f"[collect] {board.name} seed={seed} step={step}: "
                                    "프레임 훅이 한 번도 안 돎 — 라벨을 만들 수 없다")
            cmd = holder["cmd"]
            label = from_command(cmd.steer, cmd.accel, cmd.turn, env.cfg.action)
            vecs.append(vec)
            objs.append(obj)
            masks.append(mask)
            controls.append(np.asarray(label["control"], dtype=np.float32))
            turns.append(int(label["turn"]))
            if terminated or truncated:
                break
    finally:
        teacher.detach()
        env.frame_hook = None    # detach() 는 self._on_frame 과만 같음을 비교한다 — 여기서 감쌌으니 직접 뗀다
        env.close()
    meta = {"board": board.name, "seed": int(seed), "beta": float(beta),
            "outcome": info["outcome"], "steps": len(turns),
            "reward": float(total_reward), "student_steps": int(student_steps)}
    if mix != "step":
        meta.update({"mix": mix, "mix_len": int(mix_len)})
    return Shard(np.stack(vecs), np.stack(objs), np.stack(masks),
                 np.stack(controls), np.asarray(turns, dtype=np.int64), meta)
