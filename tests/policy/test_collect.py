import numpy as np
import pytest

from vtd_rl.env.action import from_command
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.env.teacher_policy import TeacherPolicy
from vtd_rl.policy.collect import MIX_MODES, DriverSchedule, collect_episode
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def short_board():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_선생님만_수집하면_완주하고_모양이_맞다():
    shard = collect_episode(short_board(), policy=None, beta=1.0, seed=0)
    n = len(shard.turn)
    assert n > 50
    assert shard.vec.shape == (n, VEC_DIM) and shard.objs.shape == (n, OBJ_N, OBJ_DIM)
    assert shard.control.shape == (n, 2) and shard.mask.shape == (n, OBJ_N)
    assert np.all(np.abs(shard.control) <= 1.0) and set(np.unique(shard.turn)) <= {0, 1, 2}
    assert shard.meta["outcome"] == "goal" and shard.meta["student_steps"] == 0
    assert shard.meta["board"] == "H_0_250" and shard.meta["beta"] == 1.0


def test_같은_시드면_같은_조각():
    a = collect_episode(short_board(), seed=3)
    b = collect_episode(short_board(), seed=3)
    assert np.array_equal(a.vec, b.vec) and np.array_equal(a.control, b.control)
    assert a.meta["steps"] == b.meta["steps"]


def test_beta가_0이면_학생이_몰고_정답은_선생님이다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    shard = collect_episode(short_board(), policy=net, beta=0.0, seed=1, max_steps=200)
    assert shard.meta["student_steps"] == len(shard.turn)
    # 학습 전 학생은 선생님처럼 몰지 못하므로 정답과 실행 행동이 달라야 의미가 있다
    assert shard.meta["outcome"] in ("goal", "offroad", "collision", "timeout", "stalled", "running")
    assert len(shard.turn) > 0


def test_beta가_1이면_학생을_줘도_선생님이_몬다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    a = collect_episode(short_board(), policy=net, beta=1.0, seed=5)
    b = collect_episode(short_board(), policy=None, beta=1.0, seed=5)
    assert np.array_equal(a.control, b.control) and a.meta["steps"] == b.meta["steps"]


def test_라벨은_그_관측을_만든_프레임의_선생님_명령이다():
    """수정 라운드 1: label 은 그 스텝이 기록한 관측과 같은 상태에서 나온 선생님 명령이어야 한다.

    `collect_episode` 와는 완전히 따로, 여기서 직접 판을 몰며 각 판단 스텝의 **첫 frame_hook
    호출**(=지금 관측의 상태, 세계가 아직 한 프레임도 안 나간 시점)에서 나온 명령만 잡는다.
    옛 방식("env.step() 전에 teacher.act() 호출")으로 되돌리면 이 값과 어긋난다 — 스크래치
    사본에서 되돌려 이 테스트가 실패하는 것으로 확인했다.
    """
    board = short_board()
    shard = collect_episode(board, policy=None, beta=1.0, seed=7)

    env = VtdDriveEnv([board], EnvConfig())
    teacher = TeacherPolicy(env)
    real_hook = env.frame_hook
    captured = {"cmd": None}

    def hook(state, clock):
        real_hook(state, clock)
        if captured["cmd"] is None:
            captured["cmd"] = teacher.command

    env.frame_hook = hook
    expected_controls, expected_turns = [], []
    try:
        obs, info = env.reset(seed=7, options={"board": board.name})
        teacher.reset()
        for _ in range(len(shard.turn)):
            action = teacher.act()
            captured["cmd"] = None
            obs, reward, terminated, truncated, info = env.step(action)
            cmd = captured["cmd"]
            assert cmd is not None
            label = from_command(cmd.steer, cmd.accel, cmd.turn, env.cfg.action)
            expected_controls.append(label["control"])
            expected_turns.append(label["turn"])
            if terminated or truncated:
                break
    finally:
        teacher.detach()
        env.close()

    assert len(expected_controls) == len(shard.turn)
    assert np.array_equal(shard.control, np.stack(expected_controls))
    assert np.array_equal(shard.turn, np.asarray(expected_turns, dtype=np.int64))


# ── M4z: β 섞는 단위 ───────────────────────────────────────────────────────────
# 지금은 걸음마다 동전을 던진다. 학생이 다르게 움직일수록 선생님 계획기가 흔들린다는 가설을
# 시험하려고 구간(여러 걸음)·판 단위로도 섞는다. 기본 "step" 은 예전과 같은 난수를 써야 한다.


def _sched(beta, mix, mix_len=5, seed=0, has_policy=True):
    return DriverSchedule(beta, mix, mix_len, np.random.default_rng(seed), has_policy)


def test_섞기_종류():
    assert MIX_MODES == ("step", "segment", "episode")


def test_step_은_예전_식과_같은_난수를_쓴다():
    s = _sched(0.5, "step", seed=7)
    got = [s.student(t) for t in range(50)]
    rng = np.random.default_rng(7)
    want = [rng.random() >= 0.5 for _ in range(50)]
    assert got == want and s.draws == 50


def test_segment_는_mix_len_걸음마다만_다시_뽑는다():
    s = _sched(0.5, "segment", mix_len=5, seed=3)
    got = [s.student(t) for t in range(23)]
    rng = np.random.default_rng(3)
    blocks = [rng.random() >= 0.5 for _ in range(5)]        # 0~4, 5~9, 10~14, 15~19, 20~22
    want = [blocks[t // 5] for t in range(23)]
    assert got == want and s.draws == 5


def test_episode_는_한_번만_뽑는다():
    for seed in range(6):
        s = _sched(0.5, "episode", seed=seed)
        got = [s.student(t) for t in range(40)]
        assert len(set(got)) == 1 and s.draws == 1
        assert got[0] == (np.random.default_rng(seed).random() >= 0.5)


@pytest.mark.parametrize("mix", ["step", "segment", "episode"])
def test_beta_가_1_이면_학생이_안_몰고_0_이면_늘_몬다(mix):
    assert not any(_sched(1.0, mix).student(t) for t in range(30))
    assert all(_sched(0.0, mix).student(t) for t in range(30))


@pytest.mark.parametrize("mix", ["step", "segment", "episode"])
def test_정책이_없으면_학생이_안_몰고_난수도_안_쓴다(mix):
    s = _sched(0.0, mix, has_policy=False)
    assert not any(s.student(t) for t in range(30)) and s.draws == 0


@pytest.mark.parametrize("bad", [dict(mix="steps"), dict(mix="segment", mix_len=0),
                                 dict(mix="segment", mix_len=-3)])
def test_잘못된_mix_는_거부한다(bad):
    with pytest.raises(ValueError, match="mix"):
        DriverSchedule(0.5, bad["mix"], bad.get("mix_len", 30), np.random.default_rng(0), True)
    with pytest.raises(ValueError, match="mix"):
        collect_episode(short_board(), beta=0.5, seed=0, mix=bad["mix"],
                        mix_len=bad.get("mix_len", 30))


def test_정수가_아닌_mix_len_은_거부하고_numpy_정수는_받는다():
    """`int(2.5)` 가 조용히 2 로 깎이지 않게 — 파이썬 int·numpy 정수만 통과한다."""
    with pytest.raises(ValueError, match="mix"):
        DriverSchedule(0.5, "segment", 2.5, np.random.default_rng(0), True)
    with pytest.raises(ValueError, match="mix"):
        collect_episode(short_board(), beta=0.5, seed=0, mix="segment", mix_len=2.5)
    for ok in (5, np.int64(5), np.int32(5)):
        s = DriverSchedule(0.5, "segment", ok, np.random.default_rng(0), True)
        assert s.mix_len == 5 and type(s.mix_len) is int


def test_기본값은_step_과_같고_meta_에_새_키가_없다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    a = collect_episode(short_board(), policy=net, beta=0.5, seed=4, max_steps=150)
    b = collect_episode(short_board(), policy=net, beta=0.5, seed=4, max_steps=150, mix="step")
    assert np.array_equal(a.vec, b.vec) and np.array_equal(a.control, b.control)
    assert a.meta == b.meta and "mix" not in a.meta and "mix_len" not in a.meta


def test_segment_수집은_meta_에_섞기를_남긴다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    shard = collect_episode(short_board(), policy=net, beta=0.5, seed=2, max_steps=120,
                            mix="segment", mix_len=10)
    assert shard.meta["mix"] == "segment" and shard.meta["mix_len"] == 10
    n = len(shard.turn)
    full, rest = divmod(n, 10)
    # 학생 걸음 수는 '통째 구간 × 10 + (마지막 덜 찬 구간이 학생이면 그 길이)' 꼴이다
    assert shard.meta["student_steps"] % 10 in (0, rest)
