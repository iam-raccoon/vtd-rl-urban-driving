import numpy as np
import pytest

from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM, flatten_obs
from vtd_rl.rl.vec_env import _boards, make_vec_env, vec_obs_to_arrays

STAGES = ("curricula/stage1.json", "curricula/stage2.json")


def test_동기_벡터_환경_한_걸음():
    venv = make_vec_env(STAGES, n_envs=2, seed=0, asynchronous=False)
    try:
        obs, info = venv.reset(seed=0)
        vec, objs, mask = vec_obs_to_arrays(obs)
        assert vec.shape == (2, VEC_DIM) and objs.shape == (2, OBJ_N, OBJ_DIM)
        assert mask.shape == (2, OBJ_N) and vec.dtype == np.float32
        action = venv.action_space.sample()
        obs, reward, term, trunc, info = venv.step(action)
        assert reward.shape == (2,) and term.shape == (2,)
    finally:
        venv.close()


def test_평탄화_순서가_단일_환경과_같다():
    venv = make_vec_env(STAGES, n_envs=1, seed=1, asynchronous=False)
    try:
        obs, _info = venv.reset(seed=1)
        vec, objs, mask = vec_obs_to_arrays(obs)
        single = {k: np.asarray(v[0]) for k, v in obs.items()}
        s_vec, s_objs, s_mask = flatten_obs(single)
        assert np.array_equal(vec[0], s_vec) and np.array_equal(objs[0], s_objs)
        assert np.array_equal(mask[0], s_mask)
    finally:
        venv.close()


def test_두_단계_판이_모두_등장한다():
    venv = make_vec_env(STAGES, n_envs=4, seed=3, asynchronous=False)
    try:
        signals = set()
        for _ in range(6):
            venv.reset()
            signals |= {e.unwrapped.board.signals for e in venv.envs}
        assert signals == {"always_green", "cycle"}
    finally:
        venv.close()


@pytest.mark.slow
def test_비동기_환경도_돈다():
    venv = make_vec_env(STAGES, n_envs=2, seed=0, asynchronous=True)
    try:
        venv.reset(seed=0)
        for _ in range(5):
            venv.step(venv.action_space.sample())
    finally:
        venv.close()


@pytest.mark.slow
def test_intent가_AsyncVectorEnv_경계를_넘는다():
    """M4c — 실학습은 `AsyncVectorEnv`(별도 프로세스)를 쓴다. `set_attr("intent", ...)` 로

    내려보낸 값이 프로세스 경계를 실제로 넘어 워커의 `VtdDriveEnv.intent` 에 닿고,
    `RewardShaper` 가 그 값으로 승차감을 계산하는지 확인한다 — `--smoke` 는 항상
    `SyncVectorEnv`(같은 프로세스)라 이 경로를 안 밟는다. `env.intent` 를 콜러블이 아니라
    값(튜플)으로 둔 이유가 바로 이것이다: 람다는 피클이 안 되지만 튜플은 된다.

    큰 행동([1.0, 1.0])을 실행해도 의도가 내내 (0.0, 0.0)으로 고정돼 있으면(리셋 직후 첫
    걸음, `_prev_intent` 콜드스타트는 "의도=자기 자신"으로 diff 0) 승차감이 0 이어야 한다 —
    0 이 아니면 `intent` 가 워커까지 안 닿아 실행 행동으로 대체 계산된 것이다.
    """
    from vtd_rl.env.drive_env import EnvConfig
    from vtd_rl.env.reward import RewardConfig

    cfg = EnvConfig(reward=RewardConfig(comfort_on_intent=True))
    venv = make_vec_env(STAGES, n_envs=2, config=cfg, seed=0, asynchronous=True)
    try:
        venv.reset(seed=0)
        venv.set_attr("intent", [(0.0, 0.0), (0.0, 0.0)])
        action = {"control": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
                  "turn": np.array([0, 0])}
        _obs, _reward, _term, _trunc, info = venv.step(action)
        comfort = np.asarray(info["reward_terms"]["comfort"], dtype=np.float64)
        assert np.allclose(comfort, 0.0, atol=1e-6), comfort
    finally:
        venv.close()


FIVE = ("curricula/stage1.json", "curricula/stage2.json", "curricula/stage3a.json",
        "curricula/stage3b.json", "curricula/stage3.json")


def _stage_of(name):
    return name.rsplit("#", 1)[1]


def test_변종_기본값은_예전_판_목록과_같다():
    a = _boards(FIVE)
    b = _boards(FIVE, variants=1)
    assert [x.name for x in a] == [x.name for x in b]
    assert len(a) == 30 and len({id(x) for x in a}) == 30
    assert all(x.name.count("#") == 1 and "@v" not in x.name for x in a)


def test_변종을_걸어도_단계마다_판_수가_같다():
    bs = _boards(FIVE, variants=4)
    from collections import Counter
    per_stage = Counter(_stage_of(b.name) for b in bs)
    assert per_stage == {"stage1": 24, "stage2": 24, "stage3a": 24, "stage3b": 24, "stage3": 24}
    # 액터 단계는 변종 v0~v3 이 다른 판이고, 액터 없는 단계는 같은 판 6 개의 되풀이다
    for st in ("stage3a", "stage3b", "stage3"):
        names = [b.name for b in bs if _stage_of(b.name) == st]
        assert len(set(names)) == 24
        assert {n.split("@v")[1].split("#")[0] for n in names} == {"0", "1", "2", "3"}
    for st in ("stage1", "stage2"):
        objs = [b for b in bs if _stage_of(b.name) == st]
        assert len({id(b) for b in objs}) == 6


def test_되풀이한_판도_접미사는_한_번만():
    bs = _boards(FIVE, variants=3)
    assert all(b.name.count("#") == 1 for b in bs), sorted({b.name for b in bs})


def test_변종_수가_1보다_작으면_거부한다():
    with pytest.raises(ValueError, match="variants"):
        _boards(FIVE, variants=0)


def test_변종_판으로도_벡터_환경이_돈다():
    venv = make_vec_env(("curricula/stage1.json", "curricula/stage3b.json"), n_envs=2, seed=0,
                        asynchronous=False, variants=2)
    try:
        obs, _info = venv.reset(seed=0)
        obs, reward, term, trunc, info = venv.step(venv.action_space.sample())
        assert reward.shape == (2,)
    finally:
        venv.close()


def test_변종_풀로도_벡터_환경이_돈다():
    venv = make_vec_env(("curricula/stage1.json", "curricula/stage3b.json"), n_envs=2, seed=0,
                        asynchronous=False, variant_pool=8)
    try:
        assert all(e.board_sampler is not None for e in venv.envs)
        obs, _info = venv.reset(seed=0)
        for _ in range(3):
            obs, reward, term, trunc, info = venv.step(venv.action_space.sample())
        assert reward.shape == (2,)
    finally:
        venv.close()


def test_변종_수와_변종_풀은_같이_못_쓴다():
    with pytest.raises(ValueError, match="variant_pool"):
        make_vec_env(("curricula/stage1.json",), n_envs=1, seed=0, asynchronous=False,
                     variants=2, variant_pool=8)
