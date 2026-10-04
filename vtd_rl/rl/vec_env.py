"""벡터 환경 — 판을 여러 프로세스에서 동시에 굴린다.

판 목록은 단계 ①(항상 초록)과 ②(신호 주기)를 **섞는다**. M3 는 단계 ①만으로 배워 단계 ②가
제로샷이었고, 그래서 학생이 빨간불을 무시했다(M3 성적표). PPO 는 두 단계를 함께 본다.
"""
import os

import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.policy.encode import VEC_KEYS
from vtd_rl.rl.variant_pool import VariantSampler
from vtd_rl.world.board import load_curriculum, load_window

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _path(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _boards(curricula, variants: int = 1):
    """커리큘럼 파일들을 읽어 한 환경이 뽑을 판 목록을 만든다.

    같은 코스가 단계마다 같은 이름으로 다시 나온다 — `curricula/stage1.json` 과
    `stage2.json` 은 둘 다 `course_A`~`course_H` 를 쓰고 신호 방식(`signals`)만 다르다.
    `VtdDriveEnv` 의 세계 캐시(`_worlds`)는 판 **이름**만을 열쇠로 쓰므로, 이름이 같은
    두 판을 그대로 섞으면 한 판을 골랐다가 다른 판을 고르는 순간
    `AssertionError: 세계 캐시가 다른 판을 준다`로 죽는다(2026-09-21, 이 커리큘럼 조합으로
    실측 재현). 판 이름에 커리큘럼 파일 이름을 붙여 물리 코스는 그대로 두고 신호 단계만
    갈라놓는다 — 심판·세계·환경의 판정에는 손대지 않는, 이 안에서 끝나는 수선이다.

    `variants > 1` 이면(M6e) 액터가 있는 커리큘럼은 `load_window(path, variants, 0)` 로 변종
    v0~v{variants-1} 판을 낸다 — 평가 창(`eval_unseen.py`)과 **같은 함수**라 변종 이름·접기
    규칙이 갈라지지 않는다. 액터가 없는 커리큘럼(①②)은 변종이 같은 판이라 `load_window` 가
    원본 한 벌로 접어 주는데, 그대로 두면 학습 판에서 ①② 몫이 줄어든다(30 개 중 12 → 84 개 중
    12). 그래서 **같은 판 객체를 `variants` 번 되풀이**해 단계 사이 비율을 지킨다. 이름 바꾸기는
    되풀이 **전에** 한 번만 한다 — 되풀이한 같은 객체에 접미사가 여러 번 붙으면 안 된다.
    """
    if int(variants) < 1:
        raise ValueError(f"variants 는 1 이상이어야 한다: {variants!r}")
    out = []
    for rel in curricula:
        path = _path(rel)
        stage = os.path.splitext(os.path.basename(path))[0]
        if variants == 1:
            _name, boards = load_curriculum(path)
            varied = True
        else:
            boards, varied = load_window(path, variants, 0)
        for board in boards:
            board.name = f"{board.name}#{stage}"
        out += boards if varied else boards * variants
    return out


def make_env_fn(curricula, config: EnvConfig | None, seed: int, rank: int, variants: int = 1,
                variant_pool: int | None = None):
    def _make():
        # 시드는 make_vec_env 의 reset(seed=[...]) 이 준다 — 여기서 리셋하면 세계를 한 번 더 짓는다.
        boards = _boards(curricula, variants)
        env = VtdDriveEnv(boards, config or EnvConfig())
        if variant_pool is not None:
            # M6f — 리셋마다 변종 판을 새로 짓는다(`variant_pool.py`). 고르개는 워커 프로세스 안에서
            # 만든다(콜러블은 피클할 필요가 없다).
            env.board_sampler = VariantSampler([_path(c) for c in curricula], variant_pool, boards)
        return env
    return _make


def make_vec_env(curricula, n_envs: int, config: EnvConfig | None = None, seed: int = 0,
                 asynchronous: bool = True, variants: int = 1, variant_pool: int | None = None):
    if variant_pool is not None and variants != 1:
        raise ValueError("variants 와 variant_pool 은 같이 못 쓴다 — 미리 지은 변종 목록과 리셋마다"
                         " 짓는 변종 풀은 다른 방식이다")
    fns = [make_env_fn(tuple(curricula), config, seed, i, variants, variant_pool) for i in range(n_envs)]
    venv = AsyncVectorEnv(fns) if asynchronous else SyncVectorEnv(fns)
    venv.reset(seed=[seed + i for i in range(n_envs)])
    return venv


def vec_obs_to_arrays(obs: dict):
    n = len(obs["ego"])
    vec = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(n, -1)
                          for k in VEC_KEYS], axis=1)
    return (vec, np.asarray(obs["objects"], dtype=np.float32),
            np.asarray(obs["object_mask"], dtype=np.float32))
