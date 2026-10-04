"""학습용 변종 풀(M6f) — 평가 창(v4~v11)을 절대 안 쓰고, 단계를 1:1 로 고른다."""
import os
import re
from collections import Counter

import numpy as np
import pytest

from vtd_rl.rl.variant_pool import VariantSampler, pool_indices
from vtd_rl.rl.vec_env import REPO, _boards

FIVE = ("curricula/stage1.json", "curricula/stage2.json", "curricula/stage3a.json",
        "curricula/stage3b.json", "curricula/stage3.json")
PATHS = [os.path.join(REPO, p) for p in FIVE]
NAME = re.compile(r"^(course_[A-Z])@v(\d+)#(stage3a|stage3b|stage3)$")


def test_풀은_평가_창을_절대_안_낸다():
    assert pool_indices(5) == [0, 1, 2, 3, 12]
    for n in (5, 32, 1000):
        idx = pool_indices(n)
        assert len(idx) == n and len(set(idx)) == n
        assert not set(idx) & set(range(4, 12))
        assert idx[:4] == [0, 1, 2, 3]


@pytest.mark.parametrize("bad", [4, 0, -3, True, 5.5])
def test_풀_크기가_잘못이면_거부한다(bad):
    with pytest.raises(ValueError):
        pool_indices(bad)


def test_고르개가_평가_창_변종을_안_짓는다():
    s = VariantSampler(PATHS, 32, _boards(FIVE))
    rng = np.random.default_rng(0)
    pool = set(pool_indices(32))
    for _ in range(300):
        b, fresh = s(rng)
        m = NAME.match(b.name)
        if m:
            assert fresh and int(m.group(2)) in pool
            assert b.cache_key.startswith("routes/HL_FMA_NEW_") and "|" in b.cache_key
        else:
            assert not fresh and b.name.endswith(("#stage1", "#stage2"))


def test_고르개는_단계를_1대1로_고르고_액터_없는_단계는_원본_객체를_준다():
    static = _boards(FIVE)
    s = VariantSampler(PATHS, 1000, static)
    rng = np.random.default_rng(1)
    stages = Counter()
    ids = {id(b) for b in static}
    for _ in range(3000):
        b, fresh = s(rng)
        st = b.name.rsplit("#", 1)[1]
        stages[st] += 1
        if st in ("stage1", "stage2"):
            assert id(b) in ids
    assert set(stages) == {"stage1", "stage2", "stage3a", "stage3b", "stage3"}
    assert all(500 <= c <= 700 for c in stages.values()), stages


def test_고르개는_같은_난수면_같은_판을_낸다():
    a = VariantSampler(PATHS, 1000, _boards(FIVE))
    b = VariantSampler(PATHS, 1000, _boards(FIVE))
    ra, rb = np.random.default_rng(7), np.random.default_rng(7)
    assert [a(ra)[0].name for _ in range(50)] == [b(rb)[0].name for _ in range(50)]


def test_원본_판_목록이_커리큘럼과_안_맞으면_거부한다():
    with pytest.raises(ValueError):
        VariantSampler(PATHS, 32, _boards(FIVE[:4]))


def test_고르개로_리셋을_거듭해도_캐시가_경로_수를_안_넘는다():
    from vtd_rl.env.board_index import _CACHE, clear_board_index_cache
    from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
    from vtd_rl.referee.core import _CONTEXT_CACHE, clear_context_cache
    clear_board_index_cache()
    clear_context_cache()
    paths = [os.path.join(REPO, "curricula/stage3b.json")]
    static = _boards(("curricula/stage3b.json",))
    env = VtdDriveEnv(static, EnvConfig())
    env.board_sampler = VariantSampler(paths, 1000, static)
    for seed in range(40):
        env.reset(seed=seed)
    # stage3b 의 경로 파일은 6 개 — 변종 이름이 40 번 새로 생겨도 캐시는 경로 수를 안 넘는다
    assert len(_CACHE) <= 6 and len(_CONTEXT_CACHE) <= 6
    assert len(env._worlds) == 0
