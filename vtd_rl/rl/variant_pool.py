"""학습용 변종 판 고르기(M6f) — 리셋마다 변종 판을 새로 짓는다.

M6e 는 변종 판 v0~v3 을 미리 다 지어 목록에 넣었다. 판 하나가 3.3 MB 라 더 못 늘린다. 여기서는
리셋마다 `load_board(entry, signals, variant=k)` 로 판을 하나 지어(약 8 ms) 쓰고 버린다.

★ **평가 창 v4~v11 은 절대 안 쓴다**(보고 창 v4~v7 · 선택 창 v8~v11, `eval_unseen.py` 와
`run_dagger.py --select-variants`). 학습 변종 번호 풀은 수집 창 v0~v3 과 v12 부터다 —
`pool_indices` 한 곳이 정한다.

단계는 1:1 로 고른다(학습 판 목록에서 단계마다 판이 6 개씩이던 M6d·M6e 와 같은 비율).
액터가 없는 단계(①②)는 변종이 같은 판이라 원본 판 객체를 그대로 준다 — 세계도 캐시된다.
새로 지은 판에는 `cache_key`(경로 파일)를 달아 색인·심판 입력 캐시를 경로마다 한 칸으로 묶는다
(`vtd_rl.world.board.cache_key`).
"""
import json
import os

from vtd_rl.world.board import load_board

EVAL_FIRST, EVAL_LAST = 4, 11      # 평가 창 v4~v11 — 학습에 안 쓴다


def pool_indices(n) -> list:
    """학습 변종 번호 n 개 — v0~v3, 그다음 v12, v13, … (평가 창 v4~v11 을 건너뛴다)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 5:
        raise ValueError(f"변종 풀 크기는 5 이상의 정수여야 한다(4 이하는 --train-variants 를 써라): {n!r}")
    return list(range(EVAL_FIRST)) + list(range(EVAL_LAST + 1, EVAL_LAST + 1 + n - EVAL_FIRST))


class VariantSampler:
    """`VtdDriveEnv.board_sampler` 로 거는 콜러블 — `rng -> (판, 새로_지은_판인가)`.

    `static_boards` 는 환경이 들고 있는 원본 판 목록(`vec_env._boards(curricula)`, 이름에
    `#단계` 가 붙은 것)이다. 액터 없는 단계는 이 객체를 그대로 준다. 커리큘럼 순서·판 순서가
    목록과 안 맞으면 거부한다 — 엉뚱한 판을 원본으로 주면 세계 캐시가 남의 세계를 준다.
    """

    def __init__(self, curricula_paths, pool, static_boards):
        self.indices = pool_indices(pool)
        self.stages = []
        pos = 0
        for path in curricula_paths:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            label = os.path.splitext(os.path.basename(path))[0]
            entries = d["boards"]
            static = list(static_boards[pos:pos + len(entries)])
            pos += len(entries)
            want = [f"{e['name']}#{label}" for e in entries]
            if [b.name for b in static] != want:
                raise ValueError(f"원본 판 목록이 커리큘럼 `{label}` 과 안 맞는다: "
                                 f"{[b.name for b in static]} != {want}")
            has_actors = any(e.get("actors") for e in entries)
            self.stages.append((label, d["signals"], entries, static, has_actors))
        if pos != len(static_boards):
            raise ValueError(f"원본 판 목록이 커리큘럼보다 길다: {len(static_boards)} != {pos}")

    def __call__(self, rng):
        label, signals, entries, static, has_actors = self.stages[int(rng.integers(len(self.stages)))]
        j = int(rng.integers(len(entries)))
        if not has_actors:
            return static[j], False
        k = self.indices[int(rng.integers(len(self.indices)))]
        entry = entries[j]
        board = load_board(entry, signals, variant=k)
        board.name = f"{board.name}#{label}"
        board.cache_key = f"{entry['route']}|{entry['lane']}"
        return board, True
