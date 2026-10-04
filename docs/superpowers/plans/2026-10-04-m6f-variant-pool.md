# M6f — 리셋마다 변종 판을 지어 학습: 변종을 4 개에서 수십·수백 개로 늘리면 더 오르나 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6e 에서 학습 판을 수집 창 변종 v0~v3 으로 넓히자 ③ 전체가 68.5 → 76.4%(선택 창)로 올랐다. 판 목록을 미리 다 짓는 방식은 판 객체가 무거워(판 하나 3.3 MB) 더 못 늘린다. M6f 는 **리셋할 때마다 변종 판을 새로 지어** 평가 창(v4~v11)을 뺀 큰 풀에서 학습하고, 변종 수를 늘리면 더 오르는지 잰다.

**Architecture:**
- `VtdDriveEnv` 에 `board_sampler` 갈고리를 둔다. `frame_hook`·`command_tags` 와 같은 평범한 속성이다.
  - 기본 `None` 이면 예전과 같다.
  - 걸려 있으면 이름을 안 준 리셋이 `board_sampler(np_random) -> (판, 새로_지은_판인가)` 로 판을 고른다.
  - 새로 지은 판의 세계는 캐시하지 않는다.
- 색인(`env/board_index.py`)과 심판 입력(`referee/core.py::context_for`) 캐시는 판 **이름**을 열쇠로 쓴다. 변종 이름이 끝없이 새로 생기면 캐시가 끝없이 자란다.
  - 두 캐시는 **경로만** 본다(액터를 안 본다). 그래서 판에 선택적 `cache_key`(경로 파일)를 두고 그것을 열쇠로 쓴다.
  - `cache_key` 가 없으면 예전처럼 이름이다.
- 새 모듈 `vtd_rl/rl/variant_pool.py`:
  - `pool_indices(n)` 은 v0~v3 과 v12 부터를 합쳐 n 개를 낸다. 평가 창 v4~v11 은 절대 안 낸다.
  - `VariantSampler` 는 단계를 1:1 로 고르고, 액터 단계면 풀에서 변종 번호를 뽑아 `load_board` 로 판을 짓는다. 액터 없는 단계면 원본 판 객체를 그대로 준다.
- `make_vec_env(..., variant_pool=N)` 과 `train_ppo.py --train-variant-pool N` 을 연다.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거는 `docs/reports/m6e-ppo-variants.md`("다음" 1).

## 사전 측정(끝남)

| 무엇 | 비용 |
|---|---|
| 판 하나 짓기(`load_board`, 변종 포함) | 약 8 ms, 3.3 MB |
| `board_index(판)` — 경로 위 신호·횡단보도·교차로 색인 | 410~490 ms(코스마다) |
| `World(...)` 짓기 + `reset` | 0 ms 대 |

- 색인을 코스(경로)마다 한 번만 지으면, 리셋마다 판을 새로 짓는 비용은 약 10 ms 다. 판 하나는 수천 걸음(10 초 이상)이라 무시할 만하다.
- 색인과 심판 입력은 경로·차로계획·신호 지도만 본다. 근거: `BoardIndex.__init__`, `oracle.match_inputs`.
  - 그래서 같은 경로 파일의 변종끼리 같이 써도 값이 같다.
  - `BoardIndex.plan_at` 은 `self.board.lane_plan` 을 쓰는데, 같은 경로 파일이면 같은 차로계획이다.

## 본 실행 (OMEN)

칸 `ARM ∈ {P32: 32, P1000: 1000}` × 시드 `S ∈ {0, 1, 2}` = 6 실행, 동시에:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-04-m6f-P$POOL-s$S --init runs/omen/2026-10-04-m6c-A-s$S/policy-best.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variant-pool $POOL \
  --steps 2000000 --eval-every 500000 --eval-seeds 1 --final-eval-seeds 3 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20
```

- M6e 칸 V 와 **학습 판 고르기만** 다르다. 출발·참조(M6c A 고른 그물), 계수, 걸음, 학습률이 같다.
  - P32 의 풀은 v0~v3 + v12~v39 다.
  - P1000 은 v0~v3 + v12~v1007 이다. 판 하나 하나가 거의 새 배치다.
- 단계 비율은 M6d·M6e 와 같다(5 단계 각 1/5, 단계 안에서 코스 1/6).
- 평가는 M6e 와 같다. 학습 풀은 v4~v11 을 안 쓰므로 두 평가 창과 겹치지 않는다.
- 먼저 OMEN 에서 짧게 돌려 메모리·속도를 본다. 판 목록이 원본 30 개뿐이라 실행 하나가 약 3 GB(M6d 와 같은 크기)일 것으로 본다.

## 판정 기준 (미리 적는다)

- 출발점은 M6c 칸 A 고른 그물이다(M6d·M6e 와 같다).
  - 선택 창 ③ 전체 62.5 · 70.8 · 72.2(평균 68.5), 보고 창 69.4 · 68.1 · 56.9(평균 64.8).
  - 선택 창 ③a 평균 98.6, ③b 평균 98.1.
- 비교 대상은 M6e 칸 V(변종 4)다. 선택 창 ③ 전체 80.6 · 86.1 · 62.5(평균 76.4), 보고 창 73.6 · 80.6 · 61.1(평균 71.8).

시드마다 `ac-best` 와 마지막 중 **선택 창 ③ 전체가 높은 쪽**을 고른다.

- **칸마다 오른다 / 내린다 / 차이 없다** — 출발점 기준, M6d·M6e 와 같은 문턱이다.
  - 오른다: 선택 창 ③ 전체 평균 ≥ 75.5%, 2/3 시드 이상이 출발점보다 높음, 선택 창 ③a ≥ 93.6 · ③b ≥ 93.1, ①·② ≥ 90%.
  - 내린다: 선택 창 ③ 전체 평균 ≤ 63.5% 이거나, ③a 또는 ③b 평균이 출발점 −10 미만이거나, ① 또는 ② < 90%.
- **변종 4 개(M6e)와 견주기** — 두 칸 중 선택 창 ③ 전체 평균이 높은 칸을 본다.
  - "변종을 늘리면 4 개보다 낫다": 선택 창 평균 ≥ 81.4%(M6e +5) **그리고** 보고 창 평균 ≥ 76.8%(M6e +5).
  - "변종을 늘리면 4 개보다 못하다": 선택 창 평균 ≤ 71.4%(M6e −5).
  - 아니면 "4 개와 비슷하다".
- 곁들여(판정과 별도): 시드 사이 폭, 마지막 체크포인트 숫자, ①② 점수, 고른 그물의 ③ 전체 실패 갈래(진단 스크립트).

## 실행 전 보강 (최종 리뷰 반영, 결과를 보기 전에 적었다)

- **P1000 은 평가 배치를 s 방향으로 거의 덮는다.**
  - 변종의 공통 s 오프셋은 황금비 저불일치 수열이다(`world/place.py::s_offset`). 변종 번호가 많아지면 범위(±40 m)를 촘촘히 채운다.
  - 리뷰 실측: P1000 의 학습 s 오프셋은 모든 코스에서 평가 창 v4~v11 의 오프셋과 0.03 m 안이다. P32 는 1.5~2.4 m 간격이다.
  - 횡 위치는 `(이름, 번호)` 마다 따로 흔들어 같은 배치는 아니다. 그래도 액터 하나 단계(③a·③b)에서는 P1000 의 "안 쓴 창" 이 사실상 분포 안이다.
  - 그래서 성적표는 P1000 이 4 개보다 나아도 "일반화" 가 아니라 **"배치를 더 촘촘히 덮은 효과"** 로 적는다. 장애물 여럿 단계(③ 전체)는 오프셋 하나에 횡 위치 4~5 개가 겹쳐 새 조합이 되므로, 판정은 ③ 전체로 한다(위 기준 그대로).
- M6e 학습 판 수를 바로잡는다. ①② 를 4 번 되풀이하므로 목록은 120 칸(단계마다 24)이고, 서로 다른 판은 84 개다. 단계 비율 1/5 은 맞다.

## Global Constraints

- numpy 는 **1.26.4** 고정 — 새 의존성을 넣지 않는다.
- `third_party/rule_stack` 은 **읽기 전용**이다.
- 커밋 금지: `runs/`, `*.pt`, `*.npz`, CSV, `*.xodr`, `*.xml`.
- 주석·독스트링·커밋 메시지·성적표는 **한국어**.
- 커밋 메시지에 **Claude 표기를 넣지 않는다**(`Co-Authored-By`, `Claude-Session`, "Generated with", 🤖, anthropic 주소 전부). 커밋 뒤 `git show -s --format=%B HEAD` 로 확인한다.
- 테스트는 `env -u PYTHONPATH .venv/bin/pytest ...` 로 돌리고 **종료 코드(rc)** 로 판정한다. "N passed" 를 grep 하지 않는다. `| tail` 로 넘기지 않고 파일로 돌린다. 한 호출이 600 초를 넘기면 파일 단위로 나눠 포그라운드로 돌린다(백그라운드 금지).
- **기본값은 지금과 같다.**
  - `board_sampler=None` 이면 리셋이 지금과 같은 판을 같은 난수 소비로 고른다.
  - `cache_key` 가 없는 판은 두 캐시의 열쇠·돌려주는 객체가 지금과 같다.
  - `variant_pool` 을 안 주면 `make_vec_env`·`train_ppo.py` 가 지금과 같다.
- **평가 창 v4~v11 은 학습 변종에 절대 안 들어간다**(`pool_indices` 한 곳이 정한다).
- 변종 판 짓기는 새로 짜지 않는다 — `vtd_rl.world.board.load_board(entry, signals, variant=k)` 를 쓴다(평가 창과 같은 배치 규칙).

## Review Focus

1. **평가 창 변종이 학습에 새는 것**(v4~v11). 한 번이라도 나오면 "안 쓴 판" 성적이 무너진다. → Task 2 `test_풀은_평가_창을_절대_안_낸다`, `test_고르개가_평가_창_변종을_안_짓는다`.
2. **캐시가 끝없이 자라는 것**(변종 이름마다 색인·심판 입력 한 칸). 200 만 걸음이면 환경마다 수백 칸 × 수 MB 다. → Task 1 `test_cache_key_가_같으면_색인을_같이_쓴다`, Task 2 `test_고르개로_리셋을_거듭해도_캐시가_경로_수를_안_넘는다`.
3. **열쇠를 같이 쓰는 변종이 남의 판을 보는 것**(`Context.board` 가 처음 판을 가리킴). → Task 1 `test_cache_key_를_같이_쓰는_판은_심판_입력을_같이_쓰되_판은_자기_것이다`.
4. **새로 지은 판의 세계가 이름으로 캐시되어 다음 리셋의 다른 판과 엉키는 것.** 세계 캐시의 단언(`world.board is board`)이 터진다. → Task 1 `test_판_고르기_갈고리가_새로_지은_판을_쓰고_세계를_캐시하지_않는다`.
5. **기본값에서 판 고르기 난수 소비가 달라지는 것.** 그러면 M6a~M6e 설정의 재현이 깨진다. → Task 1 `test_판_고르기_갈고리가_없으면_예전처럼_고른다`.

---

### Task 1: 캐시 열쇠(`cache_key`)와 환경의 판 고르기 갈고리

**Files:**
- Modify: `vtd_rl/world/board.py`(함수 `cache_key` 추가)
- Modify: `vtd_rl/env/board_index.py`
- Modify: `vtd_rl/referee/core.py`
- Modify: `vtd_rl/env/drive_env.py`
- Test: `tests/env/test_board_index.py`, `tests/env/test_prep.py`, `tests/env/test_drive_env.py`

**Interfaces:**
- Produces:
  - `vtd_rl.world.board.cache_key(board) -> str` — `board.cache_key` 가 있으면(참이면) 그것, 없으면 `board.name`.
  - `VtdDriveEnv.board_sampler`(기본 `None`) — 콜러블 `rng -> (Board, fresh: bool)`.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_board_index.py` 끝에(파일 위 임포트에 `from vtd_rl.world.board import load_board, slice_board` 가 없으면 더한다):

```python
H_ENTRY = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def test_cache_key_가_같으면_색인을_같이_쓴다():
    a = slice_board(load_board(H_ENTRY), 0.0, 250.0, "H_ck_a")
    b = slice_board(load_board(H_ENTRY), 0.0, 250.0, "H_ck_b")
    clear_board_index_cache()
    assert board_index(a) is not board_index(b)          # 열쇠가 없으면 예전처럼 이름마다 따로
    a.cache_key = b.cache_key = "H|0|250"
    clear_board_index_cache()
    assert board_index(a) is board_index(b)
    clear_board_index_cache()
```

`tests/env/test_prep.py` 끝에:

```python
def test_cache_key_를_같이_쓰는_판은_심판_입력을_같이_쓰되_판은_자기_것이다():
    clear_context_cache()
    a = slice_board(load_board(H), 0.0, 250.0, "H_ctx_a")
    b = slice_board(load_board(H), 0.0, 250.0, "H_ctx_b")
    assert context_for(a).board is a and context_for(b).board is b
    assert context_for(a).secs is not context_for(b).secs   # 열쇠가 없으면 이름마다 따로
    clear_context_cache()
    a.cache_key = b.cache_key = "H|0|250"
    ca, cb = context_for(a), context_for(b)
    assert ca.secs is cb.secs and ca.tl_stops is cb.tl_stops   # 비싼 입력은 같이 쓴다
    assert ca.board is a and cb.board is b                     # 판은 자기 것
    clear_context_cache()
```

`tests/env/test_drive_env.py` 끝에:

```python
def test_판_고르기_갈고리가_없으면_예전처럼_고른다():
    e1, e2 = VtdDriveEnv(boards()), VtdDriveEnv(boards())
    e2.board_sampler = None
    for seed in (0, 3, 7):
        o1, _ = e1.reset(seed=seed)
        o2, _ = e2.reset(seed=seed)
        assert e1.board.name == e2.board.name
        for k in o1:
            assert np.array_equal(np.asarray(o1[k]), np.asarray(o2[k])), k


def test_판_고르기_갈고리가_새로_지은_판을_쓰고_세계를_캐시하지_않는다():
    env = VtdDriveEnv(boards())
    made = []

    def sampler(rng):
        b = slice_board(load_board(H), 0.0, 250.0, f"H_fresh_{len(made)}")
        made.append(b)
        return b, True

    env.board_sampler = sampler
    for seed in range(3):
        env.reset(seed=seed)
        assert env.board is made[-1] and env.world.board is made[-1]
    assert len(env._worlds) == 0
    env.reset(seed=0, options={"board": "H_0_250"})     # 이름을 주면 예전처럼 목록에서, 캐시도 한다
    assert env.board.name == "H_0_250" and len(env._worlds) == 1


def test_판_고르기_갈고리가_원본을_주면_세계를_캐시한다():
    bs = boards()
    env = VtdDriveEnv(bs)
    env.board_sampler = lambda rng: (bs[1], False)
    env.reset(seed=0)
    env.reset(seed=1)
    assert env.board is bs[1] and list(env._worlds) == ["H_250_500"]
```

- [ ] **Step 2: 돌려서 실패를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/env/test_board_index.py tests/env/test_prep.py tests/env/test_drive_env.py -q -k "cache_key or 갈고리" > $TMP/m6f_t1_red.txt 2>&1; echo rc=$?`
Expected: rc≠0 — 열쇠를 같이 줘도 색인·입력이 따로 만들어지고, `board_sampler` 를 걸어도 무시된다.

- [ ] **Step 3: 구현한다.**

`vtd_rl/world/board.py`(`window_label` 아래 등 끝 쪽):

```python
def cache_key(board) -> str:
    """색인·심판 입력 캐시의 열쇠 — 판에 `cache_key` 가 있으면 그것, 없으면 판 이름(예전과 같다).

    M6f: 리셋마다 새로 짓는 변종 판은 이름(`course_A@v123#stage3`)이 끝없이 새로 생긴다. 색인
    (`env/board_index.py`)과 심판 입력(`referee/core.py::context_for`)은 **경로만** 본다 —
    경로·차로계획·신호 지도이고 액터는 안 본다. 그래서 같은 경로 파일의 변종은 `cache_key` 에
    경로를 적어 캐시 한 칸을 같이 쓴다. 이름을 열쇠로 두면 두 캐시가 끝없이 자란다.
    """
    return getattr(board, "cache_key", None) or board.name
```

`vtd_rl/env/board_index.py`: 임포트에 `from vtd_rl.world.board import cache_key` 를 더하고, `board_index` 의 열쇠를 바꾼다.

```python
def board_index(board) -> BoardIndex:
    # 열쇠는 `cache_key(board)`(없으면 판 이름 — 예전과 같다). 같은 경로의 변종끼리 색인을 같이
    # 쓴다(M6f) — 색인은 경로·차로계획만 보고, `plan_at` 이 읽는 차로계획도 같은 경로면 같다.
    key = (cache_key(board), len(board.route.pts))
    idx = _CACHE.get(key)
    if idx is None:
        idx = _CACHE[key] = BoardIndex(board)
    return idx
```

`vtd_rl/referee/core.py`: 임포트에 `from vtd_rl.world.board import cache_key` 를 더하고, `context_for` 를 바꾼다.

```python
def context_for(board, sections=5) -> Context:
    """(기존 독스트링 그대로)

    열쇠는 `cache_key(board)` 다(M6f, 없으면 판 이름 — 예전과 같다). 열쇠를 같이 쓰는 판(같은
    경로의 변종)은 비싼 입력(구간·제한속도·차로변경·신호)을 같이 쓰되, `Context.board` 는
    **지금 판**이어야 한다. 열쇠가 이름인 예전 판은 캐시된 판이 곧 이 판이라 그대로 둔다.
    """
    key = (cache_key(board), sections, len(board.route.pts))
    cached = _CONTEXT_CACHE.get(key)
    if cached is None:
        cached = _CONTEXT_CACHE[key] = Context.build(board, sections)
    owner = board if getattr(board, "cache_key", None) else cached.board
    return Context(owner, cached.sections, cached.secs, cached.lim_at, cached.lc_at,
                   cached.tl_stops, cached.cws)
```

`vtd_rl/env/drive_env.py`:

- `__init__` 의 `self.intent = None` 아래에:

```python
        # M6f — 판 고르기를 넘겨받는 콜러블 `rng -> (판, 새로_지은_판인가)`. 안 걸려 있으면(기본)
        # 예전처럼 `self.boards` 에서 고른다. 이름을 준 리셋(`options={"board": ...}`)은 늘 목록에서
        # 찾는다. 새로 지은 판은 이름이 리셋마다 달라 세계를 캐시하지 않는다(캐시가 끝없이 자란다).
        self.board_sampler = None
```

- `reset` 의 판 고르기와 세계 캐시를 바꾼다:

```python
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
        # (아래 단언과 나머지는 그대로)
```

- [ ] **Step 4: 돌려서 통과를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/env tests/referee -q > $TMP/m6f_t1.txt 2>&1; echo rc=$?` → rc=0(600 초를 넘기면 `tests/env`·`tests/referee` 로 나눈다).

- [ ] **Step 5: 커밋한다**

```bash
git add vtd_rl/world/board.py vtd_rl/env/board_index.py vtd_rl/referee/core.py vtd_rl/env/drive_env.py \
        tests/env/test_board_index.py tests/env/test_prep.py tests/env/test_drive_env.py
git commit -m "M6f: 판 고르기 갈고리와 경로 캐시 열쇠 — 리셋마다 새 판을 지어도 캐시가 안 자란다(기본은 예전과 같음)"
git show -s --format=%B HEAD | grep -ciE "co-authored|claude|anthropic|generated with|🤖"   # 0 이어야 한다
```

---

### Task 2: 변종 풀 고르개와 `--train-variant-pool`

**Files:**
- Create: `vtd_rl/rl/variant_pool.py`
- Modify: `vtd_rl/rl/vec_env.py`, `scripts/train_ppo.py`
- Test: `tests/rl/test_variant_pool.py`(새로), `tests/rl/test_vec_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Consumes: Task 1 의 `VtdDriveEnv.board_sampler`(콜러블 `rng -> (Board, fresh)`), `Board.cache_key`.
- Produces:
  - `pool_indices(n: int) -> list[int]`.
  - `VariantSampler(curricula_paths, pool: int, static_boards)` — 콜러블 `rng -> (Board, bool)`.
  - `make_vec_env(..., variant_pool: int | None = None)`.
  - CLI `--train-variant-pool N`, `hparams["train_variant_pool"]`.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/rl/test_variant_pool.py`(새로):

```python
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
```

`tests/rl/test_vec_env.py` 끝에:

```python
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
```

`tests/rl/test_train_ppo.py` 끝에:

```python
def test_train_variant_pool_기본값은_None():
    mod = _load_train_ppo_module()
    a = mod._build_parser().parse_args(["--out", "/tmp/불필요-존재안함"])
    assert a.train_variant_pool is None


def test_train_variant_pool_잘못이면_거부한다(capsys):
    import sys
    mod = _load_train_ppo_module()
    old_argv = sys.argv
    try:
        for bad, msg in ((["train_ppo.py", "--smoke", "--out", "/tmp/불필요-존재안함", "--train-variant-pool", "4"],
                          "5 이상"),
                         (["train_ppo.py", "--smoke", "--out", "/tmp/불필요-존재안함", "--train-variant-pool", "8",
                           "--train-variants", "2"],
                          "같이")):
            sys.argv = bad
            with pytest.raises(SystemExit):
                mod.main()
            assert msg in capsys.readouterr().err
    finally:
        sys.argv = old_argv


@pytest.mark.slow
def test_train_variant_pool_이_벡터_환경까지_닿는다(tmp_path, monkeypatch):
    import sys
    mod = _load_train_ppo_module()
    seen = {}
    real = mod.make_vec_env

    def spy(*args, **kwargs):
        seen["variant_pool"] = kwargs.get("variant_pool")
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "make_vec_env", spy)
    old_argv = sys.argv
    try:
        sys.argv = ["train_ppo.py", "--smoke", "--out", str(tmp_path / "run"), "--seed", "0",
                    "--curricula", "curricula/stage1.json", "curricula/stage3b.json",
                    "--train-variant-pool", "8"]
        mod.main()
    finally:
        sys.argv = old_argv
    assert seen["variant_pool"] == 8
    with open(tmp_path / "run" / "log.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows[0]["hparams"]["train_variant_pool"] == 8
```

- [ ] **Step 2: 돌려서 실패를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_variant_pool.py tests/rl/test_vec_env.py tests/rl/test_train_ppo.py -q -k "풀 or 고르개 or variant_pool or 원본_판_목록" > $TMP/m6f_t2_red.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `ModuleNotFoundError: vtd_rl.rl.variant_pool`.

- [ ] **Step 3: 구현한다.**

`vtd_rl/rl/variant_pool.py`(새로):

```python
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
```

`vtd_rl/rl/vec_env.py`:

```python
from vtd_rl.rl.variant_pool import VariantSampler
```

`_boards` 안의 경로 풀기를 함수로 뺀다(동작은 같다):

```python
def _path(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)
```

(`_boards` 의 `path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)` 을 `path = _path(rel)` 로.)

```python
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
```

`scripts/train_ppo.py`:

- `--train-variants` 다음에:

```python
    ap.add_argument("--train-variant-pool", type=int, default=None,
                    help="M6f — 리셋마다 변종 판을 새로 지어 학습한다. 풀은 v0~v3 + v12 부터 N 개"
                         "(평가 창 v4~v11 은 안 쓴다). 5 이상. --train-variants 와 같이 못 쓴다."
                         " 기본값 None 은 지금과 같다.")
```

- `--train-variants` 검증 다음(스모크 덮어쓰기 **전**)에:

```python
    if a.train_variant_pool is not None:
        if a.train_variant_pool < 5:
            ap.error("--train-variant-pool 은 5 이상이어야 한다(4 이하는 --train-variants 를 써라)")
        if a.train_variants != 1:
            ap.error("--train-variant-pool 과 --train-variants 는 같이 못 쓴다")
```

- `make_vec_env(...)` 호출에 `variant_pool=a.train_variant_pool` 을 더한다.

- [ ] **Step 4: 돌려서 통과를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_variant_pool.py tests/rl/test_vec_env.py -q > $TMP/m6f_t2a.txt 2>&1; echo rc=$?` → rc=0.
Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_train_ppo.py -q > $TMP/m6f_t2b.txt 2>&1; echo rc=$?` → rc=0.
그다음 전체를 파일 단위로 나눠 포그라운드로(각 호출 600 초 안), 모두 rc=0.

- [ ] **Step 5: 커밋한다**

```bash
git add vtd_rl/rl/variant_pool.py vtd_rl/rl/vec_env.py scripts/train_ppo.py \
        tests/rl/test_variant_pool.py tests/rl/test_vec_env.py tests/rl/test_train_ppo.py
git commit -m "M6f: 변종 풀 고르개와 train_ppo --train-variant-pool — 평가 창 v4~v11 을 뺀 풀에서 리셋마다 판을 짓는다"
git show -s --format=%B HEAD | grep -ciE "co-authored|claude|anthropic|generated with|🤖"   # 0 이어야 한다
```
