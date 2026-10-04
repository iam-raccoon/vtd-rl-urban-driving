# M6e — 앵커 PPO 를 변종 판으로 학습: 안 쓴 변종에서 시드 운에 덜 기대나 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6d 에서 같은 원본 판 30 개로 앵커 PPO 를 한 번 더 이었더니 ③ 전체가 그대로였다(68.5 → 71.3%). 시드 사이 편차만 커졌다(58~85%). 학습 판은 단계마다 원본 배치 6 개뿐이고, 평가는 액터 자리를 흔든 안 쓴 변종(v4~v11)으로 한다. M6e 는 **수집 창 변종(v0~v3)으로도 학습**하게 해서, 안 쓴 변종 성적이 오르는지 잰다.

**Architecture:** `vtd_rl/rl/vec_env.py` 의 `_boards(curricula, variants=1)` 가 `variants > 1` 이면 액터가 있는 커리큘럼은 `load_window(path, variants, 0)` 로 변종 v0~v{V-1} 판을 낸다. 액터가 없는 커리큘럼(①②)은 같은 판 객체를 V 번 되풀이해, 단계 사이 비율을 지금처럼 1:1 로 둔다. `make_env_fn`·`make_vec_env` 가 `variants` 를 그대로 넘긴다. `scripts/train_ppo.py --train-variants V`(기본 1 = 지금과 같음, 1~4)를 연다. 4 를 넘으면 보고 창(v4~)과 겹치므로 거부한다. 학습 중 평가(`_stage_boards`)는 그대로 원본 판이다.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거는 `docs/reports/m6d-ppo-chain.md`("다음" 1).

## 사전 측정(끝남) — 메모리

환경 하나가 판마다 세계를 캐시한다(`VtdDriveEnv._worlds`). 로컬에서 판 전부를 한 번씩 리셋해 최고 RSS 를 쟀다.

| 판 | 판 수 | 최고 RSS |
|---|---:|---:|
| 원본(V=1) | 30 | 209 MB |
| 변종(V=4) | 84 | 520 MB(같은 프로세스에서 V=1 다음에 잼 — 상한) |

환경 프로세스 10 개면 실행 하나가 약 2~3 GB 더 쓴다. 그래서 **동시 3 실행**으로 돌린다(지금 실행 하나 ≈ 3 GB → 약 5~6 GB).

## 본 실행 (OMEN)

칸 V(계수 0.03, 변종 4) × 시드 `S ∈ {0, 1, 2}` = 3 실행, 동시에:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-04-m6e-V-s$S --init runs/omen/2026-10-04-m6c-A-s$S/policy-best.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 \
  --steps 2000000 --eval-every 500000 --eval-seeds 1 --final-eval-seeds 3 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20
```

- **M6d 칸 A 와 `--train-variants 4` 하나만 다르다.** 출발·참조(M6c A 고른 그물), 판 묶음, 걸음, 학습률, 계수가 같다. M6d A 가 대조군이다.
- 학습 판 84 개: ③a·③b·③ 전체 변종 24 개씩 + ①·② 원본 6 개를 4 번 되풀이(24 개씩). 단계 비율은 M6d 와 같다(5 단계 각 1/5).
- 평가: M6d 와 같다. `ac-best`·마지막을 꺼내 ①·②(원본), ③a·③b·③(선택 창 v8~v11·보고 창 v4~v7)를 잰다. 학습 변종 v0~v3 은 두 창과 겹치지 않는다.

## 판정 기준 (미리 적는다)

출발점 = M6c 칸 A 고른 그물(M6d 와 같다). 선택 창 ③ 전체 62.5 · 70.8 · 72.2(평균 68.5), 보고 창 69.4 · 68.1 · 56.9(평균 64.8), 선택 창 ③a 평균 98.6, ③b 평균 98.1.
대조군 = M6d 칸 A 고른 그물. 선택 창 ③ 전체 58.3 · 84.7 · 70.8(평균 71.3), 보고 창 52.8 · 83.3 · 63.9(평균 66.7).

시드마다 `ac-best` 와 마지막 중 **선택 창 ③ 전체가 높은 쪽**을 고른다.

- **오른다 / 내린다 / 차이 없다** — M6d 와 같은 문턱이다.
  - 오른다: 선택 창 ③ 전체 평균 ≥ 75.5%(+7), 2/3 시드 이상이 출발점보다 높음, 선택 창 ③a·③b 평균이 각각 ≥ 93.6 · ≥ 93.1, ①·② ≥ 90%.
  - 내린다: 선택 창 ③ 전체 평균 ≤ 63.5%(−5) 이거나, ③a 또는 ③b 평균이 출발점 −10 미만이거나, ① 또는 ② 평균 < 90%.
- **"변종 학습이 ③ 전체를 올린다"** 는 다음이 모두일 때만 적는다.
  - 판정이 "오른다" 다.
  - 보고 창 ③ 전체 평균 ≥ 69.8%(출발점 +5)이고, 2/3 시드 이상이 출발점보다 높다.
  - 선택 창·보고 창 평균이 둘 다 대조군(M6d A)보다 높다.
- 곁들여(판정과 별도): 고른 그물 ③ 전체의 시드 사이 폭(최고 − 최저)을 M6d A(26.4 점)와 나란히 적는다. 마지막 체크포인트 숫자, ①② 점수도 적는다.

## Global Constraints

- numpy 는 **1.26.4** 고정 — 새 의존성을 넣지 않는다.
- `third_party/rule_stack` 은 **읽기 전용**이다.
- 커밋 금지: `runs/`, `*.pt`, `*.npz`, CSV, `*.xodr`, `*.xml`.
- 주석·독스트링·커밋 메시지·성적표는 **한국어**.
- 커밋 메시지에 **Claude 표기를 넣지 않는다**(`Co-Authored-By`, `Claude-Session`, "Generated with", 🤖, anthropic 주소 전부). 커밋 뒤 `git show -s --format=%B HEAD` 로 확인한다.
- 테스트는 `env -u PYTHONPATH .venv/bin/pytest ...` 로 돌리고 **종료 코드(rc)** 로 판정한다. "N passed" 를 grep 하지 않는다. `| tail` 로 넘기지 않고 파일로 돌린다.
- **기본값은 지금과 같다**: `_boards(curricula)`·`make_vec_env(...)` 를 `variants` 없이 부르면 판 목록(이름·순서·객체 수)이 지금과 같다. `--train-variants` 를 안 주면 `train_ppo.py` 동작이 지금과 같다.
- 판 이름 규칙(`course_X@v{k}#stage`)과 변종 접기 규칙은 **새로 짜지 않는다** — `vtd_rl.world.board.load_window` 를 쓴다(평가 창과 같은 함수).

## Review Focus

1. **되풀이한 판 객체에 `#stage` 접미사가 여러 번 붙는 것.** ①② 판을 V 번 되풀이할 때 같은 객체를 V 번 넣으므로, 이름 바꾸기를 되풀이 뒤에 하면 `course_A#stage1#stage1…` 이 된다. 그러면 세계 캐시 열쇠·로그의 판 이름이 깨진다. → `test_되풀이한_판도_접미사는_한_번만`.
2. **단계 비율이 바뀌는 것.** 변종을 액터 단계에만 걸고 ①② 를 안 되풀이하면, 학습 판의 ①② 몫이 40% → 14% 로 줄어 M6d 와 비교가 깨진다. → `test_변종을_걸어도_단계마다_판_수가_같다`.
3. **학습 변종이 평가 창과 겹치는 것**(`--train-variants 5` 이상이면 v4 가 들어간다). → `test_train_variants_범위를_벗어나면_거부한다`.
4. **기본값(V=1)에서 무언가 달라지는 것**(판 순서·객체 수). 그러면 M6a~M6d 와 같은 설정의 재현이 깨진다. → `test_변종_기본값은_예전_판_목록과_같다`.
5. **CLI 값이 실제 환경까지 안 닿는 것**(파싱만 되고 `make_vec_env` 에 안 넘어감). → `test_train_variants_가_벡터_환경까지_닿는다`(느린 스모크).

---

### Task 1: 변종 판으로 학습 — `vec_env._boards(variants)` + `train_ppo --train-variants`

**Files:**
- Modify: `vtd_rl/rl/vec_env.py`
- Modify: `scripts/train_ppo.py`
- Test: `tests/rl/test_vec_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Consumes: `vtd_rl.world.board.load_window(path, variants, offset) -> (boards, varied)`, `load_curriculum(path) -> (name, boards)`.
- Produces: `_boards(curricula, variants: int = 1) -> list[Board]`; `make_env_fn(curricula, config, seed, rank, variants=1)`; `make_vec_env(curricula, n_envs, config=None, seed=0, asynchronous=True, variants=1)`; CLI `--train-variants INT`(기본 1, 1~4), `hparams["train_variants"]`.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/rl/test_vec_env.py` 끝에(임포트 줄을 `from vtd_rl.rl.vec_env import _boards, make_vec_env, vec_obs_to_arrays` 로 고친다):

```python
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
```

`tests/rl/test_train_ppo.py` 끝에:

```python
def test_train_variants_기본값은_1이다():
    mod = _load_train_ppo_module()
    a = mod._build_parser().parse_args(["--out", "/tmp/불필요-존재안함"])
    assert a.train_variants == 1


def test_train_variants_범위를_벗어나면_거부한다(capsys):
    """5 이상이면 학습 변종에 v4 가 들어가 보고 창(v4~v7)과 겹친다 — 무거운 준비 전에 죽는다."""
    import sys
    mod = _load_train_ppo_module()
    old_argv = sys.argv
    try:
        for bad, msg in ((["train_ppo.py", "--smoke", "--out", "/tmp/불필요-존재안함", "--train-variants", "5"],
                          "보고 창"),
                         (["train_ppo.py", "--smoke", "--out", "/tmp/불필요-존재안함", "--train-variants", "0"],
                          "1 이상")):
            sys.argv = bad
            with pytest.raises(SystemExit):
                mod.main()
            assert msg in capsys.readouterr().err
    finally:
        sys.argv = old_argv


@pytest.mark.slow
def test_train_variants_가_벡터_환경까지_닿는다(tmp_path, monkeypatch):
    """파싱만 되고 `make_vec_env` 에 안 넘어가는 함정을 막는다 — 실제 `main()` 을 스모크로 돈다."""
    import sys
    mod = _load_train_ppo_module()
    seen = {}
    real = mod.make_vec_env

    def spy(*args, **kwargs):
        seen["variants"] = kwargs.get("variants")
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "make_vec_env", spy)
    old_argv = sys.argv
    try:
        sys.argv = ["train_ppo.py", "--smoke", "--out", str(tmp_path / "run"), "--seed", "0",
                    "--train-variants", "2"]
        mod.main()
    finally:
        sys.argv = old_argv
    assert seen["variants"] == 2
    with open(tmp_path / "run" / "log.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows[0]["hparams"]["train_variants"] == 2
```

- [ ] **Step 2: 돌려서 실패를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_vec_env.py tests/rl/test_train_ppo.py -q -k "변종 or train_variants or 접미사" > $TMP/m6e_red.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `TypeError: _boards() got an unexpected keyword argument 'variants'` 와 `unrecognized arguments: --train-variants`(`_boards` 는 이미 있어 임포트는 된다).

- [ ] **Step 3: 구현한다.**

`vtd_rl/rl/vec_env.py`:

```python
from vtd_rl.world.board import load_curriculum, load_window
```

`_boards` 를 바꾼다(독스트링 끝에 변종 설명을 더한다):

```python
def _boards(curricula, variants: int = 1):
    """(기존 독스트링 그대로)

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
        path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
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
```

(V=1 일 때 `varied = True` 로 두어 `boards * variants` 를 안 거친다 — 지금과 같은 목록이다.)

`make_env_fn` 과 `make_vec_env` 에 `variants=1` 을 더하고 그대로 넘긴다:

```python
def make_env_fn(curricula, config: EnvConfig | None, seed: int, rank: int, variants: int = 1):
    def _make():
        # 시드는 make_vec_env 의 reset(seed=[...]) 이 준다 — 여기서 리셋하면 세계를 한 번 더 짓는다.
        return VtdDriveEnv(_boards(curricula, variants), config or EnvConfig())
    return _make


def make_vec_env(curricula, n_envs: int, config: EnvConfig | None = None, seed: int = 0,
                 asynchronous: bool = True, variants: int = 1):
    fns = [make_env_fn(tuple(curricula), config, seed, i, variants) for i in range(n_envs)]
    venv = AsyncVectorEnv(fns) if asynchronous else SyncVectorEnv(fns)
    venv.reset(seed=[seed + i for i in range(n_envs)])
    return venv
```

`scripts/train_ppo.py`:

- `_build_parser()` 의 `--anchor-coef` 다음에:

```python
    ap.add_argument("--train-variants", type=int, default=1,
                    help="M6e — 액터가 있는 학습 판을 수집 창 변종 v0~v{N-1} 로 넓힌다(1~4). 기본 1 은"
                         " 지금과 같다(원본 판만). ①② 는 같은 판을 N 번 되풀이해 단계 비율을 지킨다."
                         " 5 이상이면 보고 창(v4~v7)과 겹쳐 거부한다.")
```

- `main()` 의 `--anchor-coef` 검증 다음(스모크 덮어쓰기 **전**)에:

```python
    if a.train_variants < 1:
        ap.error("--train-variants 는 1 이상이어야 한다")
    if a.train_variants > 4:
        ap.error("--train-variants 는 4 이하여야 한다 — 5 이상이면 학습 변종에 v4 가 들어가"
                 " 보고 창(v4~v7)과 겹친다")
```

- `make_vec_env(...)` 호출에 `variants=a.train_variants` 를 더한다:

```python
        venv = make_vec_env(a.curricula, a.envs, train_env_cfg, seed=a.seed, asynchronous=not a.smoke,
                            variants=a.train_variants)
```

(`hparams = vars(a)` 라 `train_variants` 는 로그 줄·요약 JSON 에 저절로 실린다.)

- [ ] **Step 4: 돌려서 통과를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_vec_env.py tests/rl/test_train_ppo.py -q > $TMP/m6e_t1.txt 2>&1; echo rc=$?` → rc=0.
그다음 전체를 두 번에 나눠: `tests/rl` 와 `tests --ignore=tests/rl`, 각각 파일로, 각각 rc=0.

- [ ] **Step 5: 커밋한다**

```bash
git add vtd_rl/rl/vec_env.py scripts/train_ppo.py tests/rl/test_vec_env.py tests/rl/test_train_ppo.py
git commit -m "M6e: train_ppo --train-variants — 액터 판을 수집 창 변종으로 넓혀 학습(①② 는 되풀이해 비율 유지)"
git show -s --format=%B HEAD | grep -ciE "co-authored|claude|anthropic|generated with|🤖"   # 0 이어야 한다
```
