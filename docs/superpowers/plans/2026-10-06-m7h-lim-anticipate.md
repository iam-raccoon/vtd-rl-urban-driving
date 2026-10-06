# M7h — 앞당긴 제한속도 관측: 보호구역에 들어서기 전에 줄이게 하기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 남은 감점 중 보호구역 속도(완주 판당 중대 0.7~0.8, 선생님 0)는 관측 탓이다. 관측의 `ego` 묶음은 **지금 자리의** 제한속도 `lim` 과 `v − lim` 만 준다. 보호구역(30 km/h)에 들어서는 순간 `lim` 이 갑자기 떨어지니, 그때 줄여도 첫 프레임에 +5 km/h 를 넘어 중대가 난다. 그물은 `lim` 을 잘 따른다(과속 경미 0.15~0.2 건). 그래서 M7h 는 관측의 `lim` 을 **앞당긴 제한속도**로 바꿀 수 있게 한다. 앞에 낮은 제한속도가 있으면, 거기까지 등감속으로 줄이는 곡선을 따라 미리 낮아진 값이다. 관측 차원은 그대로이므로 지금 그물을 그대로 쓸 수 있다.

**Architecture:**
- `ObsConfig`: `lim_anticipate: bool = False`, `lim_decel: float = 1.5`[m/s²], `lim_margin: float = 5.0`[m], `lim_lookahead: float = 150.0`[m]. 기본 False 면 관측이 예전과 같다.
- `BoardIndex`: `lim_drop_s`·`lim_drop_v` — 경로를 따라 차로계획의 `lim` 이 **내려가는** 자리의 s 와 내려간 뒤의 값(오름차순).
- `observation.anticipated_limit(idx, s, lim, cfg) -> float`: `s` 앞 `lim_lookahead` 안의 내려감마다 `sqrt(lim_after² + 2·lim_decel·max(ds − lim_margin, 0))` 를 구해 `lim` 과 함께 가장 작은 값을 낸다.
- `build_observation`: `cfg.lim_anticipate` 이고 `lim > 0` 이면 `ego` 묶음의 `lim`·`v − lim` 에 앞당긴 값을 쓴다. `signal` 묶음의 보호구역 표시(`zone`)는 그대로다.
- `train_ppo.py --lim-anticipate`: 학습 환경의 `ObsConfig(lim_anticipate=True)`. 학습 환경 설정은 `_build_env_cfg(a, reward_cfg)` 로 뽑는다(테스트용).
- `eval_unseen.py --lim-anticipate`: 평가 환경에 같은 관측 설정을 준다.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거 `docs/reports/m7g-block-penalty.md`("다음"), `docs/superpowers/plans/2026-10-06-m7e-continue-obs.md`("미룬 것").

## 실험

**배포 묶음** = (그물, 관측 설정). 지금 최고 배포 묶음은 (M7f 시드 2 · 12.5 만, 앞당김 끔)이다.

- **A — 학습 없이 켜 보기.** 지금 최고 그물을 앞당김을 켠 채 큰 선택 창(v1100~v1131 × 시드 2, 384 판)으로 잰다. 끈 쪽은 이미 쟀다(M7f).
- **B — 켠 채 이어 돌리기.** 지금 최고 그물에서 출발·앵커, 보상은 M7e 와 같고, 학습 환경에 `--lim-anticipate` 를 켠다. 시드 0~2, 100 만 걸음, 체크포인트 12.5 만 간격. 체크포인트 24 개 모두 앞당김을 켠 채 큰 선택 창으로 잰다.

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-06-m7h-s$S --init runs/omen/2026-10-06-m7f-s2/policy-125440.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 --lat-free-range 80 --obs-profile 1.0 --lim-anticipate
```

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 지금 최고 배포 묶음은 큰 선택 창 90.00(보호구역 중대는 M7f 자료에서 잰다)이다.

- **(가) 앞당긴 제한속도가 보호구역 속도를 고치나.** A 의 큰 선택 창 보호구역 중대(완주 판당)가 끈 쪽보다 50% 넘게 줄면 "고친다" 로 본다.
- **(나) 고르기.** 후보는 A 의 묶음 하나와 B 의 체크포인트 24 개(모두 앞당김 켬)다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3, 앞당김 켬) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 묶음이 큰 선택 창에서 지금 최고(90.00)보다 높을 때만, 지금 최고 묶음과 **새 시험 창 v20000~v20031 × 시드 2**(384 판)에서 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

---

### Task 1: 앞당긴 제한속도 관측(`lim_anticipate`)과 CLI

**Files:**
- Modify: `vtd_rl/env/board_index.py`(`lim_drop_s`·`lim_drop_v`)
- Modify: `vtd_rl/env/observation.py`(`ObsConfig.lim_*`, `anticipated_limit`, `build_observation`)
- Modify: `scripts/train_ppo.py`(`--lim-anticipate`, `_build_env_cfg`)
- Modify: `scripts/eval_unseen.py`(`--lim-anticipate`, `evaluate_stage(..., config=None)`)
- Test: `tests/env/test_board_index.py`, `tests/env/test_observation.py`, `tests/rl/test_train_ppo.py`, `tests/test_eval_unseen.py`(파일 이름은 저장소의 기존 eval_unseen 테스트 파일을 따른다)

**Interfaces:**
- Produces:
  - `BoardIndex.lim_drop_s: list[float]`, `BoardIndex.lim_drop_v: list[float]`.
  - `ObsConfig.lim_anticipate=False`, `lim_decel=1.5`, `lim_margin=5.0`, `lim_lookahead=150.0`.
  - `vtd_rl.env.observation.anticipated_limit(idx, s, lim, cfg) -> float`.
  - `train_ppo._build_env_cfg(a, reward_cfg) -> EnvConfig`, CLI `--lim-anticipate`(hparams `lim_anticipate`).
  - `eval_unseen` CLI `--lim-anticipate`.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_observation.py` 끝에(임포트가 없으면 더한다: `from types import SimpleNamespace`, `from vtd_rl.env.observation import ObsConfig, anticipated_limit`):

```python
def _idx(drops):
    return SimpleNamespace(lim_drop_s=[d[0] for d in drops], lim_drop_v=[d[1] for d in drops])


@pytest.mark.parametrize("s, lim, want", [
    (0.0, 14.0, 14.0),                                   # sqrt(64 + 3·95) ≈ 18.7 > 14
    (80.0, 14.0, (64.0 + 3.0 * 15.0) ** 0.5),            # ds 20 − margin 5 = 15
    (97.0, 14.0, 8.0),                                   # margin 안: 새 제한속도
    (101.0, 14.0, 14.0),                                 # 이미 지났다
    (0.0, 6.0, 6.0),                                     # 지금 제한이 더 낮다
])
def test_anticipated_limit(s, lim, want):
    cfg = ObsConfig(lim_anticipate=True)
    assert anticipated_limit(_idx([(100.0, 8.0)]), s, lim, cfg) == pytest.approx(want)


def test_anticipated_limit_은_lookahead_밖을_안_본다():
    cfg = ObsConfig(lim_anticipate=True, lim_lookahead=50.0)
    assert anticipated_limit(_idx([(100.0, 1.0)]), 0.0, 14.0, cfg) == pytest.approx(14.0)


def test_lim_anticipate_기본값은_꺼짐():
    assert ObsConfig().lim_anticipate is False
```

`tests/env/test_board_index.py` 끝에(파일 위 `load_board`·`board_index` 임포트를 쓴다):

```python
def test_lim_drop_은_제한속도가_내려가는_자리만_오름차순으로_담는다():
    from vtd_rl.world.board import load_board
    import json
    cur = json.load(open("curricula/stage2.json"))
    found = False
    for e in cur["boards"]:
        idx = board_index(load_board(e))
        assert idx.lim_drop_s == sorted(idx.lim_drop_s)
        assert len(idx.lim_drop_s) == len(idx.lim_drop_v)
        lims = [(p or {}).get("lim") for p in idx.board.lane_plan]
        for s, v in zip(idx.lim_drop_s, idx.lim_drop_v):
            i = idx.board.route.cum.index(s)
            prev = next(x for x in reversed(lims[:i]) if x)
            assert lims[i] == v and v < prev
        found = found or bool(idx.lim_drop_s)
    assert found                                         # 단계 ② 코스 어딘가에는 보호구역이 있다
```

같은 파일(또는 `tests/env/test_observation.py`)에:

```python
def test_lim_anticipate_를_끄면_관측이_예전과_같고_켜면_lim_이_같거나_낮다():
    import json
    import numpy as np
    from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
    from vtd_rl.env.observation import ObsConfig
    from vtd_rl.world.board import load_board
    e = json.load(open("curricula/stage2.json"))["boards"][0]
    b = load_board(e)
    off, on = VtdDriveEnv([b]), VtdDriveEnv([b], EnvConfig(obs=ObsConfig(lim_anticipate=True)))
    o1, _ = off.reset(seed=0)
    o2, _ = on.reset(seed=0)
    act = {"control": np.array([0.0, 0.6], dtype=np.float32), "turn": 0}
    for _ in range(600):
        o1, *_ = off.step(act)
        o2, *_ = on.step(act)
        for k in o1:
            if k != "ego":
                assert np.array_equal(o1[k], o2[k]), k
        assert o2["ego"][-2] <= o1["ego"][-2] + 1e-6      # 앞당긴 lim(정규화) ≤ 지금 lim
    off.close()
    on.close()
```

`tests/rl/test_train_ppo.py` 끝 쪽(`@pytest.mark.slow` 테스트들 앞)에:

```python
def test_lim_anticipate_인자가_학습_환경_관측에_닿는다():
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "x", "--lim-anticipate"])
    cfg = module._build_env_cfg(a, module._build_reward_cfg(a))
    assert cfg.obs.lim_anticipate is True
    off = module._build_parser().parse_args(["--out", "x"])
    assert module._build_env_cfg(off, module._build_reward_cfg(off)).obs.lim_anticipate is False
```

eval_unseen 테스트 파일(저장소에서 `eval_unseen` 을 부르는 테스트 파일을 찾아 그 끝)에:

```python
def test_eval_unseen_lim_anticipate_인자():
    module = _load_eval_unseen_module()      # 그 파일에 있는 로더 이름을 쓴다
    a = module.build_parser().parse_args(["--ckpt", "x", "--lim-anticipate"])   # 파서 함수 이름도 그 파일을 따른다
    assert a.lim_anticipate is True
```

(eval_unseen 에 파서 함수가 없으면 `main()` 안의 파서를 `_build_parser()` 로 뽑아내고 그것을 테스트한다.)

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_observation.py tests/env/test_board_index.py tests/rl/test_train_ppo.py <eval_unseen 테스트 파일> -q -m "not slow" > $TMP/m7h_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0.

- [ ] **Step 3: `board_index.py`.**

`BoardIndex.__init__` 의 `self.junction_s = ...` 줄 바로 뒤에:

```python
        self.lim_drop_s, self.lim_drop_v = self._lim_drops(route, board.lane_plan)
```

메서드를 더한다:

```python
    @staticmethod
    def _lim_drops(route, lane_plan):
        """차로계획의 제한속도가 **내려가는** 자리 — `(s 목록, 내려간 뒤 값 목록)`(M7h 앞당긴 제한속도).

        제한속도가 없는 점(None·0)은 건너뛰고, 직전의 있는 값보다 작아지는 첫 점만 담는다.
        """
        out_s, out_v, prev = [], [], None
        for i, p in enumerate(lane_plan):
            lim = (p or {}).get("lim")
            if not lim:
                continue
            if prev is not None and lim < prev - 1e-6:
                out_s.append(route.cum[i])
                out_v.append(float(lim))
            prev = lim
        return out_s, out_v
```

- [ ] **Step 4: `observation.py`.**

모듈 위 임포트에 `import bisect` 가 없으면 더한다. `ObsConfig` 끝(`obj_size` 뒤)에:

```python
    # M7h — 앞당긴 제한속도. True 면 `ego` 묶음의 lim·v−lim 에, 앞에 낮은 제한속도가 있을 때 거기까지
    # lim_decel 로 줄이는 곡선을 따라 미리 낮아진 값을 쓴다(관측 차원은 그대로). 기본 False 면 예전과 같다.
    lim_anticipate: bool = False
    lim_decel: float = 1.5           # [m/s²]
    lim_margin: float = 5.0          # [m] — 내려가는 자리보다 이만큼 앞에서 새 제한속도에 닿는다
    lim_lookahead: float = 150.0     # [m]
```

`_clip` 함수 뒤에:

```python
def anticipated_limit(idx, s: float, lim: float, cfg: ObsConfig) -> float:
    """앞당긴 제한속도[m/s] — `s` 앞 lim_lookahead 안의 내려감마다 등감속 곡선 값을 구해 가장 작은 것(M7h)."""
    i = bisect.bisect_right(idx.lim_drop_s, s)
    for k in range(i, len(idx.lim_drop_s)):
        ds = idx.lim_drop_s[k] - s
        if ds > cfg.lim_lookahead:
            break
        lim = min(lim, math.sqrt(idx.lim_drop_v[k] ** 2 + 2.0 * cfg.lim_decel * max(ds - cfg.lim_margin, 0.0)))
    return lim
```

`build_observation` 의 `lim = plan.get("lim") or 0.0` 줄 바로 뒤에:

```python
    if cfg.lim_anticipate and lim > 0.0:
        lim = anticipated_limit(idx, s, lim, cfg)
```

- [ ] **Step 5: `train_ppo.py`.**

`--lim-anticipate` 인자를 `--block-range` 인자 바로 뒤에:

```python
    ap.add_argument("--lim-anticipate", action="store_true",
                    help="학습 환경 관측의 제한속도를 앞당긴 값으로(M7h). 앞에 낮은 제한속도가 있으면"
                         " 거기까지 줄이는 곡선을 따라 미리 낮아진 값을 준다. 관측 차원은 그대로")
```

`_build_reward_cfg` 뒤에:

```python
def _build_env_cfg(a, reward_cfg: RewardConfig) -> EnvConfig:
    """학습 환경 설정 — 보상 설정과 관측 설정(`--lim-anticipate`)을 묶는다. `--smoke` 면 시간 제한을 줄인다."""
    obs = ObsConfig(lim_anticipate=bool(getattr(a, "lim_anticipate", False)))
    if a.smoke:
        return EnvConfig(world=WorldConfig(time_limit_scale=0.1), reward=reward_cfg, obs=obs)
    return EnvConfig(reward=reward_cfg, obs=obs)
```

`main` 의 `train_env_cfg = (EnvConfig(world=...) if a.smoke else EnvConfig(reward=reward_cfg))` 를 `train_env_cfg = _build_env_cfg(a, reward_cfg)` 로 바꾼다. 임포트에 `from vtd_rl.env.observation import ObsConfig` 를 더한다(없으면).

- [ ] **Step 6: `eval_unseen.py`.**

파서에 `ap.add_argument("--lim-anticipate", action="store_true", help="평가 환경 관측의 제한속도를 앞당긴 값으로(M7h) — 그렇게 학습한 그물을 잴 때 켠다")` 를 더한다. `evaluate_stage(net, boards, eval_seeds)` 에 `config=None` 을 더해 `evaluate_policy(..., config=config)` 로 넘기고, `main` 에서 `config = EnvConfig(obs=ObsConfig(lim_anticipate=True)) if a.lim_anticipate else None` 을 만들어 넘긴다. 성적표 머리에 앞당김을 켰으면 "관측: 앞당긴 제한속도" 한 줄을 더한다.

- [ ] **Step 7: 통과를 확인한다.**

Run: Step 2 와 같은 명령 → rc=0. 전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m7h_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 8: 커밋한다.**

```bash
git add vtd_rl/env/board_index.py vtd_rl/env/observation.py scripts/train_ppo.py scripts/eval_unseen.py <바꾼 테스트 파일들>
git commit -m "M7h — 앞당긴 제한속도 관측(lim_anticipate)과 train_ppo·eval_unseen --lim-anticipate"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `lim_anticipate=False`(기본)이면 관측이 예전과 비트 단위로 같다.
- 관측 차원(`VEC_DIM`)은 바뀌지 않는다.
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. numpy 는 1.26.4 그대로다.

## Review Focus

- 앞당김은 `ego` 의 lim·v−lim 에만 들어가고, 다른 묶음(특히 `signal` 의 보호구역 표시)은 그대로다.
- 제한속도가 없는 점(None·0)에서는 앞당기지 않는다(`lim > 0` 일 때만).
- 이미 지난 내려감은 보지 않는다(`bisect_right`).
- 학습 환경과 평가 환경이 같은 관측 설정을 쓸 수 있다(`train_ppo`·`eval_unseen` 둘 다 인자가 있다).
