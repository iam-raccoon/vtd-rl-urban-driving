# M7g — 장애물 뒤에 머무는 벌: 줄인 다음 "차로를 바꿔 비키기" 를 가르치기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 장애물 감속 곡선(M7d)은 통로 안 장애물 앞에서 "줄이기" 를 가르쳤다. 하지만 줄인 뒤 장애물 뒤에 서 있어도 벌이 없다(서 있으면 곡선을 넘지 않는다). course_G 4 번째 정지차처럼 차로를 바꿔야 지나가는 장면에서, 학생은 크게 비키는 법을 못 찾는다(M7e·M7f). 선생님은 줄인 뒤 옆 차로로 옮겨 지나간다. M7g 는 통로 안 **15 m 앞에 멈춰 있는 장애물(사람·자전거 제외)** 이 있으면 걸음마다 일정한 벌을 준다. 통로를 벗어나거나(차로 바꾸기) 지나가면 벌이 사라진다.

**Architecture:**
- `RewardConfig`: `block_profile: float = 0.0`(걸음당 벌 c), `block_range: float = 15.0`[m]. 기본 c=0 이면 항이 없고 예전과 같다.
- `VtdDriveEnv._block_gap()`: `_obs_gap` 과 같은 통로·거리 계산을 하되, **멈춰 있고(속도 < 0.5 m/s) 사람·자전거가 아닌**(`vtd_io.classify_object` 가 `VRU` 가 아닌) 물체만 본다. 그중 가장 가까운 `gap` 을 낸다. 없으면 `None`.
  - 통로 여유는 `obs_margin` 을 같이 쓴다.
- `RewardShaper.step(..., block=None)`: `block_profile > 0` 이고 `block` 이 있고 `block < block_range` 이면 `terms["block"] = −block_profile`, 아니면 `0.0`.
- `train_ppo.py --block-profile`·`--block-range`(dest 는 필드 이름과 같다), `TERM_KEYS` 에 `block`.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거 `docs/reports/m7f-variant-pool.md`("다음").

## 본 실행 (OMEN)

칸 `ARM ∈ {B2: 0.2, B10: 1.0}` × 시드 `S ∈ {0, 1, 2}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-06-m7g-$ARM-s$S --init runs/omen/2026-10-06-m7f-s2/policy-125440.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 --lat-free-range 80 --obs-profile 1.0 --block-profile $C
```

- 출발·앵커는 지금 최고 그물(M7f 시드 2 · 12.5 만)이다. 보상은 M7e 와 같고 `--block-profile` 만 더한다.
- **대조군은 M7e**(같은 보상·변종 4, 출발은 M7d 최고 — 지금 최고는 거기서 12.5 만 걸음만 이어 돌린 그물이다)다.
- 크기: 장애물 뒤 15 m 안에 60 초 서 있으면 c=0.2 에서 −120, c=1.0 에서 −600 이다. 줄인 뒤 2 초 안에 옆 차로로 빠지면 −4·−20 이다.
- 체크포인트 48 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 지금 최고 그물은 큰 선택 창 90.00(완주 363, course_G 52)이다.

- **(가) 머무는 벌이 course_G 를 고치나.** 체크포인트 24 개의 course_G 완주 평균이 대조(M7e, 39.5)보다 8 판 넘게 많은 칸이 있으면 "머무는 벌이 course_G 를 고친다" 로 본다.
  - 곁들여 칸마다 충돌·정체 평균을 대조와 비교한다.
- **(나) 고르기.** 후보는 두 칸 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체(M7f 에서 조인 규칙).** 고른 그물이 **큰 선택 창에서 지금 최고(90.00)보다 높을 때만** 새 시험 창으로 넘어간다. 고른 그물과 지금 최고 그물을 **새 시험 창 v19000~v19031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

---

### Task 1: 장애물 뒤에 머무는 벌(`block_profile`)

**Files:**
- Modify: `vtd_rl/env/reward.py`(`RewardConfig.block_profile·block_range`, 검사, `RewardShaper.step(..., block=None)`)
- Modify: `vtd_rl/env/drive_env.py`(`_block_gap`, `step` 에서 `block=` 넘기기)
- Modify: `vtd_rl/rl/diagnostics.py`(`TERM_KEYS` 에 `"block"`)
- Modify: `scripts/train_ppo.py`(`--block-profile`, `--block-range`, `_build_reward_cfg`)
- Test: `tests/env/test_reward.py`, `tests/env/test_drive_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `RewardConfig.block_profile=0.0`, `block_range=15.0`.
  - `RewardShaper.step(..., obs=None, block=None)` — `block_profile > 0` 일 때만 `terms["block"]`.
  - `VtdDriveEnv._block_gap() -> float | None`.
  - CLI `--block-profile`, `--block-range`(dest = `block_profile`, `block_range`).

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_reward.py` 끝에:

```python
def test_block_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().block_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "block" not in sh.step([], 0.0, zero, zero, "running", block=5.0).terms


@pytest.mark.parametrize("block, want", [(None, 0.0), (20.0, 0.0), (15.0, 0.0), (14.9, -0.5), (0.0, -0.5)])
def test_block_profile_은_가까운_멈춘_장애물_뒤에서만_깎는다(block, want):
    sh = RewardShaper(h_board(), RewardConfig(block_profile=0.5))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", block=block)
    assert out.terms["block"] == pytest.approx(want)
    assert str(out.terms["block"]) != "-0.0"
    assert out.total == pytest.approx(sum(out.terms.values()))


@pytest.mark.parametrize("kw", [{"block_profile": -0.1}, {"block_profile": float("nan")},
                                {"block_range": -1.0}, {"block_range": float("inf")}])
def test_잘못된_block_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)
```

`tests/env/test_drive_env.py` 끝에(`rammer_board()` 는 경로 60 m 에 멈춘 차량을 세운다):

```python
def test_block_gap_은_멈춘_차량은_보고_움직이는_물체는_안_본다():
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    gap = env._block_gap()
    assert gap is not None and abs(gap - env._obs_gap()[0]) < 1e-9     # 같은 통로·거리 계산
    env.close()
    empty = VtdDriveEnv([boards()[0]])
    empty.reset(seed=0)
    empty.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    assert empty._block_gap() is None
    empty.close()


def test_block_gap_은_사람과_움직이는_물체를_뺀다(monkeypatch):
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    car = env.state.objects[0]
    from dataclasses import replace
    moving = replace(car, speed=3.0)
    person = replace(car, length=0.5, width=0.5, height=1.7)
    for obj in (moving, person):
        monkeypatch.setattr(env.state, "objects", [obj])
        assert env._block_gap() is None
    env.close()
```

`tests/rl/test_train_ppo.py` 의 `test_obs_profile_인자가_RewardConfig에_반영되고_hparams_로_되살아난다` 바로 뒤에:

```python
def test_block_profile_인자가_RewardConfig에_반영되고_hparams_로_되살아난다():
    from vtd_rl.rl.reward_cfg import reward_config_from_hparams
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "x", "--block-profile", "1.0", "--block-range", "20"])
    cfg = module._build_reward_cfg(a)
    assert (cfg.block_profile, cfg.block_range) == (1.0, 20.0)
    back = reward_config_from_hparams(vars(a))
    assert (back.block_profile, back.block_range) == (1.0, 20.0)
    off = module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"]))
    assert (off.block_profile, off.block_range) == (0.0, 15.0)
```

(`Obj` 가 dataclass 가 아니라 `dataclasses.replace` 가 안 되면, 같은 필드로 `rs.Obj(...)` 를 새로 지어 쓴다.)

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m7g_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `block_profile` 이 없다.

- [ ] **Step 3: `reward.py`.**

`RewardConfig` 의 `obs_margin` 필드 바로 뒤에:

```python
    # M7g — 장애물 뒤에 머무는 벌. 통로 안 block_range 앞에 멈춰 있는 장애물(사람·자전거 제외)이 있으면
    # 걸음마다 block_profile 을 깎는다(항 `block`). 통로를 벗어나거나 지나가면 사라진다. 기본 0 이면 꺼짐.
    block_profile: float = 0.0
    block_range: float = 15.0        # [m]
```

`__post_init__` 끝에:

```python
        if not math.isfinite(self.block_profile) or self.block_profile < 0.0:
            raise ValueError(f"block_profile 은 유한한 0 이상이어야 한다: {self.block_profile}")
        if not math.isfinite(self.block_range) or self.block_range < 0.0:
            raise ValueError(f"block_range 는 유한한 0 이상이어야 한다: {self.block_range}")
```

`RewardShaper.step` 시그니처에 `block=None` 을 `obs=None` 뒤에 더하고, `obs` 항 블록 바로 뒤에:

```python
        if cfg.block_profile > 0.0:   # 끈 실행은 항 자체가 없다
            near = block is not None and block < cfg.block_range
            terms["block"] = 0.0 - cfg.block_profile if near else 0.0
```

- [ ] **Step 4: `drive_env.py`.**

상수 자리(`ROAD_SURFACE_H` 아래)에:

```python
BLOCK_STILL_V = 0.5              # 이보다 느린 물체를 '멈춘 장애물' 로 본다[m/s](M7g)
```

`step` 의 `obs = ...` 줄 바로 뒤에:

```python
        block = self._block_gap() if self.cfg.reward.block_profile > 0.0 else None
```

`self._shaper.step(...)` 호출에 `block=block` 을 더한다. `_obs_gap` 메서드 바로 뒤에:

```python
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
```

(`import vtd_io` 를 모듈 위로 올려도 된다 — `vtd_rl.rule_stack` 임포트 뒤에만 두면 된다.)

- [ ] **Step 5: 로그와 CLI.**

`vtd_rl/rl/diagnostics.py`: `TERM_KEYS` 끝에 `"block"` 을 더하고 주석에 "`block` 은 M7g 장애물 뒤 머무는 벌" 을 더한다.

`scripts/train_ppo.py` 의 `--obs-margin` 인자 바로 뒤에:

```python
    ap.add_argument("--block-profile", dest="block_profile", type=float, default=default_reward_cfg.block_profile,
                    help="장애물 뒤에 머무는 벌(M7g). 통로 안 --block-range 앞에 멈춘 장애물(사람 제외)이 있으면"
                         " 걸음마다 이 값을 깎는다. 0 이면 꺼짐")
    ap.add_argument("--block-range", dest="block_range", type=float, default=default_reward_cfg.block_range,
                    help="머무는 벌을 거는 거리[m]")
```

`_build_reward_cfg` 의 `dataclasses.replace(...)` 에 `block_profile=a.block_profile, block_range=a.block_range` 를 더하고, 독스트링의 필드 수("열여덟 필드" → "스무 필드")와 나열을 고친다. `term_*` 키 나열 주석·`_reward_terms_per_env` 독스트링의 항목 수도 `block` 을 넣어 고친다.

- [ ] **Step 6: 통과를 확인한다.**

Run: Step 2 와 같은 명령 → rc=0.
전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m7g_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 7: 커밋한다.**

```bash
git add vtd_rl/env/reward.py vtd_rl/env/drive_env.py vtd_rl/rl/diagnostics.py scripts/train_ppo.py \
  tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py
git commit -m "M7g — 장애물 뒤에 머무는 벌(block_profile)과 train_ppo --block-profile"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `block_profile=0.0`(기본)이면 보상 합과 `reward_terms` 가 예전과 같다(`block` 키 없음, `_block_gap` 을 안 부른다).
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. numpy 는 1.26.4 그대로다.

## Review Focus

- 보행자·자전거(VRU)와 움직이는 물체는 머무는 벌에서 빠진다.
- 경계: `block == block_range` 이면 벌이 없다.
- 통로·거리 계산이 `_obs_gap` 과 어긋나지 않는다.
