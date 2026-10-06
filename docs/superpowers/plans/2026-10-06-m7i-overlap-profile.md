# M7i — 통로와 겹친 폭에 비례하는 벌: 옆으로 비키는 도중에도 기울기 주기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 남은 가장 큰 손실은 장애물 충돌(course_G 4 번째 정지차·course_D, 384 판 중 17~18 판)이다. 지금까지의 장애물 벌은 둘이다. 감속 곡선(M7d)은 "줄이기" 에, 머무는 벌(M7g)은 "통로 밖으로 나가기" 에 기울기를 준다. 하지만 옆으로 반쯤 비킨 상태는 둘 다 벌이 같다(통로 안이면 똑같이 벌, 밖이면 0). course_G 에서 학생은 통로 안에 0.4~0.8 m 겹친 채 지나가다 부딪친다. M7i 는 **통로와 겹친 폭**에, 가까울수록 커지는 가중을 곱해 걸음마다 벌한다. 옆으로 조금씩 옮길수록 벌이 연속으로 준다.

**Architecture:**
- `RewardConfig`: `ovl_profile: float = 0.0`(c), `ovl_range: float = 30.0`[m]. 기본 c=0 이면 항이 없고 예전과 같다.
- `VtdDriveEnv._ovl_gap()`: 머무는 벌(`_block_gap`)과 같은 물체(높이 ≥ 0.25, 멈춤 < 0.5 m/s, 사람·자전거 아님)를 본다. 앞(fx > 0)이고 `gap < ovl_range` 인 것마다:
  - `overlap = max(0, o.width/2 + EGO_HALF_W + obs_margin − |fy|)`[m], `w = 1 − gap/ovl_range`.
  - 그중 `overlap · w` 가 가장 큰 값을 낸다. 없으면 `None`.
- `RewardShaper.step(..., ovl=None)`: `ovl_profile > 0` 이면 `terms["ovl"] = 0.0 − c · (ovl or 0)`.
- `train_ppo.py --ovl-profile`·`--ovl-range`(dest 는 필드 이름), `TERM_KEYS` 에 `ovl`.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거 `docs/reports/m7h-lim-anticipate.md`("다음"), `docs/reports/m7g-block-penalty.md`.

## 본 실행 (OMEN)

칸 `ARM ∈ {V2: 2.0, V5: 5.0}` × 시드 `S ∈ {0, 1, 2}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-06-m7i-$ARM-s$S --init runs/omen/2026-10-06-m7f-s2/policy-125440.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 --lat-free-range 80 --obs-profile 1.0 --ovl-profile $C
```

- 출발·앵커는 지금 최고 그물(M7f 시드 2 · 12.5 만)이다. 보상은 M7e 와 같고 `--ovl-profile` 만 더한다. 관측 앞당김은 끈다(효과를 섞지 않으려고).
- **대조군은 M7e**(같은 보상·변종 4, 출발은 M7d 최고 — 지금 최고는 거기서 12.5 만 걸음만 이어 돌린 그물)다.
- 크기: course_G 정지차에 0.8 m 겹친 채 30 m 를 13 m/s 로 다가가면(약 23 걸음, 가중 평균 0.5) 한 번에 c=2 에서 약 −18, c=5 에서 약 −46 이다. 0.8 m 비키면 0 이다.
- 체크포인트 48 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 지금 최고 묶음은 큰 선택 창 90.00(완주 363, course_G 52, 충돌 21)이다.

- **(가) 겹친 폭 벌이 충돌을 줄이나.** 칸마다 체크포인트 24 개의 충돌 평균을 낸다. 대조(M7e, 57.3)보다 20 판 넘게 적은 칸이 있으면 "겹친 폭 벌이 충돌을 줄인다" 로 본다. 곁들여 course_G 완주 평균(대조 39.5)을 본다.
- **(나) 고르기.** 후보는 두 칸 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물이 큰 선택 창에서 지금 최고(90.00)보다 높을 때만, **새 시험 창 v21000~v21031 × 시드 2**(384 판)에서 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

---

### Task 1: 통로와 겹친 폭 벌(`ovl_profile`)

**Files:**
- Modify: `vtd_rl/env/reward.py`(`RewardConfig.ovl_profile·ovl_range`, 검사, `RewardShaper.step(..., ovl=None)`)
- Modify: `vtd_rl/env/drive_env.py`(`_ovl_gap`, `step` 에서 `ovl=` 넘기기)
- Modify: `vtd_rl/rl/diagnostics.py`(`TERM_KEYS` 에 `"ovl"`)
- Modify: `scripts/train_ppo.py`(`--ovl-profile`, `--ovl-range`, `_build_reward_cfg`)
- Test: `tests/env/test_reward.py`, `tests/env/test_drive_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `RewardConfig.ovl_profile=0.0`, `ovl_range=30.0`.
  - `RewardShaper.step(..., block=None, ovl=None)` — `ovl_profile > 0` 일 때만 `terms["ovl"]`.
  - `VtdDriveEnv._ovl_gap() -> float | None`.
  - CLI `--ovl-profile`, `--ovl-range`(dest = `ovl_profile`, `ovl_range`).

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_reward.py` 끝에:

```python
def test_ovl_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().ovl_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "ovl" not in sh.step([], 0.0, zero, zero, "running", ovl=0.5).terms


@pytest.mark.parametrize("ovl, want", [(None, 0.0), (0.0, 0.0), (0.4, -0.8)])
def test_ovl_profile_은_겹친_폭에_비례해_깎는다(ovl, want):
    sh = RewardShaper(h_board(), RewardConfig(ovl_profile=2.0))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", ovl=ovl)
    assert out.terms["ovl"] == pytest.approx(want)
    assert str(out.terms["ovl"]) != "-0.0"
    assert out.total == pytest.approx(sum(out.terms.values()))


@pytest.mark.parametrize("kw", [{"ovl_profile": -0.1}, {"ovl_profile": float("nan")},
                                {"ovl_range": 0.0}, {"ovl_range": float("inf")}])
def test_잘못된_ovl_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)
```

`tests/env/test_drive_env.py` 끝에:

```python
def test_ovl_gap_은_겹친_폭에_가까움_가중을_곱한다(monkeypatch):
    from dataclasses import replace
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    assert env._ovl_gap() is None                     # 60 m 앞 — ovl_range 30 m 밖
    ego = env.world.ego
    car = env.state.objects[0]
    import math
    def put(fx, fy, **kw):                            # 내 차 기준 (fx, fy) 에 물체를 놓는다
        c, s = math.cos(ego.heading), math.sin(ego.heading)
        return replace(car, x=ego.x + fx * c - fy * s, y=ego.y + fx * s + fy * c, **kw)
    gap_of = lambda fx: fx - rs.score_fma.FRONT - car.length / 2.0
    lim = car.width / 2.0 + 0.943 + 0.3
    monkeypatch.setattr(env.state, "objects", [put(20.0, -1.0)])
    assert env._ovl_gap() == pytest.approx((lim - 1.0) * (1.0 - gap_of(20.0) / 30.0))
    monkeypatch.setattr(env.state, "objects", [put(20.0, -(lim + 0.1))])      # 통로 밖
    assert env._ovl_gap() is None or env._ovl_gap() == pytest.approx(0.0)
    monkeypatch.setattr(env.state, "objects", [put(20.0, -1.0, speed=3.0)])  # 움직이는 물체
    assert env._ovl_gap() is None
    monkeypatch.setattr(env.state, "objects", [put(-5.0, -1.0)])              # 뒤
    assert env._ovl_gap() is None
    env.close()


def test_ovl_profile_을_켜면_멈춘_차에_겹친_채_다가갈_때_ovl_항이_깎인다():
    def run(cfg):
        env = VtdDriveEnv([rammer_board()], EnvConfig(reward=cfg))
        env.reset(seed=0)
        vals = []
        for _ in range(400):
            _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 0.3], dtype=np.float32), "turn": 0})
            vals.append(info["reward_terms"].get("ovl"))
            if term or trunc:
                break
        env.close()
        return vals
    on = run(RewardConfig(ovl_profile=2.0))
    assert min(on) < 0.0
    assert all(v is None for v in run(RewardConfig()))
```

`tests/rl/test_train_ppo.py` 의 `test_block_profile_인자가_RewardConfig에_반영되고_hparams_로_되살아난다` 바로 뒤에:

```python
def test_ovl_profile_인자가_RewardConfig에_반영되고_hparams_로_되살아난다():
    from vtd_rl.rl.reward_cfg import reward_config_from_hparams
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "x", "--ovl-profile", "5", "--ovl-range", "40"])
    cfg = module._build_reward_cfg(a)
    assert (cfg.ovl_profile, cfg.ovl_range) == (5.0, 40.0)
    back = reward_config_from_hparams(vars(a))
    assert (back.ovl_profile, back.ovl_range) == (5.0, 40.0)
    off = module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"]))
    assert (off.ovl_profile, off.ovl_range) == (0.0, 30.0)
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m7i_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `ovl_profile` 이 없다.

- [ ] **Step 3: `reward.py`.**

`RewardConfig` 의 `block_range` 필드 바로 뒤에:

```python
    # M7i — 통로와 겹친 폭 벌. 통로 안 ovl_range 앞의 멈춘 장애물(사람·자전거 제외)마다 겹친 폭[m] × 가까움
    # 가중(1 − gap/ovl_range) 중 가장 큰 값에 ovl_profile 을 곱해 걸음마다 깎는다(항 `ovl`). 옆으로 비킬수록
    # 연속으로 준다. 기본 0 이면 꺼짐.
    ovl_profile: float = 0.0
    ovl_range: float = 30.0          # [m]
```

`__post_init__` 끝에:

```python
        if not math.isfinite(self.ovl_profile) or self.ovl_profile < 0.0:
            raise ValueError(f"ovl_profile 은 유한한 0 이상이어야 한다: {self.ovl_profile}")
        if not math.isfinite(self.ovl_range) or self.ovl_range <= 0.0:
            raise ValueError(f"ovl_range 는 유한한 양수여야 한다: {self.ovl_range}")
```

`RewardShaper.step` 시그니처에 `ovl=None` 을 `block=None` 뒤에 더하고, `block` 항 블록 바로 뒤에:

```python
        if cfg.ovl_profile > 0.0:     # 끈 실행은 항 자체가 없다
            terms["ovl"] = 0.0 - cfg.ovl_profile * (ovl or 0.0)
```

- [ ] **Step 4: `drive_env.py`.**

`step` 의 `block = ...` 줄 바로 뒤에:

```python
        ovl = self._ovl_gap() if self.cfg.reward.ovl_profile > 0.0 else None
```

`self._shaper.step(...)` 호출에 `ovl=ovl` 을 더한다. `_block_gap` 메서드 바로 뒤에:

```python
    def _ovl_gap(self):
        """통로와 겹친 폭 × 가까움 가중의 최댓값(M7i), 볼 물체가 없으면 None.

        물체는 `_block_gap` 과 같다(높이 ≥ 0.25 m, 멈춤 < 0.5 m/s, 사람·자전거 아님). 앞(fx > 0)이고
        `gap < ovl_range` 인 것마다 겹친 폭 `max(0, 물체 반폭 + 내 차 반폭 + obs_margin − |fy|)` 에
        `1 − gap/ovl_range` 를 곱한다.
        """
        if self._info is None:
            return None
        import vtd_io                                   # rule_stack 이 경로를 잡아 둔다
        ego = self.world.ego
        ch, sh = math.cos(-ego.heading), math.sin(-ego.heading)
        margin, rng = self.cfg.reward.obs_margin, self.cfg.reward.ovl_range
        best = None
        for o in self.state.objects:
            if o.height < ROAD_SURFACE_H or abs(float(o.speed)) >= BLOCK_STILL_V:
                continue
            if vtd_io.classify_object(o.length, o.width, o.height) == vtd_io.ObjectClass.VRU:
                continue
            dx, dy = o.x - ego.x, o.y - ego.y
            fx, fy = dx * ch - dy * sh, dx * sh + dy * ch
            if fx <= 0.0:
                continue
            gap = max(0.0, fx - rs.score_fma.FRONT - o.length / 2.0)
            if gap >= rng:
                continue
            overlap = max(0.0, o.width / 2.0 + EGO_HALF_W + margin - abs(fy))
            val = overlap * (1.0 - gap / rng)
            if best is None or val > best:
                best = val
        return best
```

- [ ] **Step 5: 로그와 CLI.**

`vtd_rl/rl/diagnostics.py`: `TERM_KEYS` 끝에 `"ovl"` 을 더하고 주석에 "`ovl` 은 M7i 통로와 겹친 폭" 을 더한다.

`scripts/train_ppo.py` 의 `--block-range` 인자 바로 뒤에:

```python
    ap.add_argument("--ovl-profile", dest="ovl_profile", type=float, default=default_reward_cfg.ovl_profile,
                    help="통로와 겹친 폭 벌(M7i). 통로 안 --ovl-range 앞 멈춘 장애물과 겹친 폭[m] × 가까움 가중에"
                         " 이 값을 곱해 걸음마다 깎는다. 0 이면 꺼짐")
    ap.add_argument("--ovl-range", dest="ovl_range", type=float, default=default_reward_cfg.ovl_range,
                    help="겹친 폭 벌을 거는 거리[m]")
```

`_build_reward_cfg` 의 `dataclasses.replace(...)` 에 `ovl_profile=a.ovl_profile, ovl_range=a.ovl_range` 를 더하고, 독스트링의 필드 수("스무 필드" → "스물두 필드")와 나열을 고친다. `term_*` 키 나열 주석·`_reward_terms_per_env` 독스트링의 항목 수도 `ovl` 을 넣어 고친다.

- [ ] **Step 6: 통과를 확인한다.**

Run: Step 2 와 같은 명령 → rc=0. 전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m7i_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 7: 커밋한다.**

```bash
git add vtd_rl/env/reward.py vtd_rl/env/drive_env.py vtd_rl/rl/diagnostics.py scripts/train_ppo.py \
  tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py
git commit -m "M7i — 통로와 겹친 폭 벌(ovl_profile)과 train_ppo --ovl-profile"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `ovl_profile=0.0`(기본)이면 보상 합과 `reward_terms` 가 예전과 같다(`ovl` 키 없음, `_ovl_gap` 을 안 부른다).
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. numpy 는 1.26.4 그대로다.

## Review Focus

- 겹친 폭은 통로 밖이면 0 이고, 옆으로 비킬수록 연속으로 준다.
- 가까움 가중은 `gap → ovl_range` 에서 0, `gap → 0` 에서 1 이다.
- 사람·자전거·움직이는 물체·뒤의 물체는 보지 않는다.
