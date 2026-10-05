# M7d — 앞길 위 장애물 감속 곡선: 장애물에 반응하게 하기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6w 이후 모든 최고 그물이 course_G 4 번째 정지차에서 부딪친다. 실패한 판 하나(course_G@v1106, 시드 0)를 학생과 선생님으로 걸음 단위로 추적했다(`m7d_trace.py`).

- **선생님:** 정지차가 경로 위(물체 옆 거리 −0.1~−0.4 m)에 있으면 25 m 앞에서 13 m/s → 2 m/s 로 크게 줄인다. 그다음 천천히 왼쪽 차로로 옮겨(횡 +3.2 m) 지나가고 돌아온다.
- **학생(지금 최고, M7b Fz 시드 1 · 50 만):** 이 구간 내내 경로 왼쪽 0.8~1.0 m 에 머문 채 13 m/s 로 달린다. 정지차가 보여도 줄이지도 더 비키지도 않는다. 정지차 중심이 내 차 기준 오른쪽 1.4 m 에 있어 0.4 m 겹친 채 부딪친다.
- 큰 선택 창에서 course_G 실패 12 판은 모두 이 정지차가 경로 쪽에 가깝게(횡 −0.3~−0.6) 선 판이다. 멀리(−1.0 안팎) 선 판은 고정된 치우침만으로 넘는다. **학생은 장애물에 반응하지 않는다.**

M7d 는 적색 감속 곡선(M6v·M6w)과 같은 꼴의 벌을 장애물에 준다. 차 앞 통로 안에 물체가 있으면, 그 앞에 설 수 있는 속도를 넘은 만큼 걸음마다 벌한다. 속도를 줄이거나 통로에서 비켜나면 벌이 사라진다. 그래서 감속과 비키기 양쪽으로 기울기가 생긴다.

**Architecture:**
- `RewardConfig`: `obs_profile: float = 0.0`(c), `obs_decel: float = 2.0`[m/s²], `obs_buffer: float = 3.0`[m], `obs_margin: float = 0.3`[m]. 기본 c=0 이면 항이 없고 예전과 같다.
- `obs_excess(obs, decel, buffer) -> float`: `obs = (gap, v_close)` 또는 `None`. `max(0, v_close − sqrt(2·decel·max(gap − buffer, 0)))`.
- `VtdDriveEnv._obs_gap()`: 걸음 끝 상태에서, 내 차 기준(뒷축, 진행 방향 x)으로 물체마다:
  - 높이 < 0.25 m(노면 물체, `vtd_io.classify_object` 의 ROAD_SURFACE 경계)는 뺀다.
  - 앞(fx > 0)이고 통로 안(`|fy| < o.width/2 + HALF_W + obs_margin`, HALF_W = 0.943)인 것만 본다.
  - `gap = max(0, fx − FRONT − o.length/2)`(FRONT = `score_fma.FRONT` 3.808), `v_close = ego.v − max(0, o.speed·cos(o.heading − ego.heading))`.
  - 그런 물체 중 `gap` 이 가장 작은 것의 `(gap, v_close)` 를 낸다. 없으면 `None`.
- `RewardShaper.step(..., obs=None)`: `obs_profile > 0` 이면 `terms["obs"] = 0.0 − c · obs_excess(obs, obs_decel, obs_buffer)`.
- `train_ppo.py --obs-profile`·`--obs-decel`·`--obs-buffer`·`--obs-margin`(dest 는 필드 이름과 같다), `TERM_KEYS` 에 `obs`.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거 위 추적과 `docs/reports/m7c-hard-courses.md`.

## 본 실행 (OMEN)

칸 `ARM ∈ {O3: 0.3, O10: 1.0}` × 시드 `S ∈ {0, 1, 2}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-06-m7d-$ARM-s$S --init runs/omen/2026-10-05-m7b-Fz-s1/policy-501760.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 --lat-free-range 80 --obs-profile $C
```

- **대조군은 M7c 칸 C**(같은 출발·보상, 장애물 곡선 없음)다. 이미 쟀다.
- 크기: 13 m/s 로 통로 안 정지차에 다가가면 약 30 m 앞부터 곡선을 넘는다. 부딪칠 때까지 약 40 걸음 동안 넘은 속도가 평균 약 5 m/s 다. c=0.3 이면 한 번에 약 −60, c=1.0 이면 약 −200 이다.
- 체크포인트 48 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 지금 최고 그물은 큰 선택 창 87.98(완주 358, 충돌 22)이다.

- **(가) 장애물 곡선이 충돌을 줄이나.** 칸마다 체크포인트 24 개의 충돌 평균을 낸다. O3 또는 O10 이 대조(M7c C, 53.9)보다 20 판 넘게 적으면 "장애물 곡선이 충돌을 줄인다" 로 본다.
  - 곁들여 course_G 완주 평균과 정체 평균을 본다(줄이기만 하고 못 지나가면 정체가 는다).
- **(나) 고르기.** 후보는 두 칸 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v16000~v16031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

---

### Task 1: 앞길 위 장애물 감속 곡선(`obs_profile`)

**Files:**
- Modify: `vtd_rl/env/reward.py`(`obs_excess`, `RewardConfig.obs_*`, 검사, `RewardShaper.step(..., obs=None)`)
- Modify: `vtd_rl/env/drive_env.py`(`_obs_gap`, `step` 에서 `obs=` 넘기기)
- Modify: `vtd_rl/rl/diagnostics.py`(`TERM_KEYS` 에 `"obs"`)
- Modify: `scripts/train_ppo.py`(`--obs-profile`·`--obs-decel`·`--obs-buffer`·`--obs-margin`, `_build_reward_cfg`)
- Test: `tests/env/test_reward.py`, `tests/env/test_drive_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `vtd_rl.env.reward.obs_excess(obs, decel, buffer) -> float`.
  - `RewardConfig.obs_profile=0.0`, `obs_decel=2.0`, `obs_buffer=3.0`, `obs_margin=0.3`.
  - `RewardShaper.step(..., intent=None, red=None, lane=None, lat=None, obs=None)` — `obs_profile > 0` 일 때만 `terms["obs"]`.
  - `VtdDriveEnv._obs_gap() -> tuple[float, float] | None`.
  - CLI `--obs-profile`, `--obs-decel`, `--obs-buffer`, `--obs-margin`(dest = `obs_profile`, `obs_decel`, `obs_buffer`, `obs_margin`).

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_reward.py`: 위 임포트에 `obs_excess` 를 더한다. 끝에:

```python
@pytest.mark.parametrize("obs, want", [
    (None, 0.0),
    ((60.0, 13.0), 0.0),                      # v_ok = sqrt(4·57) ≈ 15.1
    ((12.0, 10.0), 10.0 - 6.0),               # v_ok = sqrt(4·9) = 6
    ((2.0, 3.0), 3.0),                        # buffer 안: v_ok = 0
    ((2.0, 0.0), 0.0),                        # 서 있으면 벌 없음
    ((12.0, -2.0), 0.0),                      # 멀어지는 중
])
def test_obs_excess(obs, want):
    assert obs_excess(obs, 2.0, 3.0) == pytest.approx(want)


def test_obs_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().obs_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "obs" not in sh.step([], 0.0, zero, zero, "running", obs=(12.0, 10.0)).terms


def test_obs_profile_은_넘은_속도에_비례해_깎는다():
    sh = RewardShaper(h_board(), RewardConfig(obs_profile=0.5))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", obs=(12.0, 10.0))
    assert out.terms["obs"] == pytest.approx(-0.5 * 4.0)
    assert out.total == pytest.approx(sum(out.terms.values()))
    calm = sh.step([], 0.0, zero, zero, "running", obs=None)
    assert calm.terms["obs"] == 0.0 and str(calm.terms["obs"]) == "0.0"


@pytest.mark.parametrize("kw", [{"obs_profile": -0.1}, {"obs_profile": float("nan")}, {"obs_decel": 0.0},
                                {"obs_decel": float("inf")}, {"obs_buffer": -1.0}, {"obs_margin": -0.1},
                                {"obs_margin": float("nan")}])
def test_잘못된_obs_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)
```

`tests/env/test_drive_env.py` 끝에(정지 차량은 `rammer_board()` 가 경로 60 m 지점에 세운다):

```python
def test_obs_gap_은_앞길_위_물체의_거리와_다가가는_속도를_준다():
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    gap, v_close = env._obs_gap()
    # 경로 60 m 의 차(길이 4.5) — 뒷축 s≈0 에서 앞범퍼(3.808)·차 반길이(2.25)를 뺀 거리
    assert 50.0 < gap < 56.0
    assert v_close == pytest.approx(env.world.ego.v, abs=1e-6)
    env.close()
    empty = VtdDriveEnv([boards()[0]])
    empty.reset(seed=0)
    empty.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    assert empty._obs_gap() is None
    empty.close()


def test_obs_profile_을_켜면_정지차에_달려들_때_obs_항이_깎인다():
    def run(cfg):
        env = VtdDriveEnv([rammer_board()], EnvConfig(reward=cfg))
        env.reset(seed=0)
        vals = []
        for _ in range(300):
            _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 1.0], dtype=np.float32), "turn": 0})
            vals.append(info["reward_terms"].get("obs"))
            if term or trunc:
                break
        env.close()
        return vals
    on = run(RewardConfig(obs_profile=0.5))
    assert min(on) < 0.0
    assert all(v is None for v in run(RewardConfig()))
```

`tests/rl/test_train_ppo.py` 의 `test_실패_벌_인자는_hparams_로도_되살아난다` 바로 뒤에:

```python
def test_obs_profile_인자가_RewardConfig에_반영되고_hparams_로_되살아난다():
    from vtd_rl.rl.reward_cfg import reward_config_from_hparams
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "x", "--obs-profile", "1.0", "--obs-decel", "2.5",
                                           "--obs-buffer", "4", "--obs-margin", "0.2"])
    cfg = module._build_reward_cfg(a)
    assert (cfg.obs_profile, cfg.obs_decel, cfg.obs_buffer, cfg.obs_margin) == (1.0, 2.5, 4.0, 0.2)
    back = reward_config_from_hparams(vars(a))
    assert (back.obs_profile, back.obs_margin) == (1.0, 0.2)
    off = module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"]))
    assert (off.obs_profile, off.obs_decel, off.obs_buffer, off.obs_margin) == (0.0, 2.0, 3.0, 0.3)
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m7d_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `obs_excess` 임포트 실패.

- [ ] **Step 3: `reward.py`.**

`lat_excess` 함수 바로 뒤에:

```python
def obs_excess(obs, decel: float, buffer: float) -> float:
    """앞길 위 물체 `(gap, v_close)` → 감속 곡선을 넘은 속도[m/s]. None 이면 0(M7d).

    `gap` 은 앞범퍼에서 물체 뒤끝까지[m], `v_close` 는 다가가는 속도[m/s](멀어지면 음수).
    허용 속도 `v_ok = sqrt(2·decel·max(gap − buffer, 0))` 는 물체 buffer 앞에 서는 등감속 곡선이다.
    """
    if obs is None:
        return 0.0
    gap, v_close = obs
    v_ok = math.sqrt(2.0 * decel * max(float(gap) - buffer, 0.0))
    return max(0.0, float(v_close) - v_ok)
```

`RewardConfig` 의 `stall` 필드 바로 뒤에:

```python
    # M7d — 앞길 위 장애물 감속 곡선. 차 앞 통로(물체 반폭 + 내 차 반폭 + obs_margin) 안의 가장 가까운
    # 물체 앞에 설 수 있는 속도(`obs_excess`)를 넘으면 걸음마다 넘은 속도[m/s] × obs_profile 을 깎는다
    # (항 `obs`). 기본 0 이면 꺼짐(항도 없다).
    obs_profile: float = 0.0
    obs_decel: float = 2.0           # [m/s²]
    obs_buffer: float = 3.0          # [m] — 물체 이만큼 앞에 서는 곡선
    obs_margin: float = 0.3          # [m] — 통로 여유
```

`__post_init__` 끝에:

```python
        if not math.isfinite(self.obs_profile) or self.obs_profile < 0.0:
            raise ValueError(f"obs_profile 은 유한한 0 이상이어야 한다: {self.obs_profile}")
        if not math.isfinite(self.obs_decel) or self.obs_decel <= 0.0:
            raise ValueError(f"obs_decel 은 유한한 양수여야 한다: {self.obs_decel}")
        for name in ("obs_buffer", "obs_margin"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} 는 유한한 0 이상이어야 한다: {v}")
```

`RewardShaper.step` 시그니처에 `obs=None` 을 `lat=None` 뒤에 더하고, `stall` 항 블록 바로 뒤에:

```python
        if cfg.obs_profile > 0.0:     # 끈 실행은 항 자체가 없다
            terms["obs"] = 0.0 - cfg.obs_profile * obs_excess(obs, cfg.obs_decel, cfg.obs_buffer)
```

- [ ] **Step 4: `drive_env.py`.**

모듈 위 상수 자리(`RUNNING = "running"` 아래)에:

```python
EGO_HALF_W = 0.943               # 내 차 반폭[m] — 채점기 run_logger.EGO_HALF_W 와 같다(M7d 장애물 통로)
ROAD_SURFACE_H = 0.25            # 이보다 낮은 물체는 노면 물체 — vtd_io.classify_object 의 경계(M7d)
```

`step` 의 `lat = ...` 줄 바로 뒤에:

```python
        obs = self._obs_gap() if self.cfg.reward.obs_profile > 0.0 else None
```

`self._shaper.step(...)` 호출에 `obs=obs` 를 더한다. `_lat_gap` 메서드 바로 뒤에:

```python
    def _obs_gap(self):
        """차 앞 통로 안 가장 가까운 물체의 `(앞범퍼~물체 뒤끝 거리[m], 다가가는 속도[m/s])`, 없으면 None(M7d).

        내 차 기준(뒷축, 진행 방향 x)으로 물체를 돌려, 앞(fx > 0)에 있고 옆 거리가
        `물체 반폭 + 내 차 반폭 + obs_margin` 안인 것만 본다. 노면 물체(높이 < 0.25 m)는 뺀다.
        """
        if self._info is None:
            return None
        ego = self.world.ego
        ch, sh = math.cos(-ego.heading), math.sin(-ego.heading)
        margin = self.cfg.reward.obs_margin
        best = None
        for o in self.state.objects:
            if o.height < ROAD_SURFACE_H:
                continue
            dx, dy = o.x - ego.x, o.y - ego.y
            fx, fy = dx * ch - dy * sh, dx * sh + dy * ch
            if fx <= 0.0 or abs(fy) >= o.width / 2.0 + EGO_HALF_W + margin:
                continue
            gap = max(0.0, fx - rs.score_fma.FRONT - o.length / 2.0)
            if best is None or gap < best[0]:
                v_close = float(ego.v) - max(0.0, float(o.speed) * math.cos(o.heading - ego.heading))
                best = (gap, v_close)
        return best
```

- [ ] **Step 5: 로그와 CLI.**

`vtd_rl/rl/diagnostics.py`: `TERM_KEYS` 끝에 `"obs"` 를 더하고 주석에 "`obs` 는 M7d 장애물 감속 곡선" 을 더한다.

`scripts/train_ppo.py` 의 `--stall-penalty` 인자 바로 뒤에:

```python
    ap.add_argument("--obs-profile", dest="obs_profile", type=float, default=default_reward_cfg.obs_profile,
                    help="앞길 위 장애물 감속 곡선 벌(M7d). 통로 안 가장 가까운 물체 앞에 설 수 있는 속도를 넘으면"
                         " 걸음마다 넘은 속도[m/s] × 이 값을 깎는다. 0 이면 꺼짐")
    ap.add_argument("--obs-decel", dest="obs_decel", type=float, default=default_reward_cfg.obs_decel,
                    help="장애물 감속 곡선의 감속[m/s²]")
    ap.add_argument("--obs-buffer", dest="obs_buffer", type=float, default=default_reward_cfg.obs_buffer,
                    help="장애물 감속 곡선이 겨누는 정지 위치 — 물체 뒤끝 이만큼 앞[m]")
    ap.add_argument("--obs-margin", dest="obs_margin", type=float, default=default_reward_cfg.obs_margin,
                    help="장애물 통로 여유[m] — 물체 반폭 + 내 차 반폭에 더한다")
```

`_build_reward_cfg` 의 `dataclasses.replace(...)` 에 `obs_profile=a.obs_profile, obs_decel=a.obs_decel, obs_buffer=a.obs_buffer, obs_margin=a.obs_margin` 을 더하고, 독스트링의 필드 수("열네 필드" → "열여덟 필드")와 나열을 고친다. `term_*` 키 나열 주석·`_reward_terms_per_env` 독스트링의 항목 수도 `obs` 를 넣어 고친다.

- [ ] **Step 6: 통과를 확인한다.**

Run: Step 2 와 같은 명령 → rc=0.
전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m7d_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 7: 커밋한다.**

```bash
git add vtd_rl/env/reward.py vtd_rl/env/drive_env.py vtd_rl/rl/diagnostics.py scripts/train_ppo.py \
  tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py
git commit -m "M7d — 앞길 위 장애물 감속 곡선 보상(obs_profile)과 train_ppo --obs-profile"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `obs_profile=0.0`(기본)이면 보상 합과 `reward_terms` 가 예전과 같다(`obs` 키 없음, `_obs_gap` 을 안 부른다).
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. numpy 는 1.26.4 그대로다.

## Review Focus

- 옆 차로의 물체(통로 밖)는 벌하지 않는다. 같은 방향으로 같은 속도로 달리는 앞차는 다가가는 속도가 0 이라 벌이 없다.
- 노면 물체(높이 < 0.25 m)는 뺀다.
- 이미 겹친 물체(`fx − FRONT − 반길이 < 0`)는 거리 0 으로 둔다.
- CLI dest 가 `RewardConfig` 필드 이름과 같아 `hparams` 로 되살아난다.
