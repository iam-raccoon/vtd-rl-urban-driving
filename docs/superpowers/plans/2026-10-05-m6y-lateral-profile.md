# M6y — 차로 유지: 경로 기준 횡오차에 그 걸음에 바로 오는 벌(물체가 가까우면 끔) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6x 에서 채점기가 세는 차로 침범 깊이에 벌을 줬지만 차로 유지가 안 줄었다. 그 벌은 2.5 초 늦게 왔다. M6y 는 그 걸음에 바로 오는 기울기를 준다. 경로 기준 횡오차가 0.4 m 를 넘은 만큼 벌한다. 장애물을 비킬 때는 경로를 벗어나야 하므로, 40 m 안에 물체가 있으면 끈다.

**Architecture:**
- `RewardConfig` 에 `lat_profile: float = 0.0`(c), `lat_deadband: float = 0.4`[m], `lat_free_range: float = 40.0`[m] 를 더한다. 기본 c=0 이면 항이 없고 예전과 같다.
- `VtdDriveEnv._lat_gap()` 은 걸음 끝 상태에서 `(|횡오차|, 가장 가까운 물체까지 거리)` 를 낸다. 물체가 없으면 거리는 `inf` 다.
- `RewardShaper.step(..., lat=None)`: 물체 거리 > `lat_free_range` 이면 `terms["lat"] = 0.0 − c · max(0, |횡오차| − lat_deadband)`, 아니면 0.
- `train_ppo.py --lat-profile C`(`--lat-deadband`·`--lat-free-range` 도 연다), 판당 누적 로그 `TERM_KEYS` 에 `lat`.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거는 `docs/reports/m6x-lane-profile.md`("다음").

## 사전 측정(끝남)

경로 기준 횡오차(`m6y_lateral.py`): 결정 걸음마다(속도 > 3 m/s), 40 m 안에 물체가 없는 걸음만.

| 그물 | 판 | 평균 횡오차 | \|횡오차\| 중앙값 | \|횡오차\| 90% | 0.4 m 넘는 걸음 | 0.6 m 넘는 걸음 |
|---|---|---:|---:|---:|---:|---:|
| 선생님 | 단계 ② 원본 6 판 | −0.022 | 0.012 | 0.200 | 3.4% | 1.2% |
| 새 최고(M6w) | 단계 ② 원본 6 판 | **−0.480** | 0.426 | 1.203 | **53.5%** | 29.9% |
| 선생님 | 큰 선택 창 v1100 24 판 | −0.024 | 0.014 | 0.219 | 3.5% | 1.4% |
| 새 최고(M6w) | 큰 선택 창 v1100 24 판 | **−0.427** | 0.420 | 1.383 | **52.4%** | 32.0% |

- 선생님은 경로를 거의 그대로 따라간다. 학생은 평균 약 0.45 m **오른쪽**(− 쪽)으로 치우쳐 달린다. M6x 의 ③ 히트 97% 가 오른쪽이던 것과 맞는다.
- 0.4 m 문턱이면 선생님은 걸음의 3.5% 만 벌을 받는다(차로변경 끝자락 등).
- 크기: 학생의 문턱 초과분은 걸음당 평균 약 0.2 m 다(추정). 단계 ② 판 하나가 약 3,000 걸음이니 c=0.1 이면 판당 약 −60, c=0.5 이면 약 −300 이다.

## 본 실행 (OMEN)

칸 `ARM ∈ {A1: 0.1, A5: 0.5}` × 시드 `S ∈ {0, 1, 2}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-05-m6y-$ARM-s$S --init runs/omen/2026-10-05-m6w-W10-s0/policy-750080.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile $C
```

- M6x 칸 C(같은 출발·앵커·보상, 횡오차 벌 없음)와 `--lat-profile` 만 다르다. **대조군은 M6x 칸 C** 다(이미 쟀다).
- 체크포인트 48 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 지금 최고 그물은 큰 선택 창 84.61, 차로 유지 중대 3.91 이다.

- **(가) 차로를 지키나.** 칸마다 체크포인트 24 개의 차로 유지 중대 평균을 낸다. A1 또는 A5 칸 평균이 대조(M6x C, 4.14)보다 1.0 건 넘게 낮으면 "횡오차 벌이 차로 유지를 줄인다" 로 본다.
  - 곁들여 학습 로그의 판당 `lat` 항이 줄었는지 본다.
- **(나) 고르기.** 후보는 A1·A5 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v11000~v11031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

## 이 실험이 답하지 않는 것

- 횡오차 벌은 대회 채점에 없는 항이다. 경로가 차로 가운데라는 가정에 기댄다(선생님 측정으로는 맞다).
- 물체 근처 40 m 안에서의 침범은 다루지 않는다.

---

### Task 1: 경로 기준 횡오차 벌(`lat_profile`)

**Files:**
- Modify: `vtd_rl/env/reward.py`(`lat_excess`, `RewardConfig.lat_profile·lat_deadband·lat_free_range`, 검사, `RewardShaper.step(..., lat=None)`)
- Modify: `vtd_rl/env/drive_env.py`(`_lat_gap`, `step` 에서 `lat=` 넘기기)
- Modify: `vtd_rl/rl/diagnostics.py`(`TERM_KEYS` 에 `"lat"`)
- Modify: `scripts/train_ppo.py`(`--lat-profile`, `--lat-deadband`, `--lat-free-range`, `_build_reward_cfg`)
- Test: `tests/env/test_reward.py`, `tests/env/test_drive_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `vtd_rl.env.reward.lat_excess(lat, deadband, free_range) -> float` — `lat` 는 `(|횡오차|[m], 물체 거리[m])` 또는 `None`.
  - `RewardConfig.lat_profile: float = 0.0`, `lat_deadband: float = 0.4`, `lat_free_range: float = 40.0`.
  - `RewardShaper.step(..., intent=None, red=None, lane=None, lat=None)` — `lat_profile > 0` 일 때만 `terms["lat"]`.
  - `VtdDriveEnv._lat_gap() -> tuple[float, float] | None`.
  - CLI `--lat-profile`, `--lat-deadband`, `--lat-free-range`.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_reward.py`: 파일 위 임포트에 `lat_excess` 를 더한다(`from vtd_rl.env.reward import ..., red_excess, lat_excess`). 끝에:

```python
@pytest.mark.parametrize("lat, want", [
    (None, 0.0),
    ((0.3, 999.0), 0.0),          # 문턱 안
    ((0.9, 999.0), 0.5),          # 0.9 − 0.4
    ((0.9, 30.0), 0.0),           # 40 m 안에 물체 — 끈다
    ((0.9, float("inf")), 0.5),   # 물체 없음
])
def test_lat_excess(lat, want):
    assert lat_excess(lat, 0.4, 40.0) == pytest.approx(want)


def test_lat_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().lat_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "lat" not in sh.step([], 0.0, zero, zero, "running", lat=(1.0, 999.0)).terms


def test_lat_profile_은_문턱_넘은_횡오차에_비례해_깎는다():
    sh = RewardShaper(h_board(), RewardConfig(lat_profile=0.5))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", lat=(1.0, 999.0))
    assert out.terms["lat"] == pytest.approx(-0.5 * 0.6)
    assert out.total == pytest.approx(sum(out.terms.values()))
    calm = sh.step([], 0.0, zero, zero, "running", lat=(0.1, 999.0))
    assert calm.terms["lat"] == 0.0 and str(calm.terms["lat"]) == "0.0"


@pytest.mark.parametrize("kw", [{"lat_profile": -0.1}, {"lat_profile": float("nan")},
                                {"lat_deadband": -0.1}, {"lat_deadband": float("inf")},
                                {"lat_free_range": -1.0}, {"lat_free_range": float("nan")}])
def test_잘못된_lat_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)
```

`tests/env/test_drive_env.py` 끝에:

```python
def test_lat_gap_은_횡오차_크기와_물체_거리를_준다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    lat, near = env._lat_gap()
    assert lat == pytest.approx(abs(env._info.lateral))
    assert near == float("inf")                     # 이 판에는 물체가 없다
    env.close()
    ram = VtdDriveEnv([rammer_board()])
    ram.reset(seed=0)
    ram.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    _lat, near = ram._lat_gap()
    assert 50.0 < near < 70.0                       # 60 m 앞 정지 차량
    ram.close()


def test_lat_profile_을_켜면_경로를_벗어날_때_lat_항이_깎인다():
    def run(cfg):
        env = VtdDriveEnv([boards()[0]], EnvConfig(reward=cfg))
        env.reset(seed=0)
        lats = []
        for k in range(120):
            steer = -0.03 if k > 20 else 0.0
            _o, _r, term, trunc, info = env.step({"control": np.array([steer, 0.3], dtype=np.float32), "turn": 0})
            lats.append(info["reward_terms"].get("lat"))
            if term or trunc:
                break
        env.close()
        return lats
    on = run(RewardConfig(lat_profile=0.5))
    assert min(on) < 0.0
    assert all(v is None for v in run(RewardConfig()))
```

`tests/rl/test_train_ppo.py` 의 `test_lane_profile_인자가_RewardConfig에_반영된다` 바로 뒤에:

```python
def test_lat_profile_인자가_RewardConfig에_반영된다():
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "/tmp/불필요-존재안함", "--lat-profile", "0.5",
                                           "--lat-deadband", "0.3", "--lat-free-range", "50"])
    cfg = module._build_reward_cfg(a)
    assert (cfg.lat_profile, cfg.lat_deadband, cfg.lat_free_range) == (0.5, 0.3, 50.0)
    off = module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"]))
    assert (off.lat_profile, off.lat_deadband, off.lat_free_range) == (0.0, 0.4, 40.0)
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m6y_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `lat_excess` 임포트 실패.

- [ ] **Step 3: `reward.py`.**

`red_excess` 함수 바로 뒤에:

```python
def lat_excess(lat, deadband: float, free_range: float) -> float:
    """`(|횡오차|, 가장 가까운 물체 거리)` → 문턱을 넘은 횡오차[m]. None 이거나 물체가 가까우면 0.

    장애물을 비킬 때는 경로를 벗어나야 하므로 `free_range` 안에 물체가 있으면 벌하지 않는다(M6y).
    """
    if lat is None:
        return 0.0
    dev, near = lat
    if near <= free_range:
        return 0.0
    return max(0.0, float(dev) - deadband)
```

`RewardConfig` 의 `lane_profile` 필드 바로 뒤에:

```python
    # M6y — 경로 기준 횡오차 벌. 앞뒤 lat_free_range 안에 물체가 없을 때 |횡오차| 가 lat_deadband 를
    # 넘은 만큼[m] × lat_profile 을 그 걸음에 깎는다(항 `lat`). 기본 0 이면 꺼짐(항도 없다).
    lat_profile: float = 0.0
    lat_deadband: float = 0.4        # [m] — 선생님은 장애물 없는 길에서 걸음의 3.5% 만 넘는다
    lat_free_range: float = 40.0     # [m]
```

`__post_init__` 끝에:

```python
        if not math.isfinite(self.lat_profile) or self.lat_profile < 0.0:
            raise ValueError(f"lat_profile 은 유한한 0 이상이어야 한다: {self.lat_profile}")
        if not math.isfinite(self.lat_deadband) or self.lat_deadband < 0.0:
            raise ValueError(f"lat_deadband 는 유한한 0 이상이어야 한다: {self.lat_deadband}")
        if not math.isfinite(self.lat_free_range) or self.lat_free_range < 0.0:
            raise ValueError(f"lat_free_range 는 유한한 0 이상이어야 한다: {self.lat_free_range}")
```

`RewardShaper.step` 시그니처에 `lat=None` 을 `lane=None` 뒤에 더하고, `lane` 항 블록 바로 뒤에:

```python
        if cfg.lat_profile > 0.0:     # 끈 실행은 항 자체가 없다
            terms["lat"] = 0.0 - cfg.lat_profile * lat_excess(lat, cfg.lat_deadband, cfg.lat_free_range)
```

- [ ] **Step 4: `drive_env.py`.**

`import math` 가 없으면 더한다. `step` 의 `lane = ...` 줄 바로 뒤에:

```python
        lat = self._lat_gap() if self.cfg.reward.lat_profile > 0.0 else None
```

`self._shaper.step(...)` 호출에 `lat=lat` 를 더한다. `_red_gap` 메서드 바로 뒤에:

```python
    def _lat_gap(self):
        """`(|경로 기준 횡오차|[m], 가장 가까운 물체까지 거리[m])` — M6y 횡오차 벌용. 물체가 없으면 거리 inf."""
        if self._info is None:
            return None
        ego = self.world.ego
        near = min((math.hypot(o.x - ego.x, o.y - ego.y) for o in self.state.objects), default=math.inf)
        return abs(float(self._info.lateral)), float(near)
```

- [ ] **Step 5: 로그와 CLI.**

`vtd_rl/rl/diagnostics.py`: `TERM_KEYS = ("progress", "time", "violation", "comfort", "red", "lane", "lat")`, 주석에 "`lat` 은 M6y 횡오차" 를 더한다.

`scripts/train_ppo.py` 의 `--lane-profile` 인자 바로 뒤에:

```python
    ap.add_argument("--lat-profile", type=float, default=default_reward_cfg.lat_profile,
                    help="경로 기준 횡오차 벌(M6y). 물체가 --lat-free-range 안에 없을 때 |횡오차| 가"
                         " --lat-deadband 를 넘은 만큼[m] × 이 값을 그 걸음에 깎는다. 0 이면 꺼짐")
    ap.add_argument("--lat-deadband", type=float, default=default_reward_cfg.lat_deadband,
                    help="횡오차 벌의 문턱[m]")
    ap.add_argument("--lat-free-range", type=float, default=default_reward_cfg.lat_free_range,
                    help="이 거리[m] 안에 물체가 있으면 횡오차 벌을 끈다")
```

`_build_reward_cfg` 의 `dataclasses.replace(...)` 에 `lat_profile=a.lat_profile, lat_deadband=a.lat_deadband, lat_free_range=a.lat_free_range` 를 더하고, 독스트링 "여덟 필드" 를 "열한 필드" 로, 나열에 세 필드를 더한다. `term_*` 키를 나열한 주석과 `_reward_terms_per_env` 독스트링의 항목 수도 `lat` 을 넣어 고친다.

- [ ] **Step 6: 통과를 확인한다.**

Run: Step 2 와 같은 명령 → rc=0.
전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m6y_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 7: 커밋한다.**

```bash
git add vtd_rl/env/reward.py vtd_rl/env/drive_env.py vtd_rl/rl/diagnostics.py scripts/train_ppo.py \
  tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py
git commit -m "M6y — 경로 기준 횡오차 벌(lat_profile, 물체 가까우면 끔)과 train_ppo --lat-profile"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `lat_profile=0.0`(기본)이면 보상 합과 `reward_terms` 가 예전과 같다(`lat` 키 없음, `_lat_gap` 을 안 부른다).
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. numpy 는 1.26.4 그대로다.

## Review Focus

- 물체가 `lat_free_range` 안이면 벌이 0 이다(장애물 회피를 벌하지 않는다). 경계값(`near == free_range`)은 끈다.
- 물체 목록이 빈 판에서 거리가 `inf` 이고 예외가 없다.
- `lat` 키가 기본 실행에 생기지 않는다.
