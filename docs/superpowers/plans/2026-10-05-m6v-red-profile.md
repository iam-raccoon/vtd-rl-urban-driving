# M6v — 적색 감속 곡선: 서는 쪽으로 가는 길에 기울기를 준다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6u 에서 적색 위반 벌을 5 배·10 배로 키워도 적색 중대 위반은 그대로였다(완주 판당 약 3.5 건). 학습 중 판당 위반 항(−125·−210)도 줄지 않았다. 정책이 끝까지 서 보지 않으니, 덜 빨리 지나가도 벌이 같아서 감속 쪽으로 가는 기울기가 없다. M6v 는 적색 신호 앞에서 "정지선 앞에 설 수 있는 속도" 를 넘은 만큼 걸음마다 벌을 준다. 반쯤 줄인 감속도 벌을 줄이므로 서는 쪽으로 가는 길이 생긴다.

**Architecture:**
- 감속 곡선: 앞범퍼에서 정지선까지 거리 `gap` 에서 허용 속도는 `v_ok = sqrt(2 · red_decel · max(gap − 1, 0))` 다(정지선 앞 1 m 에 서는 등감속 곡선).
  - 신호가 적색이고 `gap ≥ −1`(채점기가 "넘었다" 고 보기 전)이면, 걸음마다 `−red_profile · max(0, v − v_ok)` 를 보상 항 `red` 로 더한다.
  - 멈춰 있으면(`v = 0`) 벌이 없다. 그래서 어디서 서든 곡선 벌은 0 이다. 정지선 바로 앞(2 m 안)에 서야 하는 것은 기존 ⑦ 항(경미·중대)이 맡는다.
- `RewardConfig` 에 `red_profile: float = 0.0`, `red_decel: float = 2.0` 을 더한다. 기본 0 이면 `red` 항이 없고 예전과 비트 단위로 같다.
- `VtdDriveEnv._red_gap()` 이 걸음 끝 상태에서 `(gap, v)` 를 낸다. 신호가 적색이 아니면 `None` 이다.
  - 신호 정지선 s 는 세계가 보고하는 신호(`World.reporter.signals` 에서 같은 `tl_id`)에서 찾는다. 앞범퍼 오프셋은 채점기와 같은 `score_fma.FRONT`(3.808 m)다.
- `train_ppo.py --red-profile C --red-decel A` 를 연다. 판당 누적 로그(`TERM_KEYS`)에 `red` 를 더한다.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거는 `docs/reports/m6u-red-light.md`("다음").

## 크기 맞추기

- 결정 걸음은 0.1 초다. 12 m/s 로 적색을 지나가면 `v_ok` 가 12 m/s 아래로 내려가는 곳(정지선 약 40 m 앞)부터 약 30 걸음 동안 넘은 속도가 0 → 12 m/s 로 자란다. 평균 약 5 m/s 다.
  - `red_profile = 0.1` 이면 판정 한 번에 약 −15, `0.3` 이면 약 −45 다.
- 기다리는 비용은 약 −4 다(M6u 계획의 계산). 곡선 벌이 그보다 크므로 서는 쪽이 낫다.
- 감속 2 m/s² 는 차의 제동 한계(−5 m/s²)보다 넉넉히 작다. 적색으로 바뀔 때 정지선이 가까우면(딜레마 구간) 피할 수 없는 벌이 생긴다. 황색(3 초)을 보고 미리 줄이는 것을 배우는 신호가 된다.

## 본 실행 (OMEN)

칸 `ARM ∈ {R1: 0.1, R3: 0.3}` × 시드 `S ∈ {0, 1, 2}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-05-m6v-$ARM-s$S --init runs/omen/2026-10-05-m6r-s4/policy-501760.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 250000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --red-profile $C
```

- M6u 칸 K5 와 **`--red-profile` 만** 다르다. 대조군은 M6u K5(곡선 없음)와 M6s 시드 0~2(배율도 없음)다. 둘 다 이미 채점표로 쟀다.
- 체크포인트(25 만 간격 넷)마다 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(완주 못 한 판 0 점, 384 판 평균). 지금 최고 그물은 82.36 이다.

- **(가) 적색을 지키나.** 고른 그물의 큰 선택 창 적색 중대 위반이 완주 판 하나당 1.0 건 이하이면 "적색을 지킨다" 로 본다(지금 최고 3.63).
  - 곁들여 칸마다 체크포인트 12 개의 적색 중대 평균과, 학습 로그의 판당 `red`·`violation` 항이 줄어드는지 본다.
- **(나) 고르기.** 후보는 R1·R3 체크포인트 24 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v8000~v8031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

## 이 실험이 답하지 않는 것

- 곡선 벌은 대회 채점에 없는 항이다. 적색을 지키게 되면, 그 뒤 곡선 벌을 끄고 이어 돌려도 지키는지는 따로 봐야 한다.
- 황색에 미리 서게 하는 벌은 안 넣는다.

---

### Task 1: 적색 감속 곡선 보상(`red_profile`)

**Files:**
- Modify: `vtd_rl/env/reward.py`(`RED_STOP_TARGET`, `red_excess`, `RewardConfig.red_profile·red_decel`, 검사, `RewardShaper.step(..., red=None)`)
- Modify: `vtd_rl/env/drive_env.py`(`_red_gap`, `step` 에서 `red=` 넘기기)
- Modify: `vtd_rl/rl/diagnostics.py`(`TERM_KEYS` 에 `"red"`)
- Modify: `scripts/train_ppo.py`(`--red-profile`, `--red-decel`, `_build_reward_cfg`)
- Test: `tests/env/test_reward.py`, `tests/env/test_drive_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `vtd_rl.env.reward.red_excess(red, decel) -> float` — `red` 는 `(gap[m], v[m/s])` 또는 `None`.
  - `RewardConfig.red_profile: float = 0.0`, `RewardConfig.red_decel: float = 2.0`.
  - `RewardShaper.step(hits, ds, action, prev_action, outcome, intent=None, red=None)` — `red_profile > 0` 일 때만 `terms["red"]` 가 생긴다.
  - `VtdDriveEnv._red_gap() -> tuple[float, float] | None`.
  - CLI `--red-profile`, `--red-decel`.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_reward.py` 끝에(파일 위 임포트를 `from vtd_rl.env.reward import COLLISION_ITEMS, RewardConfig, RewardShaper, ViolationTracker, red_excess` 로 바꾼다):

```python
@pytest.mark.parametrize("red, want", [
    (None, 0.0),
    ((50.0, 10.0), 0.0),                     # v_ok = sqrt(4·49) = 14 > 10
    ((10.0, 10.0), 10.0 - 6.0),              # v_ok = sqrt(4·9) = 6
    ((0.5, 0.0), 0.0),                       # 정지선 앞에 서 있다
    ((0.5, 3.0), 3.0),                       # 1 m 안쪽: v_ok = 0
    ((-0.5, 3.0), 3.0),                      # 앞범퍼가 선을 조금 넘었지만 아직 '넘었다' 가 아니다
    ((-2.0, 10.0), 0.0),                     # 이미 넘었다 — ⑦ 이 맡는다
])
def test_red_excess(red, want):
    assert red_excess(red, 2.0) == pytest.approx(want)


def test_red_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().red_profile == 0.0
    b = h_board()
    sh = RewardShaper(b, RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", red=(10.0, 10.0))
    assert "red" not in out.terms


def test_red_profile_은_넘은_속도에_비례해_깎는다():
    b = h_board()
    cfg = RewardConfig(red_profile=0.1)
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", red=(10.0, 10.0))
    assert out.terms["red"] == pytest.approx(-0.1 * 4.0)
    assert out.total == pytest.approx(sum(out.terms.values()))
    calm = sh.step([], 0.0, zero, zero, "running", red=None)
    assert calm.terms["red"] == 0.0


@pytest.mark.parametrize("kw", [{"red_profile": -0.1}, {"red_profile": float("nan")},
                                {"red_decel": 0.0}, {"red_decel": -1.0}, {"red_decel": float("inf")}])
def test_잘못된_red_설정은_거부한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)
```

`tests/env/test_drive_env.py` 끝에:

```python
def _signal_board(mode):
    """H 0~250 — s≈96 m 에 신호(151)가 있다. 신호 운용만 바꾼다."""
    return slice_board(load_board(H), 0.0, 250.0, f"H_0_250_{mode}", signals=mode)


def test_red_gap_은_적색일_때만_앞범퍼_거리와_속도를_준다():
    env = VtdDriveEnv([_signal_board("always_red")])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    gap, v = env._red_gap()
    sig_s = min(sig.s for sig in env.world.reporter.signals)
    assert gap == pytest.approx(sig_s - env._info.s - rs.score_fma.FRONT)
    assert v == pytest.approx(env.world.ego.v)
    env.close()
    green = VtdDriveEnv([_signal_board("always_green")])
    green.reset(seed=0)
    green.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    assert green._red_gap() is None
    green.close()


def test_red_profile_을_켜면_적색을_달려_지날_때_red_항이_깎인다():
    def run(cfg):
        env = VtdDriveEnv([_signal_board("always_red")], EnvConfig(reward=cfg))
        env.reset(seed=0)
        reds = []
        for _ in range(400):
            _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 1.0], dtype=np.float32), "turn": 0})
            reds.append(info["reward_terms"].get("red"))
            if term or trunc or info["s"] > 110.0:
                break
        env.close()
        return reds
    on = run(RewardConfig(red_profile=0.1))
    assert min(on) < 0.0
    off = run(RewardConfig())
    assert all(r is None for r in off)
```

`tests/rl/test_train_ppo.py` 의 `test_item_scale_인자의_항목이_틀리면_거부한다` 바로 뒤에:

```python
def test_red_profile_인자가_RewardConfig에_반영된다():
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args([
        "--out", "/tmp/불필요-존재안함", "--red-profile", "0.3", "--red-decel", "2.5"])
    cfg = module._build_reward_cfg(a)
    assert cfg.red_profile == 0.3 and cfg.red_decel == 2.5
    off = module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"]))
    assert off.red_profile == 0.0 and off.red_decel == RewardConfig().red_decel
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m6v_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `red_excess` 임포트 실패.

- [ ] **Step 3: `reward.py` 를 고친다.**

`COLLISION_ITEMS` 줄 아래에:

```python
RED_STOP_TARGET = 1.0            # M6v — 적색 감속 곡선이 겨누는 정지 위치: 앞범퍼가 정지선 앞 1 m
                                 # (채점기 ⑦ 정지 인정 구간 [0, 2 m) 의 가운데)


def red_excess(red, decel: float) -> float:
    """적색 접근 `(gap, v)` → 감속 곡선을 넘은 속도[m/s]. `red` 가 None(적색 아님)이면 0.

    `gap` 은 앞범퍼에서 정지선까지[m](앞이 +). 허용 속도 `v_ok = sqrt(2·decel·max(gap−1, 0))`
    는 정지선 앞 1 m 에 서는 등감속 곡선이다. 채점기가 '넘었다' 고 보는 −1 m 너머는 ⑦ 이 맡으므로 0.
    """
    if red is None:
        return 0.0
    gap, v = red
    if gap < -1.0:
        return 0.0
    v_ok = math.sqrt(2.0 * decel * max(gap - RED_STOP_TARGET, 0.0))
    return max(0.0, float(v) - v_ok)
```

`RewardConfig` 의 `item_scale` 필드 바로 뒤에(`__post_init__` 앞):

```python
    # M6v — 적색 감속 곡선. 적색 신호 앞에서 허용 속도(`red_excess`)를 넘으면 걸음마다
    # 넘은 속도[m/s] × red_profile 을 깎는다(항 `red`). 기본 0 이면 꺼짐(항도 없다, 예전과 같다).
    red_profile: float = 0.0
    red_decel: float = 2.0           # 곡선의 감속[m/s²]
```

`__post_init__` 끝(마지막 `object.__setattr__` 뒤)에:

```python
        if not math.isfinite(self.red_profile) or self.red_profile < 0.0:
            raise ValueError(f"red_profile 은 유한한 0 이상이어야 한다: {self.red_profile}")
        if not math.isfinite(self.red_decel) or self.red_decel <= 0.0:
            raise ValueError(f"red_decel 은 유한한 양수여야 한다: {self.red_decel}")
```

`RewardShaper.step` 의 시그니처를 `def step(self, hits, ds: float, action, prev_action, outcome: str, intent=None, red=None) -> RewardStep:` 로 바꾸고, `terms = {...}` 바로 뒤(`return` 앞)에:

```python
        if cfg.red_profile > 0.0:     # 끈 실행은 항 자체가 없다(합·로그가 예전과 같다)
            terms["red"] = -cfg.red_profile * red_excess(red, cfg.red_decel)
```

- [ ] **Step 4: `drive_env.py` 를 고친다.**

임포트에 `from vtd_rl.world.signals import PASSED_MARGIN` 을 더한다. `step` 의 `shaped = self._shaper.step(...)` 호출을 다음으로 바꾼다:

```python
        red = self._red_gap() if self.cfg.reward.red_profile > 0.0 else None
        shaped = self._shaper.step(hits, max(0.0, self._info.s - s0), action, self._prev_action,
                                   outcome, intent=self.intent, red=red)
```

`_frozen_step` 앞에 메서드를 더한다:

```python
    def _red_gap(self):
        """신호가 적색이면 `(앞범퍼~그 신호 정지선 거리[m], 속도[m/s])`, 아니면 None — M6v 감속 곡선용.

        정지선 s 는 세계가 보고하는 신호(`SignalReporter` 와 같은 규칙: 같은 tl_id, 뒷축이 선을
        PASSED_MARGIN 넘기 전)에서 찾고, 앞범퍼 오프셋은 채점기와 같은 `score_fma.FRONT` 다.
        """
        st = self.state
        if self._info is None or st.tl_id <= 0 or st.tl_state != rs.TL_RED:
            return None
        s = self._info.s
        for sig in self.world.reporter.signals:          # s 순서
            if sig.tl_id == st.tl_id and sig.s + PASSED_MARGIN >= s:
                return sig.s - s - rs.score_fma.FRONT, float(self.world.ego.v)
        return None
```

- [ ] **Step 5: 로그와 CLI.**

`vtd_rl/rl/diagnostics.py`: `TERM_KEYS = ("progress", "time", "violation", "comfort", "red")`. 바로 위 주석 끝에 "(`red` 는 M6v 감속 곡선 — 끈 실행은 0)" 을 더한다. 기존 테스트가 키 목록을 못박았으면 함께 고친다.

`scripts/train_ppo.py` 의 `--item-scale` 인자 바로 뒤에:

```python
    ap.add_argument("--red-profile", type=float, default=default_reward_cfg.red_profile,
                    help="적색 감속 곡선 벌(M6v). 적색 신호 앞에서 정지선 앞 1 m 에 설 수 있는 속도를"
                         " 넘으면 걸음마다 넘은 속도[m/s] × 이 값을 깎는다. 0 이면 꺼짐")
    ap.add_argument("--red-decel", type=float, default=default_reward_cfg.red_decel,
                    help="적색 감속 곡선의 감속[m/s²]")
```

`_build_reward_cfg` 의 `dataclasses.replace(...)` 에 `red_profile=a.red_profile, red_decel=a.red_decel` 을 더하고, 독스트링의 "다섯 필드" 를 "일곱 필드" 로, 나열에 두 필드를 더한다.

- [ ] **Step 6: 통과를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py tests/rl/test_diagnostics.py -q -m "not slow" > $TMP/m6v_t1.txt 2>&1; echo rc=$?` → rc=0.
전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m6v_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 7: 커밋한다.**

```bash
git add vtd_rl/env/reward.py vtd_rl/env/drive_env.py vtd_rl/rl/diagnostics.py scripts/train_ppo.py \
  tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py
git commit -m "M6v — 적색 감속 곡선 보상(red_profile)과 train_ppo --red-profile"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `red_profile=0.0`(기본)이면 보상 합과 `reward_terms` 가 예전과 같아야 한다(`red` 키 없음).
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다.
- numpy 는 1.26.4 그대로다.

## Review Focus

- 적색이 아닌 걸음(녹색·황색·신호 없음)에서 `red` 항이 0 이다.
- 선을 넘은 뒤(`gap < −1`)에는 곡선 벌이 멈춘다 — ⑦ 과 겹쳐 두 번 벌하지 않는다.
- 같은 tl_id 를 쓰는 신호가 경로에 둘 이상이면, 아직 안 지난 첫 신호를 고른다.
- 판이 끝난 걸음(`_frozen_step`)은 보상 0 그대로다.
- `red_profile=0` 이면 `_red_gap` 을 부르지도 않는다(기본 경로 그대로).
