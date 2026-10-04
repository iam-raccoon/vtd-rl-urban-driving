# M6x — 차로 유지: 채점기가 세는 차로 침범 깊이에 걸음마다 벌 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6w 의 새 최고 그물은 완주 판당 차로 유지(항목 3) 중대가 3.9 건이다(단계 ② 에서 선생님 0). 보상에는 차로 가운데를 지키라는 항이 없다. 항목 3 은 구간마다 최종 등급 한 번(최대 −6)이라, 한 구간에서 두 번 물리고 나면 더 물려도 공짜다. M6x 는 채점기가 세는 차로 경계 침범의 **깊이**에 걸음마다 벌을 준다. 적색 감속 곡선(M6v·M6w)처럼 반쯤 고친 것도 벌을 줄이게 해서 기울기를 만든다.

**Architecture:**
- `LaneGeometryJudge`(우리 코드, `vtd_rl/referee/judges/lane_geometry.py`)는 ③ 을 판정하려고 행을 2.5 초 붙잡아 둔다. 풀어 줄 때 `_edge_pred` 로 "이 행이 ③ 물림인가" 를 본다. 이 판단에는 앞뒤 2.5 초의 차로변경 사건과 차로변경 구역 면책이 이미 들어 있다.
  - `_edge_pred` 가 참일 때 그 행의 `lane_intrusion − LANE_EDGE_M`(0.1 m 넘는 깊이)을 `self.edge_excess` 에 쌓는다. 판정(Hit)은 바뀌지 않는다.
- `Referee.take_edge_excess()` 는 판정기들이 쌓은 값을 돌려주고 0 으로 되돌린다.
- `RewardConfig.lane_profile: float = 0.0`(c). 기본 0 이면 항이 없고 예전과 같다.
  - 켜면 `VtdDriveEnv.step` 이 걸음 끝에 `take_edge_excess() × 세계 dt(0.05 s)` = 깊이×시간[m·s] 를 넘긴다. `RewardShaper.step(..., lane=)` 이 `terms["lane"] = 0.0 − c × lane` 을 더한다.
  - 벌은 행이 풀리는 2.5 초 뒤에 온다. 판이 끝날 때 남은 행은 `referee.finish()` 가 풀어 마지막 걸음에 들어간다(그 걸음은 `finish` 뒤에 `take_edge_excess` 를 부른다).
- `train_ppo.py --lane-profile C`, 판당 누적 로그 `TERM_KEYS` 에 `lane`.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거는 `docs/reports/m6w-red-hold.md`("다음").

## 사전 측정(끝남)

단계 ② 원본 판 6 개(시드 0)에서, 새 최고 그물의 ③ 히트 270 개가 난 곳을 분류했다(`m6x_lanetrace.py`):

| 곳 | 히트 | 비율 |
|---|---:|---:|
| 교차로 앞뒤 3 초 안 | 117 | 43% |
| 직선 | 110 | 41% |
| 저속(<3 m/s, 서기·출발) | 21 | 8% |
| 교차로 안 | 12 | 4% |
| 곡선 | 10 | 4% |

- 270 개 중 261 개(97%)가 경로 **오른쪽**으로 벗어난 것이다(평균 |횡오차| 0.6~0.8 m, 속도 약 11 m/s). 앞선 최고 그물도 116 개 중 90 개가 오른쪽이다.
- 차로를 바꾸며 다른 차도로 넘어가는 것은 채점기가 ③ 에서 뺀다(차로변경 사건 ±2.5 초). 그래서 ③ 은 대개 오른쪽 길 가장자리를 무는 것이다(추정).

프레임마다 `lane_intrusion` 초과분(>0.1 m)의 판당 합[m·s]을, 채점기가 세는 것과 빼는 것(차로변경 ±2.5 초·변경 구역)으로 나눴다(`m6x_intrusion.py`):

| 그물 | 단계 ② 6 판: 세는 것 | 단계 ② 6 판: 빼는 것 | 큰 선택 창 v1100 24 판: 세는 것 | 큰 선택 창 v1100 24 판: 빼는 것 |
|---|---:|---:|---:|---:|
| 선생님 | 0.47 | 6.32 | 1.97 | 20.69 |
| 앞선 최고(M6r) | 3.97 | 7.81 | 14.94 | 14.04 |
| 새 최고(M6w) | 8.52 | 11.44 | 19.50 | 16.65 |

- 선생님도 차로변경 앞뒤로는 침범이 크다(판당 6~21 m·s). 침범 깊이를 그대로 벌하면 정상적인 차로변경까지 벌한다. 그래서 채점기가 **세는** 침범만 벌한다.
- 크기: c=5 이면 새 최고 기준 판당 약 −40(단계 ②)~−100(③ 전체), 선생님 기준 약 −2~−10 이다.

## 본 실행 (OMEN)

칸 `ARM ∈ {C: 0, L5: 5, L20: 20}` × 시드 `S ∈ {0, 1, 2}` = 9 실행, 동시 3 개씩 세 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-05-m6x-$ARM-s$S --init runs/omen/2026-10-05-m6w-W10-s0/policy-750080.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lane-profile $C
```

- 출발·앵커 참조는 **새 최고 그물**(M6w W10 시드 0 · 75 만)이다. 앵커 참조는 `--init` 그물이라 저절로 그렇게 된다.
- 보상은 M6w W10 과 같고 `--lane-profile` 만 더한다. 칸 C 는 차로 벌 없이 이어 돌리는 대조다.
- 체크포인트 72 개(실행당 8 개) 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 새 최고 그물은 큰 선택 창 84.61, 차로 유지 중대 3.91(완주 판당)이다.

- **(가) 차로를 지키나.** 칸마다 체크포인트 24 개의 차로 유지 중대 평균을 낸다. L5 또는 L20 칸 평균이 C 칸 평균보다 1.0 건 넘게 낮으면 "차로 벌이 차로 유지를 줄인다" 로 본다.
- **(나) 고르기.** 후보는 세 칸 체크포인트 72 개 전부다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v10000~v10031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.
- 곁들여: 적색 중대가 다시 늘지 않는지(새 최고 1.85) 칸마다 본다.

## 이 실험이 답하지 않는 것

- 장애물을 비킬 때의 침범과 그냥 치우친 침범을 가르지 않는다. 채점기가 세는 것은 다 벌한다.
- 벌이 2.5 초 늦게 오는 것이 학습을 얼마나 늦추는지는 따로 안 잰다.

---

### Task 1: 채점기가 세는 차로 침범 깊이 벌(`lane_profile`)

**Files:**
- Modify: `vtd_rl/referee/judges/lane_geometry.py`(`edge_excess` 쌓기)
- Modify: `vtd_rl/referee/core.py`(`Referee.take_edge_excess`)
- Modify: `vtd_rl/env/reward.py`(`RewardConfig.lane_profile`, 검사, `RewardShaper.step(..., lane=None)`)
- Modify: `vtd_rl/env/drive_env.py`(`step` 에서 `lane=` 넘기기)
- Modify: `vtd_rl/rl/diagnostics.py`(`TERM_KEYS` 에 `"lane"`)
- Modify: `scripts/train_ppo.py`(`--lane-profile`, `_build_reward_cfg`)
- Test: `tests/referee/test_lane_geometry_synthetic.py`, `tests/referee/test_core.py`, `tests/env/test_reward.py`, `tests/env/test_drive_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `LaneGeometryJudge.edge_excess: float` — `_edge_pred` 가 참인 행마다 `lane_intrusion − sf.LANE_EDGE_M` 를 더한 합[m·행].
  - `Referee.take_edge_excess() -> float` — 판정기 중 `edge_excess` 를 가진 것들의 합을 돌려주고 0 으로 되돌린다.
  - `RewardConfig.lane_profile: float = 0.0`, `RewardShaper.step(..., intent=None, red=None, lane=None)` — `lane_profile > 0` 일 때만 `terms["lane"]`.
  - CLI `--lane-profile`.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/referee/test_lane_geometry_synthetic.py` 끝에:

```python
def _excess(rows, lc_at):
    ctx = SimpleNamespace(secs=_Secs(), lc_at=lc_at)
    j = LaneGeometryJudge(ctx)
    for r in copy.deepcopy(rows):
        j.step(r)
    j.finish()
    return j.edge_excess


def test_edge_excess_는_채점기가_세는_물림만_쌓는다():
    # 물림 8 행(깊이 0.5 m) — 2.5 초 안에 차로변경이 오면 면책돼 0, 멀면 8 × (0.5 − 0.1).
    assert _excess(_edge_then_change_rows(gap=1.0), _no_lc) == 0.0
    far = _excess(_edge_then_change_rows(gap=3.0), _no_lc)
    assert abs(far - 8 * (0.5 - sf.LANE_EDGE_M)) < 1e-9


def test_edge_excess_를_쌓아도_판정은_그대로다():
    rng = random.Random(11)
    for _ in range(50):
        rows = _gen(rng)
        assert _judge(rows, _no_lc) == _score(rows, _no_lc)
```

`tests/referee/test_core.py` 끝에(필요한 임포트는 파일 위에 있는 것을 쓰고, 없으면 `from types import SimpleNamespace` 를 더한다):

```python
def test_take_edge_excess_는_합을_돌려주고_비운다():
    from vtd_rl.referee.core import Referee
    ref = Referee.__new__(Referee)
    ref.judges = [SimpleNamespace(edge_excess=1.5), SimpleNamespace(), SimpleNamespace(edge_excess=0.25)]
    assert ref.take_edge_excess() == 1.75
    assert ref.judges[0].edge_excess == 0.0 and ref.judges[2].edge_excess == 0.0
    assert ref.take_edge_excess() == 0.0
```

`tests/env/test_reward.py` 끝에:

```python
def test_lane_profile_기본값은_꺼짐이고_항이_없다():
    assert RewardConfig().lane_profile == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "lane" not in sh.step([], 0.0, zero, zero, "running", lane=2.0).terms


def test_lane_profile_은_침범_깊이_시간에_비례해_깎는다():
    sh = RewardShaper(h_board(), RewardConfig(lane_profile=5.0))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], 0.0, zero, zero, "running", lane=0.3)
    assert out.terms["lane"] == pytest.approx(-1.5)
    assert out.total == pytest.approx(sum(out.terms.values()))
    calm = sh.step([], 0.0, zero, zero, "running", lane=None)
    assert calm.terms["lane"] == 0.0 and str(calm.terms["lane"]) == "0.0"   # −0.0 이 아니다


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_잘못된_lane_profile_은_거부한다(bad):
    with pytest.raises(ValueError):
        RewardConfig(lane_profile=bad)
```

`tests/env/test_drive_env.py` 끝에:

```python
def test_lane_profile_을_켜면_오른쪽으로_치우쳐_달릴_때_lane_항이_깎인다():
    """H 0~250 을 오른쪽으로 조금씩 꺾으며 달려 길 가장자리를 문다 — 켜면 어느 걸음엔가 lane < 0."""
    def run(cfg):
        env = VtdDriveEnv([boards()[0]], EnvConfig(reward=cfg))
        env.reset(seed=0)
        lanes = []
        for k in range(300):
            steer = -0.05 if k > 40 else 0.0
            _o, _r, term, trunc, info = env.step({"control": np.array([steer, 0.3], dtype=np.float32), "turn": 0})
            lanes.append(info["reward_terms"].get("lane"))
            if term or trunc:
                break
        env.close()
        return lanes
    on = run(RewardConfig(lane_profile=5.0))
    assert min(on) < 0.0
    assert all(v is None for v in run(RewardConfig()))
```

`tests/rl/test_train_ppo.py` 의 `test_red_profile_인자가_RewardConfig에_반영된다` 바로 뒤에:

```python
def test_lane_profile_인자가_RewardConfig에_반영된다():
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "/tmp/불필요-존재안함", "--lane-profile", "20"])
    assert module._build_reward_cfg(a).lane_profile == 20.0
    assert module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"])).lane_profile == 0.0
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/referee/test_lane_geometry_synthetic.py tests/referee/test_core.py tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m6x_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `edge_excess`·`take_edge_excess`·`lane_profile` 이 없다.

- [ ] **Step 3: 판정기에 쌓기.**

`vtd_rl/referee/judges/lane_geometry.py` 의 `LaneGeometryJudge.__init__` 끝에:

```python
        # M6x — 보상용(판정에는 안 쓴다). ③ 물림으로 세는 행(`_edge_pred` 가 참)마다 0.1 m 를 넘는
        # 침범 깊이를 더한다. 행은 2.5 초 붙잡혔다 풀릴 때 판정되므로 차로변경 면책이 이미 들어 있다.
        self.edge_excess = 0.0
```

`_edge_pred` 의 마지막 `return not any(...)` 를 다음으로 바꾼다:

```python
        if any(abs(r["t"] - ct) < LC_WIN for ct in self.changes):
            return False
        self.edge_excess += r["lane_intrusion"] - sf.LANE_EDGE_M
        return True
```

(`SpanTracker.update` 는 행마다 `pred` 를 정확히 한 번 부른다. `_edge_pred` 를 부르는 곳이 그것뿐인지 확인한다.)

- [ ] **Step 4: 심판에서 꺼내기.**

`vtd_rl/referee/core.py` 의 `Referee.finish` 뒤에:

```python
    def take_edge_excess(self) -> float:
        """M6x — 판정기들이 쌓은 ③ 침범 깊이 합[m·행]을 돌려주고 0 으로 되돌린다(보상용)."""
        total = 0.0
        for j in self.judges:
            if hasattr(j, "edge_excess"):
                total += j.edge_excess
                j.edge_excess = 0.0
        return total
```

- [ ] **Step 5: 보상.**

`vtd_rl/env/reward.py` 의 `RewardConfig` 에서 `red_decel` 필드 바로 뒤에:

```python
    # M6x — 차로 침범 깊이 벌. 채점기가 ③ 물림으로 세는 행의 0.1 m 넘는 깊이×시간[m·s] × lane_profile 을
    # 걸음마다 깎는다(항 `lane`, 행이 풀리는 2.5 초 뒤에 온다). 기본 0 이면 꺼짐(항도 없다).
    lane_profile: float = 0.0
```

`__post_init__` 끝에:

```python
        if not math.isfinite(self.lane_profile) or self.lane_profile < 0.0:
            raise ValueError(f"lane_profile 은 유한한 0 이상이어야 한다: {self.lane_profile}")
```

`RewardShaper.step` 시그니처에 `lane=None` 을 `red=None` 뒤에 더하고, `red` 항 블록 바로 뒤에:

```python
        if cfg.lane_profile > 0.0:    # 끈 실행은 항 자체가 없다
            terms["lane"] = 0.0 - cfg.lane_profile * (lane or 0.0)   # 0.0− : 침범 없는 걸음이 −0.0 이 안 되게
```

- [ ] **Step 6: 환경.**

`vtd_rl/env/drive_env.py` 의 `step` 에서 `red = ...` 줄 바로 뒤에:

```python
        lane = (self.referee.take_edge_excess() * self.cfg.world.dt
                if self.cfg.reward.lane_profile > 0.0 else None)       # [m·행] × 프레임 dt = [m·s]
```

`self._shaper.step(...)` 호출에 `lane=lane` 을 더한다. (`referee.finish()` 를 부르는 줄이 이보다 위에 있어 마지막 걸음에는 풀린 행까지 들어간다 — 순서를 확인한다.)

- [ ] **Step 7: 로그와 CLI.**

`vtd_rl/rl/diagnostics.py`: `TERM_KEYS = ("progress", "time", "violation", "comfort", "red", "lane")`, 주석에 "(`lane` 은 M6x 차로 침범 깊이 — 끈 실행은 0)" 을 더한다.

`scripts/train_ppo.py` 의 `--red-decel` 인자 바로 뒤에:

```python
    ap.add_argument("--lane-profile", type=float, default=default_reward_cfg.lane_profile,
                    help="차로 침범 깊이 벌(M6x). 채점기가 ③ 물림으로 세는 0.1 m 넘는 깊이×시간[m·s] 에"
                         " 이 값을 곱해 깎는다(2.5 초 늦게 온다). 0 이면 꺼짐")
```

`_build_reward_cfg` 의 `dataclasses.replace(...)` 에 `lane_profile=a.lane_profile` 을 더하고, 독스트링의 "일곱 필드" 를 "여덟 필드" 로, 나열에 `a.lane_profile` 을 더한다. `term_*` 키를 나열한 주석(`terms.stats()` 줄, `_reward_terms_per_env` 독스트링의 항목 수)도 `lane` 을 넣어 고친다.

- [ ] **Step 8: 통과를 확인한다.**

Run: Step 2 와 같은 명령 → rc=0.
판정 일치 테스트 전부: `env -u PYTHONPATH .venv/bin/python -m pytest tests/referee -q > $TMP/m6x_ref.txt 2>&1; echo rc=$?` → rc=0(느린 것 포함, 시간이 걸리면 파일별로 나눈다).
전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m6x_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 9: 커밋한다.**

```bash
git add vtd_rl/referee/judges/lane_geometry.py vtd_rl/referee/core.py vtd_rl/env/reward.py vtd_rl/env/drive_env.py \
  vtd_rl/rl/diagnostics.py scripts/train_ppo.py tests/referee/test_lane_geometry_synthetic.py tests/referee/test_core.py \
  tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py
git commit -m "M6x — 채점기가 세는 차로 침범 깊이 벌(lane_profile)과 train_ppo --lane-profile"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- 심판의 판정(Hit)은 한 글자도 바뀌면 안 된다 — `tests/referee` 의 판정 일치 테스트가 모두 그대로 통과해야 한다.
- `lane_profile=0.0`(기본)이면 보상 합과 `reward_terms` 가 예전과 같다(`lane` 키 없음).
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. numpy 는 1.26.4 그대로다.

## Review Focus

- `_edge_pred` 가 행마다 한 번만 불리는지(두 번 불리면 깊이가 두 배로 쌓인다).
- 판이 끝나는 걸음: `referee.finish()` 가 붙잡힌 행을 풀고, 그 뒤에 `take_edge_excess()` 를 불러야 마지막 2.5 초가 빠지지 않는다.
- `take_edge_excess` 를 안 부르는 기본 실행(`lane_profile=0`)에서 판정기 값이 쌓이기만 하는 것은 해가 없는지(판정에 안 쓰인다).
- 리셋마다 새 `Referee` 를 지으므로 판 사이에 값이 새지 않는지.
