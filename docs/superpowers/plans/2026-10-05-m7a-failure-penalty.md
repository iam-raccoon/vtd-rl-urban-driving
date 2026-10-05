# M7a — 실패 벌 키우기: 충돌·이탈 벌을 크게, 정체를 벌이 있는 실패로 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6z 에서 운전 품질은 거의 선생님 수준이 됐다(완주 판 점수 95.4, 단계 ② 98.5). 하지만 이어 돌릴수록 충돌·이탈·정체가 늘어 완주가 흔들렸다. 벌 항을 더하면서 판 하나의 되돌림이 음수가 됐다(기본 +8.7 → −36 ~ −118). 그런데 충돌·이탈 벌은 −50 그대로이고 정체는 벌이 없다. M7a 는 실패 벌을 키워 완주를 되찾는지 본다.

**Architecture:**
- `RewardConfig` 에 `stall: float = 0.0` 을 더한다. 기본 0 이면 예전과 같다(정체는 잘리기만 하고 항도 없다).
  - 0 이 아니면 정체(`outcome == "stalled"`)를 **실패로 끝낸다**(`terminated=True`, `truncated=False`). 그 걸음에 `terms["stall"] = stall` 을 더한다.
  - 시간 초과(`timeout`)는 그대로 잘리기만 한다.
- `RewardConfig.__post_init__` 에서 `collision`·`offroad`·`stall` 이 유한한 0 이하인지 검사한다.
- `train_ppo.py` 에 `--collision-penalty`·`--offroad-penalty`·`--stall-penalty` 를 연다. 이미 있던 `collision`·`offroad` 필드와 새 `stall` 필드로 잇는다.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거는 `docs/reports/m6z-continue-lat.md`("다음").

## 본 실행 (OMEN)

칸 `ARM ∈ {C: 기본(−50·−50·0), F200: −200·−200·−200, F500: −500·−500·−500}` × 시드 `S ∈ {0, 1, 2}` = 9 실행, 동시 3 개씩 세 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-05-m7a-$ARM-s$S --init runs/omen/2026-10-05-m6z-s2/policy-501760.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 $FAIL
```

- `$FAIL` 은 C 에서 비고, F200 은 `--collision-penalty -200 --offroad-penalty -200 --stall-penalty -200`, F500 은 같은 꼴로 −500 이다.
- 출발·앵커는 M6z 고른 그물(시드 2 · 50 만)이다. 운전 품질이 가장 좋은 그물이다(시험 창 완주 판 점수 95.4).
- 체크포인트 72 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 출발 그물은 큰 선택 창 87.69(완주 353), 지금 최고 그물(M6w W10 시드 0 · 75 만)은 84.61(완주 369)이다.

- **(가) 실패 벌이 완주를 지키나.** 칸마다 체크포인트 24 개의 완주 평균을 낸다. F200 또는 F500 칸 평균이 C 칸 평균보다 20 판(384 판 중) 넘게 많으면 "실패 벌이 완주를 지킨다" 로 본다.
  - 곁들여 칸마다 종료 사유(충돌·이탈·정체)의 합과 완주 판 점수 평균을 본다.
- **(나) 고르기.** 후보는 세 칸 체크포인트 72 개 전부다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v13000~v13031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

## 이 실험이 답하지 않는 것

- 시간 초과에 벌을 주는 것은 안 본다(시간 초과는 지금 드물다).
- 실패 벌이 너무 크면 정책이 지나치게 조심스러워질 수 있다(느린 주행, 시간 초과). 종료 사유로 본다.

---

### Task 1: 실패 벌(`--collision-penalty`·`--offroad-penalty`·`--stall-penalty`, 정체를 실패로)

**Files:**
- Modify: `vtd_rl/env/reward.py`(`RewardConfig.stall`, 검사, `RewardShaper.step` 의 `stall` 항)
- Modify: `vtd_rl/env/drive_env.py`(`step` 의 `terminated`)
- Modify: `scripts/train_ppo.py`(`--collision-penalty`, `--offroad-penalty`, `--stall-penalty`, `_build_reward_cfg`)
- Test: `tests/env/test_reward.py`, `tests/env/test_drive_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `RewardConfig.stall: float = 0.0`. `stall != 0` 이면 정체는 실패로 끝나고 그 걸음에 `terms["stall"]`.
  - CLI `--collision-penalty`(기본 `RewardConfig().collision`), `--offroad-penalty`(기본 `RewardConfig().offroad`), `--stall-penalty`(기본 0).

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_reward.py` 끝에:

```python
def test_stall_기본값은_0이고_항이_없다():
    assert RewardConfig().stall == 0.0
    sh = RewardShaper(h_board(), RewardConfig())
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    assert "stall" not in sh.step([], 0.0, zero, zero, "stalled").terms


def test_stall_을_주면_정체한_걸음에만_깎는다():
    sh = RewardShaper(h_board(), RewardConfig(stall=-200.0))
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    run = sh.step([], 0.0, zero, zero, "running")
    assert run.terms["stall"] == 0.0 and str(run.terms["stall"]) == "0.0"
    out = sh.step([], 0.0, zero, zero, "stalled")
    assert out.terms["stall"] == -200.0
    assert out.total == pytest.approx(sum(out.terms.values()))


@pytest.mark.parametrize("kw", [{"stall": 1.0}, {"stall": float("nan")}, {"collision": 10.0},
                                {"collision": float("-inf")}, {"offroad": 5.0}, {"offroad": float("nan")}])
def test_실패_벌은_유한한_0_이하여야_한다(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)
```

`tests/env/test_drive_env.py` 끝에:

```python
def _stall_run(cfg):
    """H 0~250 에서 가속 0 으로 서 있는다 — 세계의 정체 판정(stall_seconds)까지 간다."""
    env = VtdDriveEnv([boards()[0]], EnvConfig(world=WorldConfig(stall_seconds=3.0), reward=cfg))
    env.reset(seed=0)
    for _ in range(200):
        _o, r, term, trunc, info = env.step({"control": np.array([0.0, -1.0], dtype=np.float32), "turn": 0})
        if term or trunc:
            break
    env.close()
    return r, term, trunc, info


def test_stall_이_0이면_정체는_예전처럼_잘리기만_한다():
    r, term, trunc, info = _stall_run(RewardConfig())
    assert info["outcome"] == "stalled" and trunc and not term
    assert "stall" not in info["reward_terms"]


def test_stall_을_주면_정체는_실패로_끝나고_벌을_문다():
    r, term, trunc, info = _stall_run(RewardConfig(stall=-200.0))
    assert info["outcome"] == "stalled" and term and not trunc
    assert info["reward_terms"]["stall"] == -200.0
    assert r == pytest.approx(sum(info["reward_terms"].values()))
```

(파일 위 임포트에 `from vtd_rl.world.world import WorldConfig` 가 없으면 더한다.)

`tests/rl/test_train_ppo.py` 의 `test_lat_profile_인자가_RewardConfig에_반영된다` 바로 뒤에:

```python
def test_실패_벌_인자가_RewardConfig에_반영된다():
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "/tmp/불필요-존재안함", "--collision-penalty", "-200",
                                           "--offroad-penalty", "-300", "--stall-penalty", "-400"])
    cfg = module._build_reward_cfg(a)
    assert (cfg.collision, cfg.offroad, cfg.stall) == (-200.0, -300.0, -400.0)
    off = module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"]))
    assert (off.collision, off.offroad, off.stall) == (RewardConfig().collision, RewardConfig().offroad, 0.0)
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m7a_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `stall` 필드가 없다.

- [ ] **Step 3: `reward.py`.**

`RewardConfig` 의 `lat_free_range` 필드 바로 뒤에:

```python
    # M7a — 정체 벌. 0 이 아니면 정체(60 초 동안 1 m 도 못 감)를 실패로 끝내고(terminated) 그 걸음에
    # 이 값을 더한다(항 `stall`). 기본 0 이면 예전처럼 잘리기만 한다(truncated, 항도 없다).
    stall: float = 0.0
```

`__post_init__` 끝에:

```python
        for name in ("collision", "offroad", "stall"):
            v = getattr(self, name)
            if not math.isfinite(v) or v > 0.0:
                raise ValueError(f"{name} 는 유한한 0 이하여야 한다: {v}")
```

`RewardShaper.step` 의 `lat` 항 블록 바로 뒤에:

```python
        if cfg.stall != 0.0:          # 끈 실행은 항 자체가 없다
            terms["stall"] = cfg.stall if outcome == "stalled" else 0.0
```

- [ ] **Step 4: `drive_env.py`.**

`step` 의 다음 줄을

```python
        terminated = outcome in ("goal", "offroad") or shaped.collision
```

다음으로 바꾼다:

```python
        # M7a — 정체 벌을 주면 정체는 실패로 끝난다(기본 0 이면 예전처럼 아래에서 잘리기만 한다).
        stall_fail = outcome == "stalled" and self.cfg.reward.stall != 0.0
        terminated = outcome in ("goal", "offroad") or shaped.collision or stall_fail
```

(바로 아래 `truncated = outcome in ("timeout", "stalled") and not terminated` 는 그대로 둔다 — `terminated` 가 참이면 거짓이 된다.)

- [ ] **Step 5: CLI.**

`scripts/train_ppo.py` 의 `--lat-free-range` 인자 바로 뒤에:

```python
    ap.add_argument("--collision-penalty", type=float, default=default_reward_cfg.collision,
                    help="충돌 벌(M7a 에서 연다). 유한한 0 이하")
    ap.add_argument("--offroad-penalty", type=float, default=default_reward_cfg.offroad,
                    help="도로 이탈 벌(M7a 에서 연다). 유한한 0 이하")
    ap.add_argument("--stall-penalty", type=float, default=default_reward_cfg.stall,
                    help="정체 벌(M7a). 0 이 아니면 정체를 실패로 끝내고 이 값을 더한다. 0 이면 예전처럼 잘리기만 한다")
```

`_build_reward_cfg` 의 `dataclasses.replace(...)` 에 `collision=a.collision_penalty, offroad=a.offroad_penalty, stall=a.stall_penalty` 를 더하고, 독스트링의 "열한 필드" 를 "열네 필드" 로, 나열에 세 인자를 더한다.

- [ ] **Step 6: 통과를 확인한다.**

Run: Step 2 와 같은 명령 → rc=0.
전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m7a_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 7: 커밋한다.**

```bash
git add vtd_rl/env/reward.py vtd_rl/env/drive_env.py scripts/train_ppo.py \
  tests/env/test_reward.py tests/env/test_drive_env.py tests/rl/test_train_ppo.py
git commit -m "M7a — 실패 벌(--collision/offroad/stall-penalty), 정체 벌을 주면 정체를 실패로 끝낸다"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `stall=0.0`(기본)이고 충돌·이탈 벌이 기본값이면 보상·`reward_terms`·종료 플래그가 예전과 같다(`stall` 키 없음, 정체는 truncated).
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. numpy 는 1.26.4 그대로다.

## Review Focus

- 정체가 실패로 끝날 때 `terminated` 와 `truncated` 가 동시에 참이 아니다(Gymnasium 규약).
- 정체와 충돌이 같은 걸음에 오면 둘 다 벌이 들어가고 종료 사유는 기존 규칙대로다.
- 기본 실행에서 `reward_terms` 에 `stall` 키가 없다.
- `train_ppo` 의 GAE 부트스트랩이 `terminated` 를 보고 끊는지(정체를 실패로 끝내면 부트스트랩하지 않는다).
