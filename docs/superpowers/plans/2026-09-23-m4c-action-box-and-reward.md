# M4c 구현 계획 — 행동 상자를 닫고, 보상이 탐색을 벌하지 않게 한다

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M4b 가 특정한 실패 기전을 제거한다 — ① 승차감 대 위반의 배분을 **측정으로 확정**하고, ② 정책이 행동 평균을 상자 밖으로 밀어 탐색을 죽이는 탈출구를 **tanh 스쿼시로 닫고**, ③ 보상이 탐색 잡음 자체를 벌하지 않게 고친다.

**Architecture:** 세 갈래다. `vtd_rl/rl/diagnostics.py` 에 항목별 보상 누적기를 더해 `log.jsonl` 이 `progress`·`violation`·`comfort` 를 따로 찍게 한다. `vtd_rl/policy/net.py` 의 가우시안을 **tanh 로 스쿼시**하고 로그확률에 야코비안 보정을 넣는다(평균이 상자를 못 벗어난다) — M3 체크포인트는 새 매개화로 **재적합**해야 한다. `vtd_rl/env/reward.py` 는 승차감을 **실행 행동이 아니라 정책의 의도(평균)** 로 잴 수 있는 선택지를 갖는다.

**Tech Stack:** Python 3.10, PyTorch 2.14, Gymnasium 1.3, numpy 1.26.4, pytest. 개발·테스트는 lab-main(16 코어), 본 학습은 OMEN(`user-OMEN` 100.87.135.28, 32 코어).

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md` (§5 보상, §6.3 PPO, §6.4 진급)

**직전 성적표(반드시 읽을 것):** `docs/reports/m4b-ppo.md` 와 해석 `docs/reports/m4b-ppo-notes.md`

## Global Constraints

- 커밋 메시지에 Claude 표기(`Co-Authored-By`, `Claude-Session`, "Generated with")를 **넣지 않는다**.
- 규칙 스택 서브모듈(`third_party/rule_stack`)은 **수정하지 않는다**.
- 공개 레포에 주행 CSV, `runs/`, 체크포인트(`*.pt`), 데이터 조각(`*.npz`), 주최측 자료(`*.xodr`, `*.xml`)를 커밋하지 않는다. **성적표 md 만 커밋한다.**
- **numpy 1.26.4 고정, torch >=2.4,<3.**
- 판정은 **종료 코드 0**. `"N passed"` 를 grep 하지 마라 — `"1 failed"` 를 삼킨다.
- **전체 스위트는 Bash 도구의 10 분 한도를 넘는다.** 겹치지 않는 세 조각으로 나눠 **전부 앞단에서**, 각각 `timeout` **600000**: ① `-m "not slow"` ② `tests/rl/` ③ `-m slow --ignore=tests/rl`. `run_in_background`·Monitor 를 쓰지 마라.
- 문서·주석·커밋 메시지는 **한국어**로 쓴다.
- **성적표 숫자는 전부 스크립트가 생성한다.** 손으로 옮겨 적지 마라.
- **기준을 낮추지 않는다.**

### ★ 이번에 바뀌는 제약

M4a·M4b 는 "심판·채점기·세계·환경의 문턱과 규칙은 바꾸지 않는다" 였다. **M4c 는 보상을 고치는 것이 목적이므로 `vtd_rl/env/reward.py` 가 범위 안이다.**

다만 **여전히 건드리지 않는 것**: `vtd_rl/referee/`(온라인 심판), `third_party/rule_stack/eval/score_fma.py`(대회 채점기), `vtd_rl/world/`(동역학·신호·액터), 그리고 `vtd_rl/env/` 중 **관측·행동 공간·종료 판정**(`observation.py`·`action.py`·`drive_env.py` 의 종료 로직). 대회 점수는 보상과 **독립**으로 계산되므로 성적표의 점수·위반 표는 이 변경에 영향받지 않는다 — 그게 이 실험이 성립하는 이유다.

## M4b 가 남긴 사실 (이 계획의 출발점)

9 회(설정 3 × 시드 3, 3M 스텝) **전부 목표 미달**, 각 설정 0/3 시드. PPO 가 출발점(M3 학생)을 한 번도 못 넘었다.

- 보상의 `comfort = -0.10|Δ조향| - 0.05|Δ가속|` 는 **실행된 행동의 차이**로 계산된다(`reward.py:74-75`). PPO 는 매 결정마다 가우시안을 독립으로 뽑으므로 **연속 행동의 차이가 곧 탐색의 크기**다.
- 계산: 하한 σ=(0.135, 0.358)에서 판(3606 걸음)당 승차감 벌점 **−128**. 과제 전체 가치는 **+114**(진행 100 + 완주 50 − 시간 36).
- 실측: 완주해도 **−110**(`base-s0` 첫 구간은 129 판 중 128 완주하고도 −110.1), **멈추면 −6.2**.
- **PPO 가 고른 탈출구는 "멈추기" 가 아니라 "평균 포화" 였다.** `sample()` 이 `raw.clamp(-1,1)` 을 환경에 넣는데 **평균 머리에 상한이 없다**. 실측(관측 256 개):

| 체크포인트 | \|평균\| 최대 (조향/가속) | 행동 상자 밖 비율 |
|---|---|---|
| `base` 최종 | 2.15 / 3.80 | 49% / 46% |
| `sigma-detach-ent` 최종 | 4.09 / 7.40 | 67% / 75% |

평균이 ±4 이고 σ=0.787 이면 거의 모든 표본이 같은 경계값으로 잘려 **탐색이 사라진다.** 정지는 그 부작용이다.
- **승차감이 전부는 아니다**: `113.9 + 승차감 + 위반 = −110.1` → 위반 몫이 약 **−96(43%)**. 이 배분은 아직 **추정**이다.

## 목표(성공 기준)

M4b 의 목표 네 줄(`vtd_rl/eval/verdict.py`)을 **그대로 쓴다** — 실질 판정은 3·4 번(출발점 대비 점수, 전 항목 중대 위반 합계)이고 1·2 번은 전제다. 여기에 이 마일스톤의 기전 목표를 더한다.

| # | 목표 | 기준 | 근거 |
|---|---|---|---|
| 1 | **배분 확정** | `log.jsonl` 에 판당 `comfort`·`violation`·`progress` 가 실측으로 찍힌다 | 지금 −128 대 −96 은 추정이다 |
| 2 | **행동 상자가 닫힌다** | 학습 끝 체크포인트에서 관측 256 개에 대해 \|행동\| ≤ 1 이 **100%** | M4b 는 49~75% 가 밖이었다 |
| 3 | **롤아웃 리턴이 양수** | 마지막 평가 구간의 `rollout_return_mean` > 0 | M4b 는 −66~−150 이었다. 이게 안 되면 PPO 는 계속 나쁜 데이터로 배운다 |
| 4 | **실질 판정** | `verdict.judge()` 의 3·4 번을 **시드 3 개 중 2 개 이상**에서 달성 | M4b 는 0/3 이었다 |

**1·2 번은 이 계획의 코드가 보장한다. 3·4 번은 실험 결과다.** 3 번을 못 채우면 원인이 하나 더 있다는 뜻이고, 그 숫자와 관찰을 성적표에 적는다 — **기준을 고치지 않는다.**

## 파일 구조

| 파일 | 책임 |
|---|---|
| `vtd_rl/rl/diagnostics.py` (수정) | `RewardTermTracker` — 판당 항목별 보상 누적 |
| `scripts/train_ppo.py` (수정) | 항목별 보상 로깅, `--comfort-*` 인자, 행동 상자 점검 |
| `vtd_rl/policy/net.py` (수정) | tanh 스쿼시 + 야코비안 보정, 스쿼시 엔트로피 |
| `vtd_rl/policy/train.py` (수정) | 스쿼시 매개화에 맞춘 모방 손실(선생님 행동을 `atanh` 로) |
| `scripts/refit_m3.py` | M3 라벨로 스쿼시 정책을 재적합해 새 출발점 체크포인트를 만든다 |
| `vtd_rl/env/reward.py` (수정) | 승차감을 **의도(평균)** 로 재는 선택지, 계수 노출 |
| `vtd_rl/env/drive_env.py` (수정) | `intent_hook` — 정책이 의도를 넘기는 통로(`command_tags` 와 같은 패턴) |
| `docs/reports/m4c-*.md` | 완료 증거 |

---

### Task 1: 항목별 보상을 롤아웃에서 누적해 로깅한다

**Files:**
- Modify: `vtd_rl/rl/diagnostics.py`, `scripts/train_ppo.py`
- Test: `tests/rl/test_diagnostics.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Consumes: `info["reward_terms"]` — `VtdDriveEnv` 가 **매 걸음** 싣는다(`drive_env.py:142`). 키: `progress`·`time`·`violation`·`collision`·`offroad`·`goal`·`comfort`. 리셋 직후는 빈 dict 다(`:105`, `:160`)
- Produces:
  - `class RewardTermTracker(n_envs, window=None)` — `ReturnTracker` 와 같은 모양
    - `.add(infos)` — 벡터 환경 `infos` 에서 `reward_terms` 를 꺼내 환경별로 누적
    - `.add_done(done)` — 판이 끝난 환경의 누적을 담고 0 으로 되돌린다
    - `.stats() -> dict` — `{"term_progress_mean": float|None, "term_violation_mean": …, "term_comfort_mean": …, "term_time_mean": …, "term_n": int}`. 끝난 판이 없으면 평균은 `None`
  - `scripts/train_ppo.py` 의 `log.jsonl` **모든 줄**에 위 키가 실린다
- **왜 필요한가:** M4b 의 "승차감 −128 대 위반 −96" 은 **계산이지 측정이 아니다.** 이 값이 없으면 Task 4 의 승차감 수정을 어느 방향·어느 크기로 할지 정할 수 없다. 측정이 이미 `info` 에 있는데 한 번도 안 썼다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_diagnostics.py` 에 더한다:
```python
def test_항목별_보상을_판마다_누적한다():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(2)
    t.add({"reward_terms": [{"progress": 1.0, "comfort": -0.5, "violation": 0.0},
                            {"progress": 2.0, "comfort": -0.1, "violation": -3.0}]})
    t.add_done(np.array([False, False]))
    t.add({"reward_terms": [{"progress": 1.0, "comfort": -0.5, "violation": -6.0},
                            {"progress": 2.0, "comfort": -0.1, "violation": 0.0}]})
    t.add_done(np.array([True, False]))
    s = t.stats()
    assert s["term_n"] == 1
    assert abs(s["term_progress_mean"] - 2.0) < 1e-9      # 0번 환경: 1+1
    assert abs(s["term_comfort_mean"] - (-1.0)) < 1e-9    # -0.5 + -0.5
    assert abs(s["term_violation_mean"] - (-6.0)) < 1e-9


def test_끝난_판이_없으면_None():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(1)
    t.add({"reward_terms": [{"progress": 1.0}]})
    t.add_done(np.array([False]))
    s = t.stats()
    assert s["term_n"] == 0 and s["term_progress_mean"] is None


def test_리셋_직후_빈_reward_terms_를_견딘다():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(1)
    t.add({"reward_terms": [{}]})          # VtdDriveEnv.reset() 이 내는 모양
    t.add({"reward_terms": [{"progress": 3.0}]})
    t.add_done(np.array([True]))
    assert abs(t.stats()["term_progress_mean"] - 3.0) < 1e-9


def test_reward_terms_키가_없으면_아무것도_안_센다():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(1)
    t.add({})
    t.add_done(np.array([True]))
    assert t.stats()["term_n"] == 1 and t.stats()["term_progress_mean"] == 0.0


def test_이동창이_최근_것만_남긴다():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(1, window=2)
    for v in (1.0, 2.0, 3.0):
        t.add({"reward_terms": [{"progress": v}]})
        t.add_done(np.array([True]))
    assert t.stats()["term_n"] == 2
    assert abs(t.stats()["term_progress_mean"] - 2.5) < 1e-9   # (2+3)/2
```

`tests/rl/test_train_ppo.py` 에 더한다:
```python
@pytest.mark.slow
def test_연습_모드가_항목별_보상을_남긴다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "train_ppo.py"),
                          "--smoke", "--out", str(tmp_path / "run"), "--seed", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    rows = [json.loads(l) for l in open(tmp_path / "run" / "log.jsonl", encoding="utf-8")]
    for r in rows:
        for k in ("term_progress_mean", "term_violation_mean", "term_comfort_mean", "term_n"):
            assert k in r, k
    done = [r for r in rows if r["term_n"] > 0]
    assert done, "끝난 판이 한 번도 안 잡혔다"
    # 승차감은 벌점이므로 음수여야 한다 — 부호가 뒤집히면 배분 해석이 통째로 틀어진다
    assert done[-1]["term_comfort_mean"] <= 0.0
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_diagnostics.py -q -k 항목별`
Expected: FAIL — `ImportError: cannot import name 'RewardTermTracker'`

- [ ] **Step 3: 구현**

`vtd_rl/rl/diagnostics.py` 에 더한다:
```python
# 판당 누적을 볼 항목 — `VtdDriveEnv` 가 info["reward_terms"] 로 내는 키들 중
# 배분 논쟁에 필요한 것만 고른다(collision·offroad·goal 은 종료 사유로 이미 보인다).
TERM_KEYS = ("progress", "time", "violation", "comfort")


class RewardTermTracker:
    """판 하나가 쌓는 **항목별** 보상을 누적한다.

    M4b 는 "승차감 −128 대 위반 −96" 을 **계산**으로만 말했다 — `info["reward_terms"]` 가
    걸음마다 이미 있는데 한 번도 안 썼기 때문이다. 이 누적기가 그 배분을 측정으로 바꾼다.

    `ReturnTracker` 와 달리 `add(infos)` 와 `add_done(done)` 이 나뉘어 있다 — 보상 항목은
    `infos` 에서, 종료는 `term|trunc` 에서 오기 때문이다. 호출부는 매 걸음 둘 다 부른다.
    """

    def __init__(self, n_envs: int, window: int | None = None):
        self.n_envs = n_envs
        self.window = window
        self._running = {k: np.zeros(n_envs, dtype=np.float64) for k in TERM_KEYS}
        self.reset()

    def reset(self):
        self._done: dict = {k: deque(maxlen=self.window) for k in TERM_KEYS}

    def add(self, infos: dict):
        terms = infos.get("reward_terms")
        if terms is None:
            return
        for i, t in enumerate(terms):
            if not t:                      # 리셋 직후는 빈 dict 다
                continue
            for k in TERM_KEYS:
                self._running[k][i] += float(t.get(k, 0.0))

    def add_done(self, done):
        for i in np.nonzero(np.asarray(done, dtype=bool))[0]:
            for k in TERM_KEYS:
                self._done[k].append(float(self._running[k][i]))
                self._running[k][i] = 0.0

    def stats(self) -> dict:
        n = len(self._done[TERM_KEYS[0]])
        out = {"term_n": n}
        for k in TERM_KEYS:
            vals = self._done[k]
            out[f"term_{k}_mean"] = (sum(vals) / n) if n else None
        return out
```
(`from collections import deque` 가 없으면 더한다. `ReturnTracker` 가 이미 쓰고 있을 것이다.)

`scripts/train_ppo.py`:
- 임포트에 `RewardTermTracker` 를 더한다.
- `tracker = ReturnTracker(...)` 옆에 `terms = RewardTermTracker(a.envs, window=30)` 을 만든다(같은 이동창).
- `venv.step` 뒤, `tracker.add(...)` 옆에 `terms.add(info)` 와 `terms.add_done(done_mask)` 를 부른다.
- 로그 줄에 `**terms.stats()` 를 펼쳐 넣는다. **키 충돌을 대조해라** — `term_` 접두가 붙어 있어 기존 키와 안 겹치는 것을 확인하고 보고해라.

- [ ] **Step 4: 통과 확인**

Run(세 조각, 전부 앞단·각각 `timeout` 600000):
`env -u PYTHONPATH .venv/bin/pytest -q -m "not slow"; echo rc=$?` · `env -u PYTHONPATH .venv/bin/pytest -q tests/rl/; echo rc=$?` · `env -u PYTHONPATH .venv/bin/pytest -q -m slow --ignore=tests/rl; echo rc=$?`
Expected: 셋 다 `rc=0`

그리고 **연습 모드를 돌려 `log.jsonl` 한 줄을 눈으로 읽어라** — 항목별 값이 말이 되는지(승차감 음수, 진행 양수).

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/rl/diagnostics.py scripts/train_ppo.py tests/rl/test_diagnostics.py tests/rl/test_train_ppo.py
git commit -m "M4c 계측 — 항목별 보상을 판마다 누적해 로깅(승차감 대 위반 배분을 측정으로)"
```

---

### Task 2: 배분을 실측하고 기록한다

**Files:**
- Create: `docs/reports/m4c-split.md`(짧은 측정 기록)

**Interfaces:**
- Consumes: Task 1 의 `term_*` 로깅
- Produces: `docs/reports/m4c-split.md` — M4b 의 추정(−128 대 −96)과 실측을 나란히 놓은 표
- **이 작업은 코드를 안 만든다.** 짧은 진단 실행 하나와 그 기록이다. **Task 4 의 승차감 수정이 이 숫자에 근거한다.**

- [ ] **Step 1: 진단 실행 (OMEN)**

코드를 올린다.
```bash
for d in vtd_rl scripts tests; do
  rsync -a --delete --exclude '__pycache__' $d/ 100.87.135.28:vtd-rl-urban-driving/$d/
done
ssh 100.87.135.28 'cd ~/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q -m "not slow"; echo rc=$?'
```
Expected: `rc=0`

**50 만 스텝짜리 짧은 실행 하나**를 기본 설정으로 돌린다(약 3 분). 배분을 보는 것이 목적이라 3M 이 필요 없다.
```bash
ssh 100.87.135.28 'cd ~/vtd-rl-urban-driving && nice -n 5 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-09-23-m4c-split \
  --init runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt \
  --dagger-data runs/lab-main/2026-09-17-dagger-fix2/data \
  --envs 30 --steps 500000 --device cpu --eval-every 250000' | tail -c 400
rsync -a 100.87.135.28:vtd-rl-urban-driving/runs/omen/2026-09-23-m4c-split/ runs/omen/2026-09-23-m4c-split/
```

- [ ] **Step 2: 배분을 읽어 기록한다**

`log.jsonl` 의 마지막 줄에서 `term_progress_mean`·`term_time_mean`·`term_violation_mean`·`term_comfort_mean` 과 `rollout_return_mean`·`rollout_len_mean` 을 읽어 `docs/reports/m4c-split.md` 를 쓴다. 담을 것:

- **M4b 의 추정 대 M4c 의 실측**을 같은 표에: 판당 승차감 / 위반 / 진행 / 시간, 그리고 넷의 합이 `rollout_return_mean` 과 맞는지
- **합이 안 맞으면 그 사실을 적어라** — `collision`·`offroad`·`goal` 이 `TERM_KEYS` 에 없어 차이가 나는 것이 정상이다. 얼마나 나는지 숫자로 적는다
- **승차감이 음수 총합의 몇 %인지**
- 이 실행의 설정(스텝 수·시드·머신)과 **이것이 짧은 진단 실행이라는 한계**

숫자는 전부 `log.jsonl` 에서 뽑아라. 손으로 적지 마라.

- [ ] **Step 3: 커밋**

```bash
git add docs/reports/m4c-split.md
git commit -m "M4c 측정 — 판당 항목별 보상 실측(승차감 대 위반 배분)"
```

---

### Task 3: 행동 상자를 닫는다 (tanh 스쿼시 + 야코비안 보정)

**Files:**
- Modify: `vtd_rl/policy/net.py`, `vtd_rl/policy/train.py`
- Test: `tests/policy/test_net.py`, `tests/rl/test_policy_fixes.py`, `tests/policy/test_train.py`

**Interfaces:**
- Produces:
  - `PolicyConfig.squash: bool = False` — `True` 면 행동을 `tanh` 로 스쿼시한다. **기본값 `False` 라 기존 동작이 안 바뀐다**(M3·M4a·M4b 체크포인트가 그대로 읽힌다)
  - `DrivePolicy.sample(...)` 이 `squash=True` 일 때: `raw` 는 **스쿼시 전** 값, `control = tanh(raw)`, `log_prob` 에 **야코비안 보정**이 들어간다
  - `DrivePolicy.evaluate_actions(vec, objs, mask, raw, turn)` — 같은 보정. 저장된 `raw` 로 계산하므로 PPO 비율이 일관된다
  - `DrivePolicy.act(..., deterministic=True)` → `tanh(mean)`
  - `TrainConfig.squash: bool = False` — `policy_loss` 가 선생님 행동을 `atanh` 로 옮겨 스쿼시 밀도로 로그가능도를 계산한다
- **왜 필요한가:** M4b 실측으로 `base` 는 표본의 **49%**, `sigma-detach-ent` 는 **67~75%** 가 행동 상자 밖이다. `clamp` 가 그 잡음을 먹어 탐색이 0 이 된다. `tanh` 는 평균이 아무리 커도 행동을 상자 안에 두고, **보정된 로그확률이 포화를 비용으로 만든다**(포화 지점에서 밀도가 발산하므로 엔트로피 보너스가 밀어낸다).

**주의 — 엔트로피가 닫힌 형태가 아니다.** 스쿼시된 분포의 엔트로피에는 닫힌 식이 없다. 표준 관행대로 **표본 하나짜리 추정 `entropy = -log_prob`** 를 쓴다(SAC 와 같다). `squash=False` 일 때는 지금처럼 가우시안 닫힌 형태를 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_policy_fixes.py` 에 더한다:
```python
def test_스쿼시하면_행동이_항상_상자_안이다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True, log_std_max=2.0))
    with torch.no_grad():
        net.log_std.copy_(torch.tensor([1.5, 1.5]))     # σ≈4.5 — 안 자르면 상자를 크게 벗어난다
    vec, objs, mask = obs_batch(64)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(1))
    assert torch.all(out["control"].abs() <= 1.0)
    assert torch.any(out["raw"].abs() > 1.0)            # 스쿼시 전은 상자 밖으로 나간다
    assert torch.allclose(out["control"], torch.tanh(out["raw"]), atol=1e-6)


def test_스쿼시_로그확률에_야코비안_보정이_들어간다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    vec, objs, mask = obs_batch(8)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(2))
    normal = torch.distributions.Normal(net(vec, objs, mask)[0], net.log_std.exp())
    base = normal.log_prob(out["raw"]).sum(dim=-1)
    corr = torch.log1p(-torch.tanh(out["raw"]) ** 2 + 1e-12).sum(dim=-1)
    turn_lp = torch.log_softmax(net(vec, objs, mask)[2], dim=-1).gather(
        1, out["turn"][:, None]).squeeze(1)
    assert torch.allclose(out["log_prob"], base - corr + turn_lp, atol=1e-4)
    # 보정이 없으면 값이 다르다 — 이 단언이 보정 누락을 잡는다
    assert not torch.allclose(out["log_prob"], base + turn_lp, atol=1e-3)


def test_스쿼시에서도_evaluate_actions_가_일치한다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    vec, objs, mask = obs_batch(8)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(3))
    again, entropy = net.evaluate_actions(vec, objs, mask, out["raw"], out["turn"])
    assert torch.allclose(again, out["log_prob"], atol=1e-5)
    assert torch.all(torch.isfinite(entropy))


def test_스쿼시_결정적_행동도_상자_안이다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    space = observation_space(ObsConfig()); space.seed(5)
    a = net.act(space.sample(), deterministic=True)
    assert abs(a["control"][0]) <= 1.0 and abs(a["control"][1]) <= 1.0


def test_스쿼시_기본값은_꺼짐이고_끄면_예전과_같다(obs_batch):
    assert PolicyConfig().squash is False
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    vec, objs, mask = obs_batch(8)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(4))
    assert torch.allclose(out["control"], out["raw"].clamp(-1.0, 1.0), atol=1e-6)
```

`tests/policy/test_train.py` 에 더한다:
```python
def test_스쿼시_모방손실은_선생님_행동을_atanh_로_옮긴다():
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), squash=True))
    batch = next(iter(toy_dataset(64).batches(32, generator=torch.Generator().manual_seed(0))))
    loss, parts = policy_loss(net, batch, TrainConfig(squash=True))
    assert torch.isfinite(loss) and parts["total"] == parts["total"]   # NaN 아님
    # 스쿼시 손실을 줄이면 tanh(mean) 이 선생님 행동에 가까워진다
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(50):
        l, _p = policy_loss(net, batch, TrainConfig(squash=True))
        opt.zero_grad(set_to_none=True); l.backward(); opt.step(); net.clamp_log_std()
    vec, objs, mask, control, _turn = batch
    with torch.no_grad():
        err = (torch.tanh(net(vec, objs, mask)[0]) - control).abs().mean()
    assert float(err) < 0.3


def test_스쿼시_기본값은_꺼짐():
    assert TrainConfig().squash is False
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_policy_fixes.py -q -k 스쿼시`
Expected: FAIL — `TypeError: PolicyConfig.__init__() got an unexpected keyword argument 'squash'`

- [ ] **Step 3: 구현**

`vtd_rl/policy/net.py`:
- `PolicyConfig` 에 `squash: bool = False` 를 더한다(주석: 왜 필요한지 — 평균 포화로 탐색을 죽이는 탈출구를 닫는다).
- 야코비안 보정 헬퍼를 더한다. **수치 안정형을 써라** — `log(1 - tanh(u)²)` 를 그대로 계산하면 `|u|` 가 크면 `log(0)` 이 된다.
```python
def _tanh_log_det(raw):
    """tanh 변환의 로그 야코비안 합 — `log(1 - tanh(u)^2)` 의 수치 안정형.

    항등식: log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u)).
    그대로 계산하면 |u| 가 조금만 커져도 1 - tanh(u)^2 이 0 으로 내려가 -inf 가 된다.
    """
    return (2.0 * (math.log(2.0) - raw - nn.functional.softplus(-2.0 * raw))).sum(dim=-1)
```
- `sample()` 을 고친다:
```python
        raw = normal.mean + noise * normal.stddev
        if self.cfg.squash:
            control = torch.tanh(raw)
            gauss_lp = normal.log_prob(raw).sum(dim=-1) - _tanh_log_det(raw)
            gauss_ent = -gauss_lp          # 스쿼시 분포는 닫힌 엔트로피가 없다(표본 1개 추정)
        else:
            control = raw.clamp(-1.0, 1.0)
            gauss_lp = normal.log_prob(raw).sum(dim=-1)
            gauss_ent = normal.entropy().sum(dim=-1)
        log_prob = gauss_lp + cat.log_prob(turn)
        entropy = gauss_ent + cat.entropy()
```
- `evaluate_actions()` 에 같은 분기를 넣는다(저장된 `raw` 를 받으므로 `tanh` 를 다시 씌우지 않는다).
- `act(deterministic=True)` 는 `torch.tanh(mean[0])` 를 쓴다(`squash=True` 일 때). 결정적 경로도 상자 안이어야 한다.
- `save`/`load` 는 `asdict(cfg)` 를 쓰므로 새 필드가 자동으로 실린다. **옛 체크포인트에는 `squash` 키가 없으므로** `PolicyConfig(**blob["cfg"])` 가 기본값 `False` 를 쓴다 — 하위호환이 유지된다.

`vtd_rl/policy/train.py`:
- `TrainConfig` 에 `squash: bool = False`.
- `policy_loss` 에 분기를 넣는다:
```python
    if cfg.squash:
        # 선생님 행동은 상자 안 값이다. 스쿼시 매개화에서는 그 행동의 로그가능도가
        # 가우시안 밀도(atanh 지점) 빼기 야코비안이다 — atanh 은 ±1 에서 발산하므로 잘라 준다.
        a = control.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(a)
        nll_ls = nll_log_std                      # 기존 sigma_grad 분기를 그대로 탄다
        var = (2.0 * nll_ls).exp()
        nll = 0.5 * (((u - mean) ** 2) / var + 2.0 * nll_ls + LOG_2PI)
        control_loss = (nll.sum(dim=-1) + _tanh_log_det_from(u)).mean()
    else:
        ... 기존 그대로 ...
```
`_tanh_log_det_from` 은 `net.py` 의 `_tanh_log_det` 를 임포트해 쓴다(복사하지 마라). 부호에 주의해라 — **손실**은 음의 로그가능도이므로 야코비안이 **더해진다**.

- [ ] **Step 4: 통과 확인**

세 조각(위 Global Constraints 의 명령) 전부 `rc=0`. **기존 테스트가 하나도 깨지면 안 된다** — `squash` 기본값이 `False` 다.

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/policy/net.py vtd_rl/policy/train.py tests/policy/test_net.py tests/policy/test_train.py tests/rl/test_policy_fixes.py
git commit -m "M4c — tanh 스쿼시와 야코비안 보정(행동 평균이 상자를 못 벗어난다), 기본값은 꺼짐"
```

---

### Task 4: 승차감을 의도(평균)로 잴 수 있게 한다

**Files:**
- Modify: `vtd_rl/env/reward.py`, `vtd_rl/env/drive_env.py`, `scripts/train_ppo.py`
- Test: `tests/env/test_reward.py`, `tests/env/test_drive_env.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `RewardConfig.comfort_on_intent: bool = False` — `True` 면 승차감을 **실행 행동이 아니라 정책의 의도(결정적 평균)** 의 변화량으로 잰다
  - `VtdDriveEnv.intent_hook: Callable[[], tuple[float, float]] | None = None` — `command_tags` 와 **같은 패턴**. 정책이 이번 걸음의 의도(조향, 가속)를 주는 통로. 안 걸려 있으면 실행 행동을 쓴다(= 지금 동작)
  - `scripts/train_ppo.py` 인자 `--comfort-steer`(기본 −0.10)·`--comfort-accel`(기본 −0.05)·`--comfort-on-intent`(기본 꺼짐)
- Consumes: Task 3 의 `squash`(의도는 `tanh(mean)` 이다)
- **왜 이렇게 하나:** 승차감은 "차가 실제로 얼마나 덜컹였나" 를 재려는 항이다. 그런데 PPO 가 매 걸음 독립 표본을 뽑으면 **그 덜컹임의 대부분이 탐색 잡음**이고, 정책은 그걸 줄이려고 탐색을 죽인다(M4b 실측). 의도로 재면 **탐색은 공짜가 되고 진짜 급조작만 벌한다.** 계수 노출은 Task 2 의 실측 배분에 맞춰 크기를 조정하기 위해서다.

**주의:** 이것은 `vtd_rl/env/reward.py` 를 바꾸는 것이고, M4a·M4b 에서 금지했던 일이다. **이번 마일스톤은 보상 수정이 목적이라 범위 안이다**(Global Constraints 의 "이번에 바뀌는 제약" 참고). **대회 점수(`score_fma`)는 보상과 독립이므로 성적표의 점수·위반 표는 이 변경에 영향받지 않는다** — 그 독립성을 테스트로 확인해라.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/env/test_reward.py` 에 더한다. **이 파일은 이미 `h_board()` 헬퍼와 `pytest.approx` 를 쓴다**(`test_승차감은_변화량에_붙는다` 참고) — 그 패턴을 따라라. 새 임포트는 필요 없다.
```python
def test_의도로_재면_표본_잡음이_승차감에_안_잡힌다():
    sh = RewardShaper(h_board(), RewardConfig(comfort_on_intent=True))
    sh.reset()
    intent = (0.20, 0.10)                              # 의도는 두 걸음 내내 같다
    a1 = {"control": [0.20, 0.10], "turn": 0}
    a2 = {"control": [0.60, 0.50], "turn": 0}          # 표본이 크게 튀었다
    sh.step([], 0.0, a1, a1, "running", intent=intent)
    out = sh.step([], 0.0, a2, a1, "running", intent=intent)
    assert out.terms["comfort"] == 0.0                 # 의도가 안 변했으니 0


def test_의도가_변하면_그_변화량으로_잰다():
    cfg = RewardConfig(comfort_on_intent=True)
    sh = RewardShaper(h_board(), cfg)
    sh.reset()
    a = {"control": [0.0, 0.0], "turn": 0}             # 실행 행동은 내내 같다
    sh.step([], 0.0, a, a, "running", intent=(0.0, 0.0))
    out = sh.step([], 0.0, a, a, "running", intent=(0.5, -0.4))
    assert out.terms["comfort"] == pytest.approx(cfg.comfort_steer * 0.5 + cfg.comfort_accel * 0.4)


def test_의도를_안_주면_예전처럼_실행_행동으로_잰다():
    cfg = RewardConfig()
    sh = RewardShaper(h_board(), cfg)
    sh.reset()
    prev = {"control": [0.0, 0.0], "turn": 0}
    now = {"control": [0.5, -0.4], "turn": 0}
    out = sh.step([], 0.0, now, prev, "running")
    assert out.terms["comfort"] == pytest.approx(cfg.comfort_steer * 0.5 + cfg.comfort_accel * 0.4)


def test_승차감_계수를_바꿀_수_있다():
    sh = RewardShaper(h_board(), RewardConfig(comfort_steer=-0.01, comfort_accel=-0.005))
    sh.reset()
    prev = {"control": [0.0, 0.0], "turn": 0}
    now = {"control": [1.0, 1.0], "turn": 0}
    out = sh.step([], 0.0, now, prev, "running")
    assert out.terms["comfort"] == pytest.approx(-0.015)


def test_comfort_on_intent_기본값은_꺼짐():
    assert RewardConfig().comfort_on_intent is False
```

`tests/env/test_drive_env.py` 에 더한다. **그 파일은 이미 `numpy as np`·`pytest`·`EnvConfig`·`VtdDriveEnv` 를 모듈 수준에서 임포트한다** — `RewardConfig` 와 `load_curriculum` 만 임포트 줄에 더하면 되고, 아래 함수 안의 임포트는 지워도 된다.
```python
def test_intent_hook_이_승차감에_닿는다():
    from vtd_rl.env.reward import RewardConfig
    from vtd_rl.world.board import load_curriculum
    boards = load_curriculum("curricula/stage1.json")[1]
    env = VtdDriveEnv(boards, EnvConfig(reward=RewardConfig(comfort_on_intent=True)))
    try:
        env.intent_hook = lambda: (0.0, 0.0)          # 의도는 내내 0
        env.reset(seed=0, options={"board": boards[0].name})
        big = {"control": np.array([1.0, 1.0], np.float32), "turn": 0}
        _o, _r, _t, _tr, info = env.step(big)
        assert abs(info["reward_terms"]["comfort"]) < 1e-9
    finally:
        env.close()


@pytest.mark.slow
def test_대회_점수는_보상_설정에_영향받지_않는다():
    """보상을 바꿔도 `score_fma` 점수·위반은 그대로여야 한다 — 그게 이 실험이 성립하는 근거다.

    보상은 학습 신호고 점수는 채점기가 따로 낸다. 이 독립성이 깨지면 M4c 의 성적표를
    M4a·M4b 와 비교할 수 없다.
    """
    import numpy as np
    from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
    from vtd_rl.env.reward import RewardConfig
    from vtd_rl.world.board import load_curriculum

    boards = load_curriculum("curricula/stage1.json")[1]
    results = []
    for cfg in (RewardConfig(),
                RewardConfig(comfort_steer=-99.0, comfort_accel=-99.0, comfort_on_intent=True)):
        env = VtdDriveEnv(boards, EnvConfig(reward=cfg))
        try:
            env.reset(seed=0, options={"board": boards[0].name})
            info = None
            for _ in range(20000):        # 판이 끝날 때까지 — 같은 행동이면 같은 궤적이다
                _o, _r, term, trunc, info = env.step(
                    {"control": np.array([0.05, 0.6], np.float32), "turn": 0})
                if term or trunc:
                    break
            assert info is not None and "result" in info, "판이 안 끝났다"
            results.append((info["result"]["score"], info["result"]["sheet"]))
        finally:
            env.close()
    assert results[0] == results[1]
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/env/test_reward.py -q -k 의도`
Expected: FAIL — `TypeError: RewardConfig.__init__() got an unexpected keyword argument 'comfort_on_intent'`

- [ ] **Step 3: 구현**

`vtd_rl/env/reward.py`:
- `RewardConfig` 에 `comfort_on_intent: bool = False` 를 더한다(주석: 왜 — 탐색 잡음이 승차감에 잡히면 정책이 탐색을 죽인다, M4b 실측).
- `RewardShaper.step(self, hits, ds, action, prev_action, outcome, intent=None)` 로 인자를 더한다(기본 `None`).
- 승차감 계산을 고친다:
```python
        if cfg.comfort_on_intent and intent is not None and self._prev_intent is not None:
            d_steer = abs(float(intent[0]) - float(self._prev_intent[0]))
            d_accel = abs(float(intent[1]) - float(self._prev_intent[1]))
        else:
            d_steer = abs(float(action["control"][0]) - float(prev_action["control"][0]))
            d_accel = abs(float(action["control"][1]) - float(prev_action["control"][1]))
        if intent is not None:
            self._prev_intent = (float(intent[0]), float(intent[1]))
```
`self._prev_intent` 는 `reset()` 에서 `None` 으로 둔다.

`vtd_rl/env/drive_env.py`:
- `self.intent_hook = None` 을 `self.command_tags = None` 옆에 더한다(같은 패턴이라는 주석과 함께).
- `self._shaper.step(...)` 호출에 `intent=self.intent_hook() if self.intent_hook is not None else None` 을 넘긴다.

`scripts/train_ppo.py`:
- 인자 `--comfort-steer`·`--comfort-accel`·`--comfort-on-intent` 를 더하고 `RewardConfig` 를 만들어 `EnvConfig(reward=...)` 로 넘긴다. **`hparams` 에 자동으로 실린다**(`vars(a)`).
- `--comfort-on-intent` 를 줄 때 **`intent_hook` 을 실제로 걸어라** — 벡터 환경이라 워커 안에서 걸어야 한다. 정책의 의도는 그 워커가 모르므로, **대신 `vec_env` 가 만드는 환경마다 직전 결정 행동의 평균을 쓸 수 없다.** 이 제약을 보고서에 적고, **가능한 구현을 하나 골라 근거와 함께 보고해라**:
  - (가) `--comfort-on-intent` 를 **비동기 벡터 환경에서 지원하지 않는다**고 막고(`ap.error`), 동기 환경에서만 쓴다
  - (나) 의도를 **행동 dict 에 실어** 보낸다(행동 공간 변경 — 이 계획의 금지 범위인 `action.py` 를 건드리므로 **하지 마라**)
  - (다) 계수 조정(`--comfort-steer`)만으로 이번 실험을 하고 의도 기반은 다음으로 미룬다
  **(가) 또는 (다)를 권한다.** 판단이 안 서면 물어라.

- [ ] **Step 4: 통과 확인**

세 조각 전부 `rc=0`. 그리고 **연습 모드를 `--comfort-steer -0.01` 로 한 번 돌려** `log.jsonl` 의 `term_comfort_mean` 이 실제로 작아지는지 눈으로 확인해라.

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/env/reward.py vtd_rl/env/drive_env.py scripts/train_ppo.py tests/env tests/rl/test_train_ppo.py
git commit -m "M4c — 승차감을 의도로 재는 선택지와 계수 노출(탐색이 벌받지 않게)"
```

---

### Task 5: M3 학생을 스쿼시 매개화로 재적합한다

**Files:**
- Create: `scripts/refit_m3.py`, `tests/policy/test_refit_m3.py`

**Interfaces:**
- Consumes: Task 3 의 `PolicyConfig.squash`·`TrainConfig.squash`, `vtd_rl.policy.dataset.load_dir`, `vtd_rl.policy.train.train_epochs`, `vtd_rl.policy.evaluate.evaluate_policy`
- Produces:
  - `scripts/refit_m3.py` — 인자 `--data runs/lab-main/2026-09-17-dagger-fix2/data --out runs/<머신>/<날짜>-m3-squash/policy.pt --epochs 8 --seed 0 --device auto [--eval]`
    - M3 라벨로 **`squash=True` 인 새 `DrivePolicy`** 를 처음부터 학습한다(기존 체크포인트를 이식하지 않는다 — 매개화가 다르다)
    - `--eval` 을 주면 단계 ①②를 **각각 따로** 평가해 요약 JSON 에 넣는다
    - 마지막 줄에 요약 JSON 한 줄. `--out` 재사용 가드는 `train_ppo.py` 와 같은 방식으로
- **왜 필요한가:** `tanh` 스쿼시는 평균의 **의미**를 바꾼다(`tanh(mean)` ≠ `clamp(mean)`). M3 체크포인트를 그대로 쓰면 출발점이 왜곡된다. 스쿼시 매개화로 다시 적합해야 M4c 의 PPO 가 **M4b 와 같은 출발선**에서 시작한다.

**목표:** 재적합한 학생이 **M3 학생과 같은 수준**이어야 한다 — 단계① 점수 ≥ 97, 단계② ≥ 92, 완주율 100%(M3 는 98.5 / 94.2 / 100%). 크게 낮으면 스쿼시 학습이 안 된 것이니 **숫자를 적고 보고해라.**

**★ 이 숫자가 목표 4 번의 난이도를 바꾼다는 것을 알고 시작해라.** `verdict.judge()` 의 3 번은 **원래 M3**(98.5 / 94.2)를 기준으로 삼는다(`M3_SCORE`). 재적합이 97 / 92 로 내려앉으면 PPO 는 **재적합에서 잃은 만큼을 먼저 되찾은 뒤에야** 목표를 향해 갈 수 있다. 그래서 위 하한(97 / 92)은 "이 정도면 실험이 성립한다" 는 선이지 만족스러운 값이 아니다.

**`M3_SCORE` 를 재적합 값으로 낮추지 마라.** 기준을 옮기면 "PPO 가 출발점을 넘었나" 라는 질문이 "PPO 가 자기가 만든 낮은 출발점을 넘었나" 로 바뀐다. 재적합 손실이 크면 **그 손실 자체를 성적표에 적고** 목표는 그대로 둔다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/policy/test_refit_m3.py`
```python
import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(REPO, "runs", "lab-main", "2026-09-17-dagger-fix2", "data")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_재적합이_스쿼시_체크포인트를_만든다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "refit_m3.py"),
                          "--data", DATA, "--out", str(tmp_path / "policy.pt"),
                          "--epochs", "1", "--seed", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["epochs"] == 1 and summary["samples"] > 0

    from vtd_rl.policy.net import DrivePolicy
    net = DrivePolicy.load(str(tmp_path / "policy.pt"))
    assert net.cfg.squash is True          # 스쿼시 매개화로 저장됐다

    # 결정적 행동이 항상 상자 안이다 — 이게 이 체크포인트의 존재 이유다
    import torch
    from vtd_rl.env.observation import ObsConfig, observation_space
    from vtd_rl.policy.encode import stack_obs, to_tensors
    sp = observation_space(ObsConfig()); sp.seed(0)
    vec, objs, mask = to_tensors(*stack_obs([sp.sample() for _ in range(64)]),
                                 torch.device("cpu"))
    with torch.no_grad():
        action = torch.tanh(net(vec, objs, mask)[0])
    assert torch.all(action.abs() <= 1.0)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_같은_폴더에_두_번_쓰지_않는다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    args = [os.path.join(REPO, ".venv", "bin", "python"),
            os.path.join(REPO, "scripts", "refit_m3.py"),
            "--data", DATA, "--out", str(tmp_path / "policy.pt"), "--epochs", "1"]
    assert subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO,
                          timeout=1800).returncode == 0
    second = subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO, timeout=300)
    assert second.returncode != 0 and "이미" in (second.stdout + second.stderr)
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/policy/test_refit_m3.py -q`
Expected: FAIL — `can't open file 'scripts/refit_m3.py'`

- [ ] **Step 3: 구현**

`scripts/refit_m3.py` 를 쓴다. `scripts/run_dagger.py` 의 인자 처리·요약 출력 관례를 따라라. 뼈대:
```python
    cfg = PolicyConfig(squash=True)
    net = DrivePolicy(cfg).to(dev)
    ds = load_dir(a.data)
    log = train_epochs(net, ds, TrainConfig(epochs=a.epochs, seed=a.seed, squash=True), device=dev)
    net.save(a.out)
    summary = {"epochs": a.epochs, "samples": len(ds), "seed": a.seed, **log}
    if a.eval:
        # 단계마다 따로 — 판 이름이 같아 합쳐 넘기면 세계 캐시 어서션으로 죽는다
        summary["stages"] = {label: _public(evaluate_policy(net, boards, seeds=(0, 1, 2)))
                             for label, boards in stages}
    print(json.dumps(summary, ensure_ascii=False))
```
`_public` 은 `evaluate_policy` 결과에서 `episodes`(직렬화 불가)를 거르는 헬퍼다 — `train_ppo.py` 에 같은 것이 있으니 **그 방식을 따르되 복사하지 말고 작게 다시 써라**(두 스크립트가 서로를 임포트하지 않는다).

- [ ] **Step 4: 통과 확인과 실제 재적합**

세 조각 전부 `rc=0`. 그 다음 **실제 재적합을 OMEN 에서 돌리고 평가해라**(약 10 분):
```bash
ssh 100.87.135.28 'cd ~/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/python scripts/refit_m3.py \
  --data runs/lab-main/2026-09-17-dagger-fix2/data \
  --out runs/omen/2026-09-23-m3-squash/policy.pt --epochs 8 --seed 0 --device cpu --eval' | tail -c 600
```
**목표(단계① ≥ 97, 단계② ≥ 92, 완주율 100%)를 못 채우면 숫자를 적고 보고해라.** 기준을 낮추지 마라.

- [ ] **Step 5: 커밋**

```bash
git add scripts/refit_m3.py tests/policy/test_refit_m3.py
git commit -m "M4c — M3 라벨로 스쿼시 정책을 재적합하는 스크립트(새 출발점)"
```

---

### Task 6: 본 실험과 성적표

**Files:**
- Create: `docs/reports/m4c-ppo.md`(스크립트 생성), `docs/reports/m4c-ppo-notes.md`(사람이 쓰는 해석)
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1~5 전부, `scripts/sweep_ppo.py`, `scripts/report_m4a.py`
- 돌릴 설정(전부 OMEN 30 환경·3M 스텝·`--device cpu`·**Task 5 의 스쿼시 체크포인트**에서 출발, **시드 0·1·2**):
  - `squash` — 스쿼시만. 행동 상자가 닫히면 그것만으로 달라지는지
  - `squash-comfort` — 스쿼시 + Task 2 실측에 근거한 승차감 계수(값은 그 측정이 정한다)
  - `squash-comfort-ent` — 위에 `--entropy-coef 0.05`
- **`--out-root` 는 설정마다 다르게 주거나 `--name` 을 다르게 줘라** — 집계 파일이 `<name>-sweep.json` 이라 안 섞이지만, M4b 에서 이 자리를 한 번 밟았다.
- **`--final-eval-seeds` 를 3 에서 줄이지 마라** — 줄이면 실질 판정 두 줄이 전부 "보류" 로 고정돼 9 회가 무의미해진다.

- [ ] **Step 1: 실행 전 확인**

코드를 올리고 빠른 스위트가 OMEN 에서 도는지 본다.
```bash
for d in vtd_rl scripts tests; do
  rsync -a --delete --exclude '__pycache__' $d/ 100.87.135.28:vtd-rl-urban-driving/$d/
done
ssh 100.87.135.28 'cd ~/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q -m "not slow"; echo rc=$?'
```
Expected: `rc=0`

- [ ] **Step 2: 세 설정 × 시드 3 을 돌린다 (약 3 시간)**

설정마다 `scripts/sweep_ppo.py` 를 `--seeds 0 1 2 --resume` 으로 돌린다. `--init` 는 **Task 5 의 스쿼시 체크포인트**다. 결과를 회수한다.

- [ ] **Step 3: 목표 2 번(행동 상자)을 확인한다**

각 설정의 최종 체크포인트에서 관측 256 개에 대해 `|행동| ≤ 1` 이 **100%** 인지 재고 기록해라. M4b 는 49~75% 가 밖이었다. **이게 100% 가 아니면 Task 3 이 제대로 안 된 것이다.**

- [ ] **Step 4: 해석 노트를 쓰고 성적표를 만든다**

`docs/reports/m4c-ppo-notes.md` 를 쓴다. **숫자는 성적표가 생성하니 해석만.** 반드시 담을 것:
- 목표 네 줄 각각의 결과(배분 확정 · 행동 상자 · 롤아웃 리턴 · 실질 판정)
- **롤아웃 리턴이 양수로 올라왔나.** 안 올라왔으면 항목별 보상 실측이 **무엇이 남았다고 말하는지**
- 행동 상자를 닫았는데도 정책이 다른 탈출구를 찾았나(예: 포화 대신 다른 퇴화)
- 시드 3 개의 편차
- **이 성적표가 주장하지 않는 것**(설정 3 개·시드 3 개·커리큘럼은 단계 ①②뿐·승차감 계수는 한 값만 시험)
- M4d 로 넘기는 것

성적표를 만든다(`--run` 은 **목표를 가장 잘 맞춘 설정**으로 고르되 **그렇게 고른 사실을 노트에 적어라**, `--sweep` 은 세 개 다, `--student-label "M4c 학생"`, `--title "M4c 성적표 — 행동 상자와 보상"`).

**목표를 못 채우면 기준을 고치지 마라.** 숫자와 관찰을 적는다.

- [ ] **Step 5: README 와 커밋**

`README.md` 의 PPO 단락에 M4c 결과 링크를 더한다.

```bash
git add docs/reports/m4c-ppo.md docs/reports/m4c-ppo-notes.md README.md
git commit -m "M4c 증거 — 행동 상자·보상 수정 뒤 설정 3 × 시드 3 결과"
```

---

## 이 계획이 미루는 것

- **커리큘럼 단계 ③④⑤**(사물·정지차 / 보행자·교통 / 연습코스 전체). 목표 3·4 번을 채운 뒤에 간다 — **PPO 가 단계 ①②에서 출발점을 못 넘는 동안 판을 어렵게 만들면 실패 원인만 늘어난다.**
- 학습률·클립·롤아웃 길이 탐색.
- 시간 상관 탐색(OU 잡음)·행동 반복 — 승차감 문제의 또 다른 해법이지만, 이번엔 스쿼시와 계수 두 가지만 시험한다. 그것으로 목표 3 번이 안 되면 다음 후보다.
- M4b 최종 리뷰가 넘긴 것: `_goal_lines` 의 `"보류" in v.line` 문자열 매칭(구조적 플래그로 바꿀 것), `OutcomeCounter` 가 0 인 사유의 키를 안 만드는 것, `ReturnTracker` 의 판 길이 +1 편향, `RewardConfig.rule_scale` 이 커리큘럼에서 설정되지 않는 것(스펙 §5).
