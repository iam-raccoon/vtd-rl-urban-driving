# M4e — 결정적 정책이 멈추는 것을 고친다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 학습이 끝난 정책이 **평균 행동만으로도** 판을 완주하게 만든다 — 지금은 확률적으로는 거의 100% 완주하는데 결정적으로는 22~48% 이고, **실패가 전부 "정체"** 다.

**Architecture:** 원인을 먼저 계측으로 좁히고(정체 순간의 결정적 가속이 실제로 모자란지), 그 다음 **두 가지 개입을 나란히** 시험한다 — ① 학습 후반에 σ 를 바닥으로 어닐링해 최적화 대상이 결정적 모드로 수렴하게 하기 ② 모방 앵커가 0 으로 사라지지 않게 바닥을 두기. 둘 다 CLI 플래그로 열고 기본값은 꺼짐이라 기존 실행이 안 바뀐다.

**Tech Stack:** Python 3.10, PyTorch 2.14(cpu), gymnasium 1.3, numpy **1.26.4 고정**, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`

**직전 증거:** `docs/reports/m4d-ppo.md`(+`-notes`), `docs/reports/m4c-ppo-notes.md`

## 왜 이 순서인가 (M4d 가 남긴 숫자)

M4d 가 보상을 대회 채점기와 맞춰 **판당 롤아웃 리턴을 −65.3 → +12.0**(이 프로젝트 첫 양수)으로 뒤집었다. 그런데 목표 판정은 여전히 0/2 이고 병목이 **결정적 완주율** 하나로 좁혀졌다.

2026-09-28 실측(3M 체크포인트, `scripts/eval_det_vs_stoch.py` + 종료 사유 집계):

| 체크포인트 | 결정적 종료 사유 | 확률적 |
|---|---|---|
| `align-sigma-s0` @3M | goal 9 / **stalled 9** | goal 18 / stalled 0 |
| `m4c-base-s0` @3M | goal 3 / **stalled 15** | goal 18 / stalled 0 |
| 출발점(스쿼시 재적합) | goal 18 / stalled 0 | goal 18 |

**결정적 실패가 100% 정체다. 시간초과는 0 건.** 느린 것이 아니라 **멈춘다.** 완주한 판의 점수는 96~98 로 멀쩡하다 — 주행의 질이 아니라 **차를 굴리는 힘**의 문제다.

**σ 는 원인이 아니다** — M4d 가 `--log-std-max -0.6` 으로 상한을 묶었고(3M 에서 실제로 −0.600 유지) 결정적 완주율이 정렬만 한 것과 같았다(44% 대 44%).

남은 가설은 **"학습이 평균을 배포 불가능한 곳으로 옮긴다"** 이다. PPO 는 `E[R(tanh(μ+σξ))]` 를 최대화하므로 **평균 자체가 좋은 행동일 필요가 없다.** 모방 앵커가 그 평균을 붙들고 있다가(출발점은 정체 0) 계수가 `0.5^(step/2M)` 로 내려가며 놓는다.

## Global Constraints

- 커밋·PR 에 **Claude 표기 금지**(`Co-Authored-By` / `Claude-Session` / "Generated with" / 🤖 / anthropic 주소). **커밋 뒤 `git show -s --format=%B HEAD` 를 직접 실행해 눈으로 확인하고 보고할 것** — M4c 에서 한 번 뚫렸고 M4d 에서 한 번 더 들어갈 뻔했다(그때는 이 확인 요구 덕에 본인이 잡았다).
- `third_party/rule_stack` **수정 금지**(읽기만).
- `runs/`·`*.pt`·`*.npz`·CSV·`*.xodr`·`*.xml` **커밋 금지**. 성적표 `.md` 만.
- numpy **1.26.4 고정**. 문서·주석·커밋 메시지는 **한국어**.
- **성적표 숫자는 전부 스크립트가 생성한다.** 손으로 적지 말 것.
- **점수는 `mean_score_completed`(완주 판만)를 주 자로 쓴다.** `mean_score_raw` 는 채점기가 미방문 구간을 100 점으로 채워 **일찍 멈춘 정책일수록 높다**(M4d 최종 리뷰 C1 — 이 함정으로 주장 셋이 뒤집혔다). `mean_score` 는 반대로 미완주를 0 점 처리한다. **셋을 다 보이되 판정은 완주 판 기준으로.**
- `vtd_rl/eval/verdict.py` 의 `M3_SCORE = {"stage1": 98.5, "stage2": 94.2}` 를 **낮추지 말 것**.
- **테스트를 돌리는 모든 Bash 호출에 `timeout: 600000`(ms)을 빠짐없이 붙일 것.** 하나라도 빠지면 하네스가 자동으로 배경으로 넘기고, 그러면 보고 없이 멈추게 된다(이 프로젝트에서 **여덟 번**). `run_in_background`·Monitor 금지.
- 테스트는 겹치지 않는 **세 조각**으로 나눠 돌리고 **종료 코드(rc)** 로 판정한다:
  - `env -u PYTHONPATH .venv/bin/python -m pytest -q -m "not slow"`
  - `env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/ -q` ← **약 650 초**로 도구 상한을 넘어 배경으로 갈 수 있다. 그러면 **알림을 기다렸다 출력 파일을 Read 로 읽어 rc 를 확인할 것. 멈추지 말 것.**
  - `env -u PYTHONPATH .venv/bin/python -m pytest -q -m slow --ignore=tests/rl`
  - **기준선(M4d 병합 시점, main `4fa67c0`): 485 / 138 / 26, 전부 rc=0.**
- 이 머신(lab-main)에는 `uv` 가 없다. `.venv` 를 직접 쓴다.
- **커밋을 먼저 하고 그 다음에 돌연변이 실험을 할 것.** 미커밋 상태에서 `git checkout` 을 해 구현이 날아간 일이 **세 번** 있었다. 돌연변이는 작업 트리에 심고 `git checkout -- <경로>` 로 **경로를 지정해** 원복한다 — 이 패키지는 editable 설치라 `/tmp` 복사본은 진짜 패키지를 임포트해 "전부 살아남음" 이라는 거짓 결과를 준다.
- 긴 학습 실행(3M)은 **OMEN**(`user-OMEN`, 32 코어, `--device cpu`)에서 계획 주관자가 직접 돌린다. 구현자는 돌리지 않는다.

---

## File Structure

| 파일 | 책임 | 작업 |
|---|---|---|
| `scripts/diag_stall.py`(신규) | 정체 순간의 결정적 행동을 선생님·확률 표본과 대조 | 1 |
| `vtd_rl/rl/ppo.py` | `PPOConfig.imitation_floor`, `imitation_coef()` 에 바닥 | 3 |
| `scripts/train_ppo.py` | `--sigma-anneal-from`·`--imitation-floor` | 2·3 |
| `vtd_rl/policy/net.py` | `set_log_std()`(어닐링이 쓸 설정자) | 2 |
| `docs/reports/m4e-*.md` | 진단 기록·성적표·해석 | 1·5 |

**같은 파일을 여러 작업이 쓰는 곳**: `scripts/train_ppo.py` 를 작업 2·3 이 모두 고친다 → **순차로 돌린다(병렬 금지).**

---

### Task 1: 정체 순간에 무엇이 모자란지 잰다

**Files:**
- Create: `scripts/diag_stall.py`, `tests/policy/test_diag_stall.py`
- Create: `docs/reports/m4e-stall.md`(짧은 측정 기록)

**Interfaces:**
- Produces: `scripts/diag_stall.py --checkpoint <ac.pt> --stage stage1 --seeds 0 1 2 [--out <jsonl>]`
  판을 **결정적으로** 몰면서 걸음마다 `tanh(mean)` 의 가속·조향과 실제 전진 속도를 모으고, 판이 `stalled` 로 끝났으면 **멈추기 직전 N 걸음**의 값을 따로 낸다. 같은 관측에서 **선생님이 무엇을 냈는지**도 함께 낸다.
- Consumes: `vtd_rl.policy.evaluate.run_policy_episode`, `vtd_rl.env.teacher_policy`(선생님 행동), `DrivePolicy.act`

**왜 이게 먼저인가:** "평균 가속이 모자라 멈춘다" 는 아직 **가설**이다. 종료 사유가 전부 정체라는 것까지만 쟀다. 개입 둘(σ 어닐링·모방 바닥)은 **둘 다 가속을 키우는 방향**이라, 원인이 가속이 아니면 둘 다 헛돈다. **하루를 쓰기 전에 반나절로 확인한다** — M4d 에서 50 만 스텝 확인 사격이 3 시간을 아낀 것과 같은 이유다.

**꼭 답해야 할 질문 셋:**
1. 정체 직전 결정적 가속(`tanh(mean)[1]`)이 **선생님보다 낮은가**? 얼마나?
2. 같은 관측에서 **확률 표본**은 그보다 높은가(잡음이 문턱을 넘겨 주는가)?
3. 조향은 멀쩡한가(문제가 가속 한 축인지 둘 다인지)?

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/policy/test_diag_stall.py`
```python
import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
CK = os.path.join(REPO, "runs", "omen", "2026-09-26-m3-squash-warm", "policy.pt")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(CK), reason="스쿼시 재적합 체크포인트가 있어야 한다")
def test_정체_진단이_판별_요약을_낸다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "diag_stall.py"),
                          "--checkpoint", CK, "--stage", "stage1", "--seeds", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["checkpoint"] == CK
    assert summary["episodes"] >= 1
    # 이 체크포인트는 정체가 없다(M4d 실측: 18/18 완주) — 그래도 필드는 있어야 한다
    assert "stalled" in summary and "accel" in summary
    assert set(summary["accel"]) >= {"policy_mean", "teacher_mean", "sample_mean"}
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/policy/test_diag_stall.py -q` (timeout 600000)
Expected: FAIL — `can't open file 'scripts/diag_stall.py'`

- [ ] **Step 3: 구현**

`scripts/diag_stall.py` — `scripts/eval_det_vs_stoch.py` 의 인자 처리·요약 JSON 관례를 따라라(그 파일을 먼저 읽어라). 뼈대:

```python
    # 판을 결정적으로 몰면서 걸음마다 모은다. 선생님 행동은 같은 관측에서 따로 뽑는다.
    # `run_policy_episode` 를 그대로 쓰지 않고 직접 루프를 돈다 — 걸음별 값이 필요하다.
    obs, info = env.reset(seed=seed, options={"board": board})
    rows = []
    for _ in range(max_steps):
        act = net.act(obs, deterministic=True)
        smp = net.act(obs, deterministic=False, generator=gen)
        rows.append({"accel_det": float(act["control"][1]),
                     "accel_sample": float(smp["control"][1]),
                     "steer_det": float(act["control"][0]),
                     "speed": <관측에서 읽은 전진 속도>})
        obs, _r, term, trunc, info = env.step(act)
        if term or trunc:
            break
```

**선생님 행동 — 주의해서 정해라.** `vtd_rl/env/teacher_policy.py:70` 의 `run_teacher_in_env(env, seed, options, max_steps)` 는 **판 하나를 통째로** 몬다. 즉 선생님은 **자기 궤적**을 달리므로, 정책이 멈춘 그 지점의 관측을 선생님이 본 적이 없다 — **걸음 번호로 짝지으면 다른 상황을 비교하는 것이다.**

**그러니 선생님 평균은 "같은 판에서 선생님이 낸 가속의 전체 분포" 로만 쓰고, 그렇게 쓴다는 것을 요약과 보고서에 명시해라.** 정체 지점의 직접 대조가 필요하다고 판단되면 **그 방법을 제안만 하고 구현하지 마라** — 내가 정한다.

**진짜 대조군은 선생님이 아니라 `accel_sample` 이다** — 같은 관측·같은 걸음에서 확률 표본이 결정적 값보다 높은지가 "잡음이 문턱을 넘겨 준다" 를 직접 재는 비교다. 그건 위 루프가 이미 같은 `obs` 로 둘 다 뽑으므로 공정하다.

요약 JSON 에 담을 것: 체크포인트, 판 수, 정체 판 수, 그리고 `accel` 아래에 `policy_mean`·`sample_mean`·`teacher_mean`(전체 걸음 평균)과 **정체 판의 마지막 100 걸음** 평균을 따로.

**전진 속도는 관측의 `ego` 벡터 0 번**이다 — `vtd_rl/env/observation.py:86` 의 `ego_vec = [_clip(ego.v, cfg.v_max), ...]`(확인했다). `cfg.v_max` 로 나뉜 값이니 **정규화된 값이라는 것을 요약에 적어라.**

- [ ] **Step 4: 통과 확인**

세 조각 전부 `rc=0`. 기준선 485 / 138 / 26.

- [ ] **Step 5: 커밋**

```bash
git add scripts/diag_stall.py tests/policy/test_diag_stall.py
git commit -m "M4e 진단 — 정체 순간의 결정적 가속을 선생님·표본과 대조하는 스크립트"
git show -s --format=%B HEAD
```

**실행과 기록(`docs/reports/m4e-stall.md`)은 계획 주관자가 OMEN 에서 직접 한다.**

---

### Task 2: 학습 후반에 σ 를 바닥으로 어닐링한다

**Files:**
- Modify: `vtd_rl/policy/net.py`, `scripts/train_ppo.py`
- Test: `tests/policy/test_net.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `DrivePolicy.set_log_std(value: float)` — `log_std` 를 그 값으로 덮어쓰고 `clamp_log_std()` 로 범위 안에 눌러 준다. 기울기 경로와 무관하게 in-place 로 쓴다(`torch.no_grad()`).
  - `scripts/train_ppo.py` 인자 `--sigma-anneal-from`(기본 `None` = 안 함) — 이 스텝부터 학습 끝까지 `log_std` 를 **현재 값에서 `log_std_min` 까지 선형으로** 내린다.
- Consumes: `PolicyConfig.log_std_min`(−2.0), `net.clamp_log_std()`

**가설:** PPO 는 `E[R(tanh(μ+σξ))]` 를 최대화한다 — σ 가 크면 **평균 자체가 좋을 필요가 없다.** σ 를 학습 후반에 0 쪽으로 몰면 최적화 대상이 결정적 모드에 수렴하므로, 끝날 때의 평균이 그대로 배포 가능해야 한다.

**M4d 의 `--log-std-max` 와 무엇이 다른가:** M4d 는 **처음부터 상한을 고정**했다(탐색 폭을 내내 좁힘). 이것은 **후반에만 내린다**(탐색은 하되 마지막에 수렴시킴). 서로 다른 개입이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/policy/test_net.py` 에 더한다:
```python
def test_set_log_std가_값을_덮어쓰고_범위에_넣는다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std(-1.5)
    assert torch.allclose(net.log_std, torch.tensor([-1.5, -1.5]))
    net.set_log_std(-99.0)                      # 하한 아래
    assert torch.allclose(net.log_std, torch.full((2,), net.cfg.log_std_min))
    net.set_log_std(99.0)                       # 상한 위
    assert torch.allclose(net.log_std, torch.full((2,), net.cfg.log_std_max))


def test_set_log_std는_기울기를_안_남긴다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std(-1.5)
    assert net.log_std.grad is None
    assert net.log_std.requires_grad          # 파라미터인 것은 그대로다
```

`tests/rl/test_train_ppo.py` 에 더한다:
```python
def test_sigma_anneal_기본값은_None이라_아무_일도_안_한다():
    mod = _load_train_ppo_module()
    a = mod._build_parser().parse_args(["--out", "/tmp/x", "--steps", "1"])
    assert a.sigma_anneal_from is None


def test_sigma_anneal_스케줄이_시작전엔_그대로_끝엔_하한이다():
    """`_sigma_anneal_target(step, start, total, cur, floor)` 의 값을 직접 잠근다."""
    mod = _load_train_ppo_module()
    f = mod._sigma_anneal_target
    assert f(step=100, start=1000, total=2000, cur=-0.5, floor=-2.0) is None   # 아직 아님
    assert f(step=1000, start=1000, total=2000, cur=-0.5, floor=-2.0) == pytest.approx(-0.5)
    assert f(step=1500, start=1000, total=2000, cur=-0.5, floor=-2.0) == pytest.approx(-1.25)
    assert f(step=2000, start=1000, total=2000, cur=-0.5, floor=-2.0) == pytest.approx(-2.0)
    assert f(step=9999, start=1000, total=2000, cur=-0.5, floor=-2.0) == pytest.approx(-2.0)


@pytest.mark.slow
def test_sigma_anneal을_주면_끝에서_하한에_닿는다(tmp_path):
    import json
    import os
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..", "..")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run(
        [os.path.join(repo, ".venv", "bin", "python"),
         os.path.join(repo, "scripts", "train_ppo.py"),
         "--out", str(tmp_path / "run"), "--smoke", "--steps", "4000",
         "--sigma-anneal-from", "1000"],
        capture_output=True, text=True, env=env, cwd=repo, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    rows = [json.loads(x) for x in open(tmp_path / "run" / "log.jsonl") if x.strip()]
    assert rows, "로그가 비었다"
    assert max(rows[-1]["log_std"]) == pytest.approx(-2.0, abs=0.05), rows[-1]["log_std"]
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/policy/test_net.py -q -k set_log_std` (timeout 600000)
Expected: FAIL — `AttributeError: 'DrivePolicy' object has no attribute 'set_log_std'`

- [ ] **Step 3: 구현**

`vtd_rl/policy/net.py`:
```python
    @torch.no_grad()
    def set_log_std(self, value: float):
        """`log_std` 를 통째로 `value` 로 덮어쓰고 범위 안에 눌러 준다.

        M4e 의 σ 어닐링이 쓴다 — PPO 는 `E[R(tanh(μ+σξ))]` 를 최대화하므로 σ 가 크면 평균
        자체가 좋을 필요가 없다. 후반에 σ 를 바닥으로 몰아 최적화 대상을 결정적 모드에
        수렴시킨다. `forward()` 는 여전히 clamp 하지 않는다(기울기를 죽인다) — 여기서만 쓴다.
        """
        self.log_std.fill_(float(value))
        self.clamp_log_std()
```

`scripts/train_ppo.py`:
- 스케줄 함수를 **이름 있는 함수로** 뺀다(테스트가 직접 부른다):
```python
def _sigma_anneal_target(step: int, start: int | None, total: int, cur: float, floor: float):
    """σ 어닐링의 이번 걸음 목표값. 안 켰거나 아직 시작 전이면 `None`(건드리지 않는다).

    `start` 부터 `total` 까지 `cur`(어닐링 시작 시점의 값)에서 `floor` 까지 선형으로 내린다.
    """
    if start is None or step < start:
        return None
    span = max(total - start, 1)
    t = min(max((step - start) / span, 0.0), 1.0)
    return cur + (floor - cur) * t
```
- 인자 `--sigma-anneal-from`(int, 기본 `None`)을 더한다.
- 학습 루프에서 **`opt.step()` 뒤**(σ 하한 규약과 같은 자리) 목표값이 `None` 이 아니면 `net.policy.set_log_std(target)` 을 부른다. **어닐링 시작 시점의 `log_std` 를 한 번 기억해 두고 그 값을 `cur` 로 쓴다** — 매 걸음 현재값을 쓰면 지수 감쇠가 되어 스케줄이 달라진다.

- [ ] **Step 4: 통과 확인** — 세 조각 전부 `rc=0`.

- [ ] **Step 5: 돌연변이로 확인**

커밋한 **뒤**: ① `set_log_std` 의 `clamp_log_std()` 호출 삭제 ② `_sigma_anneal_target` 의 `cur` 을 매 걸음 현재값으로 바꾸기. 각각 잡히는지 보고해라.

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/policy/net.py scripts/train_ppo.py tests/policy/test_net.py tests/rl/test_train_ppo.py
git commit -m "M4e — 학습 후반에 σ 를 하한으로 어닐링하는 선택지(기본값은 꺼짐)"
git show -s --format=%B HEAD
```

---

### Task 3: 모방 앵커에 바닥을 둔다

**Files:**
- Modify: `vtd_rl/rl/ppo.py`, `scripts/train_ppo.py`
- Test: `tests/rl/test_ppo.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `PPOConfig.imitation_floor: float = 0.0` — `imitation_coef()` 가 이 값 아래로 안 내려간다.
  - `scripts/train_ppo.py` 인자 `--imitation-floor`(기본 `0.0` = 지금과 같음)
- Consumes: 기존 `imitation_coef(step, cfg) = cfg.imitation_coef0 * 0.5 ** (step / cfg.imitation_half_life)`

**가설:** 출발점(스쿼시 재적합 학생)은 결정적으로 **18/18 완주**한다. 앵커가 있는 동안에도 완주가 유지되다가 계수가 `0.5^(step/2M)` 로 내려가며(3M 에서 0.354) 무너진다. **앵커가 평균을 배포 가능한 곳에 붙들고 있었다면**, 작은 바닥을 남기는 것만으로 결정적 모드가 살아야 한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_ppo.py` 에 더한다(이 파일은 이미 `PPOConfig`·`imitation_coef` 를 임포트한다 — 확인하고, 없으면 임포트 줄에 더해라):
```python
def test_모방_계수에_바닥을_둘_수_있다():
    cfg = PPOConfig(imitation_half_life=1_000_000, imitation_floor=0.2)
    assert imitation_coef(0, cfg) == pytest.approx(1.0)
    assert imitation_coef(1_000_000, cfg) == pytest.approx(0.5)
    # 바닥 아래로 안 내려간다
    assert imitation_coef(10_000_000, cfg) == pytest.approx(0.2)
    assert imitation_coef(100_000_000, cfg) == pytest.approx(0.2)


def test_모방_바닥_기본값은_0이라_예전과_같다():
    cfg = PPOConfig(imitation_half_life=1_000_000)
    assert cfg.imitation_floor == 0.0
    assert imitation_coef(10_000_000, cfg) == pytest.approx(0.5 ** 10)
```

`tests/rl/test_train_ppo.py` 에 더한다:
```python
def test_imitation_floor_인자가_PPOConfig에_반영된다():
    mod = _load_train_ppo_module()
    a = mod._build_parser().parse_args(
        ["--out", "/tmp/x", "--steps", "1", "--imitation-floor", "0.25"])
    assert a.imitation_floor == pytest.approx(0.25)
    cfg = mod._build_cfg(a)
    assert cfg.imitation_floor == pytest.approx(0.25)
```
**`_build_cfg(a) -> PPOConfig`(`scripts/train_ppo.py:274`)가 CLI 값을 `PPOConfig` 로 옮기는 자리다 — 확인했다.** `PPOConfig` 는 **frozen dataclass** 라 `dataclasses.replace` 로 덮어쓴다. 그 함수의 `replace(...)` 호출에 `imitation_floor=a.imitation_floor` 를 더해라. `tests/rl/test_ppo.py:10` 이 이미 `PPOConfig`·`imitation_coef` 를 임포트한다 — **새 임포트가 필요 없다.**

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/test_ppo.py -q -k 모방` (timeout 600000)
Expected: FAIL — `TypeError: PPOConfig.__init__() got an unexpected keyword argument 'imitation_floor'`

- [ ] **Step 3: 구현**

`vtd_rl/rl/ppo.py`:
```python
    imitation_floor: float = 0.0   # 모방 계수가 이 값 아래로 안 내려간다. M4e: 출발점은
                                   # 결정적으로 18/18 완주하는데 앵커가 사라지며 무너진다 —
                                   # 앵커가 평균을 배포 가능한 곳에 붙들고 있었다는 가설.
```
```python
def imitation_coef(step: int, cfg: PPOConfig) -> float:
    decayed = cfg.imitation_coef0 * 0.5 ** (step / max(cfg.imitation_half_life, 1))
    return max(decayed, cfg.imitation_floor)
```

`scripts/train_ppo.py`: 인자 `--imitation-floor`(float, 기본 `PPOConfig().imitation_floor`)를 더하고 `PPOConfig` 를 만드는 자리에 넘긴다.

- [ ] **Step 4: 통과 확인** — 세 조각 전부 `rc=0`. **기본값이 0.0 이라 기존 테스트가 하나도 안 깨져야 한다.**

- [ ] **Step 5: 돌연변이로 확인**

커밋한 **뒤**: ① `max(decayed, cfg.imitation_floor)` 를 `decayed` 로 되돌리기 ② `--imitation-floor` 를 `PPOConfig` 로 안 넘기기(하드코딩 0.0). 각각 잡히는지 보고해라. **②는 M4d 에서 같은 유형이 두 번 뚫린 자리다**(`--violation-mode`·`config=`) — 로그엔 값이 찍히는데 학습은 안 쓰는 조합이 생긴다.

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/rl/ppo.py scripts/train_ppo.py tests/rl/test_ppo.py tests/rl/test_train_ppo.py
git commit -m "M4e — 모방 계수에 바닥을 두는 선택지(기본값 0.0 = 예전과 같음)"
git show -s --format=%B HEAD
```

---

### Task 4: 본 실험

**Files:** 없음(실행과 회수만). **계획 주관자가 OMEN 에서 직접 돌린다.**

**돌릴 설정** — 전부 OMEN 30 환경·3M 스텝·`--device cpu`·시드 0·1·2, `--init runs/omen/2026-09-26-m3-squash-warm/policy.pt`, `--comfort-on-intent --entropy-coef 0.05 --violation-mode once_per_section`(= M4d 의 `align`), `--final-eval-seeds 3` 유지:

| 이름 | 바뀌는 것 | 무엇을 가르나 |
|---|---|---|
| `m4d-align` | 없음(M4d `align` 재현) | 기준선 |
| `sigma-anneal` | `--sigma-anneal-from 2000000` | 후반 σ 수렴이 결정적 모드를 살리나 |
| `imit-floor` | `--imitation-floor 0.3` | 앵커를 남기면 결정적 모드가 유지되나 |

**Task 1 의 진단 결과가 "가속이 모자란 게 아니다" 로 나오면 이 두 설정을 그대로 돌리지 마라** — 계획 주관자에게 보고하고 지시를 기다려라.

- [ ] **Step 1: 코드를 올리고 OMEN 에서 빠른 스위트 확인** (`rc=0`)
- [ ] **Step 2: 세 설정 × 시드 3(약 3 시간).** `scripts/sweep_ppo.py --seeds 0 1 2 --resume`, 설정마다 `--name` 다르게. `nohup` 분리 실행. **감시 필터에 실패 신호(`Traceback|Killed|OOM`)를 넣고, 앵커(`^`)·대괄호 이스케이프는 쓰지 마라**(ssh 를 거치면 깨진다 — M4c 에서 두 번 당했다). **예상 시각에 이벤트가 없으면 감시를 믿지 말고 로그를 직접 확인해라.**
- [ ] **Step 3: 결정적·확률적 평가를 최선·3M 체크포인트 **둘 다**.** `scripts/eval_det_vs_stoch.py`. **M4d 는 최선만 재서 결론을 틀렸다 — 개입은 3M 에서야 문다.**
- [ ] **Step 4: 종료 사유 집계.** 정체가 줄었는지가 이 마일스톤의 직접 지표다.

---

### Task 5: 성적표

**Files:**
- Create: `docs/reports/m4e-ppo.md`(스크립트 생성), `docs/reports/m4e-ppo-notes.md`(해석)
- Modify: `README.md`

성적표는 `scripts/report_m4a.py` 로 만든다(`--sweep` 셋, `--title "M4e 성적표 — 결정적 모드"`, `--student-label "M4e 학생"`, `--run` 은 **선택 없이 시드 0**).

**해석 노트에 반드시 담을 것:**
- 목표 판정 각 줄, **결정적·확률적 둘 다**, 그리고 **종료 사유(정체 수)**
- 점수는 **`mean_score_completed`** 를 주 자로(세 자를 다 보이되)
- **개입 둘 중 어느 것이 들었나 — 안 들었으면 "안 들었다" 고 적어라.** 이 마일스톤은 가설 둘을 가리는 것이 목적이지 통과시키는 것이 목적이 아니다.
- **이 성적표가 주장하지 않는 것**(설정 3 개·시드 3 개·단계①②뿐·`--sigma-anneal-from` 과 `--imitation-floor` 각각 한 값만)
- **M4c 성적표가 `mean_score_raw` 를 썼는지 확인한 결과**(M4d 가 넘긴 숙제 — `docs/reports/m4c-ppo.md` 를 읽고 그 주장들이 자를 바꿔도 서는지 한 문단으로)
- M4f 로 넘기는 것

**목표를 못 채우면 기준을 고치지 마라.** 숫자와 관찰을 적는다.

---

## 이 계획이 미루는 것

- **스펙 걱정("한 번 깎인 뒤 계속 어긴다")을 제대로 재는 계측.** M4d 가 쓴 구간-슬롯 수는 구조적으로 못 잰다(`sheet` 는 슬롯당 등급 하나). 걸음 단위 위반 지속을 재야 하는데, 결정적 모드가 먼저다 — 지금은 배포 가능한 정책 자체가 없다.
- **항목 ⑮(리스폰)을 채점기와 맞추는 것.** 알고도 다르게 둔 자리이고 오프라인 세계에서 아직 안 밟혔다.
- 커리큘럼 단계 ③④⑤.
- 학습률·클립·롤아웃 길이 탐색. 시간 상관 탐색(OU 잡음)·행동 반복.
- 앞 마일스톤들이 넘긴 것: `_goal_lines` 의 `"보류" in v.line` 문자열 매칭, `OutcomeCounter` 가 0 인 사유의 키를 안 만드는 것, `ReturnTracker` 의 판 길이 +1 편향, `RewardConfig.rule_scale` 이 커리큘럼에서 설정되지 않는 것, `RewardShaper` 의 판 첫 걸음 승차감이 0 인 것, `tests/rl/` 조각이 650 초로 도구 상한에 걸리는 것, `report_m4a.py` 성적표에서 항목⑦이 두 번 찍히고 열 합이 합계와 안 맞는 것.
