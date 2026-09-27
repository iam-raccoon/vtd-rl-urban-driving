# M4d — 보상을 대회 채점과 맞춘다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 보상의 위반 항을 대회 채점기와 같은 규칙(같은 항목은 **구간당 한 번**, 심화되면 차액만)으로 바꾸고, 그것이 PPO 가 점수를 올리게 만드는지 잰다.

**Architecture:** `ViolationTracker` 에 채점기의 `Sheet` 와 **같은 누적 규칙**을 넣는다(기존 `repeat_gap` 방식은 기본값으로 남긴다). 평가 경로가 실행의 보상 설정을 받게 고쳐 측정이 학습 목적함수와 같은 자를 쓰게 하고, `log_std_max` 를 CLI 로 열어 M4c 가 못 가른 "결정적 모드 붕괴가 σ 때문인가" 를 **개입**으로 가른다.

**Tech Stack:** Python 3.10, PyTorch 2.14(cpu), gymnasium 1.3, numpy **1.26.4 고정**, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`

**직전 증거:** `docs/reports/m4c-ppo.md`(+`-notes`), `docs/reports/m4c-split.md`, `docs/reports/m4c-refit.md`

## 왜 이 순서인가 (M4c 가 남긴 숫자)

- 판당 항목별 보상 실측: 진행 +99.7 · 시간 −36.0 · **위반 −96 ~ −115** · 승차감 −63.4(의도 기반으로 고친 뒤).
- 승차감을 **완전히 0** 으로 만들어도 완주 판 리턴이 −10.5 라 "멈추면 −6.2" 를 못 넘는다. **위반이 남은 병목이다.**
- 그 위반 항은 **과제와 어긋나 있다** — `vtd_rl/env/reward.py` 는 같은 (항목, 구간)을 `repeat_gap=1s` 마다 다시 깎는데, 대회 채점기(`third_party/rule_stack/eval/score_fma.py` 의 `Sheet`)는 **구간당 한 번**만 깎고 심화되면 차액(+3)만 더 깎는다. 그래서 대회 기준 94.2/100 인 학생이 보상에서 −115 를 맞는다.
- **PPO 가 점수를 올릴 이유가 없다** — 최대화하는 그 리턴이 점수와 다른 것을 재고 있다.

## ★ 이 계획은 스펙을 뒤집는다 — 그 논거

스펙 §5(`docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md:151`)는 지금 동작을 **의도**로 못 박아 놨다:

> 대회는 같은 항목을 구간당 1회만 깎지만, 학습 보상은 **위반마다** 깎는다(한 번 깎인 뒤 계속 어기는 걸 막는다).

**그러니 이건 버그 수정이 아니라 설계 변경이다.** 스펙이 든 이유는 진짜 위험이다 — 구간당 한 번만 깎으면 **그 구간에서 한 번 깎인 뒤에는 같은 항목을 공짜로 계속 어길 수 있다.**

뒤집는 근거:

1. **스펙의 그 문장은 가정이었고, 그 대가가 이제 측정됐다.** 대회 기준 94.2/100 인 학생이 보상에서 **−115** 를 맞는다(M4c 실측). 과제 가치(+113.6)보다 위반 하나가 더 크다. 그 상태에서 PPO 가 최대화하는 리턴은 점수와 **다른 것**이고, 실제로 3M 을 돌고도 점수를 못 올렸다(확률적끼리 95.2 → 92.9~94.3).
2. **스펙의 걱정도 아직 가정이다.** "한 번 깎인 뒤 계속 어긴다" 가 실제로 일어나는지는 안 재봤다.
3. **그래서 둘 다 실험으로 가린다.** 기본값은 `"repeat"`(스펙 그대로)로 두고 `"once_per_section"` 을 **선택지**로 넣는다. 본 실험이 두 설정을 나란히 돌린다.

**Ruling: 스펙 §5 의 그 줄은 이 실험 결과가 나온 뒤에 고친다.** 지금 고치면 아직 안 나온 결과를 미리 적는 것이 된다. 결과가 `once_per_section` 손을 들어주면 스펙을 고치고, 아니면 스펙이 옳았다는 증거로 남긴다.

**Ruling: 스펙이 걱정한 실패 모드를 Task 5 에서 반드시 검사한다.** 성적표에 다음을 넣는다 — `align` 설정이 `m4c-base` 보다 **구간당 서로 다른 (항목) 수**나 **한 항목을 어긴 총 시간**이 늘었는가. 늘었으면 스펙의 걱정이 현실이 된 것이고, 처방은 "구간당 한 번 + 작은 잔여 벌점" 이다(이번 범위 밖, M4e).

## Global Constraints

- 커밋·PR 에 **Claude 표기 금지**(`Co-Authored-By` / `Claude-Session` / "Generated with" / 🤖 / anthropic 주소). **커밋 뒤 `git show -s --format=%B HEAD` 를 직접 실행해 눈으로 확인하고 보고할 것** — M4c 에서 적어 두기만 했더니 한 번 뚫렸다.
- `third_party/rule_stack` **수정 금지**(읽기만 한다). 지도·시나리오가 그 안에 있고 대회 자료다.
- `runs/`·`*.pt`·`*.npz`·CSV·`*.xodr`·`*.xml` **커밋 금지**. 성적표 `.md` 만 커밋한다.
- numpy **1.26.4 고정**(2.x 금지).
- 문서·주석·커밋 메시지는 **한국어**.
- **성적표 숫자는 전부 스크립트가 생성한다.** 손으로 적지 말 것(M3 에서 세 번, M4c 에서 한 번 고쳤다).
- `vtd_rl/eval/verdict.py` 의 `M3_SCORE = {"stage1": 98.5, "stage2": 94.2}` 를 **낮추지 말 것**.
- **테스트를 돌리는 모든 Bash 호출에 `timeout: 600000`(ms)을 빠짐없이 붙일 것.** 하나라도 빠지면 하네스가 자동으로 배경으로 넘기고, 그러면 보고 없이 멈추게 된다(M4a~M4c 에서 **일곱 번** 일어났다). `run_in_background`·Monitor 금지.
- 테스트는 겹치지 않는 **세 조각**으로 나눠 돌리고 **종료 코드(rc)** 로 판정한다("N passed" grep 은 "1 failed" 를 삼킨다):
  - `env -u PYTHONPATH .venv/bin/python -m pytest -q -m "not slow"`
  - `env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/ -q` ← **약 602 초**로 상한에 걸려 `timeout` 을 제대로 줘도 배경으로 넘어갈 수 있다. 그러면 알림을 기다렸다 출력 파일을 Read 로 읽어 rc 를 확인할 것.
  - `env -u PYTHONPATH .venv/bin/python -m pytest -q -m slow --ignore=tests/rl`
  - **기준선(M4c 병합 시점): 458 / 120 / 25, 전부 rc=0.**
- 이 머신(lab-main)에는 `uv` 가 없다. `.venv` 를 직접 쓴다.
- **커밋을 먼저 하고 돌연변이 실험을 할 것.** 미커밋 상태에서 `git checkout` 을 하면 정상 구현이 날아간다(M4c 에서 두 번 당했다). 돌연변이는 작업 트리에 심고 `git checkout -- <경로>` 로 **경로를 지정해** 원복한다 — 이 패키지는 editable 설치라 `/tmp` 복사본은 진짜 패키지를 임포트해 "전부 살아남음" 이라는 거짓 결과를 준다.
- 긴 학습 실행(3M)은 **OMEN**(`user-OMEN`, 32 코어, `--device cpu`)에서 돈다. 구현자는 돌리지 않는다 — 계획 주관자가 돌린다.

---

## File Structure

| 파일 | 책임 | 작업 |
|---|---|---|
| `vtd_rl/env/reward.py` | `ViolationTracker` 에 채점기와 같은 누적 규칙 추가, `RewardConfig.violation_mode` | 2 |
| `vtd_rl/policy/evaluate.py` | 없음(이미 `config` 인자를 받는다) | — |
| `scripts/train_ppo.py` | 평가에 실행의 보상 설정 전달, `--violation-mode`·`--log-std-max` | 1·2·4 |
| `scripts/report_m4a.py` | `hparams` 에서 보상 설정을 복원해 평가에 전달 | 1 |
| `vtd_rl/rl/reward_cfg.py`(신규) | `hparams`↔`RewardConfig` 변환 한 곳 — `train_ppo` 와 `report_m4a` 가 공유 | 1 |
| `docs/reports/m4d-*.md` | 성적표·해석 | 3·5 |

**두 작업이 같은 파일을 쓰는 곳**: `scripts/train_ppo.py` 를 작업 1·2·4 가 모두 고친다 → **순차로 돌린다(병렬 금지).**

---

### Task 1: 평가가 실행의 보상 설정을 쓰게 한다

**Files:**
- Create: `vtd_rl/rl/reward_cfg.py`, `tests/rl/test_reward_cfg.py`
- Modify: `scripts/train_ppo.py`(`_evaluate_stages`), `scripts/report_m4a.py`
- Test: `tests/rl/test_reward_cfg.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `vtd_rl/rl/reward_cfg.py::reward_config_from_hparams(hparams: dict) -> RewardConfig` — `log.jsonl` 의 `hparams` dict 에서 보상 설정을 복원한다. 모르는 키는 무시하고, 없는 키는 `RewardConfig()` 기본값을 쓴다.
  - `scripts/train_ppo.py::_evaluate_stages(policy, stages, seeds, env_cfg)` — 인자가 하나 늘어난다.
- Consumes: 기존 `RewardConfig`, `EnvConfig`, `evaluate_policy(..., config=)`

**왜 이게 먼저인가:** `evaluate_policy` 는 `config` 인자를 이미 받는데 **아무도 안 넘긴다** — `_evaluate_stages`(`scripts/train_ppo.py:82-83`)와 `report_m4a.py` 둘 다 `EnvConfig()` 기본값이 만들어진다. 그래서 성적표의 "평균 보상" 이 학습 목적함수와 **다른 자**다. 점수·완주율은 심판이 따로 내므로 지금까지는 무해했지만, **작업 2 가 보상을 바꾸는 순간 측정이 통째로 어긋난다.** M4c 최종 리뷰가 지적한 자리다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_reward_cfg.py`
```python
from vtd_rl.env.reward import RewardConfig
from vtd_rl.rl.reward_cfg import reward_config_from_hparams


def test_hparams가_비면_기본값이다():
    assert reward_config_from_hparams({}) == RewardConfig()


def test_승차감_설정을_복원한다():
    cfg = reward_config_from_hparams({"comfort_steer": -0.01, "comfort_accel": -0.005,
                                      "comfort_on_intent": True})
    assert cfg.comfort_steer == -0.01
    assert cfg.comfort_accel == -0.005
    assert cfg.comfort_on_intent is True


def test_모르는_키는_무시한다():
    """`hparams` 는 `vars(argparse.Namespace)` 라 보상과 무관한 키가 잔뜩 들어 있다."""
    cfg = reward_config_from_hparams({"envs": 30, "steps": 3000000, "comfort_steer": -0.02})
    assert cfg.comfort_steer == -0.02 and cfg.progress_total == RewardConfig().progress_total


def test_없는_키는_기본값을_남긴다():
    cfg = reward_config_from_hparams({"comfort_steer": -0.02})
    assert cfg.comfort_accel == RewardConfig().comfort_accel
    assert cfg.comfort_on_intent is False
```

`tests/rl/test_train_ppo.py` 에 더한다(이 파일은 이미 `_load_train_ppo_module()` 헬퍼와 `pytest` 를 쓴다 — 그 패턴을 따라라):
```python
def test_평가가_실행의_보상_설정을_받는다(monkeypatch):
    """`_evaluate_stages` 가 받은 `env_cfg` 를 `evaluate_policy` 로 그대로 넘기는지 값으로 잠근다.

    M4c 최종 리뷰 I6: 여태 `evaluate_policy` 가 `EnvConfig()` 기본값으로 돌아 성적표의
    '평균 보상' 이 학습 목적함수와 다른 자였다. 작업 2 가 보상을 바꾸면 이 어긋남이
    측정을 통째로 무의미하게 만든다.
    """
    mod = _load_train_ppo_module()
    seen = []
    monkeypatch.setattr(mod, "evaluate_policy",
                        lambda policy, boards, seeds=None, config=None: seen.append(config) or {})
    from vtd_rl.env.drive_env import EnvConfig
    from vtd_rl.env.reward import RewardConfig
    cfg = EnvConfig(reward=RewardConfig(comfort_steer=-0.01))
    mod._evaluate_stages(object(), [("stage1", [])], (0,), cfg)
    assert seen == [cfg], seen
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/test_reward_cfg.py -q` (timeout 600000)
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.rl.reward_cfg'`

- [ ] **Step 3: 구현**

`vtd_rl/rl/reward_cfg.py`:
```python
"""`hparams` ↔ `RewardConfig` 변환 — `train_ppo.py` 와 `report_m4a.py` 가 함께 쓴다.

성적표를 만들 때 실행의 보상 설정을 되살려야 평가의 '평균 보상' 이 학습 목적함수와 같은 자가
된다(M4c 최종 리뷰 I6). 두 스크립트가 서로를 임포트하지 않으므로 여기 한 곳에 둔다.
"""
import dataclasses

from vtd_rl.env.reward import RewardConfig

_FIELDS = tuple(f.name for f in dataclasses.fields(RewardConfig))


def reward_config_from_hparams(hparams: dict) -> RewardConfig:
    """`hparams`(= `vars(args)`) 에서 `RewardConfig` 필드와 이름이 같은 것만 골라 덮어쓴다.

    보상과 무관한 키(`envs`·`steps` 등)는 무시하고, 없는 키는 기본값을 남긴다 — 그래서
    보상 필드가 생기기 전의 옛 실행 로그로도 안전하게 돌아간다.
    """
    known = {k: v for k, v in (hparams or {}).items() if k in _FIELDS}
    return dataclasses.replace(RewardConfig(), **known)
```

`scripts/train_ppo.py`:
- `_evaluate_stages` 시그니처에 `env_cfg` 를 더하고 `evaluate_policy(..., config=env_cfg)` 로 넘긴다.
```python
def _evaluate_stages(policy, stages, seeds, env_cfg) -> dict:
    """평가에도 **실행의 보상 설정**을 쓴다 — 안 그러면 성적표의 '평균 보상' 이 학습
    목적함수와 다른 자가 된다(M4c 최종 리뷰 I6). 점수·완주율은 심판이 따로 내므로 영향 없다.
    """
    return {label: evaluate_policy(policy, boards, seeds=seeds, config=env_cfg)
            for label, boards in stages}
```
- 호출부 두 곳(`step >= next_eval` 안의 주기 평가, 마지막 전체 평가)에 `train_env_cfg` 를 넘긴다. **`train_env_cfg` 는 `--smoke` 일 때 `time_limit_scale=0.1` 을 갖는다 — 평가에도 같은 제한이 걸리는 게 맞다**(연습 모드는 짧게 도는 것이 목적이다).

`scripts/report_m4a.py`:
- `reward_config_from_hparams` 를 임포트하고, `log.jsonl` 의 `hparams` 에서 `RewardConfig` 를 복원해 `EnvConfig(reward=...)` 를 만들어 `evaluate_policy(...)` 세 곳(`:547`·`:549`·`:565`)에 `config=` 로 넘긴다.
- **M3 학생(`--m3`)에게도 같은 설정을 쓴다** — 같은 자로 재야 비교가 성립한다.

- [ ] **Step 4: 통과 확인**

세 조각 전부 `rc=0`. 기존 테스트가 하나도 깨지면 안 된다(기본값이 그대로다).

- [ ] **Step 5: 돌연변이로 확인**

커밋한 **뒤**, `_evaluate_stages` 에서 `config=env_cfg` 를 지워 본다 → `test_평가가_실행의_보상_설정을_받는다` 가 잡아야 한다. `git checkout -- scripts/train_ppo.py` 로 원복.

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/rl/reward_cfg.py tests/rl/test_reward_cfg.py scripts/train_ppo.py scripts/report_m4a.py tests/rl/test_train_ppo.py
git commit -m "M4d — 평가가 실행의 보상 설정을 쓰게 한다(성적표의 평균 보상이 학습 목적함수와 같은 자)"
git show -s --format=%B HEAD    # 금지 표기 없음을 눈으로 확인
```

---

### Task 2: 위반을 대회 채점기와 같은 규칙으로 센다

**Files:**
- Modify: `vtd_rl/env/reward.py`, `scripts/train_ppo.py`
- Test: `tests/env/test_reward.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `RewardConfig.violation_mode: str = "repeat"` — `"repeat"`(지금까지) 또는 `"once_per_section"`(채점기와 같음)
  - `ViolationTracker(repeat_gap, violation_mode="repeat")`
  - `ViolationTracker.charge(hits, minor, major) -> tuple[list[Hit], float]` — 셀 히트와 **그 히트들이 실제로 깎는 총점**을 함께 낸다. 심화(경미→중대)는 **차액만** 깎는다.
  - `ViolationTracker.count(hits) -> list[Hit]` — 기존 시그니처 유지(`charge` 에 위임). 기존 테스트가 여기 걸려 있다.
  - `scripts/train_ppo.py` 인자 `--violation-mode {repeat,once_per_section}`(기본 `repeat`)
- Consumes: Task 1 의 `reward_config_from_hparams`(새 필드가 자동으로 실린다 — `_FIELDS` 가 `dataclasses.fields` 에서 오므로 **수정이 필요 없다**)

**채점기의 정확한 규칙**(`third_party/rule_stack/eval/score_fma.py:155-179`, 읽기만 했다):
```python
def hit(self, sec, item, level, why):
    cur = self.state[sec].get(item)
    if cur == "major" or cur == level:
        return                       # 이미 중대거나 같은 등급이면 아무 일도 안 일어난다
    self.state[sec][item] = level    # 없었거나 minor→major 로 올라간다

def penalty(self, sec, item):
    lv = self.state[sec].get(item)
    return MAJOR if lv == "major" else (MINOR if lv == "minor" else 0)
```
즉 **(항목, 구간)마다 최종 등급 하나만큼** 깎는다. 경미를 깎은 뒤 중대가 오면 **차액(−3)만** 더 깎는다. 중대 뒤 경미는 무시한다.

**주의:** 보상의 `minor`·`major` 는 채점기의 −3/−6 과 **부호·크기가 같게** 설정돼 있지만(`RewardConfig.minor=-3.0`, `major=-6.0`) 같은 상수가 아니다. 차액은 `cfg.major - cfg.minor` 로 계산해라 — 상수를 박지 마라.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/env/test_reward.py` 에 더한다. **이 파일은 이미 `Hit`·`ViolationTracker`·`RewardShaper`·`RewardConfig`·`h_board()`·`pytest.approx` 를 쓴다** — 새 임포트가 필요 없다.
```python
def test_구간당_한_번_모드는_같은_항목을_한_번만_센다():
    tr = ViolationTracker(1.0, violation_mode="once_per_section")
    hits = [Hit(t=k * 0.05, sec=0, item=1, level="minor") for k in range(200)]   # 10 초 연속
    counted = [h for k in range(200) for h in tr.count([hits[k]])]
    assert [round(h.t, 2) for h in counted] == [0.0]      # repeat 모드였다면 10 개다


def test_구간이_바뀌면_다시_센다():
    tr = ViolationTracker(1.0, violation_mode="once_per_section")
    counted = tr.count([Hit(0.0, 0, 1, "minor"), Hit(0.1, 1, 1, "minor"), Hit(0.2, 0, 1, "minor")])
    assert [(h.sec, h.t) for h in counted] == [(0, 0.0), (1, 0.1)]


def test_심화되면_차액만_깎는다():
    """경미를 깎은 뒤 중대가 오면 채점기처럼 차액만 더 깎는다(score_fma.Sheet 와 같은 규칙)."""
    cfg = RewardConfig(violation_mode="once_per_section")
    tr = ViolationTracker(cfg.repeat_gap, violation_mode=cfg.violation_mode)
    _c1, p1 = tr.charge([Hit(0.0, 0, 1, "minor")], cfg.minor, cfg.major)
    _c2, p2 = tr.charge([Hit(1.0, 0, 1, "major")], cfg.minor, cfg.major)
    _c3, p3 = tr.charge([Hit(2.0, 0, 1, "minor")], cfg.minor, cfg.major)   # 중대 뒤 경미는 무시
    assert p1 == pytest.approx(cfg.minor)
    assert p2 == pytest.approx(cfg.major - cfg.minor)
    assert p3 == pytest.approx(0.0)
    assert p1 + p2 == pytest.approx(cfg.major)        # 합치면 중대 한 번과 같다


def test_repeat_모드가_기본값이고_예전과_같다():
    assert RewardConfig().violation_mode == "repeat"
    tr = ViolationTracker(1.0)
    hits = [Hit(t=k * 0.05, sec=0, item=1, level="minor") for k in range(60)]
    counted = [h for k in range(60) for h in tr.count([hits[k]])]
    assert [round(h.t, 2) for h in counted] == [0.0, 1.0, 2.0]     # 기존 테스트와 같은 값


def test_구간당_한_번_모드의_판당_위반이_훨씬_작다():
    """같은 히트 열을 두 모드에 먹여 크기 차이를 값으로 잠근다 — 이 마일스톤의 전제다."""
    hits = [Hit(t=k * 0.05, sec=0, item=1, level="minor") for k in range(200)]
    zero = {"control": [0.0, 0.0], "turn": 0}
    base = RewardConfig()
    totals = {}
    for mode in ("repeat", "once_per_section"):
        sh = RewardShaper(h_board(), RewardConfig(violation_mode=mode))
        sh.reset()
        totals[mode] = sum(sh.step([h], 0.0, zero, zero, "running").terms["violation"]
                           for h in hits)
    assert totals["once_per_section"] == pytest.approx(base.minor)
    assert totals["repeat"] < totals["once_per_section"]        # 더 많이(음수로 크게) 깎는다
    assert totals["repeat"] == pytest.approx(base.minor * 10)   # 10 초 / repeat_gap 1 초


def test_reset하면_구간_기록이_지워진다():
    tr = ViolationTracker(1.0, violation_mode="once_per_section")
    assert tr.count([Hit(0.0, 0, 1, "minor")])
    assert not tr.count([Hit(1.0, 0, 1, "minor")])
    tr.reset()
    assert tr.count([Hit(2.0, 0, 1, "minor")])      # 새 판이면 다시 센다
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py -q -k 구간당 또는 심화` (timeout 600000)
Expected: FAIL — `TypeError: ViolationTracker.__init__() got an unexpected keyword argument 'violation_mode'`

- [ ] **Step 3: 구현**

`vtd_rl/env/reward.py`:
- `RewardConfig` 에 필드를 더한다:
```python
    violation_mode: str = "repeat"   # "repeat" = 같은 (항목,구간)을 repeat_gap 마다 다시 센다.
                                     # "once_per_section" = 대회 채점기와 같은 규칙 — (항목,구간)
                                     # 마다 한 번, 경미→중대 심화는 차액만. M4c 실측: repeat 는
                                     # 대회 기준 94.2/100 인 학생에게 판당 −115 를 물려 PPO 가
                                     # 점수를 올릴 이유를 없앤다(docs/reports/m4c-ppo-notes.md).
```
- `ViolationTracker` 를 고친다. **`count()` 는 시그니처를 유지하고 `charge()` 에 위임한다** — 기존 테스트 세 개가 `count()` 에 걸려 있다.
```python
class ViolationTracker:
    def __init__(self, repeat_gap: float, violation_mode: str = "repeat"):
        if violation_mode not in ("repeat", "once_per_section"):
            raise ValueError(f"violation_mode 는 repeat|once_per_section 이어야 한다: {violation_mode!r}")
        self.repeat_gap = repeat_gap
        self.violation_mode = violation_mode
        self._last: dict = {}     # repeat 모드: (item, sec) -> 마지막으로 센 시각
        self._level: dict = {}    # once_per_section 모드: (item, sec) -> 이미 깎은 등급

    def reset(self):
        self._last.clear()
        self._level.clear()

    def count(self, hits):
        """센 히트만 돌려준다 — 감점 크기는 `charge()` 가 함께 낸다."""
        return self.charge(hits, 0.0, 0.0)[0]

    def charge(self, hits, minor: float, major: float):
        """(센 히트, 그 히트들이 깎는 총점)."""
        if self.violation_mode == "once_per_section":
            return self._charge_once(hits, minor, major)
        out, total = [], 0.0
        for h in hits:
            key = (h.item, h.sec)
            last = self._last.get(key)
            if last is None or h.t - last >= self.repeat_gap - 1e-9:
                self._last[key] = h.t
                out.append(h)
                total += major if h.level == "major" else minor
        return out, total

    def _charge_once(self, hits, minor: float, major: float):
        """대회 채점기 `score_fma.Sheet` 와 같은 규칙 — (항목,구간)마다 최종 등급 한 번만."""
        out, total = [], 0.0
        for h in hits:
            key = (h.item, h.sec)
            cur = self._level.get(key)
            if cur == "major" or cur == h.level:
                continue                       # 이미 중대거나 같은 등급이면 아무 일도 없다
            self._level[key] = h.level
            out.append(h)
            total += (major - minor) if cur == "minor" else (
                major if h.level == "major" else minor)
        return out, total
```
- `RewardShaper.__post_init__` 에서 `ViolationTracker(self.cfg.repeat_gap, self.cfg.violation_mode)` 로 만든다.
- `RewardShaper.step` 의 위반 계산을 `charge` 로 바꾼다. **충돌 항목은 위반 합계에서 빼는 기존 규칙을 유지해야 한다** — `charge` 는 충돌도 포함해 세므로, 감점은 충돌을 뺀 것으로 다시 구한다:
```python
        # 충돌 항목(⑪⑭)은 `collision` 항이 따로 −50 을 물리므로 위반 합계에서 뺀다(기존 규칙).
        # `charge` 가 감점까지 내므로, 충돌 히트를 **처음부터 갈라서** 두 번 부른다 —
        # 그래야 "감점에서 충돌을 뺀다" 를 뺄셈으로 다시 구하지 않아도 된다.
        rule_hits = [h for h in hits if h.item not in COLLISION_ITEMS]
        col_hits = [h for h in hits if h.item in COLLISION_ITEMS]
        counted, penalty = self.tracker.charge(rule_hits, cfg.minor, cfg.major)
        col_counted, _col_penalty = self.tracker.charge(col_hits, cfg.minor, cfg.major)
        collision = bool(col_counted)
        violation = penalty * cfg.rule_scale
```
  **`RewardStep.counted` 는 둘을 합친 개수여야 한다** — `len(counted) + len(col_counted)`. 기존 테스트 `tests/env/test_reward.py:52`(`hit_out.counted == 1`)와 `:107`(`out.counted == 1`)이 여기 걸려 있다.

  `charge` 를 두 번 부르는 것은 상태를 두 번 건드리는 것처럼 보이지만, 두 호출이 **서로 다른 항목 키**만 만지므로 간섭하지 않는다(키는 `(item, sec)` 이고 충돌 항목과 규칙 항목은 겹치지 않는다).

`scripts/train_ppo.py`:
- 인자를 더한다:
```python
    ap.add_argument("--violation-mode", choices=("repeat", "once_per_section"),
                    default=default_reward_cfg.violation_mode,
                    help="위반을 세는 규칙. once_per_section 은 대회 채점기와 같다"
                         "((항목,구간)마다 한 번, 심화는 차액만)")
```
- `_reward_config_from_args` 에 `violation_mode=a.violation_mode` 를 더한다.

- [ ] **Step 4: 통과 확인**

세 조각 전부 `rc=0`. **기존 `tests/env/test_reward.py` 가 하나도 깨지면 안 된다** — 기본값이 `"repeat"` 다.

- [ ] **Step 5: 돌연변이로 확인**

커밋한 **뒤** 세 가지를 심어 잡히는지 보고해라:
1. `_charge_once` 의 심화 차액 `(major - minor)` 를 `major` 로 → `test_심화되면_차액만_깎는다` 가 잡아야 한다.
2. `_charge_once` 의 키에서 `h.sec` 를 빼 `(h.item,)` 으로 → `test_구간이_바뀌면_다시_센다` 가 잡아야 한다.
3. `RewardConfig.violation_mode` 기본값을 `"once_per_section"` 으로 → 기존 `test_같은_항목은_1초에_한_번만_센다` 가 잡아야 한다.

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/env/reward.py scripts/train_ppo.py tests/env/test_reward.py tests/rl/test_train_ppo.py
git commit -m "M4d — 위반을 대회 채점기와 같은 규칙으로 세는 모드 추가(기본값은 예전 그대로)"
git show -s --format=%B HEAD
```

---

### Task 3: 새 모드가 채점기와 정말 같은지 판 하나로 확인한다

**Files:**
- Test: `tests/env/test_drive_env.py`

**Interfaces:**
- Consumes: Task 2 의 `violation_mode="once_per_section"`
- **이 작업은 새 코드를 안 만든다.** 단위 테스트가 규칙을 잠갔으니, 이제 **진짜 판 하나를 끝까지 몰아** 보상의 위반 합계가 채점기의 감점과 맞는지 본다.

**왜 필요한가:** 단위 테스트는 내가 이해한 규칙을 잠근다. 이 테스트는 **내가 규칙을 옳게 이해했는지**를 채점기 자신에게 물어본다. M4c 에서 "계산은 맞았는데 전제가 틀렸다" 를 두 번 겪었다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/env/test_drive_env.py` 에 더한다. 이 파일은 이미 `np`·`pytest`·`EnvConfig`·`VtdDriveEnv` 를 모듈 수준에서 임포트한다.
```python
@pytest.mark.slow
def test_구간당_한_번_모드의_위반_합계가_채점기_감점과_같다():
    """보상의 위반 항이 대회 채점기가 실제로 깎은 점수와 일치해야 한다.

    `info["result"]["sheet"]` 는 심판이 낸 구간별 감점표다. 그 표의 위반 감점 합(충돌 항목
    ⑪⑭ 제외)과, 판 내내 쌓은 `reward_terms["violation"]` 의 합이 같아야 한다 — 그게
    "보상이 채점과 같은 것을 잰다" 는 이 마일스톤의 정의다.
    """
    from vtd_rl.env.reward import COLLISION_ITEMS, RewardConfig
    from vtd_rl.world.board import load_curriculum

    boards = load_curriculum("curricula/stage1.json")[1]
    cfg = RewardConfig(violation_mode="once_per_section")
    env = VtdDriveEnv(boards, EnvConfig(reward=cfg))
    try:
        env.reset(seed=0, options={"board": boards[0].name})
        total, info = 0.0, None
        for _ in range(20000):
            _o, _r, term, trunc, info = env.step(
                {"control": np.array([0.05, 0.6], np.float32), "turn": 0})
            total += info["reward_terms"]["violation"]
            if term or trunc:
                break
        assert info is not None and "result" in info, "판이 안 끝났다"
        sheet = info["result"]["sheet"]
        # sheet 는 구간별 {항목: 등급} 이다. 충돌 항목은 보상에서 따로 처리하므로 뺀다.
        expect = sum(cfg.major if lv == "major" else cfg.minor
                     for sec in sheet for item, lv in sec.items()
                     if int(item) not in COLLISION_ITEMS)
        assert total == pytest.approx(expect, abs=1e-6), (total, expect)
    finally:
        env.close()
```

- [ ] **Step 2: 실패 확인 — 그리고 `sheet` 의 실제 모양을 먼저 확인해라**

**위 테스트는 `sheet` 가 "구간마다 `{항목: 등급}` dict 의 리스트" 라고 가정한다. 이 가정을 먼저 실측으로 확인해라:**
```bash
env -u PYTHONPATH .venv/bin/python -c "
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.world.board import load_curriculum
import numpy as np
b = load_curriculum('curricula/stage1.json')[1]
e = VtdDriveEnv(b, EnvConfig()); e.reset(seed=0, options={'board': b[0].name})
for _ in range(20000):
    _o,_r,t,tr,i = e.step({'control': np.array([0.05,0.6],np.float32),'turn':0})
    if t or tr: break
print(type(i['result']['sheet']), repr(i['result']['sheet'])[:400])
e.close()"
```
**모양이 다르면 테스트를 실제 모양에 맞춰 고치고, 무엇이 달랐는지 보고해라.** 계획을 그대로 믿지 마라 — 이 계획서를 쓴 사람이 M4c 에서 벡터화 info 의 모양을 두 번 틀리게 적었다.

Expected(고친 뒤): FAIL — 두 합이 다르다(아직 `charge` 가 채점기와 안 맞거나, 내 규칙 이해가 틀렸다).

- [ ] **Step 3: 어긋나면 원인을 가려라**

두 합이 다르면 **`reward.py` 를 고치기 전에 무엇이 다른지 먼저 적어라.** 후보:
- 심판이 내는 `hits` 의 `sec` 와 채점표의 구간 번호가 다르다
- 심판이 같은 걸음에 같은 항목을 여러 번 낸다
- 충돌 항목 제외 규칙이 다르다
- 보상은 걸음마다 세는데 채점표는 판 끝에 한 번 만든다(순서 차이)

**차이를 숫자로 적고(어느 항목·구간에서 얼마가 벌어지는지) 그 다음에 고쳐라.** 고치는 쪽이 `reward.py` 인지 테스트의 기대식인지도 근거와 함께 보고해라.

- [ ] **Step 4: 통과 확인**

Run: 세 조각 전부 `rc=0`.

- [ ] **Step 5: 커밋**

```bash
git add tests/env/test_drive_env.py
git commit -m "M4d — 구간당 한 번 모드의 위반 합계가 채점기 감점과 같은지 판 하나로 확인"
git show -s --format=%B HEAD
```

---

### Task 4: σ 상한을 CLI 로 연다 (결정적 모드 붕괴의 개입 실험용)

**Files:**
- Modify: `scripts/train_ppo.py`
- Test: `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces: `scripts/train_ppo.py` 인자 `--log-std-max`(기본값 = 체크포인트/`PolicyConfig` 값을 그대로 씀, 즉 `None`)
- Consumes: `PolicyConfig.log_std_max`(기본 0.5), `net.clamp_log_std()`

**왜 필요한가:** M4c 는 3M 체크포인트의 **결정적** 모드만 무너지는 것을 발견했고(결정적 0~33%, 확률적 100%), σ 가 커진 것과 정책이 출발점에서 멀어진 것(`drift_rel` 0 → 0.312)이 **함께** 움직여 원인을 못 갈랐다.

**⚠ 끝난 체크포인트의 σ 를 낮춰서는 못 잰다.** `DrivePolicy.act(deterministic=True)` 는 `tanh(mean)` 만 쓰고 `log_std` 를 **아예 안 본다**(`vtd_rl/policy/net.py:158-161`). σ 가 듣는 곳은 **학습**이다 — PPO 가 어떤 평균으로 수렴하느냐. 그래서 개입은 **σ 상한을 조이고 다시 학습**하는 것뿐이다.

- [ ] **Step 0: 인자 파서를 함수로 뺀다(작은 리팩터, 이 작업의 전제)**

**`scripts/train_ppo.py` 에는 `_parse_args` 같은 함수가 없다 — argparse 가 `main()`(`:315`) 안에 인라인으로 70 줄쯤 들어 있어 CLI 기본값을 테스트할 길이 없다.** M4c 가 플래그를 넷(`--comfort-steer`·`--comfort-accel`·`--comfort-on-intent`·`--entropy-mode`) 더했는데 **기본값을 잠근 테스트가 하나도 없는** 이유가 이것이다.

`main()` 의 `ap = argparse.ArgumentParser()` 부터 `a = ap.parse_args()` 직전까지를 그대로 옮겨 `_build_parser() -> argparse.ArgumentParser` 로 만들고, `main()` 은 `a = _build_parser().parse_args()` 로 시작하게 한다. **인자 정의는 한 줄도 바꾸지 마라** — 순수 이동이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_train_ppo.py` 에 더한다:
```python
def test_log_std_max_기본값은_None이라_체크포인트_값을_쓴다():
    """기본값은 `None` — 기존 실행이 하나도 안 바뀐다."""
    mod = _load_train_ppo_module()
    a = mod._build_parser().parse_args(["--out", "/tmp/x", "--steps", "1"])
    assert a.log_std_max is None


def test_M4c_가_더한_플래그들의_기본값도_잠근다():
    """`_build_parser()` 를 뺀 김에 M4c 플래그 넷의 기본값을 잠근다 — 여태 하나도 안 잠겨 있었다."""
    from vtd_rl.env.reward import RewardConfig
    mod = _load_train_ppo_module()
    a = mod._build_parser().parse_args(["--out", "/tmp/x", "--steps", "1"])
    assert a.comfort_steer == RewardConfig().comfort_steer
    assert a.comfort_accel == RewardConfig().comfort_accel
    assert a.comfort_on_intent is False
    assert a.entropy_mode is None
    assert a.violation_mode == RewardConfig().violation_mode      # 작업 2 가 더한 것


@pytest.mark.slow
def test_log_std_max를_주면_학습_내내_그_아래로_묶인다(tmp_path):
    """연습 모드로 짧게 돌려 `log.jsonl` 의 모든 줄에서 log_std 가 상한 이하인지 본다.

    `forward()` 는 clamp 하지 않고(기울기를 죽인다) `opt.step()` 뒤 `clamp_log_std()` 가
    묶는다 — 그 규약이 새 상한에도 적용되는지 확인하는 것이 이 테스트의 목적이다.
    """
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
         "--log-std-max", "-1.0"],
        capture_output=True, text=True, env=env, cwd=repo, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    rows = [json.loads(x) for x in open(tmp_path / "run" / "log.jsonl") if x.strip()]
    assert rows, "로그가 비었다"
    for r in rows:
        assert max(r["log_std"]) <= -1.0 + 1e-6, (r["step"], r["log_std"])
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/test_train_ppo.py -q -k "log_std_max or 기본값"` (timeout 600000)
Expected: FAIL — `AttributeError: module has no attribute '_build_parser'`(Step 0 을 안 했으면) 또는 `'Namespace' object has no attribute 'log_std_max'`

- [ ] **Step 3: 구현**

`scripts/train_ppo.py`:
```python
    ap.add_argument("--log-std-max", type=float, default=None,
                    help="정책의 σ 상한(log 스케일)을 덮어쓴다. 기본값은 체크포인트/PolicyConfig "
                         "값 그대로(0.5). M4c 는 3M 에서 결정적 모드만 무너지는 것을 봤는데 σ 와 "
                         "정책 이동이 함께 커져 원인을 못 갈랐다 — 끝난 체크포인트의 σ 를 낮춰선 "
                         "못 잰다(결정적 평가는 log_std 를 안 본다). 이 값을 조이고 다시 학습하는 "
                         "것이 그 개입이다.")
```
- 정책을 만든 **직후**(엔트로피 모드 덮어쓰기와 같은 자리, `_apply_entropy_mode` 옆) 적용한다. `PolicyConfig` 는 frozen 이 아니므로 `dataclasses.replace` 로 새 cfg 를 만들어 `net.policy.cfg` 와 `net.cfg.policy` **둘 다** 갱신해야 한다 — M4c 에서 `net.cfg` 를 안 고쳐 저장된 체크포인트가 옛 값을 갖는 버그가 있었다(`_apply_entropy_mode` 가 그 교훈으로 만들어졌으니 같은 방식을 따라라).
- 갱신 **직후 `net.clamp_log_std()` 를 한 번 부른다** — 이어받은 체크포인트의 σ 가 새 상한 위에 있을 수 있다.

- [ ] **Step 4: 통과 확인**

세 조각 전부 `rc=0`.

- [ ] **Step 5: 돌연변이로 확인**

커밋한 **뒤**: 갱신에서 `net.cfg.policy` 쪽을 빼 본다 → 저장된 체크포인트를 다시 읽으면 옛 상한이 나와야 하고, 그걸 잡는 단언이 없다면 **그 사실을 보고해라**(테스트를 보강할지 내가 정한다).

- [ ] **Step 6: 커밋**

**Step 0 의 리팩터와 σ 상한 추가는 커밋을 나눠라** — 순수 이동과 기능 추가가 한 diff 에 섞이면 리뷰어가 "정말 순수 이동이었나" 를 확인할 수 없다.
```bash
git add scripts/train_ppo.py
git commit -m "M4d — train_ppo 의 인자 파서를 _build_parser() 로 뺀다(순수 이동, CLI 기본값을 테스트 가능하게)"
git add scripts/train_ppo.py tests/rl/test_train_ppo.py
git commit -m "M4d — σ 상한을 CLI 로 연다(결정적 모드 붕괴를 개입으로 가르기 위해)"
git show -s --format=%B HEAD
```

---

### Task 5: 본 실험과 성적표

**Files:**
- Create: `docs/reports/m4d-ppo.md`(스크립트 생성), `docs/reports/m4d-ppo-notes.md`(사람이 쓰는 해석)
- Modify: `README.md`

**Interfaces:**
- Consumes: 작업 1~4 전부, `scripts/sweep_ppo.py`, `scripts/report_m4a.py`, `scripts/eval_det_vs_stoch.py`
- **이 작업의 실행은 계획 주관자가 OMEN 에서 직접 돌린다.** 구현자에게 긴 SSH 를 앞단에서 기다리게 하지 않는다(M4a~M4c 에서 여러 번 실패했다).

**돌릴 설정** — 전부 OMEN 30 환경·3M 스텝·`--device cpu`·시드 0·1·2, `--init` 는 M4c 의 스쿼시 재적합 체크포인트(`runs/omen/2026-09-26-m3-squash-warm/policy.pt`), `--comfort-on-intent` 켬, `--final-eval-seeds 3` 유지:

| 이름 | 바뀌는 것 | 무엇을 가르나 |
|---|---|---|
| `m4c-base` | 없음(M4c 최고 설정 재현) | 기준선 |
| `align` | `--violation-mode once_per_section` | **보상을 채점과 맞추면 점수가 오르나** |
| `align-sigma` | 위 + `--log-std-max -0.6` | **결정적 모드 붕괴가 σ 때문인가** |

`--out-root` 는 하나로 두되 `--name` 을 셋 다 다르게 준다(집계 파일이 `<name>-sweep.json` 이라 안 섞인다).

- [ ] **Step 1: 코드를 올리고 OMEN 에서 빠른 스위트가 도는지 본다**

```bash
for d in vtd_rl scripts tests; do
  rsync -a --delete --exclude '__pycache__' $d/ user-OMEN:vtd-rl-urban-driving/$d/
done
ssh user-OMEN 'cd ~/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/python -m pytest -q -m "not slow"; echo rc=$?'
```
Expected: `rc=0`

- [ ] **Step 2: 세 설정 × 시드 3 (약 3 시간)**

설정마다 `scripts/sweep_ppo.py --seeds 0 1 2 --resume` 로 돌린다. OMEN 에서 `nohup` 으로 분리 실행하고 완료를 감시한다(감시 필터에 **실패 신호도** 넣을 것 — `Traceback|Error|FAILED|Killed|OOM`. 침묵이 성공으로 보이면 안 된다).

- [ ] **Step 3: 결정적·확률적 평가를 둘 다 잰다**

M4c 의 가장 큰 발견이 여기서 나왔다 — **결정적 평가만 보면 틀린 결론을 낸다.** 설정마다 최종(3M) 체크포인트와 최선 체크포인트를 둘 다:
```bash
ssh user-OMEN 'cd ~/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/python scripts/eval_det_vs_stoch.py \
  --checkpoint <각 체크포인트> --device cpu --out runs/omen/<날짜>-m4d/det-vs-stoch.jsonl'
```

- [ ] **Step 4: 항목별 보상 배분을 M4c 와 나란히 놓는다**

`log.jsonl` 의 `term_progress_mean`·`term_time_mean`·`term_violation_mean`·`term_comfort_mean` 과 `rollout_return_mean` 을 뽑아 **M4c 의 같은 표**(`docs/reports/m4c-split.md`)와 나란히 놓는다. 물어야 할 것:
- **위반 항이 얼마나 작아졌나** — 목표는 대회 감점과 같은 크기다(단계② 학생이 대회 기준 94.2 면 위반은 −5.8 근처여야 한다).
- **리턴이 양수가 됐나.** M4c 는 −59.7 이었고 "멈추면 −6.2" 를 못 넘었다.
- 안 됐으면 **무엇이 남았는지** 항목별 숫자로 말한다.

- [ ] **Step 5: 해석 노트를 쓰고 성적표를 만든다**

`docs/reports/m4d-ppo-notes.md` 에 담을 것(**숫자는 성적표와 스크립트가 생성한다 — 손으로 적지 마라**):
- 목표 판정 각 줄의 결과
- **결정적·확률적 둘 다** 보고(M4c 교훈)
- 위반 항이 채점과 맞은 뒤 리턴이 어떻게 됐나
- `align-sigma` 가 결정적 모드를 되살렸나 — **되살렸으면 원인은 σ, 아니면 앵커 감쇠에 의한 평균 드리프트다.** 이게 이 마일스톤이 가르기로 한 질문이다
- 시드 3 개의 편차
- **★ 스펙이 걱정한 실패 모드가 현실이 됐는가**(위 "이 계획은 스펙을 뒤집는다" 참고). `align` 이 `m4c-base` 보다 **구간당 서로 다른 항목 수**가 늘었거나 **한 항목을 어긴 총 시간**이 늘었으면 "한 번 깎인 뒤 계속 어긴다" 가 일어난 것이다. 숫자로 적어라 — 늘지 않았으면 그것도 그대로 적어라(스펙의 가정이 기우였다는 증거다).
  재료: `report_m4a.py` 의 항목별 위반 표(`violation_counts`)와 `info["result"]["sheet"]`.
- **스펙 §5:151 을 고칠지에 대한 권고.** 결과가 `once_per_section` 손을 들어주면 스펙을 고친다.
- **이 성적표가 주장하지 않는 것**(설정 3 개·시드 3 개·커리큘럼 ①②뿐·`--log-std-max` 는 한 값만 시험·"구간당 한 번 + 작은 잔여 벌점" 중간안은 안 시험)
- M4e 로 넘기는 것

성적표:
```bash
scripts/report_m4a.py --run <목표를 가장 잘 맞춘 설정의 시드 0> \
  --sweep <세 sweep.json 전부> --m3 runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt \
  --notes docs/reports/m4d-ppo-notes.md --out docs/reports/m4d-ppo.md \
  --eval-seeds 3 --device cpu --title "M4d 성적표 — 보상을 채점과 맞춘다" --student-label "M4d 학생"
```
**`--run` 은 선택 없이 시드 0 을 쓴다** — M4c 에서 체리피킹 논란을 원천 제거한 방식이다. 설정 선택은 노트에 근거와 함께 적는다.

**목표를 못 채우면 기준을 고치지 마라.** 숫자와 관찰을 적는다.

- [ ] **Step 6: README 와 커밋**

`README.md` 의 PPO 단락 목록에 `docs/reports/m4d-ppo.md` 를 더한다.
```bash
git add docs/reports/m4d-ppo.md docs/reports/m4d-ppo-notes.md README.md
git commit -m "M4d 증거 — 보상을 채점과 맞춘 뒤 설정 3 × 시드 3 결과"
git show -s --format=%B HEAD
```

---

## 이 계획이 미루는 것

- **커리큘럼 단계 ③④⑤**(사물·정지차 / 보행자·교통 / 연습코스 전체). 목표를 넘긴 뒤에 간다 — PPO 가 ①②에서 출발점을 못 넘는 동안 판을 어렵게 만들면 실패 원인만 늘어난다.
- **위반을 채점 항목별로 쪼개는 롤아웃 로깅.** 작업 2 뒤에는 보상의 위반 항이 대회 감점과 같아지므로 **평가 경로의 기존 항목별 표**(`report_m4a.py` 의 `violation_counts`)가 그대로 설명이 된다. 작업 3·4 의 숫자를 보고도 어느 항목인지 모르겠으면 그때 만든다.
- 학습률·클립·롤아웃 길이·모방 반감기 탐색.
- 시간 상관 탐색(OU 잡음)·행동 반복.
- M4b·M4c 최종 리뷰가 넘긴 것: `_goal_lines` 의 `"보류" in v.line` 문자열 매칭(구조적 플래그로), `OutcomeCounter` 가 0 인 사유의 키를 안 만드는 것, `ReturnTracker` 의 판 길이 +1 편향, `RewardConfig.rule_scale` 이 커리큘럼에서 설정되지 않는 것(스펙 §5), `RewardShaper` 의 판 첫 걸음 승차감이 0 인 것(콜드스타트), `tests/rl/` 조각이 602 초로 도구 상한에 걸리는 것.
