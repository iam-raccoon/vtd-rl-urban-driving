# M4o — 모델 선택을 진짜로 쓰고, DAgger 를 앵커와 함께 돌린다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 남은 **마지막 가설**을 제대로 시험한다 — β 를 낮춰 **학생이 직접 몰면** 학생이 빠지는 상태가 데이터에 들어온다. 그러려면 먼저 **모델 선택**이 진짜로 작동해야 한다.

**Architecture:** `run_dagger.py` 에 (1) 최선 라운드를 **실제로 채택**하게 하고 (2) **앵커**를 배선한다. 그 위에서 단계 ③ DAgger 를 돌린다. **그리고 선택에 쓴 판으로 성적을 내지 않는다** — 안 쓴 변종으로 다시 잰다.

**Tech Stack:** Python 3.10, PyTorch 2.14, numpy **1.26.4 고정**, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`
**직전 증거:** `docs/reports/m4n-weight.md`, `docs/reports/m4m-anchor.md`, `docs/reports/m4l-bc-damage.md`

## ★ 왜 이것이 남은 마지막 가설인가

| 마일스톤 | 개입 | 결과 |
|---|---|---|
| M4e·M4f | σ 후반 어닐링 | **기각** |
| M4e·M4f | 모방 앵커 바닥 | **기각**(시드 6 개에서 소멸) |
| M4k | 단계 ③ DAgger(앵커·선택 없음) | **실패**(전 단계 16.7%) |
| M4m | KL 앵커 | **유지 성공, 학습 0%** |
| M4n | 회피 걸음 가중 | **기각**(완주 0%, 총걸음 +3.8%) |

M4n 이 가른 것: 가중치는 **데이터에 있는** 상태의 비중만 바꾼다. r0 는 β=1.0 이라 선생님 궤적뿐이고, **"장애물 20 m 앞인데 차로 한가운데로 똑바로 간다"**(= 학생이 실제로 빠지는 상태)가 **데이터에 없다.** 그걸 넣는 방법이 **β<1 라운드**다. 추가 가설이 아니라 **DAgger 가 처음부터 존재하는 이유**이고, 아직 제대로 안 돌려 봤다.

## ★★ 지금 코드의 결함 — 최선 라운드를 만들고도 버린다

- `scripts/run_dagger.py:594` 가 `best` 를 **계산은 한다.**
- 그런데 **최종 평가는 `last` 라운드로 한다**(`:522` `policy-r{last['round']}.pt`), **요약도 `last` 다**(`:643`).
- **M4k 실측: r3 가 83.3% 였고 r4 가 16.7% 였는데 r4 가 나왔다.** 좋은 라운드를 만들고도 버린 것이다.

## Global Constraints

- 커밋에 **Claude 표기 금지**. 커밋 뒤 `git show -s --format=%B HEAD` 확인·보고.
- `third_party/rule_stack` 수정 금지. `runs/`·`*.pt`·`*.npz`·CSV·`*.xodr`·`*.xml` 커밋 금지.
- numpy **1.26.4**. 문서·주석·커밋 **한국어**. **성적표 숫자는 전부 스크립트 생성.**
- **테스트 Bash 호출마다 `timeout: 600000`.** `run_in_background` 금지. **rc 로 판정.**
- **기준선(main `a8f7a50`): 676 / 148 / 30, 전부 rc=0.**
- **⚠ `tests/rl/`(715 초)는 600 초 상한 초과 — 두 `--ignore` 조각으로 쪼개 돌린다. `slow`(439 초)도 가깝다. 새 테스트는 빠른 조각에.**
- 커밋 먼저, 그다음 돌연변이. 원복은 `git checkout -- <경로>`.
- **기본값은 기존 동작과 같아야 한다** — 선택 없음·앵커 없음이 기본이고, 그때 과거 성적표가 재현되어야 한다.
- OMEN: `cd` 명시, **`git pull` 금지**. **띄우기 전 로컬 `--dry-run` 인자 검증, 띄운 뒤 30 초 생존 확인.**

## ★★★ 계측 규칙 — 선택 편향을 피한다

**최선 라운드를 고른 그 평가로 성적을 내면 안 된다.** 라운드 5 개 중 최선을 고르면 잡음의 상단을 고르게 되고, 단계 ① 은 **독립 판이 6 개**(해상도 16.7 점)라 그 편향이 크다.

- **선택용 평가**: 학습에 쓴 변종(`v0~v3`)으로 한다.
- **보고용 평가**: **안 쓴 변종(`v4~v7`)으로 다시 잰다.** 성적표 헤드라인은 이 숫자다.
- 그리고 **학습 시드 3 개**(M4l: 배치 순서만 바꿔도 단계 ① 이 16.7 ↔ 66.7). 단일 실행으로 결론 금지.

---

### Task 1: 최선 라운드를 실제로 채택한다

**Files:** Modify `scripts/run_dagger.py`; Test: 빠른 조각

**Interfaces:**
- `--select {last,best}` — **기본 `last`**(기존 동작 그대로). `best` 면 최종 평가·요약·산출 체크포인트가 **최선 라운드**가 된다.
- `--select-metric` — 선택 기준을 **명시**한다. 지금 `:594` 는 `primary` 단계의 `goal_rate` 만 본다. **그 기준으로 단계 ③ 를 고르면 단계 ① 을 부순 라운드가 뽑힌다.** 유지와 학습을 **둘 다** 보는 기준을 쓰고, **무엇을 썼는지 로그와 성적표에 적어라.**
- 선택된 라운드를 `policy-best.pt` 로 복사(또는 심링크)해 뒤 단계가 경로 하나만 알면 되게 한다.

**구현자가 정할 것(보고에 근거를 적어라):** 선택 기준의 정확한 식. 후보 — (a) 모든 평가 단계 `goal_rate` 의 평균, (b) 단계 ① 이 문턱 이상인 라운드 중 단계 ③ 최대, (c) 가중합. **(b) 가 이 마일스톤의 목표("①을 지키면서 ③을 배운다")에 가장 가깝다고 본다.** 다른 것을 고르면 왜인지 적어라.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_기본은_last라_예전과_같다():
    """과거 성적표가 전부 이 보장 위에 있다."""
    rounds = [{"round": 0, "stage1": {"goal_rate": 0.2}}, {"round": 1, "stage1": {"goal_rate": 0.9}},
              {"round": 2, "stage1": {"goal_rate": 0.1}}]
    assert pick_round(rounds, "last", ...)["round"] == 2


def test_best는_최선_라운드를_고른다():
    rounds = [...]  # r1 이 최선
    assert pick_round(rounds, "best", ...)["round"] == 1


def test_m4k_상황을_재현한다_r3를_버리지_않는다():
    """★ M4k 실측: r3 83.3% · r4 16.7% 인데 r4 가 나왔다."""
    rounds = [{"round": r, "stage1": {"goal_rate": g}, "stage3": {"goal_rate": 0.0}}
              for r, g in [(0,.167),(1,.167),(2,.167),(3,.833),(4,.167)]]
    assert pick_round(rounds, "best", ...)["round"] == 3


def test_선택기준은_단계1을_부순_라운드를_안_고른다():
    """단계 ③ 만 보면 ① 을 부순 라운드가 뽑힌다."""
    rounds = [{"round": 0, "stage1": {"goal_rate": 1.0}, "stage3": {"goal_rate": 0.1}},
              {"round": 1, "stage1": {"goal_rate": 0.0}, "stage3": {"goal_rate": 0.3}}]
    assert pick_round(rounds, "best", ...)["round"] == 0


def test_선택한_라운드가_요약과_체크포인트에_실제로_반영된다():
    """★ 지금 결함이 정확히 여기다 — best 를 계산만 하고 last 를 쓴다(:522, :643)."""
    ...  # 작은 실행으로 summary 와 policy-best.pt 가 고른 라운드를 가리키는지 본다
```

- [ ] **Step 2~4: 실패 확인 → 구현 → 세 조각 rc=0** (`tests/rl/` 는 쪼개서)
- [ ] **Step 5: 돌연변이 넷** — ① `--select best` 무시 ② `best` 를 계산만 하고 `last` 로 저장(현재 결함 재현) ③ 선택 기준에서 단계 ① 조건 제거 ④ 동점 처리 뒤집기. **잡은 테스트 이름 보고.**
- [ ] **Step 6: 커밋**

---

### Task 2: 앵커를 DAgger 에 배선한다

**Files:** Modify `scripts/run_dagger.py`; Test: 빠른 조각

- `--anchor-coef`(기본 0.0 = 기존 동작)·`--near-m`·`--near-weight` 를 `TrainConfig` 로 흘린다.
- **참조는 `--init` 체크포인트로 고정**한다(라운드마다 바꾸지 않는다). M4m 이 검증한 구성이 그것이고, 라운드마다 이전 정책으로 바꾸면 **누적으로 흘러가** 앵커가 묶는 게 뭔지 불분명해진다. **다르게 했으면 보고해라.**
- `--anchor-coef > 0` 인데 `--init` 이 없으면 **터뜨려라**(참조가 없다).
- 라운드 기록(`log.jsonl`)에 `anchor`·`near_frac` 을 남긴다 — 설정만 켜고 아무 일도 안 일어나는 꼴을 숫자로 본다(M4e 의 no-op 개입).

- [ ] **Step 1~4**: 테스트 먼저, 세 조각 rc=0, 커밋
- [ ] **Step 5: 돌연변이 셋** — `--anchor-coef` 무시 / 참조를 라운드마다 갱신 / `--init` 없이 앵커 허용

---

### Task 3: 본 실행 — *계획 주관자가 OMEN 에서 돌린다*

- `--stage stage3 --variants 4`(수집), `--init <M3>`, `--anchor-coef 10`, `--select best`, `--eval-stage stage1 --eval-stage stage2 --eval-stage stage3`, 라운드 5, **학습 시드 3 개**(세 번 돌린다).
- **그다음 선택된 `policy-best.pt` 를 안 쓴 변종(`v4~v7`)으로 다시 평가**한다 — 성적표 헤드라인.

### Task 4: 성적표 `docs/reports/m4o-dagger-anchor.md`

- **단계 ③ 완주율**(안 쓴 변종 기준) — 출발점 0/18 대비. **1 차 지표.**
- **단계 ①② 가 버텼는가** — 98.60 / 94.13 기준.
- 라운드별 궤적과 **어느 라운드가 뽑혔는지, 어떤 기준으로**.
- **선택용 평가와 보고용 평가의 차이** — 편향 크기가 그대로 드러난다.
- **안 되면 그렇게 적는다.** 개입 다섯을 이미 기각했다.

## 이 계획이 미루는 것

- 단계 ④⑤, PPO 복귀.
- `summarize_dvs.py:22` STAGES 하드코딩, 단계 ①② 수집 중복 저장, `tests/rl` 조각 쪼개기(임시 회피 중), 평가 표본(단계 ① 판 6 개).
