# M4m — BC 손실에 "알던 것을 유지하라" 항을 넣는다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BC 학습이 **이미 잘하는 정책을 파괴하지 않게** 만든다. M4l 이 원인을 확정했다 — 손실에 이전 정책과 묶어 두는 항이 없다.

**Architecture:** `policy_loss` 에 **참조 정책과의 KL(증류) 항**을 더한다. 참조는 학습 시작 시점의 정책 사본(고정). 그리고 **이 마일스톤은 DAgger 에 배선하지 않는다** — 앵커가 듣는지부터 가장 싼 실험으로 확인한다. 안 들으면 배선은 낭비다.

**Tech Stack:** Python 3.10, PyTorch 2.14, numpy **1.26.4 고정**, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`
**직전 증거:** `docs/reports/m4l-bc-damage.md`

## ★ M4l 이 확정한 것 (이 계획의 전제)

1. **원인은 데이터 "방향" 이다.** 가중치 이동거리를 맞추고 방향만 바꾸면: 무작위 잡음 Δθ=0.54 → **83.3% 유지**, 단계 ③ 기울기 Δθ=0.54 → **0%**. 자기 말뭉치 쪽 Δθ=2.29 → 94.4%.
2. **학습률은 원인이 아니다.** 거리를 맞추면 설명력이 없고, 단계 ③ 은 **lr 3e-6 에서도 0%**(가중치당 최대 변화 0.0032). 다만 `TrainConfig.lr = 3e-4`(`train.py:21`)가 웜스타트에 과한 것은 **별개로 사실**이다.
3. **스쿼시 정렬은 멀쩡하다**(일부러 어긋내면 손실이 오히려 오른다).
4. **옛 데이터를 섞는 것으로는 안 된다**(M3+단계③ 41.7 만 행에서도 무너진다).
5. **손실에 "유지" 항이 아예 없다**(`train.py:69-95`, `policy_loss` 는 새 데이터 우도만 본다).

## ★★ 계측이 먼저다 — 안 그러면 효과를 못 잰다

M4l 이 같이 밝힌 것:

- **학습이 비결정적이다.** 배치 순서 시드만 바꿔도 완주율 **16.7% ↔ 66.7%** 인데 손실·MAE·Δθ 는 소수 셋째 자리까지 같다.
- **단계 ① 은 시드가 세계를 안 바꾼다**(액터 0·항상초록). **독립 판 6 개**, 해상도 **16.7 점**. "100%→83.3%" 는 판 하나다.

**그러므로 이 마일스톤의 어떤 주장도 학습 시드 ≥3 의 평균이어야 하고, 단계 ① 단독 숫자로는 아무것도 못 말한다.** 평가는 **단계 ①(유지 확인) + 단계 ③ 변종(학습 확인)** 을 같이 본다 — 단계 ③ 은 액터·변종이 있어 시드가 실제로 세계를 바꾼다.

## Global Constraints

- 커밋·PR 에 **Claude 표기 금지**. **커밋 뒤 `git show -s --format=%B HEAD` 로 확인하고 보고할 것.**
- `third_party/rule_stack` **수정 금지**. `runs/`·`*.pt`·`*.npz`·CSV·`*.xodr`·`*.xml` **커밋 금지**.
- numpy **1.26.4 고정**. 문서·주석·커밋 메시지 **한국어**. **성적표 숫자는 전부 스크립트가 생성한다.**
- **테스트를 돌리는 모든 Bash 호출에 `timeout: 600000`(ms).** `run_in_background` 금지. **rc 로 판정**(파이프를 태우면 `$?` 가 `tail` 것이 된다).
- **기준선(main `6e9045c`): 612 / 148 / 30, 전부 rc=0.**
- **⚠ `tests/rl/`(715 초)는 이미 600 초 상한을 넘었고 `slow`(427 초)도 가깝다. 새 테스트는 빠른 조각에 넣어라.**
- 커밋을 **먼저** 하고 돌연변이. 원복은 `git checkout -- <경로>`.
- OMEN: `cd` 명시, **`git pull` 금지**(체크아웃이 낡았고 트리는 rsync 동기화).
- **이 마일스톤은 `run_dagger.py` 를 건드리지 않는다**(배선은 앵커가 듣는 것을 확인한 뒤).

---

### Task 1: `policy_loss` 에 참조 정책 KL 항

**Files:** Modify `vtd_rl/policy/train.py`; Test `tests/policy/test_train.py`(기존) 또는 신규

**Interfaces:**
- `TrainConfig.anchor_coef: float = 0.0` — 0 이면 **지금과 완전히 같다**(기존 모든 호출부 불변).
- `TrainConfig.anchor_lr: float | None = None` — 주면 `lr` 대신 쓴다(웜스타트용). 기본 `None` = 기존 동작.
- `policy_loss(net, batch, cfg, ref=None)` — `ref` 가 주어지고 `anchor_coef > 0` 이면 **`KL(ref ‖ net)`** 를 더한다. 연속 행동은 두 가우시안의 해석적 KL, 지시등은 로짓 분포의 KL.
- `train_epochs(..., ref=None)` — `ref` 를 배치마다 그대로 넘긴다.

**설계 지침 (구현자가 판단할 것):**
- **참조는 고정이다** — `copy.deepcopy(net)` 후 `eval()`, `requires_grad_(False)`, `torch.no_grad()` 로 평가. **참조가 같이 학습되면 앵커가 아무것도 안 묶는다**(돌연변이로 확인할 것).
- KL 방향은 **`KL(ref ‖ net)`** 를 기본으로 한다(참조의 질량을 덮게 강제 — 망각 방지에 맞는 방향). 반대 방향을 골랐으면 **왜인지 보고해라.**
- 지시등 로짓은 `turn_weight` 와 어떻게 섞을지 정하고 **주석에 근거를 남겨라.**
- **스쿼시와의 상호작용**: 두 정책 모두 같은 사전-스쿼시 가우시안을 쓰므로 KL 은 스쿼시 전에서 계산하면 된다(야코비안이 상쇄된다). **맞는지 확인하고 주석에 적어라** — 틀리면 앵커가 엉뚱한 것을 묶는다.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_anchor_coef_0은_예전과_완전히_같다():
    """기존 호출부·과거 성적표가 전부 이 보장 위에 있다."""
    net = _tiny_net(seed=0); ref = copy.deepcopy(net)
    b = _tiny_batch()
    a, _ = policy_loss(net, b, TrainConfig(anchor_coef=0.0), ref=ref)
    c, _ = policy_loss(net, b, TrainConfig())           # ref 없음
    assert torch.allclose(a, c)


def test_참조와_같으면_KL이_0이다():
    net = _tiny_net(seed=0); ref = copy.deepcopy(net)
    _, parts = policy_loss(net, _tiny_batch(), TrainConfig(anchor_coef=1.0), ref=ref)
    assert parts["anchor"] == pytest.approx(0.0, abs=1e-6)


def test_참조에서_멀어지면_KL이_커진다():
    net = _tiny_net(seed=0); ref = copy.deepcopy(net)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    _, parts = policy_loss(net, _tiny_batch(), TrainConfig(anchor_coef=1.0), ref=ref)
    assert parts["anchor"] > 0.01


def test_앵커는_참조를_학습시키지_않는다():
    """★ 참조가 같이 움직이면 앵커가 아무것도 안 묶는다 — 조용한 무효화."""
    net = _tiny_net(seed=0); ref = copy.deepcopy(net)
    before = [p.detach().clone() for p in ref.parameters()]
    cfg = TrainConfig(anchor_coef=1.0, epochs=1, lr=1e-2)
    train_epochs(net, _tiny_dataset(), cfg, ref=ref)
    for p, q in zip(ref.parameters(), before):
        assert torch.equal(p, q), "참조가 움직였다"
    assert any(not torch.equal(p, q) for p, q in zip(net.parameters(), before)), "학생이 안 움직였다"


def test_앵커가_가중치_이동을_실제로_줄인다():
    """같은 데이터·시드·에포크에서 anchor_coef 가 크면 Δθ 가 작아야 한다."""
    deltas = []
    for coef in (0.0, 1.0, 10.0):
        net = _tiny_net(seed=0); ref = copy.deepcopy(net)
        start = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
        train_epochs(net, _tiny_dataset(), TrainConfig(anchor_coef=coef, epochs=2, seed=0), ref=ref)
        end = torch.cat([p.detach().flatten() for p in net.parameters()])
        deltas.append(float((end - start).norm()))
    assert deltas[0] > deltas[1] > deltas[2], deltas


def test_anchor_lr이_없으면_lr을_쓴다():
    assert TrainConfig(lr=3e-4).effective_lr() == 3e-4
    assert TrainConfig(lr=3e-4, anchor_lr=1e-5).effective_lr() == 1e-5
```

**구현자에게:** `_tiny_net`·`_tiny_batch`·`_tiny_dataset` 은 기존 테스트에 있는 것을 **먼저 찾아 재사용해라**(없으면 만들고 보고). `effective_lr()` 이름이 마음에 안 들면 바꾸고 테스트도 같이 바꿔라.

- [ ] **Step 2~4: 실패 확인 → 구현 → 세 조각 rc=0**

- [ ] **Step 5: 돌연변이 다섯** — ① `anchor_coef` 무시 ② 참조를 `deepcopy` 대신 같은 객체로(앵커 무효) ③ 참조에 `no_grad` 빼기 ④ KL 부호 뒤집기 ⑤ `anchor_lr` 무시. **각각 실제로 심고 잡은 테스트 이름을 보고해라.**

- [ ] **Step 6: 커밋**

---

### Task 2: 앵커가 듣는가 — 가장 싼 결정적 실험

**Files:** Create `scripts/probe_anchor.py` + 테스트

**이것이 이 마일스톤의 결론이다.** DAgger 를 돌리지 않는다. M4l 이 쓴 것과 같은 방식으로 **`train_epochs` + `evaluate_policy` 를 직접** 부른다(한 번에 몇 분).

**실험 설계:**
- 출발: M3 체크포인트(`runs/omen/2026-09-26-m3-squash-warm/policy.pt`, OMEN).
- 학습: **단계 ③ 데이터 1 에포크**(M4l 이 Δθ=0.54 에서 0% 를 본 바로 그 설정), `anchor_coef ∈ {0, 0.1, 1, 10}` × **학습 시드 3 개**.
- 평가: **단계 ①**(유지됐는가) + **단계 ③ 변종**(배웠는가), 결정적.
- 보고: 설정마다 **시드 3 개의 평균과 범위**. 단일 숫자 금지.

**판정 기준(미리 적는다 — 결과 보고 기준을 바꾸지 않기 위해):**
- **앵커가 듣는다** = 어떤 `anchor_coef` 에서 **단계 ① 평균이 83.3% 이상으로 유지되면서 단계 ③ 가 0% 를 넘는다.**
- **앵커가 안 듣는다** = 모든 계수에서 둘 중 하나가 깨진다(①이 무너지거나 ③이 0 에 머문다).
- **애매하다** = 시드 범위가 그 선을 가로지른다 → 시드를 늘려야지 결론 내지 마라.

- [ ] **Step 1~4**: 스크립트 + 테스트(빠른 조각), 세 조각 rc=0, 커밋
- [ ] **Step 5**: *계획 주관자가 OMEN 에서 돌린다*

---

### Task 3: 성적표 `docs/reports/m4m-anchor.md`

- 위 표(계수 × 시드 3 평균·범위), 판정 기준에 비춘 결론.
- **앵커가 안 들으면 그렇게 적는다.** 이 프로젝트는 지금까지 개입 넷을 기각했고, 다섯 번째도 기각될 수 있다.
- **이 성적표가 주장하지 않는 것** — 특히 단계 ① 독립 판이 6 개뿐이라는 것, 학습 비결정성 폭.

## 이 계획이 미루는 것

- **DAgger 배선**(`run_dagger.py --anchor`) — Task 2 가 "듣는다" 로 나온 뒤.
- **모델 선택**(지금은 마지막 라운드가 그냥 채택 — M4k 에서 r3 83.3% 를 버리고 r4 16.7% 를 냈다). 중요하지만 앵커와 독립이다.
- **단계 ①② 수집의 중복 저장**(`-s0` 와 `-s1` 이 바이트 동일 — M3 의 20 만 행은 고유 10 만 행).
- **평가 표본 늘리기**(단계 ① 은 시드가 무의미하다).
- `summarize_dvs.py:22` 의 `STAGES` 하드코딩, `ObsConfig.obj_y` 포화, `tests/rl` 조각 쪼개기.
