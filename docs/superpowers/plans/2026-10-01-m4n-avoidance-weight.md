# M4n — 회피 걸음에 무게를 준다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BC 가 **단계 ③ 회피를 실제로 배우게** 만든다. M4m 이 막힌 자리를 특정했다 — 회피가 데이터의 2% 라 손실을 98% 의 평범한 주행이 지배한다.

**Architecture:** 손실에 **표본별 가중치**를 넣고, **물체가 가까운 상태**를 올려 준다. M4m 의 `probe_anchor.py` 틀을 그대로 재사용해 **10 분 안에** 답을 낸다. DAgger 는 이번에도 안 돌린다.

**Tech Stack:** Python 3.10, PyTorch 2.14, numpy **1.26.4 고정**, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`
**직전 증거:** `docs/reports/m4m-anchor.md`, `docs/reports/m4l-bc-damage.md`

## ★ M4m 이 확정한 것 (전제)

1. **앵커는 유지에 성공한다.** 8 에포크에서 앵커 없이 Δθ 11.0·단계① 0% → **계수 10 이면 Δθ 3.80·100%**(시드 3, 범위 0). **그러므로 이번 실험은 계수 10 을 고정으로 깔고 간다.**
2. **단계 ③ 는 21 개 설정 전부 0%** 이고, **완주율이 가린 게 아니다** — 총걸음이 기준점 대비 −1.5~+2.3%, `D 268 · E 384 · H 185` 로 **한 걸음 차이까지 같다.**
3. **★ 회피가 2% 다.** r0 216,440 행 중: 물체 보임 28.2%, **물체 30 m 안 전방 11.4%**(24,674), `|조향|>0.15` 9.8%, **둘 다 = 4,383(2.03%)**. 물체가 가까운 걸음 중 **큰 조향은 20.6% 뿐** — 선생님의 회피는 대부분 작은 넛지다.
4. 최근접 물체의 전방거리는 **`objs[:, 0, 0] * 80.0` m** 다(관측 배치 `env/observation.py:135-138`, 범위 정규화 `WorldConfig.object_range=80.0`).

## ★★ 계측 규칙 (M4l·M4m 에서 확정 — 어기면 결론이 거짓이 된다)

- **학습이 비결정적이다.** 배치 순서 시드만 바꿔도 단계 ① 완주가 **16.7% ↔ 66.7%**. **모든 숫자는 학습 시드 ≥3 의 평균과 범위.** 단일 숫자 금지.
- **단계 ① 은 독립 판이 6 개다**(액터 0·항상초록이라 시드가 세계를 안 바꾼다). 해상도 16.7 점.
- **완주율만 보면 학습을 놓친다.** M4m 에서 그걸 따로 확인해야 했다. **이번엔 "얼마나 멀리 갔나"(판별 걸음 수)를 스크립트가 처음부터 같이 내야 한다** — 지난번엔 던져 버리는 스크립트로 쟀다.
- 평가 자체는 결정적이다(같은 체크포인트 → 같은 보상). 흔들림은 **학습 잡음**이다.

## Global Constraints

- 커밋에 **Claude 표기 금지**. 커밋 뒤 `git show -s --format=%B HEAD` 확인·보고.
- `third_party/rule_stack` 수정 금지. `runs/`·`*.pt`·`*.npz`·CSV·`*.xodr`·`*.xml` 커밋 금지.
- numpy **1.26.4**. 문서·주석·커밋 **한국어**. **성적표 숫자는 전부 스크립트 생성.**
- **테스트 Bash 호출마다 `timeout: 600000`.** `run_in_background` 금지. **rc 로 판정**(파이프 태우면 `$?` 가 `tail` 것).
- **기준선(main `30df49f`): 651 / 148 / 30, 전부 rc=0.**
- **⚠ `tests/rl/`(715 초)는 600 초 상한을 넘었고 `slow`(427 초)도 가깝다 — 새 테스트는 빠른 조각에.**
- 커밋 먼저, 그다음 돌연변이. 원복은 `git checkout -- <경로>`.
- **`weight_*` 기본값은 "가중치 없음" 이어야 하고 그때 지금과 비트 동일**이어야 한다.
- 이 마일스톤은 **`run_dagger.py` 를 안 건드린다.**
- OMEN: `cd` 명시, **`git pull` 금지**. **띄우기 전 로컬 dry-run 으로 인자 검증, 띄운 뒤 30 초 생존 확인**(이번 세션에 이걸 안 해서 두 번 당했다).

---

### Task 1: 손실에 표본별 가중치

**Files:** Modify `vtd_rl/policy/train.py`; Test: 빠른 조각

**Interfaces:**
- `TrainConfig.near_m: float = 0.0` — 0 이면 가중치 없음(**기존과 비트 동일**). >0 이면 "최근접 물체가 전방 `near_m` m 안" 인 표본을 고른다.
- `TrainConfig.near_weight: float = 1.0` — 그 표본의 손실 배율.
- `policy_loss` 가 표본별 가중 평균을 내게 한다. **가중치는 정규화한다**(가중 합 / 가중치 합) — 안 하면 `near_weight` 가 **실효 학습률을 같이 키워** 효과를 못 가린다. **이것이 이 Task 의 핵심 함정이다.**

**왜 라벨이 아니라 관측으로 고르나:** "물체가 가깝다" 는 **상태**의 성질이라 라벨과 독립이다. `|조향|>0.15` 로 고르면 **선생님이 크게 꺾은 표본만** 배우고 "가까운데 안 꺾는다" 를 못 배운다 — 그런데 물체 근처 걸음의 **79.4% 가 큰 조향이 아니다**(M4m). 그 다수를 버리면 안 된다.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_near_m_0은_예전과_완전히_같다():
    net = _tiny_net(seed=0); b = _tiny_batch()
    a, _ = policy_loss(net, b, TrainConfig(near_m=0.0, near_weight=7.0))
    c, _ = policy_loss(net, b, TrainConfig())
    assert torch.allclose(a, c)


def test_가중치는_정규화된다_실효학습률을_안_키운다():
    """★ 핵심 함정 — 정규화를 빼면 near_weight 가 lr 을 같이 키워 효과를 못 가린다."""
    net = _tiny_net(seed=0); b = _batch_all_near()     # 전부 '가까움' 인 배치
    a, _ = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=1.0))
    c, _ = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=100.0))
    assert torch.allclose(a, c), "전부 같은 가중치면 배율과 무관해야 한다"


def test_가까운_표본이_실제로_더_센다():
    net = _tiny_net(seed=0)
    b = _batch_half_near()        # 절반은 물체 5m, 절반은 물체 없음
    base, _ = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=1.0))
    up, _ = policy_loss(net, b, TrainConfig(near_m=30.0, near_weight=10.0))
    assert not torch.allclose(base, up), "가중이 손실을 안 바꿨다"


def test_선택은_관측의_전방거리로_한다():
    """`objs[:,0,0]*80` 이 near_m 안이고 전방일 때만 고른다 — 라벨을 안 본다."""
    net = _tiny_net(seed=0)
    near = _batch_with(fx_m=10.0, steer=0.0)      # 가깝지만 안 꺾음 → 뽑혀야 한다
    far = _batch_with(fx_m=70.0, steer=0.9)       # 멀지만 크게 꺾음 → 안 뽑혀야 한다
    assert _selected(near, TrainConfig(near_m=30.0)).all()
    assert not _selected(far, TrainConfig(near_m=30.0)).any()


def test_마스크가_죽은_슬롯은_안_고른다():
    """object_mask 가 0 인 슬롯의 fx 는 쓰레기다 — 0 이면 '아주 가까움' 으로 오인된다."""
    net = _tiny_net(seed=0)
    b = _batch_with(fx_m=0.0, mask=0)
    assert not _selected(b, TrainConfig(near_m=30.0)).any()
```

**구현자에게:** `_batch_*`·`_selected` 는 네가 만들어라(기존 테스트의 헬퍼를 먼저 찾아 재사용). `_selected` 가 노출할 함수 이름은 네가 정하고 테스트도 맞춰라. **마지막 테스트가 중요하다** — 죽은 슬롯의 `fx` 가 0 이면 "0 m 앞" 으로 읽혀 **배치 전체가 '가까움' 이 된다.**

- [ ] **Step 2~4: 실패 확인 → 구현 → 세 조각 rc=0**
- [ ] **Step 5: 돌연변이 넷** — ① `near_weight` 무시 ② 정규화 제거 ③ `object_mask` 무시 ④ 선택을 라벨(`|steer|`)로. 각각 심고 **잡은 테스트 이름 보고.**
- [ ] **Step 6: 커밋**

---

### Task 2: 탐침에 가중치 축과 "얼마나 멀리" 지표

**Files:** Modify `scripts/probe_anchor.py`; Test: 빠른 조각

- `--near-m`, `--near-weights`(쉼표 구분) 추가. **기존 기본값은 불변.**
- **표에 "단계 ③ 총 걸음" 열을 더한다**(판별 걸음 합). 기준점 대비 비율도. **완주율이 0 이어도 배우는지 이걸로 본다** — M4m 에서 이걸 따로 재느라 던져 버리는 스크립트를 썼다.
- `--coefs` 가 쉼표 구분인 것처럼 **새 목록 인자도 같은 규칙**으로(이번 세션에 내가 `--coefs 1 3 10` 으로 줬다 rc=2 로 즉사했다).

- [ ] **Step 1~4**: 테스트 먼저, 세 조각 rc=0, 커밋
- [ ] **Step 5: 돌연변이 둘** — 총걸음을 평균 대신 합으로 잘못 세기 / `--near-weights` 무시

---

### Task 3: 실험 — *계획 주관자가 OMEN 에서 돌린다*

**격자**: `anchor_coef=10` 고정(M4m 최선), `near_m=30`, `near_weight ∈ {1(대조), 5, 20, 50}` × 학습 시드 3 × 8 에포크. 데이터 `data-r0`(216,440 행).

**판정 기준(미리 적는다):**
- **듣는다** = 어떤 `near_weight` 에서 **단계 ① 83.3% 이상 유지**하면서 **단계 ③ 완주 > 0%**.
- **부분적으로 듣는다** = 완주는 0 이지만 **단계 ③ 총걸음이 기준점 대비 +10% 이상**(= 더 멀리 간다). 그러면 데이터·라운드를 늘리는 쪽이 맞다.
- **안 듣는다** = 둘 다 아니다. 그러면 가중치가 지렛대가 아니고 **DAgger 라운드(분포 어긋남)** 로 간다.

### Task 4: 성적표 `docs/reports/m4n-weight.md`

위 표 + 판정. **안 들으면 그렇게 적는다.** 주장하지 않는 것(단계 ① 판 6 개, 학습 잡음 폭, `near_m=30` 한 값만 봤다는 것).

## 이 계획이 미루는 것

- **모델 선택**(지금은 마지막 라운드가 그냥 채택 — M4k 가 r3 83.3% 를 버리고 r4 16.7% 를 냈다). **DAgger 라운드를 돌리기 전에 반드시.**
- **DAgger 라운드 + 앵커**(분포 어긋남 해결). Task 3 결과를 보고.
- `summarize_dvs.py:22` STAGES 하드코딩, 단계 ①② 수집 중복 저장, `tests/rl` 쪼개기, 평가 표본.
