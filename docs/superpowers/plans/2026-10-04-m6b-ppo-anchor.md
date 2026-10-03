# M6b — PPO 에 KL 앵커: 출발 그물 근처를 지키며 정지차 성적이 오르나 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6a 에서 앵커 없는 PPO 는 정지차 단계(③b) 회피를 25 만~50 만 걸음 안에 잃었다(선택 창 68.5 → 48.1%). PPO 손실에 **얼린 출발 정책과의 KL(ref ‖ net)** 을 더해(모방학습 M4m 과 같은 식) 출발점 근처를 지키게 하고, 그 안에서 정지차 성적이 오르는지 잰다.

**Architecture:** `vtd_rl/rl/ppo.py` 의 `PPOConfig` 에 `anchor_coef`(기본 0.0 = 꺼짐, 지금과 비트 동일)를 넣고, `update(..., ref=None)` 가 PPO 미니배치의 **상태**(롤아웃에서 실제로 간 곳)에서 `vtd_rl.policy.train.anchor_kl` 로 KL 을 재 손실에 더한다. `scripts/train_ppo.py` 에 `--anchor-coef` 를 열고, `--init` 로 받은 정책을 `make_reference` 로 얼려 참조로 넘긴다. 로그 줄에 계수를 안 곱한 KL(`anchor`)이 실린다.

**Tech Stack:** Python 3.10, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(RTX 3090 Ti, 32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거는 `docs/reports/m6a-ppo-probe.md`("다음" 1).

## 왜

- M6a: 앵커 없음·학습률 3e-4. 세 시드 모두 출발점보다 못했다. PPO 로 여지가 큰 단계에서도 오르기는커녕 배운 회피를 잃었다.
- **사전 측정(끝남)** — M6a 체크포인트가 출발 그물(M4y EMA)에서 얼마나 멀어졌나. 상태는 M4s 수집 데이터 앞 4 만 행, KL 은 `anchor_kl`(TrainConfig 기본, 지시등 가중 0.5) 그대로.

| 시드 | 25 만 걸음 KL(전체·근접 30 m) | 50 만 | 100 만 |
|---|---|---|---|
| 0 | 1.43 · 3.74 | 2.61 · 5.31 | 4.41 · 8.93 |
| 1 | 1.83 · 4.32 | 3.19 · 7.60 | 2.85 · 6.69 |
| 2 | 1.69 · 1.63 | 2.13 · 3.79 | 4.66 · 4.67 |

  시드 2 의 25 만 걸음 그물은 선생님 라벨 대비 조향 오차가 0.024 → 0.146(6 배)이다. **PPO 는 25 만 걸음 만에 평균을 크게 옮긴다.** 앵커가 이 이동을 KL 0.1 대 이하로 묶으면서도 성적을 올릴 수 있는 계수가 있는지가 질문이다.
- 계수 감 잡기(어림): 조향 σ ≈ e^-1.9 ≈ 0.15 라 평균 이동 δ 의 KL 기울기는 `c·δ/σ²` ≈ 44·c·δ 이다. 이점 정규화된 PPO 기울기의 체계 성분을 0.05~0.2/σ 로 보면 균형 δ ≈ 0.15·(0.05~0.2)/c — c=0.03 은 거의 안 묶고(δ 최대 ~1), c=3 은 거의 못 움직인다(δ ~0.001~0.01). 그래서 **0.03 · 0.3 · 3** 세 칸(10 배 간격)을 잰다.

## 실험 설계

| 칸 | `--anchor-coef` | 공통 |
|---|---|---|
| A | 0.03 | `--lr 1e-4 --warmup-updates 20`, 나머지는 M6a 그대로 |
| B | 0.3 | 〃 |
| C | 3 | 〃 |

- 출발: M4y 승자 EMA(데이터 시드 0·1·2) — M6a 와 같은 그물. 칸마다 세 시드, 모두 9 실행.
- 학습률을 3e-4 → 1e-4 로, 가치만 배우는 처음 갱신 수를 5 → 20(약 5 만 걸음)으로 모든 칸에서 같이 바꾼다. 비교 대상은 M6a 가 아니라 **출발점**이므로 이 두 변경은 칸끼리 같게만 두면 된다.
- 끝나면 M6a 와 똑같이 `ac-best` 와 마지막 체크포인트에서 정책망을 꺼내 `eval_unseen.py --device cpu` 로 단계 ①·②(원본)·③b(선택 창 v8~v11·보고 창 v4~v7)를 잰다.

## 판정 기준 (미리 적는다)

출발점 = M4y 승자 EMA(같은 시드)의 선택 창 ③b: 73.6 · 69.4 · 62.5(평균 68.5%), 보고 창 61.1 · 59.7 · 58.3(평균 59.7%). M6a 와 같다.

칸·시드마다 `ac-best` 와 마지막 중 **선택 창 ③b 가 높은 쪽**을 고른다(보고 창은 고르기에 안 쓴다).

칸마다:

- **오른다** = 고른 그물의 선택 창 ③b 평균 ≥ 73.5%(+5) **그리고** 세 시드 중 2 개 이상이 출발점보다 높음 **그리고** ①·② 완주율 평균이 둘 다 ≥ 90%.
- **내린다** = 선택 창 ③b 평균 ≤ 63.5%(−5) 이거나 ① 또는 ② 평균 < 90%.
- 아니면 **차이 없다**.

전체:

- "오른다" 칸이 있으면 그중 선택 창 평균이 가장 높은 칸의 **보고 창 ③b 평균이 62.7%(출발점 +3) 이상**일 때만 "앵커 PPO 가 정지차 성적을 올린다" 고 적는다. 아니면 "선택 창에서만 올랐다" 로 적는다(칸 셋 중 고른 것이라 선택 창 숫자는 낙관 쪽으로 치우친다).
- 칸 C(가장 센 앵커)가 "차이 없다" 이고 A 가 "내린다" 면, 앵커가 무너짐을 막는다는 것은 확인한 것으로 적는다.

곁들여 적는 것(판정과 별도): 칸마다 마지막 50 갱신의 `anchor` 평균(롤아웃 상태), 그리고 위 사전 측정과 **같은 자**(M4s 데이터 4 만 행)로 잰 고른 그물의 KL.

## 실행 전 보강 (최종 리뷰 반영, 결과를 보기 전에 적었다)

최종 리뷰가 판정 규칙의 빈틈 셋을 짚었다. 본 실행을 띄운 직후·결과가 하나도 나오기 전에 아래를 더한다.

1. **대조군 칸 Z(계수 0)** 를 더한다 — 같은 lr 1e-4·warm-up 20, 시드 0·1·2. 칸 A 가 안 무너졌을 때 "lr 1e-4 만으로 충분했다" 와 "앵커 덕" 을 가르려면 앵커만 뺀 칸이 있어야 한다. Z 도 A·B·C 와 같은 칸별 판정을 받는다. (사용 가능 메모리 ≥ 7 GB 일 때만 하나씩 띄운다 — 동시 6 개가 RAM 약 19 GB 를 쓴다.)
2. **"앵커 PPO 가 올린다" 에는 KL 조건이 붙는다.** "오른다" 칸이라도 고른 그물의 KL(사전 측정과 같은 자, M4s 데이터 4 만 행) 평균이 **0.3 이하**일 때만 "앵커가 붙잡은 채 올랐다" 로 적는다. 0.3 을 넘으면 "앵커가 거의 안 걸린 채 올랐다" 로 적고, 그때 Z 와 비교한다.
3. **칸 C 는 '못 움직이는 칸' 으로 읽는다.** 어림으로 C 의 평형 이동은 조향 0.01 안팎이라 새 회피를 배울 수 없다 — "C 차이 없다" 는 거의 정해진 결과다. C 는 앵커가 실제로 붙잡는지(마지막 50 갱신 `anchor` ≤ 0.01)와 파이프라인이 출발점을 안 깨는지 보는 점검으로 쓴다. **"앵커가 무너짐을 막는다" 는 Z 가 "내린다" 이고 B 가 "내린다" 가 아닐 때만** 적는다(위 판정 규칙의 C·A 문장은 이것으로 바꾼다).
4. **보고 창 확인을 조인다**: 고른 칸의 보고 창 ③b 평균 ≥ 62.7%(+3) **그리고** 세 시드 중 2 개 이상이 출발점보다 높음. 시드 평균의 표준오차가 약 3 점이라 +3 하나로는 운을 못 거른다.

## Global Constraints

- numpy 는 **1.26.4** 고정 — 새 의존성을 넣지 않는다.
- `third_party/rule_stack` 은 **읽기 전용**이다.
- 커밋 금지: `runs/`, `*.pt`, `*.npz`, CSV, `*.xodr`, `*.xml`.
- 주석·독스트링·커밋 메시지·성적표는 **한국어**.
- 커밋 메시지에 **Claude 표기를 넣지 않는다**(`Co-Authored-By`, `Claude-Session`, "Generated with", 🤖, anthropic 주소 전부). 커밋 뒤 `git show -s --format=%B HEAD` 로 확인한다.
- 테스트는 `env -u PYTHONPATH .venv/bin/pytest ...` 로 돌리고 **종료 코드(rc)** 로 판정한다. "N passed" 를 grep 하지 않는다. `| tail` 로 넘기지 않고 파일로 돌린다.
- **기본값은 지금과 비트 동일**: `PPOConfig().anchor_coef == 0.0` 이고, 계수가 0 이면 `update()` 가 참조를 받아도 **추가 연산이 하나도 없다**(같은 시드면 파라미터가 비트 단위로 같다). `--anchor-coef` 를 안 주면 `_build_cfg(a) == PPOConfig()`.
- 앵커 KL 은 **새로 짜지 않는다** — `vtd_rl.policy.train.anchor_kl`·`make_reference`·`_check_reference` 를 그대로 쓴다(방향 `KL(ref ‖ net)`, 지시등 균형 `TrainConfig().turn_weight`).
- 스크립트끼리는 서로를 임포트하지 않는다(레포 관행).

## Review Focus

1. **계수를 켰는데 참조가 없어 앵커가 조용히 꺼지는 것** — "앵커 효과 없음" 으로 잘못 읽힌다. → Task 1 `test_앵커를_켰는데_ref가_없으면_거부한다`, Task 2 `test_앵커는_init_없이_거부한다`.
2. **참조가 학생과 파라미터를 공유하는 것**(같은 객체·얕은 복사) — KL 이 늘 0. → Task 1 `test_ref가_학생_정책과_같은_객체면_거부한다`.
3. **참조가 학습 중 움직이거나 기울기를 쌓는 것**(`DrivePolicy.forward` 는 `log_std` Parameter 를 그대로 돌려준다 — `no_grad` 만으로는 안 막힌다). → Task 1 `test_앵커는_처음에_0이고_ref는_움직이지_않는다`.
4. **계수 0 에서 무언가 달라지는 것**(추가 forward·난수 소비) — 그러면 M6a 이전 결과와 비교가 깨진다. → Task 1 `test_앵커가_꺼져_있으면_ref를_줘도_예전과_비트단위로_같다`.
5. **앵커가 손실에 실제로 안 더해지는 것**(로그 키만 있고 역전파는 안 되는 경우) 또는 **더해져도 붙잡는 힘이 없는 것.** → Task 1 `test_앵커_손실이_실제로_loss에_합산된다`, `test_앵커가_출발점_근처에_붙잡는다`; Task 2 스모크 `test_앵커_스모크가_로그에_KL을_남긴다`.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `vtd_rl/rl/ppo.py` (수정) | `PPOConfig.anchor_coef`, `anchor_loss(net, batch, ref)`, `update(..., ref=None)` |
| `tests/rl/test_ppo.py` (수정) | 위 동작 잠그기 |
| `scripts/train_ppo.py` (수정) | `--anchor-coef`, `_build_cfg`, `--init` 요구, 참조 만들기·넘기기 |
| `tests/rl/test_train_ppo.py` (수정) | 위 동작 잠그기 + 스모크 |

---

### Task 1: `ppo.py` — PPO 미니배치 상태에서 KL 앵커

**Files:**
- Modify: `vtd_rl/rl/ppo.py`
- Test: `tests/rl/test_ppo.py`

**Interfaces:**
- Consumes: `vtd_rl.policy.train.anchor_kl(net_out, ref_out, cfg) -> Tensor`, `make_reference(net) -> 얼린 deepcopy`, `_check_reference(net, ref)`(같은 객체·공유 파라미터면 `ValueError`).
- Produces: `PPOConfig.anchor_coef: float = 0.0`; `anchor_loss(net, batch, ref) -> Tensor`(스칼라, 계수 안 곱함); `update(net, opt, buffer, dagger_iter, cfg, step, generator=None, ref=None) -> dict` — 반환 dict 에 `"anchor"`(미니배치 평균, 계수 안 곱함, 꺼져 있으면 0.0)가 늘 있다.

- [ ] **Step 1: 실패하는 테스트를 쓴다** — `tests/rl/test_ppo.py` 끝에 더한다(임포트 줄의 `from vtd_rl.rl.ppo import ...` 에 `anchor_loss` 를 더하고, 파일 위에 `import copy` 와 `from vtd_rl.policy.train import TrainConfig, anchor_kl, make_reference` 를 더한다).

```python
def _kl_to(ref, net, buf):
    """버퍼 상태 전체에서 `KL(ref ‖ net.policy)` — 테스트용 자."""
    flat = next(iter(buf.batches(10_000, generator=torch.Generator().manual_seed(0))))
    vec, objs, mask = flat[0], flat[1], flat[2]
    with torch.no_grad():
        return anchor_kl(net.policy(vec, objs, mask), ref(vec, objs, mask), TrainConfig()).item()


def test_앵커_계수_기본값은_0():
    assert PPOConfig().anchor_coef == 0.0


def test_앵커가_꺼져_있으면_ref를_줘도_예전과_비트단위로_같다(small_ac):
    torch.manual_seed(0)
    a = small_ac()
    b = copy.deepcopy(a)
    buf = filled_buffer(a, n_steps=8, n_envs=4)
    opt_a = torch.optim.Adam(a.parameters(), lr=1e-2)
    opt_b = torch.optim.Adam(b.parameters(), lr=1e-2)
    cfg = PPOConfig(epochs=2, minibatch=8)
    out_a = update(a, opt_a, buf, None, cfg, step=0, generator=torch.Generator().manual_seed(0))
    out_b = update(b, opt_b, buf, None, cfg, step=0, generator=torch.Generator().manual_seed(0),
                   ref=make_reference(b.policy))
    for (name, pa), (_n, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name
    assert out_a["anchor"] == 0.0 and out_b["anchor"] == 0.0


def test_앵커를_켰는데_ref가_없으면_거부한다(small_ac):
    net = small_ac()
    buf = filled_buffer(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="ref"):
        update(net, opt, buf, None, PPOConfig(anchor_coef=0.5), step=0,
               generator=torch.Generator().manual_seed(0))


def test_ref가_학생_정책과_같은_객체면_거부한다(small_ac):
    net = small_ac()
    buf = filled_buffer(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        update(net, opt, buf, None, PPOConfig(anchor_coef=0.5), step=0,
               generator=torch.Generator().manual_seed(0), ref=net.policy)


def test_앵커는_처음에_0이고_ref는_움직이지_않는다(small_ac):
    net = small_ac()
    ref = make_reference(net.policy)
    before = {k: v.clone() for k, v in ref.state_dict().items()}
    buf = filled_buffer(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    # 미니배치 하나·에폭 하나 — 첫(유일한) 미니배치는 학생 == 참조라 KL 이 0 이다.
    out = update(net, opt, buf, None, PPOConfig(anchor_coef=1.0, epochs=1, minibatch=1000), step=0,
                 generator=torch.Generator().manual_seed(0), ref=ref)
    assert abs(out["anchor"]) < 1e-6
    for k, v in ref.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert all(p.grad is None for p in ref.parameters())


def test_앵커_손실이_실제로_loss에_합산된다(small_ac, monkeypatch):
    net = small_ac()
    dummy = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD(list(net.parameters()) + [dummy], lr=0.0)   # lr=0: 이동 없이 grad 만 본다

    def fake_anchor_loss(_net, _batch, _ref):
        return dummy * 3.0

    monkeypatch.setattr("vtd_rl.rl.ppo.anchor_loss", fake_anchor_loss)
    buf = filled_buffer(net)
    update(net, opt, buf, None, PPOConfig(anchor_coef=0.5, epochs=1, minibatch=1000), step=0,
           generator=torch.Generator().manual_seed(0), ref=make_reference(net.policy))
    assert dummy.grad is not None and abs(dummy.grad.item() - 1.5) < 1e-6   # 3.0 × 계수 0.5


def test_앵커가_출발점_근처에_붙잡는다(small_ac):
    torch.manual_seed(0)
    start = small_ac()
    buf = filled_buffer(start, n_steps=16, n_envs=8)
    kls = {}
    for coef in (0.0, 10.0):
        net = copy.deepcopy(start)
        ref = make_reference(start.policy)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        cfg = PPOConfig(anchor_coef=coef, epochs=4, minibatch=32, target_kl=1e9)   # 조기 종료 없이
        for k in range(5):
            update(net, opt, buf, None, cfg, step=0, generator=torch.Generator().manual_seed(k),
                   ref=ref if coef > 0 else None)
        kls[coef] = _kl_to(ref, net, buf)
    assert kls[0.0] > 1e-3                 # 앵커가 없으면 실제로 멀어진다(공허한 테스트가 아님)
    assert kls[10.0] < 0.5 * kls[0.0]      # 앵커가 그 거리를 절반 아래로 묶는다


def test_anchor_loss는_anchor_kl과_같다(small_ac):
    net = small_ac()
    ref = make_reference(net.policy)
    with torch.no_grad():
        net.policy.mean.weight.add_(0.1)   # 학생만 조금 옮긴다
    buf = filled_buffer(net)
    batch = next(iter(buf.batches(1000, generator=torch.Generator().manual_seed(0))))
    got = anchor_loss(net, batch, ref).item()
    want = _kl_to(ref, net, buf)
    assert got > 0.0 and abs(got - want) < 1e-5
```

- [ ] **Step 2: 돌려서 실패를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_ppo.py -q > $TMP/m6b_t1_red.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `ImportError: cannot import name 'anchor_loss'`.

- [ ] **Step 3: 구현한다** — `vtd_rl/rl/ppo.py`.

임포트를 고친다:

```python
from vtd_rl.policy.train import TrainConfig, _check_reference, anchor_kl, policy_loss
```

`PPOConfig` 의 `value_clip` 아래에 필드를 더한다:

```python
    # M6b — 얼린 출발 정책과의 `KL(ref ‖ net)` 계수(모방학습 M4m 과 같은 식). 0 이면 꺼짐 —
    # `update()` 가 참조를 받아도 추가 연산이 하나도 없어 예전과 비트 단위로 같다.
    # M6a: 앵커 없이 돌리자 25 만 걸음 만에 KL 이 1.4~1.8 로 벌어져 정지차 회피를 잃었다.
    anchor_coef: float = 0.0
```

`_IMITATION_CFGS` 정의 아래에 상수와 함수를 더한다:

```python
# 앵커 KL 의 지시등 균형 — 모방학습 앵커(`policy_loss`)와 같은 `TrainConfig().turn_weight` 를 쓴다.
_ANCHOR_TRAIN_CFG = TrainConfig()


def anchor_loss(net, batch, ref):
    """PPO 미니배치의 **상태**에서 `KL(ref ‖ net.policy)` — 계수는 안 곱한다.

    상태는 롤아웃에서 정책이 실제로 간 곳이다(모방 데이터가 아니다) — PPO 가 옮기는 곳에서
    붙잡는다. 식은 `vtd_rl.policy.train.anchor_kl` 그대로다(방향·σ·스쿼시 논의는 그 독스트링).
    `ref` 는 `make_reference(net.policy)` 로 만든 얼린 사본이어야 한다 — 같은 객체·공유
    파라미터면 `_check_reference` 가 `ValueError` 를 낸다. `ref.log_std` 는 Parameter 를 그대로
    돌려받으므로 `no_grad` 와 별도로 떼어야 참조에 기울기가 안 쌓인다(`policy_loss` 와 같은 함정).
    """
    vec, objs, mask = batch[0], batch[1], batch[2]
    _check_reference(net.policy, ref)
    with torch.no_grad():
        ref_mean, ref_log_std, ref_logits = ref(vec, objs, mask)
    return anchor_kl(net.policy(vec, objs, mask), (ref_mean, ref_log_std.detach(), ref_logits),
                     _ANCHOR_TRAIN_CFG)
```

`update()` 의 시그니처와 본문을 고친다(독스트링 끝에 한 줄 더하기):

```python
def update(net, opt, buffer, dagger_iter, cfg: PPOConfig, step: int, generator=None, ref=None) -> dict:
    """에폭을 반복하며 PPO 손실 + 모방 손실(감쇠)을 합쳐 한 걸음씩 최적화한다.

    최적화 한 걸음마다 `net.clamp_log_std()` 를 불러 표준편차 범위를 지킨다(M3 가 σ 붕괴로
    학생이 브레이크만 밟았던 실패를 되풀이하지 않는다). `approx_kl` 이 `target_kl` 을 넘으면
    그 에폭에서 바로 멈춘다.

    `cfg.anchor_coef > 0` 이면 미니배치마다 `anchor_coef * anchor_loss(net, batch, ref)` 를 더한다
    (M6b). 계수를 켰는데 `ref` 가 없으면 앵커가 조용히 꺼지므로 거부한다.
    """
    if cfg.anchor_coef > 0.0 and ref is None:
        raise ValueError("anchor_coef > 0 인데 ref 가 없다 — 앵커가 조용히 꺼진다."
                         " `make_reference(net.policy)` 로 얼린 참조를 넘겨라.")
    coef = imitation_coef(step, cfg)
    sums, count = {}, 0
    for _epoch in range(cfg.epochs):
        stop = False
        for batch in buffer.batches(cfg.minibatch, generator=generator):
            loss, parts = ppo_losses(net, batch, cfg, generator=generator)
            parts["imitation"] = 0.0
            if dagger_iter is not None and coef > 1e-4:
                dbatch = next(dagger_iter)
                imi, iparts = imitation_loss(net, dbatch, cfg)
                loss = loss + coef * imi
                parts["imitation"] = iparts["total"]
            parts["anchor"] = 0.0
            if cfg.anchor_coef > 0.0:
                anc = anchor_loss(net, batch, ref)
                loss = loss + cfg.anchor_coef * anc
                parts["anchor"] = float(anc.item())
            opt.zero_grad(set_to_none=True)
            # (이하 지금과 같다)
```

- [ ] **Step 4: 돌려서 통과를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_ppo.py -q > $TMP/m6b_t1.txt 2>&1; echo rc=$?`
Expected: rc=0. `test_앵커가_출발점_근처에_붙잡는다` 가 실패하면 숫자(`kls`)를 보고 원인을 적는다 — 단언을 약하게 고치지 않는다.

- [ ] **Step 5: 커밋한다**

```bash
git add vtd_rl/rl/ppo.py tests/rl/test_ppo.py
git commit -m "M6b: PPO 에 KL 앵커 — 미니배치 상태에서 얼린 출발 정책과의 KL(기본 0 은 비트 동일)"
git show -s --format=%B HEAD | grep -ciE "co-authored|claude|anthropic|generated with|🤖"   # 0 이어야 한다
```

---

### Task 2: `train_ppo.py --anchor-coef` — 출발 정책을 얼려 참조로 넘긴다

**Files:**
- Modify: `scripts/train_ppo.py`
- Test: `tests/rl/test_train_ppo.py`

**Interfaces:**
- Consumes: Task 1 의 `PPOConfig.anchor_coef`, `update(..., ref=...)`; `vtd_rl.policy.train.make_reference`.
- Produces: CLI `--anchor-coef FLOAT`(기본 `PPOConfig().anchor_coef`), 로그 줄 키 `anchor`(Task 1 의 `stats` 를 `**stats` 로 그대로 싣는다), `hparams["anchor_coef"]`.

- [ ] **Step 1: 실패하는 테스트를 쓴다** — `tests/rl/test_train_ppo.py` 끝에 더한다.

```python
def test_앵커_계수_인자가_cfg에_닿는다():
    mod = _load_train_ppo_module()
    a = mod._build_parser().parse_args(["--out", "/tmp/불필요-존재안함", "--init", "x.pt",
                                        "--anchor-coef", "0.3"])
    assert mod._build_cfg(a).anchor_coef == 0.3
    b = mod._build_parser().parse_args(["--out", "/tmp/불필요-존재안함"])
    assert mod._build_cfg(b).anchor_coef == 0.0


def test_앵커는_init_없이_거부한다():
    """무작위 시작점에 묶는 것은 뜻이 없다 — 무거운 준비(venv·모델) 전에 argparse 오류로 죽는다."""
    import sys
    mod = _load_train_ppo_module()
    old_argv = sys.argv
    try:
        for bad in (["train_ppo.py", "--smoke", "--out", "/tmp/불필요-존재안함", "--anchor-coef", "0.5"],
                    ["train_ppo.py", "--smoke", "--out", "/tmp/불필요-존재안함", "--init", "x.pt",
                     "--anchor-coef", "-1"]):
            sys.argv = bad
            with pytest.raises(SystemExit):
                mod.main()
    finally:
        sys.argv = old_argv


@pytest.mark.slow
def test_앵커_스모크가_로그에_KL을_남긴다(tmp_path):
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig
    init = tmp_path / "init.pt"
    DrivePolicy(PolicyConfig(squash=True)).save(str(init))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "train_ppo.py"),
                          "--smoke", "--out", str(tmp_path / "run"), "--seed", "0",
                          "--init", str(init), "--anchor-coef", "1.0", "--lr", "0.01"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    with open(tmp_path / "run" / "log.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows and all("anchor" in r and math.isfinite(r["anchor"]) and r["anchor"] >= 0.0 for r in rows)
    assert rows[0]["hparams"]["anchor_coef"] == 1.0
    # 가치만 배우는 처음 5 갱신(기본 --warmup-updates)이 지나면 정책이 움직여 KL 이 0 보다 커진다.
    assert max(r["anchor"] for r in rows) > 0.0
```

- [ ] **Step 2: 돌려서 실패를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_train_ppo.py -q -k "앵커" > $TMP/m6b_t2_red.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `unrecognized arguments: --anchor-coef`.

- [ ] **Step 3: 구현한다** — `scripts/train_ppo.py`.

임포트에 `make_reference` 를 더한다(이미 `from vtd_rl.policy.train import TrainConfig` 같은 줄이 있으면 거기에 붙인다).

`_build_parser()` 의 `--imitation-floor` 다음에:

```python
    ap.add_argument("--anchor-coef", type=float, default=default_cfg.anchor_coef,
                    help="M6b — 얼린 출발 정책(--init)과의 KL(ref‖net) 계수. 기본값 0.0 은 지금과"
                         " 같다(꺼짐). M6a 에서 앵커 없는 PPO 가 25 만 걸음 만에 KL 1.4~1.8 로"
                         " 멀어져 정지차 회피를 잃었다. --init 이 있어야 한다.")
```

`_build_cfg()` — `anchor_coef=a.anchor_coef` 를 더하고, 독스트링의 "**여섯** 필드" 를 "**일곱** 필드" 로, 나열에 `anchor_coef` 를 더한다.

`main()` 의 `--eval-every` 검증 바로 아래(스모크 덮어쓰기 **전**)에:

```python
    if a.anchor_coef < 0.0:
        ap.error("--anchor-coef 는 0 이상이어야 한다")
    if a.anchor_coef > 0.0 and not a.init:
        ap.error("--anchor-coef 는 --init 이 있어야 한다 — 무작위 시작점에 묶는 것은 뜻이 없다")
```

`ref_state = snapshot_policy(net)` 바로 아래에:

```python
        # M6b 앵커 참조 — `--init` 로드와 σ·엔트로피 덮어쓰기가 끝난 **뒤**, 갱신 **전**에 얼린다.
        # 계수가 0 이면 만들지 않는다(`update()` 가 참조를 안 본다).
        anchor_ref = make_reference(net.policy) if cfg.anchor_coef > 0.0 else None
```

`update(...)` 호출에 `ref=anchor_ref` 를 더한다:

```python
            stats = update(net, opt, buf, dagger_iter if not warming else None, cfg, step,
                           generator=gen, ref=anchor_ref)
```

`row = {...}` 위 주석의 `stats`(policy/value/…/updates) 나열에 `anchor` 를 더한다(다른 그룹 키와 안 겹친다 — 확인해서 적는다).

- [ ] **Step 4: 돌려서 통과를 확인한다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_train_ppo.py -q > $TMP/m6b_t2.txt 2>&1; echo rc=$?`
Expected: rc=0(기존 `test_인자를_안_주면_PPOConfig_기본값과_같다` 포함).

그다음 전체: `env -u PYTHONPATH .venv/bin/pytest -q > $TMP/m6b_full.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 5: 커밋한다**

```bash
git add scripts/train_ppo.py tests/rl/test_train_ppo.py
git commit -m "M6b: train_ppo --anchor-coef — --init 정책을 얼려 PPO 앵커 참조로 넘긴다"
git show -s --format=%B HEAD | grep -ciE "co-authored|claude|anthropic|generated with|🤖"   # 0 이어야 한다
```

---

## 본 실행 — 계획 주관자가 OMEN 에서 (구현 과제 아님)

1. `for d in vtd_rl scripts tests; do rsync ...; done` 로 동기화.
2. 스모크: 칸 B 설정으로 `--steps 30000 --eval-every 10000000 --final-eval-seeds 1` 한 번 — `anchor` 가 로그에 찍히고 0 보다 큰지, 걸음 속도.
3. 본 실행(실행 하나 ≈ RAM 2.5~3 GB 실측 → **동시 6 개**, 빈자리가 나면 다음 것):

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-04-m6b-$ARM-s$S --init runs/omen/2026-10-04-m4y/K2-a2-w3-s$S/policy-ema2500.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3b.json \
  --steps 1000000 --eval-every 250000 --eval-seeds 1 --final-eval-seeds 3 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef $COEF --lr 1e-4 --warmup-updates 20
```

   순서: s0(A·B·C), s1(A·B·C), s2(A·B·C). 끝난 실행마다 M6a 와 같은 방식으로 `ac-best`·마지막을 꺼내 선택 창·보고 창 평가.
4. 숫자는 스크립트로(`m6b_tables.py`), 판정은 위 기준 그대로.
