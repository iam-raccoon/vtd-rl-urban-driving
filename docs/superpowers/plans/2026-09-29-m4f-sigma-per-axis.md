# M4f — σ 어닐링을 제대로 재고 모방 바닥을 확증한다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M4e 가 무효로 만든 σ 어닐링 실험을 **축별 구현으로 고쳐 다시 재고**, 유일하게 방향이 나온 모방 앵커 바닥을 **시드를 늘려 확증**한다.

**Architecture:** 원인이 된 두 줄(`set_log_std` 의 스칼라 `fill_`, `sigma_anneal_cur` 의 `.max()`)을 **축별**로 고치고 그 계약을 테스트로 잠근다. 그 다음 σ 어닐링을 같은 설정으로 다시 돌리고, 모방 바닥은 **시드 6 개**로 늘려 n=3·p=0.25 였던 신호를 확증한다. 덤으로 `diag_stall.py` 의 의미를 잠근다(지금은 돌연변이 6 종이 전부 살아남는다).

**Tech Stack:** Python 3.10, PyTorch 2.14(cpu), gymnasium 1.3, numpy **1.26.4 고정**, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`

**직전 증거:** `docs/reports/m4e-ppo.md`(+`-notes`), `docs/reports/m4e-stall.md`

## 왜 이 순서인가 (M4e 가 남긴 것)

M4e 는 결정적 정책이 멈추는 문제를 다뤘다. 진단: **가속이 모자란 게 아니라 제동을 학습한다**(정체 직전 가속 −0.81~−0.87, 실패의 100% 가 정체·시간초과 0). 개입 둘 중:

- **모방 앵커 바닥(`--imitation-floor 0.5`)** — 3M 결정적 완주율 **44.4% → 66.7%**, 정체 **10.0 → 6.0/18**. 확률적은 거의 그대로(98.1%). **다만 n=3, 시드별 +50/0/+17 pp 로 부호검정 p=0.25 — 방향이지 확증이 아니다.** 단계②는 한 시드 악화.
- **σ 후반 어닐링** — **무효다.** `set_log_std()` 가 두 축을 한 스칼라로 채우고 `sigma_anneal_cur` 을 `.max()` 로 잡아, 시작 시점 `log_std = [-2.000, -0.175]` 에서 **조향 σ 가 e^−2.0 → e^−0.183 으로 6.2 배 폭증**하며 시작했다. 리턴 붕괴(+17.7 → −53.5)는 그 충격과 같은 이터레이션이고, **σ 가 실제로 내려가는 구간에서는 −70 → −24 로 회복**한다. **순수한 σ 감소를 한 번도 안 쟀다.**

그리고 M4e 최종 리뷰가 `scripts/diag_stall.py` 에 **돌연변이 6 종을 심었는데 전부 살아남았다**(가속/조향 인덱스 교환, `stalled` 필터 반전, tail 구간 뒤집기 등). 그 스크립트가 M4e 진단의 유일한 근거였다.

## Global Constraints

- 커밋·PR 에 **Claude 표기 금지**(`Co-Authored-By` / `Claude-Session` / "Generated with" / 🤖 / anthropic 주소). **커밋 뒤 `git show -s --format=%B HEAD` 를 직접 실행해 눈으로 확인하고 보고할 것.**
- `third_party/rule_stack` **수정 금지**(읽기만). `runs/`·`*.pt`·`*.npz`·CSV·`*.xodr`·`*.xml` **커밋 금지**.
- numpy **1.26.4 고정**. 문서·주석·커밋 메시지는 **한국어**.
- **성적표 숫자는 전부 스크립트가 생성한다.**
- **점수는 `mean_score_completed`(완주 판만)를 주 자로 쓴다.** `mean_score_raw` 는 채점기가 미방문 구간을 100 점으로 채워 **일찍 멈춘 정책일수록 높다**(M4d 최종 리뷰 C1). `mean_score` 는 미완주를 0 점 처리한다.
- `vtd_rl/eval/verdict.py` 의 `M3_SCORE = {"stage1": 98.5, "stage2": 94.2}` 를 **낮추지 말 것**.
- **테스트를 돌리는 모든 Bash 호출에 `timeout: 600000`(ms)을 빠짐없이 붙일 것.** 빠지면 하네스가 배경으로 넘기고 보고 없이 멈추게 된다(이 프로젝트에서 **여덟 번**). `run_in_background`·Monitor 금지.
- 겹치지 않는 **세 조각**, **rc** 로 판정:
  - `env -u PYTHONPATH .venv/bin/python -m pytest -q -m "not slow"`
  - `env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/ -q` ← **약 710 초**로 도구 상한을 넘어 배경으로 갈 수 있다. **알림을 기다렸다 출력 파일을 Read 로 읽어 rc 를 확인할 것. 멈추지 말 것.**
  - `env -u PYTHONPATH .venv/bin/python -m pytest -q -m slow --ignore=tests/rl`
  - **기준선(M4e 병합 시점, main `ce1b166`): 492 / 145 / 27, 전부 rc=0.**
- 이 머신(lab-main)에는 `uv` 가 없다. `.venv` 를 직접 쓴다.
- **커밋을 먼저 하고 그 다음에 돌연변이 실험을 할 것.** 미커밋 상태에서 `git checkout` 을 해 구현이 날아간 일이 **세 번** 있었다. 돌연변이는 작업 트리에 심고 `git checkout -- <경로>` 로 **경로를 지정해** 원복 — editable 설치라 `/tmp` 복사본은 진짜 패키지를 임포트한다.
- **`runs/` 안의 파일로 테스트를 게이트하지 말 것.** 머신마다 달라 `skipif` 가 조용히 항상 참이 된다(M4e 에서 그 결함이 났다). 테스트가 체크포인트를 **만들어** 쓴다.
- **`--smoke` 는 `--steps` 를 무조건 4000 으로 덮어쓴다**(`train_ppo.py` 의 smoke 분기). 스모크 테스트에서 `--steps` 를 의도대로 못 쓴다는 것을 알고 쓸 것.
- 긴 학습(3M)은 **OMEN**(`user-OMEN`, 32 코어, `--device cpu`)에서 계획 주관자가 직접 돌린다.
- **개입을 띄우기 전에 "이 값이 실제로 무언가를 바꾸는가" 를 숫자로 확인할 것.** M4e 에서 `--imitation-floor 0.3` 이 감쇠식과 안 맞아(`0.5^(3M/2M)=0.353 > 0.3`) 한 팔이 통째로 no-op 였다.

---

## File Structure

| 파일 | 책임 | 작업 |
|---|---|---|
| `vtd_rl/policy/net.py` | `set_log_std` 를 축별로 받게 | 1 |
| `scripts/train_ppo.py` | `sigma_anneal_cur` 을 축별 벡터로, `_sigma_anneal_target` 도 축별 | 1 |
| `scripts/diag_stall.py` | 순수 함수를 떼어 의미를 잠글 수 있게 | 2 |
| `tests/policy/test_net.py`·`tests/rl/test_train_ppo.py`·`tests/policy/test_diag_stall.py` | 계약 잠금 | 1·2 |
| `docs/reports/m4f-*.md` | 성적표·해석 | 4 |

**Task 1 만 `train_ppo.py` 를 건드린다** — Task 2 는 `diag_stall.py` 뿐이라 **순서 의존이 없다.** 다만 같은 작업 트리이므로 **순차로 돌린다**(리뷰와 구현을 겹치지 않게).

---

### Task 1: σ 어닐링을 축별로 고친다 ✅ 완료 (`f17fa30`)

> **실행 결과 (2026-09-29).** 테스트 497 / 148 / 27 전부 rc=0. 돌연변이 3 종 전부 잡힘.
>
> **★ 계획서의 슬로우 테스트가 틀렸고 구현자가 고쳤다.** 아래 Step 1 에 적은 대로 스모크를
> 맨 처음부터 돌리면 4000 스텝 동안 두 축이 **0.005 밖에 안 벌어져**(실측 끝값
> `[-1.0256, -1.0226]`), `.max()` 버그를 심어도 상승폭이 허용치 0.05 안에 들어 **테스트가
> 통과해 버린다.** 구현자가 M4e 실측 시작값 `[-2.000, -0.175]` 짜리 체크포인트를 테스트
> 안에서 만들어 `--init` 으로 넘기게 바꿨고, 그러자 수정 전 코드에서 정확히 재현하며 실패했다.
> **교훈: "버그를 심어 잡히는지" 를 실제로 해 보지 않으면 잠금 테스트는 잠그지 않는다.**
>
> 부수 확인: `--sigma-anneal-from` 을 안 준 실행은 수정 전후 `log.jsonl` 32 줄이 전부 동일
> (`elapsed_s`·경로 제외). **Task 3 의 M4e 재사용 Ruling 이 성립한다.** 코드로도 자명하다 —
> `start is None` 이면 캡처 블록을 건너뛰고 `_sigma_anneal_target` 이 즉시 `None` 을 돌려줘
> `set_log_std()` 에 닿지 않는다.

**Files:**
- Modify: `vtd_rl/policy/net.py`, `scripts/train_ppo.py`
- Test: `tests/policy/test_net.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `DrivePolicy.set_log_std(value)` — `value` 가 **스칼라면 지금처럼 두 축을 같게**, **길이 2 의 시퀀스/텐서면 축별로** 채운다. 둘 다 `clamp_log_std()` 로 범위 안에 누른다.
  - `scripts/train_ppo.py::_sigma_anneal_target(step, start, total, cur, floor)` — `cur` 이 **스칼라 또는 길이 2** 를 받고, 넣은 모양 그대로 돌려준다(축별이면 축별 목표).
- Consumes: `PolicyConfig.log_std_min`(−2.0)·`log_std_max`(0.5), `net.clamp_log_std()`

**왜 이렇게 고치나:** 현재 `set_log_std(value: float)` 는 `self.log_std.fill_(float(value))` 로 **두 축을 같게** 만들고, 호출부는 `float(net.policy.log_std.detach().max())` 로 **큰 축만** 잡는다. M4e 실행에서 시작 시점이 `[-2.000, -0.175]` 였으므로 **조향 σ 가 6.2 배 뛰었다.** 축별로 고치면 각 축이 **자기 현재값에서** 하한까지 내려간다 — 그것이 원래 의도였다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/policy/test_net.py` 에 더한다:
```python
def test_set_log_std가_축별_값을_받는다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std([-1.5, -0.5])
    assert torch.allclose(net.log_std, torch.tensor([-1.5, -0.5]))
    net.set_log_std(torch.tensor([-0.7, -1.9]))
    assert torch.allclose(net.log_std, torch.tensor([-0.7, -1.9]))


def test_set_log_std_스칼라는_예전처럼_두_축을_같게():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std(-1.25)
    assert torch.allclose(net.log_std, torch.tensor([-1.25, -1.25]))


def test_set_log_std_축별도_범위에_눌린다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    net.set_log_std([-99.0, 99.0])
    assert torch.allclose(net.log_std,
                          torch.tensor([net.cfg.log_std_min, net.cfg.log_std_max]))
```

`tests/rl/test_train_ppo.py` 에 더한다:
```python
def test_sigma_anneal_목표가_축별로_따로_내려간다():
    """★ M4e 를 무효로 만든 자리 — 축마다 **자기 현재값**에서 하한까지 가야 한다.

    M4e 는 두 축의 `.max()` 하나로 둘 다 덮어써서, 시작 시점 `[-2.000, -0.175]` 에서
    **조향 σ 가 6.2 배 폭증**하며 어닐링이 시작됐다. 그 충격이 리턴을 무너뜨렸고
    실험 한 팔이 통째로 무효가 됐다.
    """
    mod = _load_train_ppo_module()
    cur = [-2.0, -0.2]
    start, total, floor = 1000, 2000, -2.0
    assert mod._sigma_anneal_target(1000, start, total, cur, floor) == pytest.approx([-2.0, -0.2])
    mid = mod._sigma_anneal_target(1500, start, total, cur, floor)
    assert mid == pytest.approx([-2.0, -1.1])          # 조향은 이미 하한이라 안 움직인다
    assert mod._sigma_anneal_target(2000, start, total, cur, floor) == pytest.approx([-2.0, -2.0])


def test_sigma_anneal_스칼라_cur_은_예전과_같다():
    mod = _load_train_ppo_module()
    f = mod._sigma_anneal_target
    assert f(100, 1000, 2000, -0.5, -2.0) is None
    assert f(1500, 1000, 2000, -0.5, -2.0) == pytest.approx(-1.25)


@pytest.mark.slow
def test_sigma_anneal이_조향_시그마를_올리지_않는다(tmp_path):
    """어닐링 구간에서 **어느 축도 시작값보다 커지면 안 된다** — M4e 의 실패를 직접 잠근다."""
    import json
    import os
    import subprocess

    repo = os.path.join(os.path.dirname(__file__), "..", "..")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run(
        [os.path.join(repo, ".venv", "bin", "python"),
         os.path.join(repo, "scripts", "train_ppo.py"),
         "--out", str(tmp_path / "run"), "--smoke", "--sigma-anneal-from", "2000"],
        capture_output=True, text=True, env=env, cwd=repo, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    rows = [json.loads(x) for x in open(tmp_path / "run" / "log.jsonl") if x.strip()]
    before = [r for r in rows if r["step"] < 2000]
    after = [r for r in rows if r["step"] >= 2000]
    assert before and after
    # 어닐링 시작 직전까지의 축별 최댓값을 천장으로 삼는다. 여유 0.05 는 캡처 직전 한 번의
    # PPO 업데이트(엔트로피 보너스가 σ 를 조금 올린다)를 견디려는 것이지 버그를 봐주려는 게
    # 아니다 — M4e 의 실제 폭증은 **1.82**(log_std -2.000 → -0.175)로 이 여유의 36 배다.
    base = [max(r["log_std"][ax] for r in before) for ax in (0, 1)]
    for r in after:
        for ax in (0, 1):
            assert r["log_std"][ax] <= base[ax] + 0.05, (ax, r["step"], r["log_std"], base)
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/policy/test_net.py -q -k 축별` (timeout 600000)
Expected: FAIL — `set_log_std` 가 시퀀스를 `float()` 에 넣어 `TypeError`

- [ ] **Step 3: 구현**

`vtd_rl/policy/net.py`:
```python
    @torch.no_grad()
    def set_log_std(self, value):
        """`log_std` 를 `value` 로 덮어쓰고 범위 안에 눌러 준다.

        `value` 가 스칼라면 두 축을 같게, **길이 2 면 축별로** 채운다.
        축별이 기본 쓰임이다 — M4e 에서 두 축을 `.max()` 하나로 덮어썼다가 조향 σ 가
        6.2 배 폭증해 실험 한 팔이 무효가 됐다(`docs/reports/m4e-ppo-notes.md`).
        `forward()` 는 여전히 clamp 하지 않는다(기울기를 죽인다) — 여기서만 누른다.
        """
        v = torch.as_tensor(value, dtype=self.log_std.dtype, device=self.log_std.device)
        self.log_std.copy_(v.expand_as(self.log_std))
        self.clamp_log_std()
```
**`expand_as` 가 스칼라와 길이 2 를 둘 다 받는지 확인해라** — 0 차원 텐서는 확장되고 `(2,)` 는 그대로다. 안 되면 분기해서 쓰고 **왜 그렇게 했는지 보고해라.**

`scripts/train_ppo.py`:
- `_sigma_anneal_target` 이 `cur` 을 스칼라·시퀀스 둘 다 받게 한다. 스칼라면 스칼라를, 시퀀스면 **같은 길이 리스트**를 돌려준다. `t` 계산은 그대로다.
- 캡처를 축별로 바꾼다:
```python
                sigma_anneal_cur = net.policy.log_std.detach().cpu().tolist()   # 축별
```
**주석에 M4e 의 실패를 적어라** — 왜 `.max()` 가 아니라 축별인지.

- [ ] **Step 4: 통과 확인** — 세 조각 전부 `rc=0`, 기준선 492 / 145 / 27.

**그리고 `--sigma-anneal-from` 을 안 준 실행이 예전과 완전히 같은지 확인해라**(기본값 `None`).

- [ ] **Step 5: 돌연변이로 확인**

커밋한 **뒤** 셋을 심어 각각 잡히는지 보고해라:
1. 캡처를 `.max()` 로 되돌리기 → `test_sigma_anneal이_조향_시그마를_올리지_않는다` 가 잡아야 한다.
2. `set_log_std` 에서 `expand_as` 대신 `fill_(float(value[0]))` → 축별 테스트가 잡아야 한다.
3. `clamp_log_std()` 호출 삭제 → 범위 테스트가 잡아야 한다.

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/policy/net.py scripts/train_ppo.py tests/policy/test_net.py tests/rl/test_train_ppo.py
git commit -m "M4f — σ 어닐링을 축별로 고친다(M4e 에서 조향 σ 가 6.2 배 폭증했다)"
git show -s --format=%B HEAD
```

---

### Task 2: `diag_stall.py` 의 의미를 잠근다 ✅ 완료 (`2550b4b`)

> **실행 결과 (2026-09-29).** 테스트 501 / 148 / 27 전부 rc=0. 기존 슬로우 테스트
> `test_정체_진단이_판별_요약을_낸다` 가 그대로 통과 — 리팩터가 동작을 안 바꿨다는 증거다.
> 돌연변이 **4 종 전부 잡힘**(계획서의 3 종 + 구현자가 추가한 1 종: 상수는 그대로 두고
> `_action_row` **안의 쓰임만** 바꾸기).
>
> 계획서보다 테스트를 하나 더 넣었다 — `test_행동_축_인덱스가_action_to_command와_같다`.
> 계획서 테스트는 `(STEER_IDX, ACCEL_IDX) == (0, 1)` 이라는 **리터럴만** 확인해서, 환경 쪽
> 축 순서가 바뀌면 조용히 거짓이 된다. 새 테스트는 상수를 실제 `to_command()` 에 통과시킨다.
>
> **남은 것(M4g 후보): M4e 돌연변이 ⑥(대조군 표본을 결정적으로)은 아직 안 잡힌다.**
> `run_policy_episode_rows` 안의 부수효과 호출이라 순수 함수로 못 뗀다. 왜 `deterministic=False`
> 여야 하는지 주석으로만 남겼다. (①`.max()`→`.min()`·②캡처 상수 0.0 은 `diag_stall.py` 에
> 없다 — `train_ppo.py` 쪽이고 Task 1 이 이미 잠갔다.)

**Files:**
- Modify: `scripts/diag_stall.py`
- Test: `tests/policy/test_diag_stall.py`

**Interfaces:**
- Produces: `scripts/diag_stall.py` 의 순수 함수 — `_tail(rows, key, n)`(이미 있다)과, 지금 `main()` 안에 섞여 있는 **정체 판 고르기**·**축 인덱스**를 떼어낸 함수.
- Consumes: 없음(이 작업은 리팩터 + 테스트다)

**왜 필요한가:** M4e 최종 리뷰가 이 파일에 돌연변이 **6 종**을 심었는데 **전부 살아남았다** — ① `.max()`→`.min()` ② 캡처 상수 0.0 ③ `_tail` 이 **앞** 100 걸음 ④ 가속/조향 **인덱스 교환** ⑤ `stalled` 필터 **반전** ⑥ 대조군 표본을 결정적으로. 지금 테스트는 요약의 **모양**만 본다. **이 스크립트가 M4e 진단("제동을 학습한다")의 유일한 근거였다.**

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/policy/test_diag_stall.py` 에 더한다(이 파일은 이미 `json`·`os`·`subprocess`·`pytest` 를 임포트한다 — `importlib` 로 스크립트를 모듈로 불러 쓰는 헬퍼가 없으면 `tests/rl/test_train_ppo.py:19` 의 `_load_train_ppo_module()` 패턴을 그대로 흉내 내라):
```python
def _load_diag_stall():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "diag_stall.py")
    spec = importlib.util.spec_from_file_location("diag_stall", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tail이_뒤쪽_n개를_가져온다():
    m = _load_diag_stall()
    rows = [{"a": i} for i in range(10)]
    assert m._tail(rows, "a", 3) == [7, 8, 9]
    assert m._tail(rows, "a", 100) == list(range(10))   # n 이 크면 전부
    assert m._tail([], "a", 3) == []


def test_정체_판만_고른다():
    m = _load_diag_stall()
    eps = [{"outcome": "goal", "rows": [1]}, {"outcome": "stalled", "rows": [2]},
           {"outcome": "timeout", "rows": [3]}, {"outcome": "stalled", "rows": [4]}]
    got = m._stalled_only(eps)
    assert [e["rows"][0] for e in got] == [2, 4]


def test_행동_축_인덱스가_조향0_가속1이다():
    """`ActionConfig` 의 순서와 같아야 한다 — 뒤집히면 진단이 통째로 거짓말한다."""
    m = _load_diag_stall()
    assert (m.STEER_IDX, m.ACCEL_IDX) == (0, 1)
    row = m._action_row({"control": [0.3, -0.7]}, {"control": [0.4, -0.6]}, speed=0.5)
    assert row["steer_det"] == pytest.approx(0.3)
    assert row["accel_det"] == pytest.approx(-0.7)
    assert row["accel_sample"] == pytest.approx(-0.6)
```

**`_stalled_only`·`_action_row`·`STEER_IDX`·`ACCEL_IDX` 는 아직 없다** — `main()` 안의 그 로직을 떼어내면서 만든다. **이름이 마음에 안 들면 바꾸되 테스트도 같이 바꾸고 보고해라.**

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/policy/test_diag_stall.py -q -k "tail or 정체 or 인덱스"` (timeout 600000)
Expected: FAIL — `AttributeError: module 'diag_stall' has no attribute '_stalled_only'`

- [ ] **Step 3: 구현**

`main()` 안에 섞여 있는 로직을 **순수 함수로 떼어낸다.** 동작을 바꾸지 마라 — **순수 이동 + 이름 붙이기**다. `STEER_IDX = 0`·`ACCEL_IDX = 1` 을 모듈 상수로 올리고 **`vtd_rl/env/action.py` 의 실제 순서를 읽어 확인한 뒤** 주석에 근거를 적어라.

- [ ] **Step 4: 통과 확인** — 세 조각 전부 `rc=0`.

**그리고 기존 슬로우 테스트(`test_정체_진단이_판별_요약을_낸다`)가 그대로 통과해야 한다** — 리팩터가 동작을 안 바꿨다는 증거다.

- [ ] **Step 5: 돌연변이로 확인**

커밋한 **뒤** M4e 리뷰가 심었던 것 중 셋을 다시 심어 **이제 잡히는지** 보고해라: `_tail` 을 앞쪽으로 / `STEER_IDX`·`ACCEL_IDX` 교환 / `_stalled_only` 필터 반전.

- [ ] **Step 6: 커밋**

```bash
git add scripts/diag_stall.py tests/policy/test_diag_stall.py
git commit -m "M4f — diag_stall 의 의미를 순수 함수로 떼어 잠근다(돌연변이 6 종이 살아남았다)"
git show -s --format=%B HEAD
```

---

### Task 3: 본 실험

**Files:** 없음. **계획 주관자가 OMEN 에서 직접 돌린다.**

**돌릴 설정** — OMEN 30 환경·3M 스텝·`--device cpu`, `--init runs/omen/2026-09-26-m3-squash-warm/policy.pt`, `--comfort-on-intent --entropy-coef 0.05 --violation-mode once_per_section`, `--final-eval-seeds 3`:

| 이름 | 바뀌는 것 | 시드 | 무엇을 가르나 |
|---|---|---|---|
| `base` | 없음 | 0·1·2 | 기준선(M4e `m4d-align` 재현이어야 한다) |
| `sigma-axis` | `--sigma-anneal-from 2000000` | 0·1·2 | **축별로 고친 σ 어닐링** |
| `floor6` | `--imitation-floor 0.5` | **0~5(6 개)** | 모방 바닥 **확증** |

**`floor6` 의 시드를 6 개로 늘리는 이유**: M4e 가 n=3 에 +50/0/+17 pp 로 부호검정 p=0.25 였다. **방향은 있는데 확증이 안 된다.** 시드 6 개면 전부 개선일 때 p=0.016 이다.

**★ Ruling(2026-09-29, 계획 주관자) — 12 회가 아니라 6 회만 새로 돌린다.** OMEN 의 `runs/omen/2026-09-29-m4e/` 에 `m4d-align-s0/1/2` 와 `imit-floor05-s0/1/2` 가 **그대로 남아 있다**(확인함). 이 두 팔은 `--sigma-anneal-from` 을 **안 주므로 `set_log_std()` 를 한 번도 안 부른다** — Task 1 의 수정이 그 경로에 닿지 않는다. 그러므로:

- `base` 시드 0·1·2 → **M4e `m4d-align-s*` 재사용**(새로 안 돌림)
- `floor6` 시드 0·1·2 → **M4e `imit-floor05-s*` 재사용**, 시드 **3·4·5 만 새로**
- `sigma-axis` 시드 0·1·2 → **전부 새로**(여기가 고친 코드가 실제로 도는 유일한 팔)

**새 실행 6 회, 약 2 시간.** 대가: 재사용한 6 회는 수정 **전** 코드가 낸 것이다. 그래서 **Task 1 Step 4 의 "`--sigma-anneal-from` 을 안 준 실행이 예전과 완전히 같은가" 확인이 이 재사용의 전제**다. 그 확인이 어긋나면 재사용을 버리고 12 회를 전부 새로 돌린다. **틀렸을 때의 비용은 2 시간 재실행이고, 맞을 때의 이득도 2 시간이다.**

- [x] **Step 1: 개입이 실제로 무는지 먼저 확인** ✅ — `--smoke` 로 σ 어닐링을 돌려 **두 축이 각자 자기 값에서 내려가는지** 로그로 본다. M4e 는 이걸 안 해서 한 팔을 날렸다.

  > **본 실행(`sigma-axis-s0`, 3M)에서 확인 완료 (2026-09-29).** 어닐링 구간의 실측 궤적:
  >
  > | 스텝 | 조향 `log_std` | 가속 `log_std` |
  > |---|---|---|
  > | 1,904,640 (어닐링 전) | −2.0000 | −0.1817 |
  > | 2,119,680 | −2.0000 | −0.3932 |
  > | 2,549,760 | −2.0000 | −1.1782 |
  > | 3,002,880 (끝) | −2.0000 | −2.0000 |
  >
  > **조향축은 이미 하한이라 안 움직이고, 가속축만 자기 값에서 하한까지 선형으로 내려간다.**
  > M4e 는 같은 자리에서 조향축이 −2.000 → −0.183 으로 **뛰었다**(σ 6.2 배). 축별 최대값 대조:
  > 어닐링 전 `[-2.0000, -0.1749]` → 후 `[-2.0000, -0.1830]`, 어느 축도 안 올라갔다.
  > **순수한 σ 감소를 이 프로젝트에서 처음 쟀다.**
- [ ] **Step 2: 코드를 올리고 OMEN 에서 빠른 스위트** (`rc=0`)
- [ ] **Step 3: 새 실행 6 회(약 2 시간) — 위 Ruling 참고.** `sweep_ppo.py --resume`, 설정마다 `--name` 다르게. `nohup` 분리 실행. **감시 필터에 앵커(`^`)·대괄호 이스케이프를 쓰지 마라**(ssh 를 거치면 깨진다 — M4c·M4e 에서 당했다). **예상 시각에 이벤트가 없으면 감시를 믿지 말고 로그를 직접 확인해라.**
- [ ] **Step 4: 평가 — 최선·3M 체크포인트 둘 다**, 결정적·확률적 둘 다(`scripts/eval_det_vs_stoch.py`). **M4d·M4e 가 이 자리에서 두 번 틀렸다.**
- [ ] **Step 5: 정체 수**(`scripts/diag_stall.py`) — 이 마일스톤의 직접 지표.

---

### Task 4: 성적표

**Files:**
- Create: `docs/reports/m4f-ppo.md`(스크립트 생성), `docs/reports/m4f-ppo-notes.md`(해석)
- Modify: `README.md`

`scripts/report_m4a.py` 로 만든다(`--sweep` 셋, `--title "M4f 성적표 — 축별 σ 어닐링과 모방 바닥 확증"`, `--student-label "M4f 학생"`, `--run` 은 **선택 없이 시드 0**).

**해석 노트에 반드시 담을 것:**
- **σ 어닐링이 이번엔 제대로 걸렸는지** — 두 축의 `log_std` 궤적을 표로. 어느 축도 시작값보다 커지면 안 된다.
- **그래서 σ 어닐링이 드는가.** M4e 는 이 질문에 답을 못 했다. **안 들면 "안 든다" 고 적어라.**
- **모방 바닥이 시드 6 개에서 확증되는가.** 시드별 변화를 전부 적고 **부호검정 p 를 계산해 적어라.**
- 결정적·확률적 둘 다, 최선·3M 둘 다, **정체 수**
- 점수는 **`mean_score_completed`** 주 자
- **개입이 배포되는 정책(`ac-best`)을 바꾸는가** — M4e 는 안 바꿨다(`best_step` 이 개입 전이었다). 이번에도 그러면 그렇게 적어라.
- **이 성적표가 주장하지 않는 것**
- M4g 로 넘기는 것

**목표를 못 채우면 기준을 고치지 마라.**

---

## 이 계획이 미루는 것

- **왜 제동을 학습하는지**(`m4e-stall.md` 의 한계). 후보: 위반을 피하는 가장 싼 길 / 승차감 항 / 값함수의 낙관. **개입 둘의 답이 나온 뒤에 판다.**
- 스펙 걱정("한 번 깎인 뒤 계속 어긴다")을 제대로 재는 계측, 항목 ⑮ 를 채점기와 맞추기.
- 커리큘럼 단계 ③④⑤.
- `--smoke` 가 `--steps` 를 덮어쓰는 것, `report_m4a.py` 성적표의 항목⑦ 중복과 열 합 불일치, `_goal_lines` 의 문자열 매칭, `OutcomeCounter` 가 0 인 사유의 키를 안 만드는 것, `ReturnTracker` 의 판 길이 +1 편향, `RewardConfig.rule_scale` 이 커리큘럼에서 설정되지 않는 것, 승차감 콜드스타트, `tests/rl/` 조각이 도구 상한을 넘는 것.
