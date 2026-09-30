# M4k — 단계 ③ 에서 DAgger 를 다시 모은다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M3 학생이 단계 ③ 을 **안 박고 끝내게** 만든다. 그러려면 (1) 액터가 시드마다 달라져야 하고, (2) DAgger 가 단계 ③ 에서 수집할 수 있어야 한다.

**Architecture:** 관측은 멀쩡하고 **정책이 물체 채널을 한 번도 학습한 적이 없다**(M4j 판정 A — 물체 슬롯 16 개가 M3 학습 내내 마스크였다). 그래서 파이프라인이 아니라 **데이터**를 고친다. 다만 지금 액터는 고정이라 시드를 늘려도 데이터가 안 는다 — 판 **변종(variant)** 을 만들어 다양성을 먼저 확보하고, 그 위에서 DAgger 를 돌린다.

**Tech Stack:** Python 3.10, PyTorch 2.14(cpu), gymnasium 1.3, numpy **1.26.4 고정**, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`

**직전 증거:** `docs/reports/m4i-student-stage3.md`, `docs/reports/m4j-obs-diag.md`, `docs/reports/m4h-stage3-notes.md`

## ★ 조사로 확인한 것 (파일:줄 확인 완료)

1. **`run_dagger.py` 는 단계 ① 에서만 수집한다.** `stage1`·`stage2` 를 경로 상수로 박아 두고(`:203-204`), 수집 작업은 **stage1 만** 쓴다(`:225-227` — `names = [b.name for b in _boards(stage1, ...)]`, jobs 도 `stage1`). 평가만 ①② 둘 다 한다.
2. **⚠ `--smoke` 는 `slice_board` 를 쓰고(`:51`), `slice_board` 는 액터를 통째로 버린다**(`world/board.py:71`: `actors=[], lights=[], zones=[]`). **단계 ③ 에 `--smoke` 를 그대로 쓰면 액터 0 개인 판에서 아무것도 시험하지 않는다.** 이 계획이 반드시 막아야 할 조용한 실패다.
3. **월드는 액터를 생성 시점에 한 번 읽고**(`world/world.py:50`), `VtdDriveEnv._worlds` 는 **판 이름으로 캐시**한다(`drive_env.py:65,89-96`, 이름이 겹치면 `:96` assert). 그래서 **같은 이름으로 액터만 바꿔 끼울 수 없다** — 변종은 **이름이 달라야** 한다. `vec_env._boards` 가 이미 `#{stage}` 를 붙이는 전례가 있다(`rl/vec_env.py:35`).
4. **액터 기술과 스키마는 M4h 가 만들었다** — `world/place.py` 의 `route_point`·`actor_from_spec`, 허용 키 `{id,kind,type,size,s,lateral,speed,pts,spawn,events}`, 모르는 키는 `ValueError`.
5. **학생의 실패는 첫 액터 한 곳에 뭉친다**(M4i: `course_H` 185/185/185 — 시드가 걸음 수를 거의 안 바꾼다). **시드는 신호 위상만 바꾼다.**
6. **선생님은 단계 ③ 을 18/18 완주한다**(96.8) — 라벨을 낼 수 있다. 다만 **⑬ 지시등 42 슬롯을 틀린다**(M4h) — 모방은 그 실수를 그대로 배운다.

## Global Constraints

- 커밋·PR 에 **Claude 표기 금지**. **커밋 뒤 `git show -s --format=%B HEAD` 로 확인하고 보고할 것.**
- `third_party/rule_stack` **수정 금지**(읽기만). `runs/`·`*.pt`·`*.npz`·CSV·`*.xodr`·`*.xml` **커밋 금지**.
- numpy **1.26.4 고정**. 문서·주석·커밋 메시지는 **한국어**.
- **성적표 숫자는 전부 스크립트가 생성한다.**
- **테스트를 돌리는 모든 Bash 호출에 `timeout: 600000`(ms).** `run_in_background`·Monitor 금지. 배경으로 넘어가면 알림을 기다렸다 **파일에서 rc 를 읽을 것.**
- **rc 로 판정**(파이프를 태우면 `$?` 가 `tail` 것이 된다 — 파일로 받고 rc 를 따로 읽는다).
- **기준선(main `ababf53`): 557 / 148 / 30, 전부 rc=0.** 전체 608 개(겹침 127).
- **⚠ 셋째 조각이 432 초 / 600 초다. 느린 테스트를 더하지 마라** — 더해야 하면 조각을 먼저 쪼개고 그 사실을 보고해라.
- 커밋을 **먼저** 하고 돌연변이 실험. 원복은 `git checkout -- <경로>`(editable 설치).
- **`runs/` 안의 파일로 테스트를 게이트하지 말 것.**
- 긴 실행은 **OMEN**(`user-OMEN`, 32 코어, `--device cpu`)에서 계획 주관자가 돌린다. **⚠ OMEN 의 git 체크아웃은 `288e12c` 로 낡았고 작업 트리는 rsync 로 맞춰져 있다 — `git pull` 금지.**
- **이 마일스톤은 PPO 를 건드리지 않는다**(`vtd_rl/rl/`·`scripts/train_ppo.py`).

---

## File Structure

| 파일 | 책임 | 작업 |
|---|---|---|
| `vtd_rl/world/place.py` | 액터 흔들기(jitter) — 결정적 rng | 1 |
| `vtd_rl/world/board.py` | 판 변종 생성(`variants`), 이름 규칙 | 1 |
| `curricula/stage3.json` | 변종 설정을 싣는다 | 1 |
| `scripts/run_dagger.py` | `--stage` 를 받는다, smoke 의 액터 소실을 막는다 | 2 |
| `docs/reports/m4k-dagger-stage3.md` | 성적표 | 4 |

---

### Task 1: 액터를 시드마다 흔든다 — 판 변종

**Files:** Modify `vtd_rl/world/place.py`, `vtd_rl/world/board.py`, `curricula/stage3.json`; Test `tests/test_place.py`, `tests/test_board_schema.py`, `tests/test_stage3.py`

**Interfaces:**
- Produces: `load_curriculum(path, variants: int = 1)` — `variants > 1` 이면 판마다 **변종 N 개**를 만든다. 이름은 `<원래이름>@v<k>`. 각 변종은 **자기 결정적 rng** 로 액터의 `s`·`lateral` 을 흔든다.
- Produces: 커리큘럼 판 항목의 새 선택 키 `jitter` — `{"s": 40.0, "lateral": 0.6}` 처럼 축별 최대 흔들기 폭(균등분포 ±).
- Consumes: M4h 의 `actor_from_spec`

**왜 변종인가:** 월드가 판 이름으로 캐시되므로(조사 3) 같은 이름의 액터를 바꿔 끼울 수 없다. **이름이 다른 판을 여러 개 만드는 것이 기존 기계장치와 싸우지 않는 유일한 길**이다.

**왜 결정적 rng 인가:** 같은 `(판, 변종 번호)` 는 **항상 같은 배치**여야 한다. 안 그러면 DAgger 라운드마다 판이 달라져 무엇을 배웠는지 못 가린다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_변종은_이름이_다르고_배치가_다르다():
    _, base = load_curriculum("curricula/stage3.json")
    _, vs = load_curriculum("curricula/stage3.json", variants=3)
    assert len(vs) == 3 * len(base)
    names = [b.name for b in vs]
    assert len(set(names)) == len(names), "이름이 겹치면 월드 캐시가 죽는다"
    assert all("@v" in n for n in names)
    # 같은 코스의 변종끼리 액터 위치가 실제로 다르다
    h = [b for b in vs if b.name.startswith("course_H@")]
    pos = [tuple(tuple(a.motion["pos"]) for a in b.scenario.actors) for b in h]
    assert len(set(pos)) == len(pos), "변종인데 배치가 같다 — jitter 가 안 먹었다"


def test_같은_변종은_항상_같은_배치다():
    """DAgger 라운드마다 판이 달라지면 무엇을 배웠는지 못 가린다."""
    _, a = load_curriculum("curricula/stage3.json", variants=3)
    _, b = load_curriculum("curricula/stage3.json", variants=3)
    for x, y in zip(a, b):
        assert x.name == y.name
        assert [i.motion["pos"] for i in x.scenario.actors] == \
               [i.motion["pos"] for i in y.scenario.actors]


def test_variants_1은_예전과_완전히_같다():
    """기존 호출부(stage1·stage2·기존 성적표)가 안 바뀐다."""
    _, old = load_curriculum("curricula/stage3.json")
    _, new = load_curriculum("curricula/stage3.json", variants=1)
    assert [b.name for b in old] == [b.name for b in new]
    for x, y in zip(old, new):
        assert [i.motion["pos"] for i in x.scenario.actors] == \
               [i.motion["pos"] for i in y.scenario.actors]


def test_흔들어도_액터_수와_종류는_그대로다():
    _, vs = load_curriculum("curricula/stage3.json", variants=4)
    for b in vs:
        assert len(b.scenario.actors) >= 3
        for a in b.scenario.actors:
            assert a.motion["kind"] == "static"
            assert "hd" in a.motion, "흔든 뒤에도 경로 방위를 다시 넣어야 한다"


def test_흔들기는_경로_밖으로_안_나간다():
    """`s` 가 음수거나 경로 길이를 넘으면 `route_point` 가 끝점으로 포화해 액터가 뭉친다.

    `rs.Actor` 는 `{id,type,size,spawn,motion,events}` 뿐이라 흔든 뒤의 호 길이를 실을 자리가
    없다(확인함). 그래서 **세계 좌표를 경로에 되투영해** 확인한다. 경로 길이는
    `board.route.total` 이다(`length` 가 아니다 — 확인함).
    """
    _, vs = load_curriculum("curricula/stage3.json", variants=6)
    for b in vs:
        for a in b.scenario.actors:
            x, y = a.motion["pos"]
            s = b.route.project(x, y)[0] if isinstance(b.route.project(x, y), tuple) \
                else b.route.project(x, y)
            assert 5.0 < s < b.route.total - 5.0, (b.name, a.id, s)
```

**구현자에게:** `route.project` 의 반환 모양을 **먼저 확인하고** 위 분기를 실제 모양에 맞게 정리해라(내가 확인하지 않았다). `route` 가 가진 것은 `cum, heading_at, point_at, project, pts, total` 이다.

- [ ] **Step 2: 실패 확인** → **Step 3: 구현** → **Step 4: 세 조각 rc=0**

- [ ] **Step 5: 돌연변이 넷** — ① 변종 이름을 안 바꾸기(→ 월드 캐시 assert) ② rng 를 시드 없이(→ 재현성 테스트) ③ jitter 를 0 으로(→ 배치가 같다) ④ 흔든 뒤 `hd` 재계산 빠뜨리기. **각각 실제로 심고 잡은 테스트 이름을 보고해라.**

- [ ] **Step 6: 커밋**

---

### Task 2: DAgger 가 단계 ③ 에서 수집하게 한다

**Files:** Modify `scripts/run_dagger.py`; Test `tests/policy/test_run_dagger.py`(있으면 거기, 없으면 신규)

**Interfaces:**
- `run_dagger.py --stage <이름>`(반복 가능, 기본값 `stage1` — **기존 동작 유지**) 과 `--variants N`.
- 평가 단계도 같이 받는다(지금 `stage1`·`stage2` 하드코딩).

**★ 반드시 막을 것 — `--smoke` 의 액터 소실.** `--smoke` 는 `slice_board` 로 H 0~250 조각을 쓰는데 **`slice_board` 가 액터를 버린다**(조사 2). 단계 ③ 에서 `--smoke` 를 쓰면 **액터 0 개인 판에서 아무것도 시험하지 않는다.** 둘 중 하나로 처리하고 **어느 쪽을 골랐는지 보고해라**:
- (a) 액터가 있는 커리큘럼 + `--smoke` 조합을 **거부**한다(`ValueError`), 또는
- (b) `slice_board` 가 구간 안에 드는 액터를 **살려서 옮긴다**.

(b) 가 더 쓸모 있지만 `slice_board` 의 좌표계를 건드려야 한다. **(a) 로 시작하고 (b) 가 필요하면 그때 하는 것을 권한다** — 다만 (a) 를 고르면 스모크로 단계 ③ 경로를 못 시험하므로, 대신 **작은 실제 판 하나로 도는 빠른 테스트**를 넣어라.

- [ ] **Step 1: 실패하는 테스트** — 기본이 `stage1` 그대로일 것 / `--stage stage3` 가 그 커리큘럼으로 풀릴 것 / 모르는 단계는 터질 것 / **액터 있는 단계 + `--smoke` 가 조용히 액터 0 개로 돌지 않을 것**
- [ ] **Step 2~4: 실패 확인 → 구현 → 세 조각 rc=0**
- [ ] **Step 5: 돌연변이 셋** — `--stage` 무시 / smoke 가드 삭제 / 평가 단계와 수집 단계 뒤바꾸기
- [ ] **Step 6: 커밋**

---

### Task 3: 본 수집·학습 — *계획 주관자가 OMEN 에서 돌린다*

- [ ] **Step 1: 개입이 무는지 먼저 확인** — 변종 4 개로 판을 만들어 **액터 위치가 실제로 흩어지는지** 출력으로 본다. M4e 에서 no-op 개입에 한 팔을 날린 적이 있다.
- [ ] **Step 2: 단계 ③ DAgger 수집·학습.** 변종 수·라운드 수·시드 수는 Step 1 을 보고 정한다.
- [ ] **Step 3: 평가** — 단계 ①②③ 전부, 결정적·확률적 둘 다(`scripts/eval_det_vs_stoch.py --stage ...`). **①② 가 무너지지 않았는지 반드시 확인**(치명적 망각).

---

### Task 4: 성적표 `docs/reports/m4k-dagger-stage3.md`

담을 것:
- **단계 ③ 완주율** — 출발점 0/18 대비. **이 마일스톤의 유일한 1 차 지표다.**
- **단계 ①② 가 안 무너졌는가** — 98.60 / 94.13 기준.
- **⑬ 지시등** — 선생님이 42 슬롯 틀린다. 학생이 **선생님을 따라 틀리는지**(모방의 한계 확인) 본다. **여기서 학생이 선생님만큼 틀리는 것이 정상이고, 그것이 PPO 의 표적이 살아 있다는 증거다.**
- **⚠ 단계 ③ 점수를 인용할 때 `mean_score_completed` 만 쓸 것.** raw 는 미방문 구간이 100 점이라 **일찍 죽을수록 높다**(M4i 에서 98.13 으로 단계 ② 보다 높게 나왔다).
- 이 성적표가 주장하지 않는 것.

## 이 계획이 미루는 것

- **PPO.** 학생이 단계 ③ 을 달릴 수 있게 된 뒤다.
- **단계 ④⑤.**
- `summarize_dvs.py:22` 의 `STAGES` 하드코딩(단계 ③ 을 조용히 버린다) — **다음 비교 표 전에 반드시.**
- `ObsConfig.obj_y` 의 ±20 m 포화, slow 조각 432 초, `report_m4a.py` 항목⑦ 중복, `--smoke` 가 `--steps` 덮어쓰기.
