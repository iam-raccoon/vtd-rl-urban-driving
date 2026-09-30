# M4h — 커리큘럼에 액터를 싣고 단계 ③(사물·정지차)을 만든다 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 커리큘럼 파일이 **액터를 기술할 수 있게** 하고, 그것으로 **단계 ③(사물·정지차) 판**을 만든 뒤, **선생님(규칙 스택)이 거기서 몇 점인지 먼저 잰다.**

**Architecture:** 지금 커리큘럼 JSON 은 액터를 못 싣는다(`"actors"` 키를 넣어도 **조용히 무시**된다). 그 구멍을 스키마 검증과 함께 메우고, `referee/scenarios.py` 가 이미 가진 경로-상대 좌표 원시함수(`_route_point`)를 재사용해 단계 ③ 판을 만든다. 그 다음 **학생을 붙이기 전에 선생님을 먼저 재는** 것이 이 계획의 핵심이다.

**Tech Stack:** Python 3.10, PyTorch 2.14(cpu), gymnasium 1.3, numpy **1.26.4 고정**, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`

**직전 증거:** `docs/reports/m4g-probe.md`, `docs/reports/m4f-ppo.md`

## 왜 이 마일스톤인가

**스펙의 M4 완료 증거는 "단계 ⑤ 진급"**(`design.md:233`)이고 커리큘럼은 **①빈 경로 ②신호·정지선 ③사물·정지차 ④보행자·교통 ⑤연습 코스 전체**(`design.md:37`)다. 그런데 **존재하는 판은 `curricula/stage1.json`·`stage2.json` 둘뿐이다.** M4a~M4g 를 전부 ①② 에서만 돌렸다.

그리고 ①② 는 **거의 포화됐다** — M3 학생 98.60 / 94.13, 선생님 99.6 / 99.5. 남은 여지가 +1.0 / +5.4 다. **PPO 가 학생을 못 넘은 것은 이길 것이 거의 없는 판에서 쟀기 때문일 수 있다.** RL 이 이길 만한 결정(추월·양보·교착 탈출)은 전부 ③④⑤ 에 있고 그 판이 없다.

**그러므로 이 마일스톤은 RL 을 건드리지 않는다.** 판을 만들고 **선생님을 재는 것**까지다.

## ★ 조사로 확인한 것 (2026-09-30, 전부 파일:줄 확인)

이 계획의 모든 설계 판단이 여기 근거한다. **구현자는 이 절을 먼저 읽어라.**

1. **커리큘럼 로더는 모르는 키를 조용히 버린다.** `load_curriculum`(`vtd_rl/world/board.py:78-81`)은 `name`·`signals`·`boards` 만 읽고, 판 항목은 `name`·`route`·`lane` 만 읽는다(`board.py:48-51`). **스키마 검증이 전혀 없다** — `"actors"` 를 넣으면 오류가 아니라 **무시**된다(실측 확인). `{"stage": 1}` 도 이미 안 읽힌다.
2. **액터 운동 모델은 읽기 전용 `third_party/rule_stack/eval/mock_vtd.py` 에 있다.** 클래스가 아니라 `kind` 문자열 분기다: `static`(`:50-53`), `lane`(`:54-56`), `constant`(`:57-59`), `crossing`(`:57-59`, 같은 분기), `waypoints`(`:60-62`). **`events` 는 운동 모델이 아니라** 어떤 kind 위에도 얹는 별도 리스트다(`:69-77`). 모르는 kind 는 **조용히** 정지 물체가 된다(`:63`).
3. **`Actor.type` 은 런타임에 버려진다.** `actors.py:30-31` 이 `rs.Obj` 로 감싸면서 type 을 안 넘긴다 — 실제 9910 패킷에 모델 종류가 없기 때문이다(`vtd_io.py:78-83`). **분류는 전부 바운딩박스 치수로** 한다. 관측(`env/observation.py:60-71`)과 심판(`judges/contact.py:29-31`)이 쓰는 문턱이 **일부러 다르다**(관측은 `PERSON_MIN_H=1.2`, 채점기는 `PED_MIN_H=0.85`) — `observation.py:61-66` 에 이유가 적혀 있다.
4. **항목 ⑩⑪⑭ 는 지금 구조적으로 발화 불가능하다.** ⑪⑭ 는 `r["clr"] < 0.0` 으로 게이트하는데(`judges/contact.py:22`), 물체가 없으면 `clr` 이 **항상 99.9**다(`run_logger.py:110`). ⑩ 은 `objs` 가 빈 문자열이라 `parse_objs` 가 `[]` 를 돌려준다. **단계 ③ 은 `RewardConfig.collision = -50.0`(`env/reward.py:17`)과 판 종료 경로(`drive_env.py:133,139,142-143`)를 처음으로 깨운다 — 정책이 한 번도 본 적 없는 신호다.**
5. **항목 ⑧ 은 물체가 없을 때 오히려 더 발화한다.** 물체는 면책을 만든다(`judges/traffic_light.py:93-95`). `env/tags.py:11-12` 의 독스트링이 이 마일스톤을 그대로 예고한다 — *"단계 ① 판은 교통이 없어 드러나지 않지만, 교통이 있는 M3 판에서는 매번 걸리고 DAgger 가 잘못 붙은 감점을 배운다."*
6. **추월 채점 항목은 없다.** `ITEMS`(`score_fma.py:122-128`)에 추월이 없다. 추월은 ④중앙선·⑥실선차로변경·⑬지시등으로만 간접 채점된다. **단계 ③ 성적표에 "추월을 잘한다" 는 칸을 만들지 마라.**
7. **재사용할 원시함수가 이미 있다** — `vtd_rl/referee/scenarios.py`. `_route_point(board, s, lateral)`(`:56-58`)이 **(호 길이, 횡 오프셋) → 세계 (x,y,heading)** 를 준다. 이것이 커리큘럼이 필요로 하는 바로 그 변환이다. `_static(board, aid, kind, s, size)`(`:61-64`)도 있다.
8. **⚠ `_static` 의 잠복 버그**: `motion["hd"]` 를 안 넣어 **방향이 0 으로 고정**된다(`mock_vtd.py:52`). `mock_vtd.py:46-50` 이 길게 경고하는 문제이고 **12.99 m 버스로 유령 −50 충돌이 났던 적이 있다.** 0.15 m 라바콘은 괜찮고 4.5 m 승용차는 아슬아슬하지만, **굽은 코스에 큰 차를 놓으면 진짜 버그다.** 이 계획 Task 1 에서 고친다.
9. **월드는 액터를 생성 시점에 한 번만 읽는다**(`world.py:50`)이고 `VtdDriveEnv._worlds` 는 **판 이름으로 캐시**한다(`drive_env.py:65,89-96`). 나중에 `scenario.actors` 를 바꿔도 안 먹고, 이름이 겹치면 `drive_env.py:96` 의 assert 에 걸린다. `vec_env._boards` 가 이름 뒤에 `#{stage}` 를 붙인다(`vec_env.py:35`).
10. **`slice_board` 는 액터를 통째로 버린다**(`board.py:71`: `actors=[], lights=[], zones=[]`).
11. **경로는 A·B·D·E·G·H 여섯 개가 전부다** — C·F 는 없다. 그리고 **단계 ①② 가 이미 전체 길이(2.1~5.2 km)를 달린다.** 따라서 **단계 ⑤ 는 길이가 아니라 구성(②③④의 합집합)으로 달라져야 한다.**
12. **선생님을 액터 있는 판에서 몰아 본 적이 한 번도 없다.** 규칙 스택의 물체 의존 로직(추월 FSM `overtake.py:30,394`, 종방향 양보 `behavior.py:712-715`, 측방 넛지 `drive.py:1779,1491,1612`)은 `s.objects` 가 비어 있어 **전부 휴면 상태**다(`drive.py:2402-2420`). 규칙 스택 자체 테스트는 40 개 있지만 **`vtd_rl` 의 월드를 통과한 적이 없다 — 진짜 미검증 이음매다.**
13. **빌려 쓸 액터 패턴 58 개가 이미 있다** — `third_party/rule_stack/eval/scenarios/` (30 개가 액터 보유: `static`×35, `constant`×13, `lane`×10, `crossing`×6, `waypoints`×2). **다만 직선 합성 도로**(`ego_start:[0,0,0]`, 200 m)라 **좌표는 못 옮기고 패턴만** 참고한다. `vtd_rl` 어디서도 참조하지 않는다.
14. **진급 로직은 없다.** 스펙이 기술하지만(`design.md:172-173`) 구현이 없다. **이 계획에서도 안 만든다**(단계 ③ 하나만 다루므로).

## Global Constraints

- 커밋·PR 에 **Claude 표기 금지**(`Co-Authored-By` / `Claude-Session` / "Generated with" / 🤖 / anthropic 주소). **커밋 뒤 `git show -s --format=%B HEAD` 를 직접 실행해 확인하고 보고할 것.**
- `third_party/rule_stack` **수정 금지**(읽기만). `runs/`·`*.pt`·`*.npz`·CSV·`*.xodr`·`*.xml` **커밋 금지**. 성적표 `.md` 만.
- numpy **1.26.4 고정**. 문서·주석·커밋 메시지는 **한국어**.
- **성적표 숫자는 전부 스크립트가 생성한다.**
- `vtd_rl/eval/verdict.py` 의 `M3_SCORE = {"stage1": 98.5, "stage2": 94.2}` 를 **낮추지 말 것**.
- **테스트를 돌리는 모든 Bash 호출에 `timeout: 600000`(ms)을 빠짐없이 붙일 것.** `run_in_background`·Monitor 금지.
- 겹치지 않는 세 조각, **rc** 로 판정(파이프를 태우면 `$?` 가 `tail` 것이 되니 파일로 받고 rc 를 따로 읽을 것):
  - `env -u PYTHONPATH .venv/bin/python -m pytest -q -m "not slow"`
  - `env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/ -q` ← **약 12 분**, 도구 상한을 넘어 배경으로 갈 수 있다. **알림을 기다렸다 파일에서 rc 를 읽을 것.**
  - `env -u PYTHONPATH .venv/bin/python -m pytest -q -m slow --ignore=tests/rl`
  - **기준선(main `0760eec`): 509 / 148 / 27, 전부 rc=0.**
- **커밋을 먼저 하고 그 다음에 돌연변이 실험.** 원복은 `git checkout -- <경로>` 로 **경로를 지정해서**(editable 설치라 `/tmp` 복사본은 진짜 패키지를 임포트한다).
- **`runs/` 안의 파일로 테스트를 게이트하지 말 것** — 머신마다 달라 `skipif` 가 조용히 항상 참이 된다.
- 긴 실행은 **OMEN**(`user-OMEN`, 32 코어, `--device cpu`)에서 계획 주관자가 직접 돌린다. **⚠ OMEN 의 git 체크아웃은 `288e12c`(M3 시절)로 낡았고 작업 트리는 rsync 로 맞춰져 있다. `git pull` 하지 말 것 — 작업 트리를 덮어쓴다. 바뀐 파일만 rsync 할 것.**
- **이 마일스톤은 RL 을 건드리지 않는다.** `vtd_rl/rl/`·`scripts/train_ppo.py` 를 수정하지 마라.

---

## File Structure

| 파일 | 책임 | 작업 |
|---|---|---|
| `vtd_rl/world/board.py` | 커리큘럼 스키마 검증 + `actors` 키 해석 | 1 |
| `vtd_rl/world/place.py` (신규) | 경로-상대 좌표로 액터를 놓는 순수 함수 | 1 |
| `vtd_rl/referee/scenarios.py` | `_static` 의 `hd` 버그 수정, `place.py` 재사용 | 1 |
| `curricula/stage3.json` (신규) | 단계 ③ 판 정의 | 2 |
| `scripts/measure_teacher.py` (신규) | 임의 단계에서 선생님 성적을 낸다 | 3 |
| `docs/reports/m4h-stage3-teacher.md` | 선생님 기준선 성적표 | 3 |

**Task 1 → 2 → 3 은 순차 의존이다**(2 는 1 의 스키마를, 3 은 2 의 판을 쓴다).

---

### Task 1: 커리큘럼이 액터를 싣게 하고, 모르는 키는 터뜨린다

**Files:**
- Create: `vtd_rl/world/place.py`
- Modify: `vtd_rl/world/board.py`, `vtd_rl/referee/scenarios.py`
- Test: `tests/test_board_schema.py`(신규), `tests/test_place.py`(신규), `tests/referee/test_scenarios.py`(기존 통과 유지)

**Interfaces:**
- Produces:
  - `vtd_rl/world/place.py::route_point(board, s, lateral=0.0) -> (x, y, heading)` — `referee/scenarios.py:56-58` 의 `_route_point` 를 **그대로 옮긴 것**(동작 동일).
  - `vtd_rl/world/place.py::actor_from_spec(board, spec: dict) -> rs.Actor` — 커리큘럼의 액터 기술을 `rs.Actor` 로 바꾼다. **경로-상대 좌표(`s`, `lateral`)를 세계 좌표로 변환하고, `static` 이면 `motion["hd"]` 에 그 지점의 경로 방향을 넣는다.**
  - `vtd_rl/world/board.py::load_curriculum` — 기존 시그니처 유지. **모르는 키가 있으면 `ValueError`.**
- Consumes: `rs.Actor`(`third_party/rule_stack/src/scenario.py:22-31`), `board.route`

**왜 검증을 넣나:** 지금은 `"actors"` 를 넣어도 **조용히 무시**된다. 이 계획이 하려는 일이 정확히 "커리큘럼에 액터를 싣는 것"이라, **오타 하나가 액터 0 개인 판을 만들고 그게 단계 ①② 와 구별이 안 된다.** 실험 한 팔이 통째로 무의미해지는 M4e 의 재판이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_place.py`:
```python
import math

import pytest

from vtd_rl.world.board import load_board
from vtd_rl.world.place import actor_from_spec, route_point

BOARD = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json",
         "lane": "routes/HL_FMA_NEW_H_lane.json"}


def test_route_point의_횡_오프셋_공식():
    """`referee/scenarios.py:56-58` 의 공식 그대로여야 한다.

    ⚠ `sc._route_point` 와 대조하지 마라 — Task 1 이 그것을 `place.route_point` 로
    위임시키므로 **동어반복이 된다.** 공식을 직접 확인한다: 왼쪽(+lateral)이
    `(-sin h, +cos h)` 방향이고, 이동 거리가 |lateral| 이어야 한다.
    """
    b = load_board(BOARD, "always_green")
    for s in (0.0, 50.0, 150.0, 400.0):
        x0, y0, h = route_point(b, s, 0.0)
        assert (x0, y0, h) == b.route.point_at(s)        # lateral 0 이면 경로 점 그대로
        for lat in (-3.0, 2.5):
            x, y, h2 = route_point(b, s, lat)
            assert h2 == h
            assert x == pytest.approx(x0 - lat * math.sin(h))
            assert y == pytest.approx(y0 + lat * math.cos(h))
            assert math.hypot(x - x0, y - y0) == pytest.approx(abs(lat))


def test_static_액터는_그_지점의_경로_방향을_받는다():
    """★ `mock_vtd.py:46-50` 이 경고하는 버그 — hd 가 0 으로 고정되면 굽은 코스에서
    큰 차의 바운딩박스가 도로를 가로질러 유령 충돌(-50)을 만든다."""
    b = load_board(BOARD, "always_green")
    a = actor_from_spec(b, {"id": 1, "kind": "static", "type": "vehicle",
                            "s": 400.0, "lateral": 0.0, "size": [4.5, 1.8, 1.5]})
    _, _, h = route_point(b, 400.0, 0.0)
    assert a.motion["hd"] == h
    assert abs(h) > 1e-6, "굽은 지점을 골라야 이 테스트가 의미 있다 — s 를 바꿔라"


def test_액터_기술은_경로_상대_좌표를_세계_좌표로_바꾼다():
    b = load_board(BOARD, "always_green")
    a = actor_from_spec(b, {"id": 2, "kind": "static", "type": "obstacle",
                            "s": 150.0, "lateral": -1.5, "size": [0.15, 0.46, 0.61]})
    x, y, _ = route_point(b, 150.0, -1.5)
    assert a.motion["pos"] == [x, y]
    assert a.size == [0.15, 0.46, 0.61]


def test_모르는_운동_kind는_터진다():
    """`mock_vtd.py:63` 은 모르는 kind 를 **조용히** 정지 물체로 만든다. 여기서 막는다."""
    import pytest
    b = load_board(BOARD, "always_green")
    with pytest.raises(ValueError, match="운동"):
        actor_from_spec(b, {"id": 3, "kind": "teleport", "s": 10.0, "size": [1, 1, 1]})
```

`tests/test_board_schema.py`:
```python
import json

import pytest

from vtd_rl.world.board import load_curriculum

BASE = {"name": "t", "signals": "always_green",
        "boards": [{"name": "course_H", "route": "routes/HL_FMA_NEW_H.json",
                    "lane": "routes/HL_FMA_NEW_H_lane.json"}]}


def _write(tmp_path, d):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_모르는_최상위_키는_터진다(tmp_path):
    """★ 지금은 조용히 무시된다. 오타 하나가 액터 0 개인 판을 만들고 단계 ①② 와
    구별이 안 된다 — 실험 한 팔이 통째로 무의미해진다."""
    with pytest.raises(ValueError, match="actorz"):
        load_curriculum(_write(tmp_path, {**BASE, "actorz": []}))


def test_모르는_판_항목_키는_터진다(tmp_path):
    bad = {**BASE, "boards": [{**BASE["boards"][0], "aktors": []}]}
    with pytest.raises(ValueError, match="aktors"):
        load_curriculum(_write(tmp_path, bad))


def test_기존_stage1_stage2는_그대로_읽힌다():
    for name in ("stage1", "stage2"):
        label, boards = load_curriculum(f"curricula/{name}.json")
        assert len(boards) == 6
        assert all(not b.scenario.actors for b in boards)


def test_stage키는_허용하되_안_읽는다(tmp_path):
    """기존 파일이 `"stage": 1` 을 갖고 있다 — 허용 목록에 넣되 의미는 없다."""
    label, boards = load_curriculum(_write(tmp_path, {**BASE, "stage": 3}))
    assert len(boards) == 1


def test_판에_액터를_실으면_scenario에_들어간다(tmp_path):
    d = {**BASE, "boards": [{**BASE["boards"][0],
         "actors": [{"id": 1, "kind": "static", "type": "obstacle",
                     "s": 150.0, "lateral": 0.0, "size": [0.15, 0.46, 0.61]}]}]}
    _, boards = load_curriculum(_write(tmp_path, d))
    assert len(boards[0].scenario.actors) == 1
    assert boards[0].scenario.actors[0].id == 1
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_place.py tests/test_board_schema.py -q` (timeout 600000)
Expected: FAIL — `ModuleNotFoundError: vtd_rl.world.place`, 그리고 스키마 테스트는 예외가 안 나서 실패

- [ ] **Step 3: 구현**

`vtd_rl/world/place.py` 를 새로 만든다:
- `route_point(board, s, lateral=0.0)` — `referee/scenarios.py:56-58` 의 본문을 **그대로** 옮긴다. 동작을 바꾸지 마라(위 테스트가 두 구현을 대조한다).
- `actor_from_spec(board, spec)` — 받는 필드: `id`(필수), `kind`(필수), `type`(기본 `"vehicle"`), `size`(필수, 길이 3), `s`(필수), `lateral`(기본 0.0), `speed`, `spawn`, `events`, 그리고 `waypoints` 용 `pts`(경로-상대 `[[s, lateral], ...]`). **허용하는 `kind` 는 `mock_vtd.py` 가 실제로 구현한 것만**: `static`·`lane`·`constant`·`crossing`·`waypoints`. 그 외는 `ValueError("... 운동 kind ...")`.
  - `static`: `motion = {"kind": "static", "pos": [x, y], "hd": h}` — **`hd` 를 반드시 넣는다**(위 버그).
  - `crossing`/`constant`: `motion = {"kind": ..., "pos": [x, y], "vel": [...]}`. 횡단 속도는 `referee/scenarios.py:122-127` 의 패턴을 따른다 — 경로 방향 `h` 에 대해 `vel = [-v*sin(h), v*cos(h)]`.
  - `spawn` 은 그대로 넘긴다(`at_time`/`ego_x`/`ego_within`+`of`, `mock_vtd.py:108-117`). 안 주면 `{"at_time": 0.0}`.
  - `events` 는 그대로 넘긴다(`mock_vtd.py:69-77`).

`vtd_rl/world/board.py`:
- 최상위 허용 키 `{"name", "signals", "boards", "stage"}`, 판 항목 허용 키 `{"name", "route", "lane", "actors"}` 를 **모듈 상수로** 두고, 벗어나면 `ValueError` 에 **그 키 이름을 담아** 던진다.
- `load_board` 가 `actors` 를 받으면 `place.actor_from_spec` 으로 만들어 `scenario.actors` 에 넣는다. **순환 임포트에 주의** — `place.py` 가 `board.py` 를 임포트하지 않도록(board 객체를 인자로만 받는다).

`vtd_rl/referee/scenarios.py`:
- `_route_point` 를 `place.route_point` 로 위임하고, `_static` 이 `hd` 를 넣게 고친다. **기존 `tests/referee/test_scenarios.py` 가 그대로 통과해야 한다** — 통과하지 않으면 `hd` 수정이 판정을 바꿨다는 뜻이니 **멈추고 보고해라**(라바콘·승용차는 안 바뀌어야 정상이다).

- [ ] **Step 4: 통과 확인** — 세 조각 전부 rc=0, 기준선 509/148/27 에서 새 테스트만큼 늘어난다.

- [ ] **Step 5: 돌연변이로 확인**

커밋한 **뒤** 넷을 심어 각각 잡히는지 이름으로 보고해라:
1. 최상위 키 검증 삭제 → `test_모르는_최상위_키는_터진다`
2. `_static`/`actor_from_spec` 에서 `hd` 를 다시 빼기 → `test_static_액터는_그_지점의_경로_방향을_받는다`
3. `lateral` 부호 뒤집기 → `test_액터_기술은_경로_상대_좌표를_세계_좌표로_바꾼다`
4. 모르는 kind 를 `ValueError` 대신 `static` 으로 조용히 떨구기 → `test_모르는_운동_kind는_터진다`

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/world/place.py vtd_rl/world/board.py vtd_rl/referee/scenarios.py \
        tests/test_place.py tests/test_board_schema.py
git commit -m "M4h — 커리큘럼이 액터를 싣게 하고 모르는 키는 터뜨린다"
git show -s --format=%B HEAD
```

---

### Task 2: 단계 ③ 판을 만든다

**Files:**
- Create: `curricula/stage3.json`
- Test: `tests/test_stage3.py`(신규)

**Interfaces:**
- Consumes: Task 1 의 `actors` 스키마
- Produces: `curricula/stage3.json` — 여섯 코스(A·B·D·E·G·H) 각각에 **정지차·장애물**을 배치. 신호는 `cycle`(단계 ② 를 포함한다).

**설계 지침:**
- **단계 ③ 은 "사물·정지차" 다**(`design.md:37`). 움직이는 교통·보행자는 단계 ④ 다. 그러므로 여기서는 **`static` 만** 쓴다(그리고 필요하면 `constant` 로 아주 느린 정체 차).
- 코스마다 **3~5 개**를 놓는다. 너무 많으면 완주 자체가 불가능해 신호가 안 나온다.
- 배치는 `third_party/rule_stack/eval/scenarios/` 의 **패턴**을 참고하되 **좌표는 옮기지 마라**(그 파일들은 200 m 직선 합성 도로다).
- **라바콘**(`size: [0.15, 0.46, 0.61]`)과 **정지 승용차**(`[4.5, 1.8, 1.5]`)를 섞는다. 관측의 분류 문턱(`observation.py:60-71`)상 라바콘은 `obstacle`, 승용차는 `vehicle` 로 잡힌다 — **둘 다 있어야 분류 채널이 학습에 쓰인다.**
- **차로 한가운데에만 놓지 마라.** `lateral` 을 섞어 (a) 비켜 갈 수 있는 것과 (b) 차로를 막아 차로변경이 필요한 것을 둘 다 만든다.
- **곡선 구간에 큰 차를 놓을 때 Task 1 의 `hd` 수정이 실제로 먹는지 확인해라**(안 먹으면 유령 −50 이 난다).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stage3.py`:
```python
from vtd_rl.world.board import load_curriculum


def test_단계3이_여섯_코스를_모두_덮는다():
    _, boards = load_curriculum("curricula/stage3.json")
    assert sorted(b.name for b in boards) == [
        "course_A", "course_B", "course_D", "course_E", "course_G", "course_H"]


def test_단계3의_모든_판에_액터가_있다():
    """액터 0 개인 판이 하나라도 있으면 그 판은 단계 ② 와 같다 — 조용한 오염."""
    _, boards = load_curriculum("curricula/stage3.json")
    for b in boards:
        assert len(b.scenario.actors) >= 3, b.name


def test_단계3은_신호_주기를_쓴다():
    """단계 ③ 은 ② 를 포함한다 — 신호를 끄면 뒷걸음질이다."""
    _, boards = load_curriculum("curricula/stage3.json")
    assert all(b.signals == "cycle" for b in boards)


def test_단계3에_장애물과_차량이_둘_다_있다():
    """관측의 분류 채널(observation.py:60-71)이 둘 다 쓰이게 한다."""
    _, boards = load_curriculum("curricula/stage3.json")
    for b in boards:
        lens = [a.size[0] for a in b.scenario.actors]
        assert any(l <= 2.2 for l in lens), f"{b.name}: 장애물 없음"
        assert any(l > 2.2 for l in lens), f"{b.name}: 차량 없음"


def test_단계3은_정지물만_쓴다():
    """움직이는 교통·보행자는 단계 ④ 다."""
    _, boards = load_curriculum("curricula/stage3.json")
    for b in boards:
        for a in b.scenario.actors:
            assert a.motion["kind"] in ("static", "constant"), (b.name, a.motion)
```

- [ ] **Step 2: 실패 확인** — `FileNotFoundError: curricula/stage3.json`

- [ ] **Step 3: `curricula/stage3.json` 작성** — 위 지침대로.

- [ ] **Step 4: 통과 확인** — 세 조각 전부 rc=0.

- [ ] **Step 5: ★ 판이 실제로 달릴 수 있는지 눈으로 확인**

**테스트만으로는 부족하다** — 액터가 경로를 완전히 막으면 테스트는 통과하고 판은 못 달린다. 선생님을 여섯 판에 태워 **완주하는지** 보고, 못 하면 배치를 고쳐라. 이건 Task 3 의 축소판이니 **`scripts/measure_teacher.py` 를 먼저 만들어 써도 된다**(그러면 Task 3 Step 1 을 여기서 하는 셈).

**선생님이 한 판이라도 못 완주하면 배치가 틀린 것이다 — 규칙 스택이 못 푸는 판을 학생에게 주지 마라.** 단, **선생님이 감점을 받는 것은 정상이고 오히려 목표다**(그게 RL 이 이길 여지다).

- [ ] **Step 6: 커밋**

```bash
git add curricula/stage3.json tests/test_stage3.py
git commit -m "M4h — 단계 ③ 판(사물·정지차) 여섯 코스"
git show -s --format=%B HEAD
```

---

### Task 3: 선생님을 단계 ③ 에서 잰다 — 이 마일스톤의 결론

**Files:**
- Create: `scripts/measure_teacher.py`, `docs/reports/m4h-stage3-teacher.md`
- Test: `tests/test_measure_teacher.py`(신규)

**Interfaces:**
- Produces: `scripts/measure_teacher.py --stage <이름> --seeds N --out <md>` — 임의 커리큘럼에서 **선생님의 완주율·점수·항목별 감점**을 낸다. `scripts/run_stage1_teacher.py` 가 단계 ① 전용으로 하는 일의 일반화다(**그 스크립트를 먼저 읽고 재사용할 것**).
- Consumes: `vtd_rl/env/teacher_policy.py`, `vtd_rl/eval/`

**왜 이게 결론인가:** 선생님이 단계 ③ 에서 몇 점인지 **모른 채로** RL 을 얹으면 M4a~M4g 를 반복한다 — **이길 여지가 있는지도 모르고 최적화하는 것.** ①② 에서 선생님은 99.6/99.5 였고 학생이 98.6/94.1 이라 여지가 거의 없었다. **③ 에서 선생님이 몇 점인지가 이 프로젝트의 다음 방향을 정한다.**

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_선생님_측정이_임의_단계를_받는다(tmp_path):
    """단계 ① 전용이던 것을 일반화한 것이므로 stage1 에서 기존 값과 맞아야 한다."""
    ...  # 구현자가 run_stage1_teacher.py 를 읽고 그 산출과 대조하는 테스트를 쓴다


@pytest.mark.slow
def test_선생님이_단계3_판을_완주한다():
    """★ 규칙 스택이 못 푸는 판을 학생에게 주지 않는다는 보증."""
    ...  # 시드 1 개, 여섯 판. 완주율 100% 를 단언한다.
```

**구현자에게:** 위 두 테스트의 본문은 `scripts/run_stage1_teacher.py` 와 `vtd_rl/eval/` 을 읽고 **네가 채워라** — 내가 그 API 를 확인하지 않고 적으면 틀린다. 채운 내용을 보고에 적어라.

- [ ] **Step 2~4: 실패 확인 → 구현 → 통과 확인** (세 조각 rc=0)

- [ ] **Step 5: 커밋**

- [ ] **Step 6: 본 측정과 성적표** — *계획 주관자가 OMEN 에서 직접 돌린다.*

`docs/reports/m4h-stage3-teacher.md` 에 담을 것:
- **단계 ①②③ 에서 선생님의 완주율·점수**를 나란히. ①② 는 이미 99.6/99.5 다.
- **항목별 감점** — 어느 항목이 ③ 에서 새로 열리는가. **⑪⑭(접촉)와 ⑧(녹색 무의미 정차)의 면책**이 처음으로 살아난다.
- **선생님이 감점을 받는 지점이 곧 RL 의 표적이다.** 그 목록을 명시적으로 적어라.
- **M3 학생을 같은 판에 태운 결과**(재학습 없이). 학생이 액터를 한 번도 본 적 없으므로 무너질 것이다 — **얼마나** 무너지는지가 다음 마일스톤(DAgger 재수집)의 크기를 정한다.
- **이 성적표가 주장하지 않는 것.**

---

## 이 계획이 미루는 것

- **단계 ④(보행자·교통)·⑤(②③④ 합집합)** — 각각 다음 마일스톤. ⑤ 는 길이가 아니라 **구성**으로 달라진다(경로는 이미 전체 길이다).
- **진급 로직** — 스펙(`design.md:172-173`)이 기술하지만 구현이 없다. 단계가 셋 다 생긴 뒤에 만든다.
- **DAgger 재수집·PPO** — 선생님 기준선을 본 뒤에 결정한다.
- **`Actor.type` 이 버려지는 것**(`actors.py:30-31`) — 의도된 설계(9910 패킷에 종류가 없다)라 고치지 않는다. 다만 성적표에 적어 둔다.
- 앞 마일스톤들이 넘긴 것: `--smoke` 가 `--steps` 를 덮어쓰는 함정, `report_m4a.py` 항목⑦ 중복, `_goal_lines` 문자열 매칭, `OutcomeCounter` 의 0 인 사유, `ReturnTracker` +1 편향, `RewardConfig.rule_scale`, 승차감 콜드스타트, `diag_stall.py` 돌연변이 ⑥, `tests/rl/` 이 도구 상한을 넘는 것, **OMEN 레포 체크아웃이 낡은 것**.
