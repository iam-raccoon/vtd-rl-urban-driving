# M1 — 오프라인 세계(world)와 선생님 그림자 주행 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 규칙 스택을 선생님으로 태운 오프라인 세계를 만들어, 선생님이 커리큘럼 단계 ① 판(연습 코스 6개, 신호 항상 녹색·액터 없음)을 완주하고 스텝 속도를 잰다.

**Architecture:** 비공개 규칙 스택 레포를 `third_party/rule_stack` 서브모듈로 고정하고 `vtd_rl/rule_stack.py` 한 곳에서 import 한다. `vtd_rl/world` 는 경로 인덱스·판·자차 동역학·신호·액터를 따로 두고 `World` 가 이들을 묶어 20 Hz 로 9910 과 같은 `State` 를 낸다. `vtd_rl/teacher/shadow.py` 가 규칙 스택 `DrivingStack` 을 시뮬 시계로 돌리고, `vtd_rl/rollout.py` 가 한 판을 끝까지 달린다.

**Tech Stack:** Python 3.10 표준 라이브러리, pytest ≥ 8. (numpy·Gymnasium·PyTorch 는 M2 이후)

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`

## Global Constraints

- 커밋 메시지·PR 본문에 Claude 표기(`Co-Authored-By`, `Claude-Session`, "Generated with") **넣지 않는다**.
- 규칙 스택 서브모듈(`third_party/rule_stack`) 코드는 **수정하지 않는다** — 읽고 import 만 한다.
- 공개 레포다. 주최측 자료(`*.xodr`, VTD 시나리오 XML, 강의 받아쓰기·주최측 답변)와 주행 CSV·체크포인트·`runs/` 는 커밋하지 않는다.
- Python 3.10. numpy 를 쓰게 되면 `numpy==1.26.4` 고정(2.x 금지 — 같은 머신의 ROS2 스택과 충돌).
- 시뮬 주기 20 Hz(`dt = 0.05`).
- 테스트 통과 판정은 pytest **종료 코드 0** 과 실패 목록으로 한다("N passed" 문자열로 판정하지 않는다).
- 규칙 스택 고정 커밋: `4a3f7037ef9ed5d7fae15107f209a86e5d550b00`(HL-FMA2026_Simulation main, 대회 코드 src 와 동일).

## 스펙과 다른 점(M1 범위)

- **지도(xodr) 캐시는 M2 로 미룬다.** M1 이 쓰는 신호 정지선(`tl_map_livinglab.json`)·도색 정지선(`stoplines_all.json`)·차로계획(`*_lane.json`)은 규칙 스택이 이미 JSON 으로 만들어 두었다. xodr 파싱은 M2 온라인 심판(차선 종류 판정)에서 처음 필요하다.
- **자동 생성 판(무작위 경로 라이브러리)은 M4 커리큘럼에서 만든다.** M1 단계 ① 판은 연습 코스 6개다.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `pyproject.toml` · `requirements-dev.txt` · `.gitignore` · `README.md` | 패키지·테스트 설정, 공개 레포 안내 |
| `third_party/rule_stack/` | 규칙 스택 서브모듈(고정 커밋) |
| `vtd_rl/rule_stack.py` | 서브모듈 import 경로 설정, 규칙 스택 심볼 재노출, 지도 DB(JSON) 로드 |
| `vtd_rl/world/route.py` | 경로 폴리라인: 누적거리·투영(s, 횡오프셋)·방위 |
| `vtd_rl/world/board.py` | 판 = 규칙 스택 Scenario + 차로계획 + 신호 운용. 불러오기·자르기 |
| `vtd_rl/world/dynamics.py` | 자차 자전거 모델 + 가속 1차 지연 + 조향 속도 제한 |
| `vtd_rl/world/signals.py` | 경로 위 신호 찾기, 신호 주기, VTD 식 신호 알림 |
| `vtd_rl/world/actors.py` | 규칙 스택 오프라인 액터 모델로 물체 목록 만들기 |
| `vtd_rl/world/world.py` | 위를 묶어 reset/step, 종료 판정 |
| `vtd_rl/teacher/shadow.py` | 규칙 스택 DrivingStack 그림자 주행(속도 추정 포함) |
| `vtd_rl/rollout.py` | 선생님으로 한 판 끝까지 달리기 + CSV |
| `curricula/stage1.json` | 단계 ① 판 목록 |
| `scripts/bench_world.py` · `scripts/run_stage1_teacher.py` | 스텝 속도 측정, 단계 ① 성적표 |
| `docs/reports/m1-stage1-teacher.md` | M1 완료 증거(성적표) |

---

### Task 1: 레포 뼈대 · 서브모듈 · 규칙 스택 연결

**Files:**
- Create: `pyproject.toml`, `requirements-dev.txt`, `.gitignore`, `README.md`, `vtd_rl/__init__.py`, `vtd_rl/rule_stack.py`, `tests/test_rule_stack.py`
- Submodule: `third_party/rule_stack` → `https://github.com/iam-raccoon/HL-FMA2026_Simulation.git` @ `4a3f7037ef9ed5d7fae15107f209a86e5d550b00`

**Interfaces:**
- Produces:
  - `vtd_rl.rule_stack.ROOT: str` — 서브모듈 절대경로
  - `vtd_rl.rule_stack.path(*parts) -> str` — 서브모듈 안 경로
  - `vtd_rl.rule_stack.commit() -> str` — 서브모듈 커밋 40자
  - `vtd_rl.rule_stack.map_db() -> dict` — `{"tl_map": {int: [x, y]}, "crosswalks": list, "stoplines": list, "stoplines_all": list[[x, y, hdg]]}` (한 번만 읽음)
  - 재노출: `State, Obj, TL_UNSET, TL_RED, TL_YELLOW, TL_GREEN, TS_OFF, TS_LEFT, TS_RIGHT, VTDLink, HEADER`(CSV 헤더), `Scenario, Actor, DrivingStack, Command, RunLogger, mock_vtd`

- [ ] **Step 1: 저장소 파일 만들기**

`pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vtd-rl-urban-driving"
version = "0.1.0"
description = "VTD LivingLab 도심 주행 end-to-end 강화학습"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
include = ["vtd_rl*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: 코스 한 판 이상을 끝까지 달리는 느린 테스트"]
```

`requirements-dev.txt`
```
pytest>=8
```

`.gitignore`
```
.venv/
__pycache__/
*.pyc
*.egg-info/
runs/
*.csv
*.pkl
*.mp4
# 주최측 자료 — 공개 레포에 올리지 않는다
*.xodr
```

`README.md`
```markdown
# vtd-rl-urban-driving

Hexagon VTD 2025.2 의 도심 지도(LivingLab)에서 도로교통법을 지키며 달리는 **end-to-end 강화학습 운전 정책**을 만든다.
2026 HL-FMA 대회에서 완주한 규칙 기반 주행 스택을 선생님으로 두고, 오프라인 세계에서 모방학습(DAgger) → PPO 로 학습한다.

- 설계: [docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md](docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md)
- 규칙 스택 공개 참고본: [HL-FMA2026-VTD](https://github.com/iam-raccoon/HL-FMA2026-VTD)

## 주의
`third_party/rule_stack` 서브모듈은 **비공개 레포**라 외부에서는 클론만으로 실행되지 않는다.
주최측 자료(지도 xodr, VTD 시나리오, 교육 자료)는 이 레포에 넣지 않는다.

## 설치
    git clone --recurse-submodules https://github.com/iam-raccoon/vtd-rl-urban-driving.git
    cd vtd-rl-urban-driving
    python3 -m venv .venv
    .venv/bin/pip install -e . -r requirements-dev.txt

## 테스트
    .venv/bin/pytest
```

`vtd_rl/__init__.py`
```python
"""VTD LivingLab 도심 주행 강화학습."""
```

- [ ] **Step 2: 서브모듈 추가·고정**

Run:
```bash
cd /home/user/vtd-rl-urban-driving
git submodule add https://github.com/iam-raccoon/HL-FMA2026_Simulation.git third_party/rule_stack
git -C third_party/rule_stack checkout 4a3f7037ef9ed5d7fae15107f209a86e5d550b00
git -C third_party/rule_stack log --oneline -1
```
Expected: `4a3f703 Merge pull request #60 ...`

- [ ] **Step 3: venv 만들기**

Run:
```bash
python3 -m venv .venv && .venv/bin/pip install -q -e . -r requirements-dev.txt && .venv/bin/pytest --version
```
Expected: `pytest 8.x`

- [ ] **Step 4: 실패하는 테스트 작성**

`tests/test_rule_stack.py`
```python
import os
import re

from vtd_rl import rule_stack as rs


def test_서브모듈_경로():
    assert os.path.isfile(rs.path("src", "drive.py"))
    assert os.path.isfile(rs.path("routes", "HL_FMA_NEW_H.json"))


def test_서브모듈_커밋은_고정값():
    assert re.fullmatch(r"[0-9a-f]{40}", rs.commit())
    assert rs.commit() == "4a3f7037ef9ed5d7fae15107f209a86e5d550b00"


def test_규칙_스택_심볼():
    s = rs.State(x=1.0, y=2.0, heading=0.5)
    assert s.objects == [] and s.tl_state == rs.TL_UNSET
    assert rs.Command().turn == rs.TS_OFF
    assert rs.HEADER.startswith("t,x,y,heading,v,")
    assert callable(rs.mock_vtd.step_actor)


def test_지도_DB():
    db = rs.map_db()
    assert len(db["tl_map"]) == 214
    assert db["tl_map"][151] and len(db["tl_map"][151]) == 2
    assert len(db["stoplines_all"]) == 710
    assert len(db["stoplines_all"][0]) == 3
    assert isinstance(db["crosswalks"], list) and isinstance(db["stoplines"], list)
    assert rs.map_db() is db          # 두 번째는 캐시
```

- [ ] **Step 5: 테스트가 실패하는지 확인**

Run: `.venv/bin/pytest tests/test_rule_stack.py -q`
Expected: FAIL — `ImportError: cannot import name 'rule_stack'`

- [ ] **Step 6: 구현**

`vtd_rl/rule_stack.py`
```python
"""규칙 스택(서브모듈) import 를 한곳에 모은다.

규칙 스택은 패키지가 아니라 `src/`·`eval/`·`vtd/` 폴더에 모듈이 흩어져 있고 서로를
최상위 이름(`from drive import ...`)으로 부른다. 그래서 세 폴더를 sys.path 앞에 넣는다.
세 폴더 사이에 같은 모듈 이름은 없다(2026-09-15 확인).
"""
import functools
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "rule_stack"))
_DIRS = [os.path.join(ROOT, d) for d in ("src", "eval", "vtd")]
sys.path[:0] = [d for d in _DIRS if d not in sys.path]

from vtd_io import (State, Obj, VTDLink, TL_UNSET, TL_RED, TL_YELLOW, TL_GREEN,  # noqa: E402
                    TS_OFF, TS_LEFT, TS_RIGHT)
from scenario import Scenario, Actor  # noqa: E402
from drive import DrivingStack, Command  # noqa: E402
from run_logger import RunLogger, HEADER  # noqa: E402
import mock_vtd  # noqa: E402

__all__ = ["ROOT", "path", "commit", "map_db", "State", "Obj", "VTDLink", "TL_UNSET", "TL_RED",
           "TL_YELLOW", "TL_GREEN", "TS_OFF", "TS_LEFT", "TS_RIGHT", "Scenario", "Actor",
           "DrivingStack", "Command", "RunLogger", "HEADER", "mock_vtd"]


def path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def commit() -> str:
    return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


@functools.lru_cache(maxsize=1)
def map_db() -> dict:
    def load(name):
        with open(path("routes", name), encoding="utf-8") as f:
            return json.load(f)
    return {
        "tl_map": {int(k): v for k, v in load("tl_map_livinglab.json").items()},
        "crosswalks": load("crosswalks.json")["crosswalks"],
        "stoplines": load("stoplines.json")["stoplines"],
        "stoplines_all": load("stoplines_all.json")["stoplines"],
    }
```

- [ ] **Step 7: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_rule_stack.py -q; echo rc=$?`
Expected: `4 passed`, `rc=0`

- [ ] **Step 8: 커밋**

```bash
git add .gitmodules third_party/rule_stack pyproject.toml requirements-dev.txt .gitignore README.md vtd_rl/__init__.py vtd_rl/rule_stack.py tests/test_rule_stack.py
git commit -m "레포 뼈대 — 규칙 스택 서브모듈(4a3f703) 고정과 import 한곳 모으기"
```

---

### Task 2: 경로 인덱스

**Files:**
- Create: `vtd_rl/world/__init__.py`, `vtd_rl/world/route.py`, `tests/test_route.py`

**Interfaces:**
- Produces:
  - `Projection(s: float, lateral: float, index: int)` — frozen dataclass. `lateral` 은 경로 진행 방향 **왼쪽이 +**
  - `RouteIndex(pts: list[tuple[float, float]])` — 속성 `pts`, `cum: list[float]`, `total: float`
  - `RouteIndex.project(x, y, hint: int | None = None, window: int = 60) -> Projection`
  - `RouteIndex.heading_at(i: int) -> float` — 앞뒤 3점으로 잰 방위[rad]
  - `RouteIndex.point_at(s: float) -> tuple[float, float, float]` — (x, y, 세그먼트 방위)

- [ ] **Step 1: 실패하는 테스트 작성**

`vtd_rl/world/__init__.py`
```python
"""오프라인 세계."""
```

`tests/test_route.py`
```python
import math

import pytest

from vtd_rl.world.route import RouteIndex


def straight():
    return RouteIndex([(float(i), 0.0) for i in range(21)])


def test_누적거리():
    r = straight()
    assert r.total == pytest.approx(20.0)
    assert r.cum[5] == pytest.approx(5.0)


def test_투영_왼쪽이_양수():
    r = straight()
    p = r.project(5.3, 2.0)
    assert p.s == pytest.approx(5.3)
    assert p.lateral == pytest.approx(2.0)
    q = r.project(15.0, -1.0)
    assert q.lateral == pytest.approx(-1.0)


def test_투영_힌트_창():
    r = straight()
    assert r.project(12.2, 0.5, hint=12).s == pytest.approx(12.2)


def test_경로_끝을_넘으면_끝에_붙는다():
    r = straight()
    p = r.project(25.0, 0.0)
    assert p.s == pytest.approx(20.0) and p.index == 20


def test_지점과_방위():
    r = RouteIndex([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    x, y, h = r.point_at(15.0)
    assert (x, y) == pytest.approx((10.0, 5.0))
    assert h == pytest.approx(math.pi / 2)
    assert r.heading_at(0) == pytest.approx(math.atan2(10.0, 10.0))


def test_점이_하나면_거부():
    with pytest.raises(ValueError):
        RouteIndex([(0.0, 0.0)])
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/test_route.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.world.route'`

- [ ] **Step 3: 구현**

`vtd_rl/world/route.py`
```python
"""경로 폴리라인 — 누적거리·투영·방위."""
import bisect
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Projection:
    s: float          # 경로 누적거리[m]
    lateral: float    # 경로 진행 방향 왼쪽 + [m]
    index: int        # 가장 가까운 경로점 번호


class RouteIndex:
    def __init__(self, pts):
        if len(pts) < 2:
            raise ValueError("경로점이 2개 이상이어야 한다")
        self.pts = [(float(x), float(y)) for x, y in pts]
        self.cum = [0.0]
        for (x0, y0), (x1, y1) in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.total = self.cum[-1]

    def heading_at(self, i: int) -> float:
        a = self.pts[max(0, i - 3)]
        b = self.pts[min(len(self.pts) - 1, i + 3)]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def project(self, x: float, y: float, hint=None, window: int = 60) -> "Projection":
        n = len(self.pts)
        rng = range(n) if hint is None else range(max(0, hint - window), min(n, hint + window + 1))
        i = min(rng, key=lambda j: (self.pts[j][0] - x) ** 2 + (self.pts[j][1] - y) ** 2)
        best = None
        for j in (i - 1, i):
            if j < 0 or j + 1 >= n:
                continue
            (x0, y0), (x1, y1) = self.pts[j], self.pts[j + 1]
            dx, dy = x1 - x0, y1 - y0
            seg2 = dx * dx + dy * dy
            u = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg2))
            px, py = x0 + u * dx, y0 + u * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if best is None or d2 < best[0]:
                seg = math.sqrt(seg2)
                lat = 0.0 if seg == 0 else (dx * (y - y0) - dy * (x - x0)) / seg
                best = (d2, self.cum[j] + u * seg, lat)
        return Projection(best[1], best[2], i)

    def point_at(self, s: float):
        s = max(0.0, min(self.total, s))
        j = max(0, min(len(self.pts) - 2, bisect.bisect_right(self.cum, s) - 1))
        (x0, y0), (x1, y1) = self.pts[j], self.pts[j + 1]
        seg = self.cum[j + 1] - self.cum[j]
        u = 0.0 if seg == 0 else (s - self.cum[j]) / seg
        return x0 + u * (x1 - x0), y0 + u * (y1 - y0), math.atan2(y1 - y0, x1 - x0)
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/test_route.py -q; echo rc=$?`
Expected: `6 passed`, `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/world/__init__.py vtd_rl/world/route.py tests/test_route.py
git commit -m "world: 경로 인덱스 — 누적거리·투영(왼쪽 +)·방위"
```

---

### Task 3: 판 불러오기·자르기와 단계 ① 목록

**Files:**
- Create: `vtd_rl/world/board.py`, `curricula/stage1.json`, `tests/test_board.py`

**Interfaces:**
- Consumes: `rs.Scenario`, `rs.path`, `RouteIndex`
- Produces:
  - `Board(name: str, scenario: Scenario, lane_plan: list[dict | None], signals: str = "always_green")` — `__post_init__` 에서 `route: RouteIndex` 를 만든다. **경로점은 `scenario.ego_route` 원본**(차로계획과 짝이 맞는 쪽)이고, 비어 있으면 `scenario.route_points()`
  - `Board.start_pose -> tuple[float, float, float]` — 경로 첫 점과 **경로 방위**(코스 JSON 의 `ego_start` 방위는 0.0 자리표시라 쓰지 않는다)
  - `Board.goal -> tuple[float, float]` — 경로 마지막 점
  - `load_board(entry: dict, signals: str = "always_green") -> Board` — `entry = {"name", "route", "lane"}`, 경로는 서브모듈 기준 상대경로
  - `slice_board(board: Board, s_from: float, s_to: float, name: str) -> Board`
  - `load_curriculum(path: str) -> tuple[str, list[Board]]` — (단계 이름, 판 목록)

- [ ] **Step 1: 단계 ① 목록 파일**

`curricula/stage1.json`
```json
{
  "stage": 1,
  "name": "빈 경로 — 액터 없음, 신호 항상 녹색",
  "signals": "always_green",
  "boards": [
    {"name": "course_A", "route": "routes/HL_FMA_NEW_A.json", "lane": "routes/HL_FMA_NEW_A_lane.json"},
    {"name": "course_B", "route": "routes/HL_FMA_NEW_B.json", "lane": "routes/HL_FMA_NEW_B_lane.json"},
    {"name": "course_D", "route": "routes/HL_FMA_NEW_D.json", "lane": "routes/HL_FMA_NEW_D_lane.json"},
    {"name": "course_E", "route": "routes/HL_FMA_NEW_E.json", "lane": "routes/HL_FMA_NEW_E_lane.json"},
    {"name": "course_G", "route": "routes/HL_FMA_NEW_G.json", "lane": "routes/HL_FMA_NEW_G_lane.json"},
    {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
  ]
}
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_board.py`
```python
import math
import os

import pytest

from vtd_rl.world.board import Board, load_board, load_curriculum, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
STAGE1 = os.path.join(os.path.dirname(__file__), "..", "curricula", "stage1.json")


def test_코스_H_불러오기():
    b = load_board(H)
    assert len(b.route.pts) == len(b.lane_plan) == 2015     # 원본 ego_route 와 차로계획이 짝
    assert b.route.total == pytest.approx(2222, abs=2)
    x, y, h = b.start_pose
    assert (x, y) == pytest.approx(b.route.pts[0])
    assert h == pytest.approx(b.route.heading_at(0))
    assert b.signals == "always_green"


def test_짝이_안_맞으면_거부():
    b = load_board(H)
    with pytest.raises(ValueError):
        Board("bad", b.scenario, b.lane_plan[:-1])


def test_자르기():
    b = load_board(H)
    s = slice_board(b, 0.0, 250.0, "H_0_250")
    assert s.name == "H_0_250"
    assert 245.0 <= s.route.total <= 251.0
    assert len(s.route.pts) == len(s.lane_plan)
    assert s.scenario.actors == [] and s.scenario.lights == []
    assert s.goal == pytest.approx(s.route.pts[-1])
    assert s.scenario.duration >= 60.0


def test_단계1_목록():
    name, boards = load_curriculum(STAGE1)
    assert name.startswith("빈 경로")
    assert [b.name for b in boards] == ["course_A", "course_B", "course_D", "course_E", "course_G", "course_H"]
    assert all(b.signals == "always_green" for b in boards)
    assert all(len(b.route.pts) == len(b.lane_plan) for b in boards)
```

- [ ] **Step 3: 실패 확인**

Run: `.venv/bin/pytest tests/test_board.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.world.board'`

- [ ] **Step 4: 구현**

`vtd_rl/world/board.py`
```python
"""판 = 규칙 스택 Scenario + 경로점별 차로계획 + 신호 운용 방식."""
import bisect
import json
from dataclasses import dataclass, field

from vtd_rl import rule_stack as rs
from vtd_rl.world.route import RouteIndex

SIGNAL_MODES = ("always_green", "cycle")


@dataclass
class Board:
    name: str
    scenario: object
    lane_plan: list
    signals: str = "always_green"
    route: RouteIndex = field(init=False)

    def __post_init__(self):
        if self.signals not in SIGNAL_MODES:
            raise ValueError(f"신호 운용은 {SIGNAL_MODES} 중 하나: {self.signals}")
        pts = self.scenario.ego_route or self.scenario.route_points()
        if len(pts) != len(self.lane_plan):
            # 차로계획은 원본 ego_route 와 짝이다. route_points() 로 촘촘하게 나누면
            # 코스 H 는 2021 점이 되어 6점 어긋난다(2026-09-15 확인).
            raise ValueError(f"{self.name}: 경로점 {len(pts)} != 차로계획 {len(self.lane_plan)}")
        self.route = RouteIndex(pts)

    @property
    def start_pose(self):
        x, y = self.route.pts[0]
        return x, y, self.route.heading_at(0)

    @property
    def goal(self):
        return self.route.pts[-1]


def _load_json(rel):
    with open(rs.path(rel), encoding="utf-8") as f:
        return json.load(f)


def load_board(entry: dict, signals: str = "always_green") -> Board:
    sc = rs.Scenario.load(rs.path(entry["route"]))
    lane = _load_json(entry["lane"])["pts"]
    return Board(entry["name"], sc, lane, signals)


def slice_board(board: Board, s_from: float, s_to: float, name: str) -> Board:
    cum = board.route.cum
    i0 = bisect.bisect_left(cum, s_from)
    i1 = bisect.bisect_right(cum, s_to) - 1
    if i1 - i0 < 2:
        raise ValueError(f"자른 구간이 너무 짧다: {s_from}~{s_to}")
    pts = [list(p) for p in board.route.pts[i0:i1 + 1]]
    length = cum[i1] - cum[i0]
    sc0 = board.scenario
    sc = rs.Scenario(
        name=name,
        ego_start=[pts[0][0], pts[0][1], board.route.heading_at(i0)],
        ego_goal=pts[-1],
        speed_limit=sc0.speed_limit,
        duration=max(60.0, length / 5.0 + 30.0),
        actors=[], lights=[], zones=[],
        ego_route=pts, respawns=[], tl_stops={},
    )
    return Board(name, sc, board.lane_plan[i0:i1 + 1], board.signals)


def load_curriculum(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["name"], [load_board(e, d["signals"]) for e in d["boards"]]
```

- [ ] **Step 5: 통과 확인**

Run: `.venv/bin/pytest tests/test_board.py -q; echo rc=$?`
Expected: `4 passed`, `rc=0`

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/world/board.py curricula/stage1.json tests/test_board.py
git commit -m "world: 판 불러오기·자르기, 단계 ① 목록(연습 코스 6개)"
```

---

### Task 4: 자차 동역학

**Files:**
- Create: `vtd_rl/world/dynamics.py`, `tests/test_dynamics.py`

**Interfaces:**
- Produces:
  - `DynamicsParams(wheelbase=2.95, max_steer=math.radians(35.0), max_steer_rate=1.0, accel_tau=0.25, accel_min=-5.0, accel_max=2.0)` — dataclass. 값의 출처: 축거·조향 한계·가속 범위는 규칙 스택(`src/control.py`, `eval/mock_vtd.py`). 지연·조향 속도는 **M5 에서 VTD 로 보정하기 전의 초기값**
  - `EgoState(x, y, heading, v=0.0, accel=0.0, steer=0.0)` — dataclass
  - `step(st: EgoState, steer_cmd: float, accel_cmd: float, dt: float, p: DynamicsParams) -> EgoState` — 새 객체를 돌려준다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dynamics.py`
```python
import math

import pytest

from vtd_rl.world.dynamics import DynamicsParams, EgoState, step

DT = 0.05


def run(st, steer, accel, seconds, p):
    for _ in range(int(round(seconds / DT))):
        st = step(st, steer, accel, DT, p)
    return st


def test_지연_없는_직선_가속은_규칙_스택_오프라인_시뮬과_같다():
    p = DynamicsParams(accel_tau=0.0, max_steer_rate=100.0)
    st = run(EgoState(0.0, 0.0, 0.0), 0.0, 2.0, 1.0, p)
    assert st.v == pytest.approx(2.0)
    assert st.x == pytest.approx(1.05)       # sum_{k=1..20} (0.1k) * 0.05
    assert st.y == pytest.approx(0.0)


def test_가속_1차_지연():
    p = DynamicsParams(accel_tau=0.25)
    st = run(EgoState(0.0, 0.0, 0.0), 0.0, 2.0, 0.25, p)
    assert st.accel == pytest.approx(2.0 * (1.0 - math.exp(-1.0)), rel=1e-6)


def test_조향_속도_제한():
    p = DynamicsParams(max_steer_rate=1.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=5.0), 0.5, 0.0, 0.1, p)
    assert st.steer == pytest.approx(0.1)


def test_조향_한계():
    p = DynamicsParams(max_steer_rate=100.0)
    st = step(EgoState(0.0, 0.0, 0.0), 2.0, 0.0, DT, p)
    assert st.steer == pytest.approx(math.radians(35.0))


def test_가속_범위와_후진_없음():
    p = DynamicsParams(accel_tau=0.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=1.0), 0.0, -9.0, 1.0, p)
    assert st.accel == pytest.approx(-5.0)
    assert st.v == 0.0 and st.x < 0.2


def test_좌회전은_왼쪽으로_간다():
    p = DynamicsParams(max_steer_rate=100.0, accel_tau=0.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=5.0), 0.3, 0.0, 1.0, p)
    assert st.heading > 0.3 and st.y > 0.5
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/test_dynamics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.world.dynamics'`

- [ ] **Step 3: 구현**

`vtd_rl/world/dynamics.py`
```python
"""자차 동역학 — 자전거 모델 + 가속 1차 지연 + 조향 속도 제한.

적분 순서는 규칙 스택 오프라인 시뮬(`eval/mock_vtd.py`)과 같다:
속도 -> 방위(새 속도로) -> 위치(새 방위로). 지연·속도제한을 끄면 결과가 같다.
"""
import math
from dataclasses import dataclass


@dataclass
class DynamicsParams:
    wheelbase: float = 2.95
    max_steer: float = math.radians(35.0)
    max_steer_rate: float = 1.0      # rad/s — M5 보정 전 초기값
    accel_tau: float = 0.25          # s — 0 이면 즉시 반영. M5 보정 전 초기값
    accel_min: float = -5.0
    accel_max: float = 2.0


@dataclass
class EgoState:
    x: float
    y: float
    heading: float
    v: float = 0.0
    accel: float = 0.0
    steer: float = 0.0


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def step(st: EgoState, steer_cmd: float, accel_cmd: float, dt: float, p: DynamicsParams) -> EgoState:
    steer_cmd = _clamp(steer_cmd, -p.max_steer, p.max_steer)
    dmax = p.max_steer_rate * dt
    steer = st.steer + _clamp(steer_cmd - st.steer, -dmax, dmax)

    accel_cmd = _clamp(accel_cmd, p.accel_min, p.accel_max)
    if p.accel_tau <= 0.0:
        accel = accel_cmd
    else:
        accel = st.accel + (accel_cmd - st.accel) * (1.0 - math.exp(-dt / p.accel_tau))

    v = max(0.0, st.v + accel * dt)
    heading = st.heading + v / p.wheelbase * math.tan(steer) * dt
    heading = (heading + math.pi) % (2.0 * math.pi) - math.pi
    x = st.x + v * math.cos(heading) * dt
    y = st.y + v * math.sin(heading) * dt
    return EgoState(x, y, heading, v, accel, steer)
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/test_dynamics.py -q; echo rc=$?`
Expected: `6 passed`, `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/world/dynamics.py tests/test_dynamics.py
git commit -m "world: 자차 동역학 — 자전거 모델 + 가속 지연 + 조향 속도 제한"
```

---

### Task 5: 신호 — 경로 위 신호 찾기·주기·알림

**Files:**
- Create: `vtd_rl/world/signals.py`, `tests/test_signals.py`

**Interfaces:**
- Consumes: `RouteIndex.project`, `RouteIndex.heading_at`, `rs.TL_*`, `rs.map_db()`, `load_board`(테스트)
- Produces:
  - 상수 `SAME_DIR_DEG = 45.0`, `MAX_LATERAL = 8.0`, `REPORT_RANGE = 150.0`, `PASSED_MARGIN = 5.0`
  - `RouteSignal(s: float, tl_id: int)` — frozen dataclass
  - `route_signals(route: RouteIndex, tl_map: dict[int, list], stoplines_all: list) -> list[RouteSignal]` — s 오름차순, 같은 정지선(같은 s)의 여러 id 중 **가장 작은 id** 하나
  - `SignalProgram(mode="always_green", green=30.0, yellow=3.0, red=30.0, offset=0.0)` + `.state(t: float) -> int`
  - `programs_for(mode: str, signals: list[RouteSignal], rng: random.Random) -> dict[int, SignalProgram]`
  - `SignalReporter(signals, lane_plan, programs)` + `.report(s_ego: float, index: int, t: float) -> tuple[int, int]` — (tl_id, 상태), 알릴 게 없으면 `(-1, TL_UNSET)`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_signals.py`
```python
import random

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.world.board import load_board
from vtd_rl.world.route import RouteIndex
from vtd_rl.world.signals import (REPORT_RANGE, RouteSignal, SignalProgram, SignalReporter,
                                  programs_for, route_signals)

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def test_코스_H_같은_방향_신호만():
    b = load_board(H)
    db = rs.map_db()
    sig = route_signals(b.route, db["tl_map"], db["stoplines_all"])
    assert [s.tl_id for s in sig[:3]] == [151, 155, 161]     # 반대 방향 150·152·157 은 빠진다
    assert sig[0].s == pytest.approx(96.0, abs=3.0)
    assert all(a.s < c.s for a, c in zip(sig, sig[1:]))


def test_신호_주기():
    p = SignalProgram("cycle", green=10.0, yellow=2.0, red=8.0, offset=0.0)
    assert p.state(0.0) == rs.TL_GREEN
    assert p.state(10.5) == rs.TL_YELLOW
    assert p.state(12.5) == rs.TL_RED
    assert p.state(20.5) == rs.TL_GREEN
    assert SignalProgram().state(999.0) == rs.TL_GREEN


def test_프로그램_묶음():
    sig = [RouteSignal(10.0, 3), RouteSignal(50.0, 9)]
    g = programs_for("always_green", sig, random.Random(0))
    assert set(g) == {3, 9} and g[3].mode == "always_green"
    c1 = programs_for("cycle", sig, random.Random(1))
    c2 = programs_for("cycle", sig, random.Random(1))
    assert c1[3].offset == c2[3].offset                      # 같은 시드 = 같은 위상
    with pytest.raises(ValueError):
        programs_for("blink", sig, random.Random(0))


def test_알림_규칙():
    sig = [RouteSignal(100.0, 7)]
    lane = [{"j": 0}] * 150 + [{"j": 1}] * 10 + [None] * 40
    rep = SignalReporter(sig, lane, {7: SignalProgram()})
    assert rep.report(100.0 - REPORT_RANGE - 1.0, 0, 0.0) == (-1, rs.TL_UNSET)   # 너무 멀다
    assert rep.report(20.0, 20, 0.0) == (7, rs.TL_GREEN)
    assert rep.report(97.0, 97, 0.0) == (7, rs.TL_GREEN)                         # 정지선 조금 지남
    assert rep.report(155.0, 155, 0.0) == (-1, rs.TL_UNSET)                      # 교차로 안
    assert rep.report(120.0, 120, 0.0) == (-1, rs.TL_UNSET)                      # 완전히 지남
    assert rep.report(20.0, 190, 0.0) == (7, rs.TL_GREEN)                        # 차로계획 칸이 None


def test_짧은_직선에는_신호가_없다():
    r = RouteIndex([(float(i), 0.0) for i in range(50)])
    assert route_signals(r, {1: [1000.0, 1000.0]}, [[1000.0, 1000.0, 0.0]]) == []
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/test_signals.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.world.signals'`

- [ ] **Step 3: 구현**

`vtd_rl/world/signals.py`
```python
"""신호 — 경로가 지나는 신호 찾기, 신호 주기, VTD 식 알림.

VTD 는 자차 앞 신호 하나의 id·상태를 주고 교차로 안에서는 주지 않는다(주최측 답변).
신호 정지선 DB(tl_map, 214개)에는 방위가 없어 가장 가까운 도색 정지선(stoplines_all, 방위 포함)의
방위로 **같은 방향 진입로**만 고른다. 코스 H 에서 이 규칙이 신호 151(s≈96 m)을 잡는 것을 확인했다.
"""
import math
from dataclasses import dataclass

from vtd_rl import rule_stack as rs

SAME_DIR_DEG = 45.0      # 도색 정지선 방위와 경로 방위 차 허용[도]
MAX_LATERAL = 8.0        # 경로에서 이만큼 안의 정지선만[m]
REPORT_RANGE = 150.0     # 이 앞까지 알린다[m] — VTD 실측 60~300 m 의 가운데쯤
PASSED_MARGIN = 5.0      # 뒷축이 정지선을 이만큼 지날 때까지는 계속 알린다[m]


@dataclass(frozen=True)
class RouteSignal:
    s: float
    tl_id: int


def route_signals(route, tl_map, stoplines_all):
    found = {}
    for tid, (x, y) in tl_map.items():
        p = route.project(x, y)
        if abs(p.lateral) > MAX_LATERAL or p.s <= 0.0 or p.s >= route.total:
            continue
        _, _, sh = min(stoplines_all, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
        dh = abs((sh - route.heading_at(p.index) + math.pi) % (2.0 * math.pi) - math.pi)
        if math.degrees(dh) > SAME_DIR_DEG:
            continue
        key = round(p.s)
        if key not in found or int(tid) < found[key].tl_id:
            found[key] = RouteSignal(p.s, int(tid))
    return sorted(found.values(), key=lambda r: r.s)


@dataclass
class SignalProgram:
    mode: str = "always_green"
    green: float = 30.0
    yellow: float = 3.0
    red: float = 30.0
    offset: float = 0.0

    def state(self, t: float) -> int:
        if self.mode == "always_green":
            return rs.TL_GREEN
        tt = (t + self.offset) % (self.green + self.yellow + self.red)
        if tt < self.green:
            return rs.TL_GREEN
        if tt < self.green + self.yellow:
            return rs.TL_YELLOW
        return rs.TL_RED


def programs_for(mode, signals, rng):
    if mode == "always_green":
        return {s.tl_id: SignalProgram("always_green") for s in signals}
    if mode == "cycle":
        base = SignalProgram("cycle")
        period = base.green + base.yellow + base.red
        return {s.tl_id: SignalProgram("cycle", offset=rng.uniform(0.0, period)) for s in signals}
    raise ValueError(f"알 수 없는 신호 운용: {mode}")


class SignalReporter:
    def __init__(self, signals, lane_plan, programs):
        self.signals = signals
        self.lane_plan = lane_plan
        self.programs = programs

    def report(self, s_ego: float, index: int, t: float):
        plan = self.lane_plan[index] if 0 <= index < len(self.lane_plan) else None
        if plan and plan.get("j"):
            return -1, rs.TL_UNSET
        for sig in self.signals:
            if sig.s + PASSED_MARGIN < s_ego:
                continue
            if sig.s - s_ego <= REPORT_RANGE:
                return sig.tl_id, self.programs[sig.tl_id].state(t)
            break
        return -1, rs.TL_UNSET
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/test_signals.py -q; echo rc=$?`
Expected: `5 passed`, `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/world/signals.py tests/test_signals.py
git commit -m "world: 신호 — 같은 방향 정지선 찾기, 주기, VTD 식 알림(교차로 안 없음)"
```

---

### Task 6: 액터 — 규칙 스택 오프라인 액터 모델 재사용

**Files:**
- Create: `vtd_rl/world/actors.py`, `tests/test_actors.py`

**Interfaces:**
- Consumes: `rs.mock_vtd.spawned`, `rs.mock_vtd.init_actor_state`, `rs.mock_vtd.step_actor`, `rs.Obj`, `rs.Actor`
- Produces:
  - `ActorSet(actors: list[Actor])` + `.reset()` + `.step(t: float, ex: float, ey: float, dt: float) -> list[Obj]` — 등장한 액터 전부(거리 거르기는 World 가 한다)
  - `nearest_objects(objs: list[Obj], ex: float, ey: float, max_range: float, max_n: int) -> list[Obj]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_actors.py`
```python
import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.world.actors import ActorSet, nearest_objects


def actor(aid, spawn, motion, size=(0.15, 0.46, 0.61), typ="object"):
    return rs.Actor(id=aid, type=typ, size=list(size), spawn=spawn, motion=motion)


def test_정지_물체():
    a = actor(401, {"at_time": 0}, {"kind": "static", "pos": [1474.915, -548.598]})
    objs = ActorSet([a]).step(0.0, 1470.0, -550.0, 0.05)
    assert len(objs) == 1
    o = objs[0]
    assert (o.id, o.x, o.y, o.speed) == (401, pytest.approx(1474.915), pytest.approx(-548.598), 0.0)
    assert (o.length, o.width, o.height) == (0.15, 0.46, 0.61)


def test_등장_트리거와_리셋():
    a = actor(7, {"ego_within": 20, "of": [100.0, 0.0]}, {"kind": "static", "pos": [100.0, 3.0]})
    acts = ActorSet([a])
    assert acts.step(0.0, 0.0, 0.0, 0.05) == []
    assert len(acts.step(0.05, 85.0, 0.0, 0.05)) == 1
    assert len(acts.step(0.10, 0.0, 0.0, 0.05)) == 1       # 한 번 등장하면 남는다
    acts.reset()
    assert acts.step(0.0, 0.0, 0.0, 0.05) == []


def test_횡단_보행자는_움직인다():
    a = actor(9, {"at_time": 0}, {"kind": "crossing", "pos": [50.0, -5.0], "vel": [0.0, 1.5]},
              size=(0.6, 0.7, 1.8), typ="pedestrian")
    acts = ActorSet([a])
    for k in range(20):
        objs = acts.step(k * 0.05, 0.0, 0.0, 0.05)
    assert objs[0].y == pytest.approx(-5.0 + 1.5 * 1.0, abs=0.01)


def test_가까운_순_거르기():
    objs = [rs.Obj(i, float(d), 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0) for i, d in ((1, 50), (2, 10), (3, 90), (4, 30))]
    out = nearest_objects(objs, 0.0, 0.0, max_range=80.0, max_n=2)
    assert [o.id for o in out] == [2, 4]
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/test_actors.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.world.actors'`

- [ ] **Step 3: 구현**

`vtd_rl/world/actors.py`
```python
"""액터 — 규칙 스택 오프라인 시뮬(`eval/mock_vtd.py`)의 등장·운동 모델을 그대로 쓴다.

호출 순서도 오프라인 시뮬과 같다: 등장 판정 -> (처음이면) 상태 초기화 -> 한 스텝 적분 -> Obj.
"""
import math

from vtd_rl import rule_stack as rs

_mv = rs.mock_vtd


class ActorSet:
    def __init__(self, actors):
        self.actors = list(actors)
        self.reset()

    def reset(self):
        self._state = {}
        self._spawn_t = {}

    def step(self, t, ex, ey, dt):
        objs = []
        for a in self.actors:
            if not _mv.spawned(a, t, ex, ey, self._spawn_t):
                continue
            st = self._state.get(a.id)
            if st is None:
                st = self._state[a.id] = _mv.init_actor_state(a)
            _mv.step_actor(a, st, ex, ey, dt)
            objs.append(rs.Obj(a.id, st["x"], st["y"], 0.0, st["hd"], st["sp"],
                               a.size[0], a.size[1], a.size[2]))
        return objs


def nearest_objects(objs, ex, ey, max_range, max_n):
    """VTD 9910 처럼 수평 max_range 안을 가까운 순으로 max_n 개."""
    ranked = sorted(((math.hypot(o.x - ex, o.y - ey), o) for o in objs), key=lambda p: p[0])
    return [o for d, o in ranked if d <= max_range][:max_n]
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/test_actors.py -q; echo rc=$?`
Expected: `4 passed`, `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/world/actors.py tests/test_actors.py
git commit -m "world: 액터 — 규칙 스택 오프라인 액터 모델 재사용, 80 m·30개 거르기"
```

---

### Task 7: World — reset/step 과 종료 판정

**Files:**
- Create: `vtd_rl/world/world.py`, `tests/test_world.py`

**Interfaces:**
- Consumes: `Board`, `RouteIndex.project`, `DynamicsParams`, `EgoState`, `dynamics.step`, `ActorSet`, `nearest_objects`, `route_signals`, `programs_for`, `SignalReporter`, `rs.State`, `rs.map_db`
- Produces:
  - `WorldConfig(dt=0.05, dynamics=DynamicsParams(), goal_radius=10.0, offroad_margin=1.5, stall_seconds=60.0, stall_progress=1.0, time_limit_scale=1.0, object_range=80.0, max_objects=30, clock_origin=1.0e9)`
  - `StepInfo(t, s, lateral, index, v, done: bool, outcome: str)` — outcome ∈ `"running" | "goal" | "offroad" | "timeout" | "stalled"`
  - `World(board: Board, config: WorldConfig | None = None, signals: list[RouteSignal] | None = None, seed: int = 0)`
    - `.reset() -> State`, `.step(steer: float, accel: float, turn: int) -> tuple[State, StepInfo]`
    - 속성 `t: float`(판 시작부터 초), `clock: float`(= `clock_origin + t`, 규칙 스택에 넘기는 now), `ego: EgoState`, `turn: int`
  - `State.speed` 는 **0 으로 둔다**(VTD 패킷에 속도가 없다). 참속도는 `StepInfo.v`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_world.py`
```python
import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.world.board import Board
from vtd_rl.world.signals import RouteSignal
from vtd_rl.world.world import World, WorldConfig

LANE = {"lane": -1, "w": 3.5, "l": 1.75, "r": 1.75, "pl": 1.75, "pr": 1.75, "xl": 0.0, "xr": 0.0,
        "need": 0.0, "sig": 0, "j": 0, "jx": 0, "lim": 13.9}


def straight_board(length=200, actors=(), duration=60.0, signals="always_green"):
    route = [[float(i), 0.0] for i in range(length + 1)]
    sc = rs.Scenario(name="straight", ego_start=[0.0, 0.0, 0.0], ego_goal=route[-1], speed_limit=13.9,
                     duration=duration, actors=list(actors), lights=[], zones=[], ego_route=route,
                     respawns=[], tl_stops={})
    return Board("straight", sc, [dict(LANE) for _ in route], signals)


def drive(world, steer, accel, seconds):
    info = None
    for _ in range(int(round(seconds / world.cfg.dt))):
        _, info = world.step(steer, accel, 0)
        if info.done:
            break
    return info


def test_리셋():
    w = World(straight_board(), signals=[])
    s = w.reset()
    assert (s.x, s.y, s.heading) == (0.0, 0.0, 0.0)
    assert s.speed == 0.0 and s.tl_id == -1
    assert s.t == pytest.approx(1.0e9) and w.t == 0.0


def test_가속하면_경로를_따라_나아간다():
    w = World(straight_board(), signals=[])
    w.reset()
    info = drive(w, 0.0, 2.0, 3.0)
    assert info.outcome == "running"
    assert 4.0 < info.s < 9.0 and info.v > 3.0
    assert info.t == pytest.approx(3.0)


def test_목표():
    w = World(straight_board(length=40), signals=[])
    w.reset()
    info = drive(w, 0.0, 2.0, 20.0)
    assert info.outcome == "goal" and info.done
    assert info.s >= 40.0 - 10.0


def test_도로_이탈():
    w = World(straight_board(), signals=[])
    w.reset()
    info = drive(w, 0.6, 2.0, 20.0)
    assert info.outcome == "offroad"
    assert info.lateral > 1.75 + 1.5


def test_시간_초과():
    w = World(straight_board(duration=1.0), signals=[])
    w.reset()
    assert drive(w, 0.0, 0.0, 5.0).outcome == "timeout"


def test_정체():
    cfg = WorldConfig(stall_seconds=2.0)
    w = World(straight_board(), config=cfg, signals=[])
    w.reset()
    info = drive(w, 0.0, 0.0, 5.0)
    assert info.outcome == "stalled" and info.t == pytest.approx(2.0, abs=0.06)


def test_신호_알림():
    w = World(straight_board(), signals=[RouteSignal(50.0, 7)])
    s = w.reset()
    assert (s.tl_id, s.tl_state) == (7, rs.TL_GREEN)


def test_물체는_80m_안만():
    near = rs.Actor(id=1, type="object", size=[1.0, 1.0, 1.0], spawn={"at_time": 0},
                    motion={"kind": "static", "pos": [40.0, 3.0]})
    far = rs.Actor(id=2, type="object", size=[1.0, 1.0, 1.0], spawn={"at_time": 0},
                   motion={"kind": "static", "pos": [150.0, 0.0]})
    w = World(straight_board(actors=[near, far]), signals=[])
    s = w.reset()
    assert [o.id for o in s.objects] == [1]


def test_리셋하면_처음부터():
    w = World(straight_board(), signals=[])
    w.reset()
    drive(w, 0.0, 2.0, 2.0)
    s = w.reset()
    assert (s.x, w.t, w.ego.v) == (0.0, 0.0, 0.0)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/test_world.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.world.world'`

- [ ] **Step 3: 구현**

`vtd_rl/world/world.py`
```python
"""오프라인 세계 — 판 하나를 20 Hz 로 굴려 9910 과 같은 State 를 낸다."""
import math
import random
from dataclasses import dataclass, field

from vtd_rl import rule_stack as rs
from vtd_rl.world import dynamics as dyn
from vtd_rl.world.actors import ActorSet, nearest_objects
from vtd_rl.world.signals import SignalReporter, programs_for, route_signals


@dataclass
class WorldConfig:
    dt: float = 0.05
    dynamics: dyn.DynamicsParams = field(default_factory=dyn.DynamicsParams)
    goal_radius: float = 10.0        # 경로 끝까지 남은 거리가 이 안이면 완주[m] — 주최측 "마지막 좌표 반경 10~20 m"
    offroad_margin: float = 1.5      # 물리 폭(pl/pr) 밖으로 이만큼 나가면 도로 이탈[m]
    stall_seconds: float = 60.0      # 이만큼 진행이 없으면 정체[s]
    stall_progress: float = 1.0      # 이 이상 나아가야 진행으로 친다[m]
    time_limit_scale: float = 1.0    # 시나리오 duration 에 곱한다
    object_range: float = 80.0       # 9910 물체 수평 범위[m]
    max_objects: int = 30            # 9910 물체 슬롯 수
    clock_origin: float = 1.0e9      # 규칙 스택 now 오프셋 — 실차에서는 벽시계 epoch 가 들어간다


@dataclass
class StepInfo:
    t: float
    s: float
    lateral: float
    index: int
    v: float
    done: bool
    outcome: str


class World:
    def __init__(self, board, config=None, signals=None, seed=0):
        self.board = board
        self.cfg = config or WorldConfig()
        if signals is None:
            db = rs.map_db()
            signals = route_signals(board.route, db["tl_map"], db["stoplines_all"])
        self.signals = signals
        self.seed = seed
        self.actors = ActorSet(board.scenario.actors)
        self.reset()

    @property
    def clock(self) -> float:
        return self.cfg.clock_origin + self.t

    def reset(self):
        rng = random.Random(self.seed)
        x, y, h = self.board.start_pose
        self.ego = dyn.EgoState(x, y, h)
        self.t = 0.0
        self.turn = rs.TS_OFF
        self.actors.reset()
        self.reporter = SignalReporter(self.signals, self.board.lane_plan,
                                       programs_for(self.board.signals, self.signals, rng))
        self._hint = None
        p = self._project()
        self._best_s, self._best_t = p.s, 0.0
        self.time_limit = self.board.scenario.duration * self.cfg.time_limit_scale
        return self._observe(p, 0.0)

    def step(self, steer: float, accel: float, turn: int):
        self.ego = dyn.step(self.ego, steer, accel, self.cfg.dt, self.cfg.dynamics)
        self.t += self.cfg.dt
        self.turn = int(turn)
        p = self._project()
        state = self._observe(p, self.cfg.dt)
        outcome = self._outcome(p)
        return state, StepInfo(self.t, p.s, p.lateral, p.index, self.ego.v, outcome != "running", outcome)

    def _project(self):
        p = self.board.route.project(self.ego.x, self.ego.y, hint=self._hint)
        self._hint = p.index
        return p

    def _observe(self, p, dt):
        objs = self.actors.step(self.t, self.ego.x, self.ego.y, dt)
        objs = nearest_objects(objs, self.ego.x, self.ego.y, self.cfg.object_range, self.cfg.max_objects)
        tl_id, tl_state = self.reporter.report(p.s, p.index, self.t)
        return rs.State(x=self.ego.x, y=self.ego.y, heading=self.ego.heading, objects=objs,
                        tl_id=tl_id, tl_state=tl_state, t=self.clock)

    def _outcome(self, p) -> str:
        if self.board.route.total - p.s <= self.cfg.goal_radius:
            return "goal"
        plan = self.board.lane_plan[p.index] or {}
        if not (plan.get("j") or plan.get("jx")):
            left = (plan.get("pl") or plan.get("l") or 1.75) + self.cfg.offroad_margin
            right = (plan.get("pr") or plan.get("r") or 1.75) + self.cfg.offroad_margin
            if p.lateral > left or p.lateral < -right:
                return "offroad"
        if self.t >= self.time_limit - 1e-9:
            return "timeout"
        if p.s > self._best_s + self.cfg.stall_progress:
            self._best_s, self._best_t = p.s, self.t
        elif self.t - self._best_t >= self.cfg.stall_seconds - 1e-9:
            return "stalled"
        return "running"
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/test_world.py -q; echo rc=$?`
Expected: `9 passed`, `rc=0`

- [ ] **Step 5: 전체 테스트**

Run: `.venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 6: 커밋**

```bash
git add vtd_rl/world/world.py tests/test_world.py
git commit -m "world: reset/step 과 종료 판정(완주·도로 이탈·시간 초과·정체)"
```

---

### Task 8: 선생님 그림자 주행과 한 판 달리기

**Files:**
- Create: `vtd_rl/teacher/__init__.py`, `vtd_rl/teacher/shadow.py`, `vtd_rl/rollout.py`, `tests/test_teacher.py`

**Interfaces:**
- Consumes: `World`, `WorldConfig`, `Board`, `load_board`, `slice_board`, `rs.DrivingStack`, `rs.VTDLink`, `rs.RunLogger`, `rs.map_db`
- Produces:
  - `ShadowTeacher(board: Board)` + `.act(state: State, now: float) -> Command` — **매 프레임** 불러야 한다(내부 상태·속도 추정이 이어진다). `state.speed` 를 채운다
  - `EpisodeResult(board: str, outcome: str, sim_time: float, steps: int, wall_time: float, distance: float)` + `.steps_per_sec`
  - `run_teacher_episode(board: Board, config: WorldConfig | None = None, log_csv: str | None = None, seed: int = 0) -> EpisodeResult`
  - `run_teacher_episode` 의 CSV 는 규칙 스택 `run_logger.HEADER` 열 그대로(규칙 스택 채점기가 읽는다)

- [ ] **Step 1: 실패하는 테스트 작성**

`vtd_rl/teacher/__init__.py`
```python
"""선생님 — 규칙 스택 그림자 주행."""
```

`tests/test_teacher.py`
```python
from vtd_rl import rule_stack as rs
from vtd_rl.rollout import run_teacher_episode
from vtd_rl.teacher.shadow import ShadowTeacher
from vtd_rl.world.board import load_board, slice_board
from vtd_rl.world.world import World

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_선생님은_출발한다():
    b = h_slice()
    w = World(b)
    s = w.reset()
    t = ShadowTeacher(b)
    for _ in range(40):
        cmd = t.act(s, w.clock)
        s, info = w.step(cmd.steer, cmd.accel, cmd.turn)
    assert info.v > 1.0
    t.act(s, w.clock)                     # 방금 받은 State 의 speed 는 선생님이 읽을 때 채워진다
    assert s.speed > 0.5                  # 위치 미분 추정


def test_선생님은_신호_151을_보고_지난다():
    b = h_slice()
    w = World(b)
    s = w.reset()
    t = ShadowTeacher(b)
    seen = set()
    info = None
    while info is None or not info.done:
        cmd = t.act(s, w.clock)
        s, info = w.step(cmd.steer, cmd.accel, cmd.turn)
        seen.add(s.tl_id)
    assert 151 in seen
    assert info.outcome == "goal"


def test_한_판_달리기와_CSV(tmp_path):
    csv = tmp_path / "run.csv"
    r = run_teacher_episode(h_slice(), log_csv=str(csv))
    assert r.outcome == "goal"
    assert 240.0 - 10.0 <= r.distance <= 251.0
    assert r.steps == round(r.sim_time / 0.05)
    assert r.steps_per_sec > 0
    lines = csv.read_text(encoding="utf-8").splitlines()
    assert lines[0] + "\n" == rs.HEADER
    assert len(lines) - 1 == r.steps
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/test_teacher.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.teacher.shadow'`

- [ ] **Step 3: 선생님 구현**

`vtd_rl/teacher/shadow.py`
```python
"""규칙 스택 그림자 주행.

대회 제어PC(`src/main.py`)와 같은 인자로 DrivingStack 을 만든다. 차이는 둘뿐이다:
- now 에 벽시계 대신 **시뮬 시계**(World.clock)를 넣는다 — 규칙 스택 판단 코드는 벽시계를 직접 읽지 않는다
  (`overtake.py` 는 now 가 None 일 때만 time.time() 을 쓰고, drive 가 now 를 넘긴다. 2026-09-15 확인).
- 속도는 VTDLink 의 위치 미분 추정기를 소켓 없이 그대로 쓴다(VTD 패킷에 자차 속도가 없다).
"""
from vtd_rl import rule_stack as rs


class ShadowTeacher:
    def __init__(self, board):
        db = rs.map_db()
        self.stack = rs.DrivingStack(
            scenario=board.scenario, base_limit=board.scenario.speed_limit,
            tl_stops_extra=db["tl_map"], max_speed=None, lane_plan=board.lane_plan,
            turn_lane=True, crosswalks=db["crosswalks"], stoplines=db["stoplines"],
            stoplines_all=db["stoplines_all"], allow_centerline_escape=False)
        self._speed = rs.VTDLink("127.0.0.1")          # connect() 하지 않는다
        self._last_now = None

    def act(self, state, now):
        self._speed._update_speed(state, now)
        dt = 0.05 if self._last_now is None else max(1e-3, now - self._last_now)
        self._last_now = now
        return self.stack.step(state, dt, now)
```

- [ ] **Step 4: 한 판 달리기 구현**

`vtd_rl/rollout.py`
```python
"""선생님으로 한 판 끝까지 달리기."""
import time
from dataclasses import dataclass

from vtd_rl import rule_stack as rs
from vtd_rl.teacher.shadow import ShadowTeacher
from vtd_rl.world.world import World


@dataclass
class EpisodeResult:
    board: str
    outcome: str
    sim_time: float
    steps: int
    wall_time: float
    distance: float

    @property
    def steps_per_sec(self) -> float:
        return self.steps / max(self.wall_time, 1e-9)


def run_teacher_episode(board, config=None, log_csv=None, seed=0) -> EpisodeResult:
    world = World(board, config, seed=seed)
    state = world.reset()
    teacher = ShadowTeacher(board)
    logger = rs.RunLogger(log_csv)
    steps, info = 0, None
    t0 = time.perf_counter()
    try:
        while info is None or not info.done:
            cmd = teacher.act(state, world.clock)
            logger.log(world.t, state, cmd)
            state, info = world.step(cmd.steer, cmd.accel, cmd.turn)
            steps += 1
    finally:
        logger.close()
    return EpisodeResult(board.name, info.outcome, info.t, steps, time.perf_counter() - t0, info.s)
```

- [ ] **Step 5: 통과 확인**

Run: `.venv/bin/pytest tests/test_teacher.py -q; echo rc=$?`
Expected: `3 passed`, `rc=0`

선생님이 H 0~250 m 를 완주하지 못하면(`stalled`·`offroad`·`timeout`) **world 구현이 틀린 것**이다(스펙 §8.1). 규칙 스택을 고치지 말고 `run_teacher_episode(..., log_csv=...)` 로 CSV 를 남겨 `reason`·`cap_by`·`sig`·`off` 칸에서 원인을 찾고, world 의 좌표·신호·차로계획 정렬을 먼저 의심한다.

- [ ] **Step 6: 전체 테스트**

Run: `.venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 7: 커밋**

```bash
git add vtd_rl/teacher/__init__.py vtd_rl/teacher/shadow.py vtd_rl/rollout.py tests/test_teacher.py
git commit -m "선생님 그림자 주행 — 규칙 스택을 시뮬 시계로, 한 판 달리기와 CSV"
```

---

### Task 9: 스텝 속도 측정과 단계 ① 성적표(M1 완료 증거)

**Files:**
- Create: `scripts/bench_world.py`, `scripts/run_stage1_teacher.py`, `docs/reports/m1-stage1-teacher.md`(스크립트가 만든다)

**Interfaces:**
- Consumes: `World`, `load_board`, `load_curriculum`, `run_teacher_episode`, `EpisodeResult`, `rs.commit`

- [ ] **Step 1: 스텝 속도 측정 스크립트**

`scripts/bench_world.py`
```python
"""world 단독·선생님 포함 스텝 속도를 잰다.

    .venv/bin/python scripts/bench_world.py --board course_H --seconds 120
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vtd_rl.teacher.shadow import ShadowTeacher  # noqa: E402
from vtd_rl.world.board import load_board  # noqa: E402
from vtd_rl.world.world import World  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="course_H")
    ap.add_argument("--seconds", type=float, default=120.0, help="시뮬 시간[s]")
    a = ap.parse_args()
    letter = a.board.split("_")[-1]
    board = load_board({"name": a.board, "route": f"routes/HL_FMA_NEW_{letter}.json",
                        "lane": f"routes/HL_FMA_NEW_{letter}_lane.json"})
    n = int(a.seconds / 0.05)

    w = World(board)
    w.reset()
    t0 = time.perf_counter()
    for k in range(n):
        _, info = w.step(0.0, 1.0 if w.ego.v < 8.0 else 0.0, 0)
        if info.done:
            w.reset()
    world_sps = n / (time.perf_counter() - t0)

    w = World(board)
    s = w.reset()
    teacher = ShadowTeacher(board)
    t0 = time.perf_counter()
    for k in range(n):
        cmd = teacher.act(s, w.clock)
        s, info = w.step(cmd.steer, cmd.accel, cmd.turn)
        if info.done:
            break
    both_sps = (k + 1) / (time.perf_counter() - t0)

    print(json.dumps({"board": a.board, "sim_seconds": a.seconds,
                      "world_steps_per_sec": round(world_sps),
                      "world_plus_teacher_steps_per_sec": round(both_sps)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 측정 실행**

Run: `.venv/bin/python scripts/bench_world.py --board course_H --seconds 120`
Expected: JSON 한 줄. `world_steps_per_sec` 와 `world_plus_teacher_steps_per_sec` 가 0 보다 크다. 값을 Step 4 성적표에 쓴다.

- [ ] **Step 3: 단계 ① 성적표 스크립트**

`scripts/run_stage1_teacher.py`
```python
"""선생님이 단계 ① 판을 전부 달리고 성적표를 쓴다 — M1 완료 증거.

    .venv/bin/python scripts/run_stage1_teacher.py --bench '<bench_world.py 출력 JSON>'
CSV 는 runs/m1/ 에 남긴다(git 제외).
"""
import argparse
import datetime
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.rollout import run_teacher_episode  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="scripts/bench_world.py 가 출력한 JSON 한 줄")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "reports", "m1-stage1-teacher.md"))
    a = ap.parse_args()
    bench = json.loads(a.bench)
    name, boards = load_curriculum(os.path.join(REPO, "curricula", "stage1.json"))
    os.makedirs(os.path.join(REPO, "runs", "m1"), exist_ok=True)

    rows = []
    for b in boards:
        r = run_teacher_episode(b, log_csv=os.path.join(REPO, "runs", "m1", f"{b.name}.csv"))
        rows.append((b, r))
        print(f"{b.name}: {r.outcome} {r.distance:.0f}/{b.route.total:.0f} m "
              f"{r.sim_time:.1f}s {r.steps_per_sec:.0f} steps/s", flush=True)

    ok = sum(1 for _, r in rows if r.outcome == "goal")
    lines = [
        "# M1 성적표 — 선생님(규칙 스택)이 오프라인 세계에서 단계 ① 판을 달린 결과", "",
        f"- 날짜 {datetime.date.today().isoformat()} · 머신 `{platform.node()}` · Python {platform.python_version()}",
        f"- 규칙 스택 커밋 `{rs.commit()}`",
        f"- 단계 ① `{name}` · 판 {len(rows)}개 · **완주 {ok}/{len(rows)}**", "",
        "| 판 | 경로 길이 [m] | 결과 | 달린 거리 [m] | 시뮬 시간 [s] | 스텝 | 벽시계 [s] | 스텝/초 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for b, r in rows:
        lines.append(f"| {b.name} | {b.route.total:.0f} | {r.outcome} | {r.distance:.0f} | {r.sim_time:.1f} "
                     f"| {r.steps} | {r.wall_time:.1f} | {r.steps_per_sec:.0f} |")
    lines += [
        "", "## 스텝 속도(`scripts/bench_world.py`)", "",
        f"- 판 `{bench['board']}` · 시뮬 {bench['sim_seconds']:.0f} 초",
        f"- world 단독 **{bench['world_steps_per_sec']} 스텝/초**",
        f"- world + 선생님 **{bench['world_plus_teacher_steps_per_sec']} 스텝/초**", "",
        "동역학 지연·조향 속도는 M5 보정 전 초기값(`DynamicsParams` 기본값)이다.",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"성적표: {a.out}")
    sys.exit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 성적표 만들기**

Run:
```bash
B=$(.venv/bin/python scripts/bench_world.py --board course_H --seconds 120)
.venv/bin/python scripts/run_stage1_teacher.py --bench "$B"; echo rc=$?
```
Expected: 판마다 한 줄씩 출력하고 `docs/reports/m1-stage1-teacher.md` 를 쓴다. **`rc=0`(6/6 완주)이면 M1 완료.**
`rc=1` 이면 완주 못 한 판의 CSV(`runs/m1/<판>.csv`)로 원인을 찾아 보고한다 — 규칙 스택을 고치거나 판을 빼서 숫자를 맞추지 않는다.

- [ ] **Step 5: 전체 테스트**

Run: `.venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 6: 커밋**

```bash
git add scripts/bench_world.py scripts/run_stage1_teacher.py docs/reports/m1-stage1-teacher.md
git commit -m "M1 증거 — 스텝 속도 측정, 선생님 단계 ① 성적표"
git push origin main
```
