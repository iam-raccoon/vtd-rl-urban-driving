"""규칙 스택(서브모듈) import 를 한곳에 모은다.

규칙 스택은 패키지가 아니라 `src/`·`eval/`·`vtd/` 폴더에 모듈이 흩어져 있고 서로를
최상위 이름(`from drive import ...`)으로 부른다. 그래서 세 폴더를 sys.path 앞에 넣는다.
세 폴더 사이에 같은 모듈 이름은 없다(2026-09-15 확인).

⚠️ 이 모듈을 import 하면 규칙 스택 모듈(`main`, `evaluate`, `control`, `scenario` 등)이 sys.path **맨 앞**에
   오므로, 나중에 같은 이름의 패키지·모듈을 import 하면 규칙 스택 쪽이 먼저 잡혀 가려질 수 있다.
"""
import functools
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "rule_stack"))
_DIRS = [os.path.join(ROOT, d) for d in ("src", "eval", "vtd")]
sys.path[:0] = [d for d in _DIRS if d not in sys.path]

from vtd_io import (State, Obj, VTDLink, TL_UNSET, TL_RED, TL_YELLOW, TL_GREEN, TL_GREEN_LEFT,  # noqa: E402
                    TS_OFF, TS_LEFT, TS_RIGHT)
from scenario import Scenario, Actor  # noqa: E402
from drive import DrivingStack, Command  # noqa: E402
from run_logger import RunLogger, HEADER  # noqa: E402
import mock_vtd  # noqa: E402
import score_fma  # noqa: E402
import check_lanes  # noqa: E402
import control  # noqa: E402

__all__ = ["ROOT", "path", "commit", "map_db", "State", "Obj", "VTDLink", "TL_UNSET", "TL_RED",
           "TL_YELLOW", "TL_GREEN", "TL_GREEN_LEFT", "TS_OFF", "TS_LEFT", "TS_RIGHT", "Scenario", "Actor",
           "DrivingStack", "Command", "RunLogger", "HEADER", "mock_vtd", "score_fma", "check_lanes",
           "control", "XODR", "load_map"]


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


XODR = os.path.join(ROOT, "map", "HL_FMA_VTD_LivingLab.xodr")


@functools.lru_cache(maxsize=1)
def load_map():
    """LivingLab 지도. 실측 0.34 초(2026-09-15)라 캐시 파일 없이 프로세스당 한 번 읽는다."""
    from build_lane_plan import Map
    return Map(XODR)
