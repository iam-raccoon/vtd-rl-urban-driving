"""M7l — 단계 ③s(어려운 장면 바로 앞에서 출발하는 짧은 판) 커리큘럼."""
import importlib.util
import json
import os

REPO = os.path.join(os.path.dirname(__file__), "..")


def _load(name):
    return json.load(open(os.path.join(REPO, "curricula", f"{name}.json")))


def test_단계3s_는_여섯_구간이고_액터는_단계3_그대로다():
    s3 = {b["name"]: b for b in _load("stage3")["boards"]}
    s3s = _load("stage3s")
    assert s3s["signals"] == "cycle" and s3s["stage"] == 3
    names = [b["name"] for b in s3s["boards"]]
    assert names == ["course_G_s2665", "course_G_s640", "course_D_s2050", "course_D_s730", "course_A_s1470", "course_E_s1740"]
    for b in s3s["boards"]:
        base = s3[b["name"].split("_s")[0]]
        assert b["actors"] == base["actors"] and b["jitter"] == base["jitter"]
        assert b["route"] == base["route"] and b["lane"] == base["lane"]
        s0, s1 = b["slice"]
        assert s1 - s0 >= 250.0


def test_단계3s_변종마다_구간_안에_액터가_있다():
    from vtd_rl.world.board import load_window
    boards, varied = load_window(os.path.join(REPO, "curricula", "stage3s.json"), 4, 0)
    assert varied
    for b in boards:
        assert len(b.scenario.actors) >= 1, b.name


def test_단계3s_는_평가_기본_단계에_없다():
    for script, names in (("eval_unseen.py", ("DEFAULT_STAGES",)),
                          ("run_dagger.py", ("DEFAULT_COLLECT_STAGES", "DEFAULT_EVAL_STAGES")),
                          ("refit_round.py", ("DEFAULT_STAGES",))):
        spec = importlib.util.spec_from_file_location(f"chk3s_{script[:-3]}", os.path.join(REPO, "scripts", script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for n in names:
            assert "stage3s" not in getattr(mod, n), (script, n)


def test_단계3s_는_같은_경로의_판과_점_수가_겹치지_않는다():
    """변종 판의 색인·심판 캐시 열쇠는 (경로|차로, 경로점 수) 다(`rl/variant_pool.py`, `world/board.py::cache_key`).

    자른 판이 같은 경로의 전체 판이나 다른 구간과 점 수까지 같으면 캐시가 남의 지형을 준다.
    """
    from vtd_rl.world.board import load_board
    seen = {}
    for stage in ("stage3", "stage3s"):
        d = _load(stage)
        for e in d["boards"]:
            n = len(load_board(e, d["signals"]).route.pts)
            key = (e["route"], e["lane"])
            assert n not in seen.setdefault(key, {}), (e["name"], seen[key][n])
            seen[key][n] = e["name"]
