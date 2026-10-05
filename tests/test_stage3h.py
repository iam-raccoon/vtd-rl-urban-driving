"""M7c — 단계 ③h(학습용 어려운 코스 덧보기) 커리큘럼.

③ 전체에서 실패가 나는 코스(A·D·E·G)만 골라 **같은 액터·같은 흔들기**로 다시 담는다. 학습에서
이 코스들을 더 자주 보게 하는 것이 목적이다. 판 이름 뒤에 커리큘럼 파일 이름이 붙으므로
(`rl/vec_env._boards`) ③ 전체의 같은 판과 세계 캐시가 섞이지 않는다.
"""
import json
import os

REPO = os.path.join(os.path.dirname(__file__), "..")


def _load(name):
    return json.load(open(os.path.join(REPO, "curricula", f"{name}.json")))


def test_단계3h_는_단계3_의_A_D_E_G_를_그대로_담는다():
    s3, s3h = _load("stage3"), _load("stage3h")
    assert [b["name"] for b in s3h["boards"]] == ["course_A", "course_D", "course_E", "course_G"]
    by_name = {b["name"]: b for b in s3["boards"]}
    for b in s3h["boards"]:
        assert b == by_name[b["name"]], b["name"]        # 경로·차로·액터·흔들기 모두 같다
    assert s3h["signals"] == s3["signals"] and s3h["stage"] == s3["stage"]


def test_단계3h_변종_이름은_학습_판_목록에서_단계3_과_갈린다():
    from vtd_rl.rl.vec_env import _boards
    names = [b.name for b in _boards(["curricula/stage3.json", "curricula/stage3h.json"], variants=2)]
    assert len(names) == len(set(names))
    assert any(n.endswith("#stage3h") for n in names)


def test_단계3h_는_평가_기본_단계에_없다():
    """③h 는 학습 전용 — 평가 스크립트의 기본 단계에 끼면 안 된다."""
    import importlib.util
    for script, names in (("eval_unseen.py", ("DEFAULT_STAGES",)),
                          ("run_dagger.py", ("DEFAULT_COLLECT_STAGES", "DEFAULT_EVAL_STAGES")),
                          ("refit_round.py", ("DEFAULT_STAGES",))):
        spec = importlib.util.spec_from_file_location(f"chk3h_{script[:-3]}", os.path.join(REPO, "scripts", script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for n in names:
            assert "stage3h" not in getattr(mod, n), (script, n)
