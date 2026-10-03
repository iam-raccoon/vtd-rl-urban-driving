"""`scripts/summarize_dvs.py` 의 의미를 잠근다.

이 스크립트가 내는 부호검정 p 와 짝짓기 결과가 성적표의 주장을 떠받친다. 특히 **짝지을
시드가 없을 때 조용히 평균끼리 비교하지 않는 것**이 핵심이다 — M4f 에서 `floor6`(시드 3~5)을
기준선(시드 0~2)과 짝 없이 비교하면 시드 운을 개입 효과로 읽게 된다.
"""
import importlib.util
import json
import os

import pytest


def _load():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "summarize_dvs.py")
    spec = importlib.util.spec_from_file_location("summarize_dvs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_체크포인트_경로에서_설정과_시드를_뽑는다():
    m = _load()
    assert m.parse_run("runs/omen/2026-09-29-m4f/sigma-axis-s0/ac-3002880.pt") == ("sigma-axis", 0)
    assert m.parse_run("runs/omen/x/imit-floor05-s12/ac-1.pt") == ("imit-floor05", 12)
    # 이름에 숫자와 대시가 섞여도 **마지막** -s<숫자> 만 시드다
    assert m.parse_run("runs/a/base-v2-s3/ac.pt") == ("base-v2", 3)
    assert m.parse_run("runs/a/noseed/ac.pt") == ("noseed", None)


def test_부호검정_한쪽으로_전부_쏠리면_유의하다():
    m = _load()
    # 6 개 전부 개선 → 양측 p = 2 * (1/64) = 0.03125
    assert m.sign_test_p([0.1] * 6) == (6, 0, pytest.approx(0.03125))
    # 3 개 전부 개선 → 2 * (1/8) = 0.25. M4e 가 방향만 있고 확증은 아니라고 한 그 값이다.
    assert m.sign_test_p([0.1] * 3) == (3, 0, pytest.approx(0.25))


def test_부호검정_반반이면_유의하지_않다():
    m = _load()
    pos, neg, p = m.sign_test_p([0.1, 0.1, -0.1, -0.1])
    assert (pos, neg) == (2, 2)
    assert p == pytest.approx(1.0)


def test_부호검정_묶인_차이는_버리고_n을_줄인다():
    m = _load()
    # 0 인 차이 2 개는 버려서 n=2 가 된다(전부 개선) → 2 * (1/4) = 0.5
    pos, neg, p = m.sign_test_p([0.1, 0.1, 0.0, 0.0])
    assert (pos, neg) == (2, 0)
    assert p == pytest.approx(0.5)
    # 전부 묶이면 p 를 낼 수 없다 — 0.0 이나 1.0 으로 꾸며 내면 안 된다
    assert m.sign_test_p([0.0, 0.0]) == (0, 0, None)


def test_짝지을_시드가_없으면_평균끼리_비교하지_않는다(tmp_path, capsys):
    """★ M4f 의 설계 구멍을 직접 잠근다 — 시드가 안 겹치면 표에 경고가 나와야 한다."""
    m = _load()
    rows = []
    for name, seeds in (("base", (0, 1, 2)), ("floor6", (3, 4, 5))):
        for s in seeds:
            rows.append({
                "checkpoint": f"runs/x/{name}-s{s}/ac-3002880.pt",
                "stage1": {"det_goal": 0.4, "sto_goal": 1.0,
                           "det_score_completed": 90.0, "sto_score_completed": 90.0},
                "stage2": {"det_goal": 0.4, "sto_goal": 1.0,
                           "det_score_completed": 90.0, "sto_score_completed": 90.0},
            })
    p = tmp_path / "dvs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    import sys
    argv = sys.argv
    sys.argv = ["summarize_dvs.py", "--jsonl", str(p), "--pair", "base"]
    try:
        m.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "짝지을 시드가 없다" in out
    # 앞쪽 "설정별 평균" 표에도 같은 접두사의 줄이 있으니 **짝짓기 절로 범위를 좁힌다**
    paired = out.split("# 시드로 짝지은 비교", 1)[1]
    floor_rows = [ln for ln in paired.splitlines() if ln.startswith("| floor6 |")]
    assert floor_rows, paired
    for ln in floor_rows:
        assert "0.0" not in ln and "1.000" not in ln, ln


def _write(tmp_path, spec):
    """`spec` = [(설정이름, 시드, det_goal), ...] → JSONL 파일 경로."""
    rows = [{
        "checkpoint": f"runs/x/{name}-s{s}/ac-3002880.pt",
        "stage1": {"det_goal": g, "sto_goal": 1.0,
                   "det_score_completed": 90.0, "sto_score_completed": 90.0},
        "stage2": {"det_goal": g, "sto_goal": 1.0,
                   "det_score_completed": 90.0, "sto_score_completed": 90.0},
    } for name, s, g in spec]
    p = tmp_path / "dvs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_별칭이_다른_이름의_같은_설정을_한_팔로_묶는다(tmp_path):
    """M4f 의 기준선은 시드 0~2 가 `m4d-align`, 시드 3~5 가 `base` 로 이름이 다르지만
    **설정은 같다**(둘 다 개입 없음). 별칭으로 묶어야 n=6 짝짓기가 된다."""
    m = _load()
    p = _write(tmp_path, [("m4d-align", 0, 0.4), ("m4d-align", 1, 0.4),
                          ("base", 2, 0.4), ("base", 3, 0.4)])
    data = m.load(str(p), {"m4d-align": "base"})
    assert sorted(k for k in data) == [("base", 0), ("base", 1), ("base", 2), ("base", 3)]


def test_별칭이_서로_다른_실행을_덮으면_터진다(tmp_path):
    """조용히 덮어쓰면 어느 실행의 숫자가 성적표에 실렸는지 알 수 없게 된다."""
    m = _load()
    p = _write(tmp_path, [("m4d-align", 0, 0.4), ("base", 0, 0.9)])   # 둘 다 시드 0
    with pytest.raises(SystemExit, match="두 번"):
        m.load(str(p), {"m4d-align": "base"})


def test_겹치는_시드만_짝짓는다(tmp_path, capsys):
    m = _load()
    rows = []
    for name, seeds in (("base", (0, 1, 2)), ("arm", (1, 2, 9))):
        for s in seeds:
            rows.append({
                "checkpoint": f"runs/x/{name}-s{s}/ac-3002880.pt",
                "stage1": {"det_goal": 0.2 if name == "base" else 0.5,
                           "sto_goal": 1.0, "det_score_completed": 90.0,
                           "sto_score_completed": 90.0},
                "stage2": {"det_goal": 0.2 if name == "base" else 0.5,
                           "sto_goal": 1.0, "det_score_completed": 90.0,
                           "sto_score_completed": 90.0},
            })
    p = tmp_path / "dvs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    import sys
    argv = sys.argv
    sys.argv = ["summarize_dvs.py", "--jsonl", str(p), "--pair", "base"]
    try:
        m.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    paired = out.split("# 시드로 짝지은 비교", 1)[1]
    arm = [ln for ln in paired.splitlines() if ln.startswith("| arm |")][0]
    assert "1,2 (n=2)" in arm, arm      # 시드 9 는 기준선에 없으니 빠진다
    assert "+30.0pp" in arm, arm        # 0.5 - 0.2 = +0.3


def test_단계_이름을_박지_않고_JSONL_에서_읽는다(tmp_path, capsys):
    """③ 단계로 잰 줄의 `stage3b` 가 표에서 조용히 사라지면 안 된다(예전 `STAGES` 박아두기)."""
    m = _load()
    met = {"det_goal": 0.5, "sto_goal": 1.0, "det_score_completed": 90.0, "sto_score_completed": 90.0}
    rows = [{"checkpoint": f"runs/x/base-s{s}/ac-1.pt", "stage1": met, "stage3b": dict(met, det_goal=0.25)}
            for s in (0, 1)]
    p = tmp_path / "dvs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    data = m.load(str(p))
    assert m.stages_of(data) == ["stage1", "stage3b"]
    assert "checkpoint" not in data[("base", 0)]
    import sys
    argv = sys.argv
    sys.argv = ["summarize_dvs.py", "--jsonl", str(p)]
    try:
        m.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "## stage3b" in out and "25.0%" in out
    assert "## stage2" not in out      # 없는 단계를 빈 표로 지어내지 않는다
