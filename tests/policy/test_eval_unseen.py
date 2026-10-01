"""`scripts/eval_unseen.py` — **고른 판으로 성적을 내지 않는다.**

M4o 계측 규칙: `--select best` 는 라운드 5 개 중 최선을 고르므로 **그 평가는 잡음의
상단**이다. 단계 ① 은 독립 판이 6 개(해상도 16.7 점)라 그 편향이 작지 않다. 그래서
채택된 `policy-best.pt` 를 **수집이 쓰지 않은 변종**으로 다시 재고, 성적표 헤드라인은
그 숫자다.

여기서 잠그는 것은 둘이다.
① 평가 판이 **수집이 본 판과 안 겹친다**(오프셋이 조용히 0 이 되면 편향이 그대로 돌아온다).
② 무엇을 평가했는지가 **성적표와 요약에 숫자로** 남는다(변종 창·판 수).
"""
import importlib.util
import json
import os
import sys
import types

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "eval_unseen.py")


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("eval_unseen_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """작은 진짜 체크포인트 — 스크립트가 `DrivePolicy.load` 로 읽는 길을 그대로 탄다."""
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig
    path = tmp_path_factory.mktemp("ck") / "policy-best.pt"
    DrivePolicy(PolicyConfig(trunk=(8, 8), squash=True)).save(str(path))
    return path


def _fake_ev(goal_rate=0.5, mean_score=90.0, steps=100):
    ep = types.SimpleNamespace(board="course_A", seed=0, outcome="goal", steps=steps,
                               reward=0.0, score=mean_score, sheet=[])
    return {"goal_rate": goal_rate, "mean_score": mean_score, "mean_score_raw": mean_score,
            "mean_score_completed": mean_score, "mean_reward": 0.0, "episodes": [ep]}


@pytest.fixture
def seen_boards(ev, monkeypatch):
    """평가에 **실제로 넘어간 판 목록**을 단계별로 적어 둔다 — 판은 진짜로 짓는다."""
    got = {}

    def fake_eval(net, boards, seeds=(0,)):
        got.setdefault(len(got), None)
        got[tuple(b.name for b in boards)] = len(boards)
        return _fake_ev()

    monkeypatch.setattr(ev, "evaluate_policy", fake_eval)
    return got


def _names(seen_boards):
    """`fake_eval` 이 받은 판 이름 전부(단계 구분 없이 합친 집합)."""
    out = set()
    for key in seen_boards:
        if isinstance(key, tuple):
            out |= set(key)
    return out


def _run(ev, monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["eval_unseen.py", *args])
    assert ev.main() == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# -------------------------------------------------------- 안 쓴 변종으로 간다

def test_기본_변종창은_수집이_안_쓴_구간이다(ev, seen_boards, ckpt, tmp_path, monkeypatch,
                                            capsys):
    """★★★ 오프셋이 **조용히 0** 이 되면 바로 이 테스트가 죽는다.

    수집이 `--variants 4` 로 v0~v3 를 썼으므로 기본 평가 창은 **v4 부터**여야 한다.
    """
    summary = _run(ev, monkeypatch, capsys,
                   ["--ckpt", str(ckpt), "--stage", "stage3", "--eval-seeds", "1",
                    "--device", "cpu", "--out", str(tmp_path / "u.md")])
    assert summary["variant_offset"] == 4 and summary["seen_variants"] == 4
    names = _names(seen_boards)
    assert names == {f"course_{c}@v{k}" for c in "ABDEGH" for k in range(4, 8)}
    assert not any(n.endswith(("@v0", "@v1", "@v2", "@v3")) for n in names), \
        "수집이 본 변종으로 평가했다 — 선택 편향이 그대로 성적표로 간다"
    assert summary["stages"]["stage3"]["boards"] == 24


def test_겹치는_창은_거부한다(ev, ckpt, tmp_path, monkeypatch):
    """★ 실수로 겹치게 부르면 **돌기 전에** 막는다 — 다 돌고 나면 숫자만 멀쩡해 보인다."""
    monkeypatch.setattr(sys, "argv",
                        ["eval_unseen.py", "--ckpt", str(ckpt), "--stage", "stage3",
                         "--variant-offset", "0", "--device", "cpu"])
    with pytest.raises(SystemExit):
        ev.main()


def test_일부러_겹치게_하려면_명시해야_한다(ev, seen_boards, ckpt, tmp_path, monkeypatch,
                                            capsys):
    """선택용 숫자(= 편향이 든 쪽)를 일부러 다시 재는 길은 **이름이 붙어** 있어야 한다."""
    summary = _run(ev, monkeypatch, capsys,
                   ["--ckpt", str(ckpt), "--stage", "stage3", "--variant-offset", "0",
                    "--include-seen", "--eval-seeds", "1", "--device", "cpu",
                    "--out", str(tmp_path / "u.md")])
    assert summary["include_seen"] is True and summary["variant_offset"] == 0
    assert _names(seen_boards) == {f"course_{c}@v{k}" for c in "ABDEGH" for k in range(4)}


def test_수집_변종_수를_바꾸면_창도_따라간다(ev, seen_boards, ckpt, tmp_path, monkeypatch,
                                            capsys):
    summary = _run(ev, monkeypatch, capsys,
                   ["--ckpt", str(ckpt), "--stage", "stage3", "--seen-variants", "2",
                    "--variants", "2", "--eval-seeds", "1", "--device", "cpu",
                    "--out", str(tmp_path / "u.md")])
    assert summary["variant_offset"] == 2
    assert _names(seen_boards) == {f"course_{c}@v{k}" for c in "ABDEGH" for k in (2, 3)}


def test_액터_없는_단계는_변종을_접는다(ev, seen_boards, ckpt, tmp_path, monkeypatch, capsys):
    """변종은 액터만 흔든다 — 단계 ①② 는 변종 N 벌이 **글자 그대로 같은 판 N 벌**이다.

    접지 않으면 일만 N 배로 늘고 숫자는 그대로다(단계 ①② 는 긴 코스를 끝까지 모는 평가다).
    접었다는 사실은 요약과 표에 남는다.
    """
    summary = _run(ev, monkeypatch, capsys,
                   ["--ckpt", str(ckpt), "--stage", "stage1", "--eval-seeds", "1",
                    "--device", "cpu", "--out", str(tmp_path / "u.md")])
    assert summary["stages"]["stage1"]["boards"] == 6
    assert summary["stages"]["stage1"]["varied"] is False
    assert _names(seen_boards) == {f"course_{c}" for c in "ABDEGH"}


def test_단계3은_접지_않는다(ev, ckpt, tmp_path, monkeypatch, capsys):
    """위의 '접기' 규칙이 단계 ③ 까지 삼키면 이 도구 전체가 무의미해진다."""
    summary = _run(ev, monkeypatch, capsys,
                   ["--ckpt", str(ckpt), "--stage", "stage3", "--eval-seeds", "1",
                    "--device", "cpu", "--out", str(tmp_path / "u.md")])
    assert summary["stages"]["stage3"]["varied"] is True


# ------------------------------------------------- 무엇을 쟀는지 문서에 남는다

def test_성적표가_변종창과_판_수를_적는다(ev, ckpt, tmp_path, monkeypatch, capsys):
    out = tmp_path / "u.md"
    summary = _run(ev, monkeypatch, capsys,
                   ["--ckpt", str(ckpt), "--stage", "stage3", "--eval-seeds", "1",
                    "--device", "cpu", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "v4~v7" in text, "어느 변종으로 쟀는지가 성적표에 없다"
    assert "안 쓴 변종" in text and str(ckpt) in text
    assert "| 단계 ③ |" in text and "24" in text
    assert summary["ckpt"] == str(ckpt) and summary["report"] == str(out)


def test_없는_체크포인트는_아무것도_하기_전에_거부한다(ev, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval_unseen.py", "--ckpt", str(tmp_path / "nope.pt"),
                                      "--out", str(tmp_path / "u.md")])
    with pytest.raises(SystemExit):
        ev.main()
    assert not (tmp_path / "u.md").exists()


def test_dry_run_은_아무것도_안_돌리고_창만_찍는다(ev, ckpt, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eval_unseen.py", "--ckpt", str(ckpt),
                                      "--stage", "stage3", "--dry-run"])
    assert ev.main() == 0
    out = capsys.readouterr().out
    assert "v4~v7" in out and "24" in out
    assert "stage3" in out


def test_잘못된_인자는_거부한다(ev, ckpt, monkeypatch):
    for extra in (["--variants", "0"], ["--seen-variants", "-1"], ["--eval-seeds", "0"],
                  ["--stage", "stage9"]):
        monkeypatch.setattr(sys, "argv", ["eval_unseen.py", "--ckpt", str(ckpt),
                                          "--dry-run", *extra])
        with pytest.raises(SystemExit):
            ev.main()
