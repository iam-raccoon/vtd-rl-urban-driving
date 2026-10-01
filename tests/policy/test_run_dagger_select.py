"""`scripts/run_dagger.py` 의 **모델 선택** — 최선 라운드를 계산만 하고 버리던 결함.

M4k 실측(`runs/omen/2026-09-30-m4k-dagger-stage3/log.jsonl`): 라운드 3 이 단계 ① 83.3%,
라운드 4 가 16.7% 였는데 **라운드 4 가 나왔다.** `best` 는 계산됐지만 최종 평가는
`policy-r{last}.pt` 를 읽었고 요약도 `last` 였다 — 좋은 라운드를 만들고도 버린 것이다.

★ 그래서 이 파일의 핵심은 `pick_round()` 가 옳은 라운드를 **돌려주는가** 가 아니다.
그 라운드가 **실제로 밖으로 나오는가** 다 — 최종 평가가 읽는 체크포인트, 요약의 숫자,
`policy-best.pt` 의 내용까지 따라간다. 이 레포는 "도우미는 시험했는데 호출부가 그 도우미를
안 쓰는" 잠금 실패를 세 번 겪었다.
"""
import importlib.util
import json
import os
import re
import sys
import types

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "run_dagger.py")


@pytest.fixture(scope="module")
def rd():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다."""
    spec = importlib.util.spec_from_file_location("run_dagger_select_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ 도구

def _rows(stages: dict, scores: dict | None = None) -> list:
    """`{"stage1": [라운드별 완주율...], ...}` → 로그 행 목록."""
    n = len(next(iter(stages.values())))
    out = []
    for r in range(n):
        row = {"round": r, "beta": 0.0}
        for name, goals in stages.items():
            row[name] = {"goal_rate": goals[r],
                         "mean_score": (scores or {}).get(name, [98.6] * n)[r],
                         "mean_reward": 0.0}
        out.append(row)
    return out


def _fake_ev(goal_rate=1.0, mean_score=98.6):
    ep = types.SimpleNamespace(board="course_A", seed=0, outcome="goal", steps=100,
                               reward=0.0, score=mean_score, sheet=[])
    return {"goal_rate": goal_rate, "mean_score": mean_score, "mean_score_raw": mean_score,
            "mean_score_completed": mean_score, "mean_reward": 0.0, "episodes": [ep]}


class _FakeDataset:
    def __len__(self):
        return 8


MIN = ["--rounds", "5", "--seeds", "1", "--epochs", "1", "--eval-seeds", "1",
       "--workers", "1", "--device", "cpu"]


@pytest.fixture
def harness(rd, monkeypatch):
    """`main()` 을 통째로 돌리되 판·수집·학습·평가만 가짜로 바꾼다.

    바꾸지 **않는** 것이 이 파일의 대상이다 — 라운드 선택, 최종 평가가 **어느 체크포인트를
    읽는가**, 요약에 **어느 라운드 숫자가 들어가는가**, `policy-best.pt` 가 **어느 파일의
    복사본인가**. 체크포인트는 진짜로 저장하고 진짜로 다시 읽는다.
    """
    state = {"rnd": -1, "loaded": [], "goals": {}}

    monkeypatch.setattr(rd, "collect_targets",
                        lambda paths, smoke, variants: [("(fake)", "course_X")])
    monkeypatch.setattr(rd, "_collect_job", lambda job: {"outcome": "goal"})
    monkeypatch.setattr(rd, "_stall_start_count", lambda *args, **kw: 0)
    monkeypatch.setattr(rd, "load_dir", lambda data_dir: _FakeDataset())
    # 판 자리에 **단계 이름**을 넣는다 — 가짜 평가가 "어느 단계냐" 를 알아야 라운드마다
    # 단계별로 다른 숫자를 돌려줄 수 있다(선택 기준이 두 단계를 같이 본다).
    monkeypatch.setattr(rd, "eval_boards", lambda names, smoke: [(n, [n]) for n in names])

    def fake_train(net, dataset, cfg, device=None, ref=None):
        state["rnd"] += 1
        return {"epochs": cfg.epochs, "samples": len(dataset), "loss": 0.0, "control": 0.0,
                "turn": 0.0, "anchor": 0.0, "near_frac": 0.0, "seconds": 0.0}

    def fake_eval(net, boards, seeds=(0,)):
        rnd = net._pick_round if hasattr(net, "_pick_round") else state["rnd"]
        return _fake_ev(state["goals"][(rnd, boards[0])])

    monkeypatch.setattr(rd, "train_epochs", fake_train)
    monkeypatch.setattr(rd, "evaluate_policy", fake_eval)
    monkeypatch.setattr(rd, "evaluate_teacher", lambda boards, seeds=(0,): _fake_ev(1.0, 99.6))

    orig_load = rd.DrivePolicy.load

    def rec_load(path, device=None):
        """어느 체크포인트를 읽었는지 기록하고, 읽은 그물에 라운드 번호를 붙여 둔다."""
        state["loaded"].append(str(path))
        net = orig_load(path, device=device)
        m = re.search(r"policy-r(\d+)\.pt$", str(path))
        if m:
            net._pick_round = int(m.group(1))
        return net

    monkeypatch.setattr(rd.DrivePolicy, "load", staticmethod(rec_load))
    return state


def _set_goals(state, **by_stage):
    """`_set_goals(state, stage1=[...], stage3=[...])` — 값은 라운드 순서다."""
    state["goals"] = {(r, s): g for s, gs in by_stage.items() for r, g in enumerate(gs)}


def _run_main(rd, monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", *args])
    assert rd.main() == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# ------------------------------------------------------- pick_round 그 자체

def test_기본은_last라_예전과_같다(rd):
    """★ 과거 성적표가 전부 이 보장 위에 있다 — 아무것도 안 주면 마지막 라운드다."""
    rounds = _rows({"stage1": [0.2, 0.9, 0.1]})
    assert rd.pick_round(rounds, "last", None, ["stage1"], rd.SELECT_FLOOR)["round"] == 2
    # 기준을 줘도 `last` 면 기준을 아예 안 본다.
    assert rd.pick_round(rounds, "last", "retain-then-learn", ["stage1"], 0.8)["round"] == 2


def test_best는_최선_라운드를_고른다(rd):
    rounds = _rows({"stage1": [0.2, 0.9, 0.1]})
    assert rd.pick_round(rounds, "best", "primary", ["stage1"], rd.SELECT_FLOOR)["round"] == 1


def test_m4k_상황을_재현한다_r3를_버리지_않는다(rd):
    """★ M4k 실측: r3 83.3% · r4 16.7% 인데 r4 가 나왔다."""
    rounds = _rows({"stage1": [0.167, 0.167, 0.167, 0.833, 0.167],
                    "stage3": [0.0] * 5})
    got = rd.pick_round(rounds, "best", "retain-then-learn", ["stage1", "stage3"], rd.SELECT_FLOOR)
    assert got["round"] == 3
    # 문턱이 **실측값 83.3% 를 받아들여야** 한다 — 5/6 를 그대로 문턱에 쓰면 부동소수
    # 표현 하나로 이 라운드가 떨어진다.
    assert rd.SELECT_FLOOR <= 0.833


def test_선택기준은_단계1을_부순_라운드를_안_고른다(rd):
    """★ 단계 ③ 만 보면 ① 을 부순 라운드가 뽑힌다 — 문턱이 그것을 막는다."""
    rounds = _rows({"stage1": [1.0, 0.0], "stage3": [0.1, 0.3]})
    assert rd.pick_round(rounds, "best", "retain-then-learn",
                         ["stage1", "stage3"], 0.8)["round"] == 0


def test_문턱을_넘은_라운드가_하나도_없으면_유지부터_살린다(rd):
    """전부 부서졌으면 '단계 ③ 가 제일 높은 폐허' 가 아니라 **덜 부서진 쪽**을 고른다."""
    rounds = _rows({"stage1": [0.5, 0.167], "stage3": [0.0, 0.4]})
    assert rd.pick_round(rounds, "best", "retain-then-learn",
                         ["stage1", "stage3"], 0.8)["round"] == 0


def test_동점이면_더_뒤_라운드다(rd):
    """★ 동점 처리 — 뒤 라운드가 데이터가 더 많다. 두 기준 모두 같은 규칙이다."""
    tied = _rows({"stage1": [1.0, 1.0, 1.0], "stage3": [0.3, 0.3, 0.3]})
    assert rd.pick_round(tied, "best", "primary", ["stage1", "stage3"], 0.8)["round"] == 2
    assert rd.pick_round(tied, "best", "retain-then-learn",
                         ["stage1", "stage3"], 0.8)["round"] == 2
    # 학습 단계가 동점이면 **유지가 더 좋은** 쪽, 그다음이 더 뒤 라운드다.
    rounds = _rows({"stage1": [1.0, 0.9], "stage3": [0.3, 0.3]})
    assert rd.pick_round(rounds, "best", "retain-then-learn",
                         ["stage1", "stage3"], 0.8)["round"] == 0


def test_기준_기본값은_select_를_따라간다(rd):
    """`last` 는 예전 기준(`primary`), `best` 는 이 마일스톤 기준이다 — 둘을 섞지 않는다."""
    assert rd.DEFAULT_SELECT_METRIC["last"] == "primary"
    assert rd.DEFAULT_SELECT_METRIC["best"] == "retain-then-learn"
    rounds = _rows({"stage1": [1.0, 0.9], "stage3": [0.1, 0.9]})
    assert rd.pick_round(rounds, "best", None, ["stage1", "stage3"], 0.8)["round"] == 1


def test_모르는_선택값은_조용히_넘어가지_않는다(rd):
    rounds = _rows({"stage1": [1.0]})
    with pytest.raises(ValueError, match="select"):
        rd.pick_round(rounds, "latest", None, ["stage1"], 0.8)
    with pytest.raises(ValueError, match="select-metric"):
        rd.pick_round(rounds, "best", "goal", ["stage1"], 0.8)


def test_유지와_학습_단계는_평가_순서에서_온다(rd):
    """앞이 유지, **뒤가 학습**이다 — 뒤바꾸면 기준이 통째로 뒤집힌다."""
    assert rd.select_stages(["stage1", "stage2", "stage3"]) == ("stage1", "stage3")
    assert rd.select_stages(["stage1"]) == ("stage1", "stage1")


# ------------------------------------------- ★ 고른 라운드가 실제로 밖으로 나오는가

#: Task 3 과 같은 모양(평가 ①②③). r1 은 단계 ③ 가 제일 높지만 단계 ① 을 부쉈고,
#: r4(마지막)는 쓰레기다. **r3 가 답이다.**
E2E = {"stage1": [1.0, 0.167, 1.0, 1.0, 0.167],
       "stage2": [0.95, 0.20, 0.95, 0.95, 0.20],
       "stage3": [0.0, 0.5, 0.167, 0.333, 0.0]}
E2E_STAGES = ["--eval-stage", "stage1", "--eval-stage", "stage2", "--eval-stage", "stage3"]


def test_선택한_라운드가_요약과_체크포인트와_최종평가에_전부_반영된다(
        rd, harness, tmp_path, monkeypatch, capsys):
    """★★★ 지금 결함이 정확히 여기다 — `best` 를 **계산만** 하고 `last` 를 내보낸다.

    `pick_round()` 만 시험하면 이 결함을 못 잡는다. 그래서 고른 라운드를 **세 군데까지**
    따라간다: 최종 평가가 읽은 체크포인트 경로, 요약의 단계별 숫자, `policy-best.pt` 의 바이트.
    """
    _set_goals(harness, **E2E)
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(tmp_path / "r.md"),
                         "--select", "best", *MIN, *E2E_STAGES])

    assert summary["selected_round"] == 3, "고른 라운드가 요약에 안 적혔다"
    assert summary["select"] == "best" and summary["select_metric"] == "retain-then-learn"
    # ① 요약의 숫자가 **그 라운드의** 숫자여야 한다(마지막 라운드 r4 는 0.167/0.0 이다).
    assert summary["stage1"]["goal_rate"] == pytest.approx(1.0)
    assert summary["stage3"]["goal_rate"] == pytest.approx(0.333)
    assert summary["target_met"] is True, "목표 판정도 고른 라운드로 해야 한다(r4 면 미달이다)"
    # ② 최종 상세 평가가 **그 라운드 체크포인트**를 읽어야 한다.
    assert harness["loaded"], "최종 평가가 체크포인트를 아예 안 읽었다"
    assert harness["loaded"][-1].endswith(os.path.join("run", "policy-r3.pt")), \
        f"최종 평가가 엉뚱한 체크포인트를 읽었다: {harness['loaded'][-1]}"
    # ③ `policy-best.pt` 가 그 라운드 파일과 **바이트까지 같아야** 한다.
    best = out / rd.BEST_CKPT_NAME
    assert best.is_file() and not best.is_symlink(), "policy-best.pt 가 진짜 파일이 아니다"
    assert best.read_bytes() == (out / "policy-r3.pt").read_bytes()
    assert best.read_bytes() != (out / "policy-r4.pt").read_bytes(), \
        "policy-best.pt 가 마지막 라운드 복사본이다 — 바로 이 결함이다"


def test_기본_실행은_여전히_마지막_라운드를_낸다(rd, harness, tmp_path, monkeypatch, capsys):
    """★★ 기본값(`--select last`)이 바뀌면 과거 성적표가 전부 재현 불가가 된다."""
    _set_goals(harness, **E2E)
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(tmp_path / "r.md"), *MIN, *E2E_STAGES])
    assert summary["select"] == "last" and summary["selected_round"] == 4
    assert summary["stage3"]["goal_rate"] == pytest.approx(0.0)
    assert summary["target_met"] is False
    assert harness["loaded"][-1].endswith(os.path.join("run", "policy-r4.pt"))
    assert (out / rd.BEST_CKPT_NAME).read_bytes() == (out / "policy-r4.pt").read_bytes()


def test_기본_성적표_문구는_한_글자도_안_달라진다(rd, harness, tmp_path, monkeypatch, capsys):
    """★★ 기본 실행의 "가장 좋았던 라운드" 줄은 **예전 기준**(단계 ① 완주율 최대)이다.

    커밋된 M3 성적표가 그 줄을 들고 있다(`가장 좋았던 라운드: 4`). 여기서 기본 기준을
    새 기준으로 바꾸면 그 문서를 `--report-only` 로 다시 못 만든다. 그래서 두 기준이
    **서로 다른 답**을 내는 판을 일부러 만들어 확인한다.
    """
    # 예전 기준(단계 ① 최대) → r0. 새 기준(①≥문턱 중 ② 최대) → r1.
    _set_goals(harness, stage1=[1.0, 0.9], stage2=[0.1, 0.9])
    out, report = tmp_path / "run", tmp_path / "r.md"
    args = [a for a in MIN]
    args[args.index("--rounds") + 1] = "2"
    summary = _run_main(rd, monkeypatch, capsys, ["--out", str(out), "--report", str(report), *args])
    text = report.read_text(encoding="utf-8")
    assert "- 가장 좋았던 라운드: 0(" in text, "기본 기준이 예전과 달라졌다"
    assert "채택:" not in text, "기본 실행의 성적표에 없던 줄이 생겼다"
    assert summary["selected_round"] == 1, "그래도 채택은 마지막 라운드다"


def test_성적표가_무슨_기준으로_골랐는지_적는다(rd, harness, tmp_path, monkeypatch, capsys):
    """기준이 암묵적이면 안 된다 — 성적표와 요약이 **문장으로** 들고 있어야 한다."""
    _set_goals(harness, **E2E)
    out, report = tmp_path / "run", tmp_path / "r.md"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(report),
                         "--select", "best", *MIN, *E2E_STAGES])
    text = report.read_text(encoding="utf-8")
    assert "채택: 라운드 3" in text
    assert "retain-then-learn" in text
    assert "단계 ③" in summary["select_note"] and "단계 ①" in summary["select_note"]
    assert summary["select_note"] in text, "요약과 성적표가 서로 다른 기준을 말한다"
    assert summary["select_floor"] == pytest.approx(rd.SELECT_FLOOR)
    assert summary["best_ckpt"] == str(out / rd.BEST_CKPT_NAME)


def test_문턱을_인자로_옮길_수_있다(rd, harness, tmp_path, monkeypatch, capsys):
    """문턱을 0.1 까지 내리면 단계 ① 을 부순 r1(단계 ③ 0.5)도 통과해 **r1 이 뽑힌다**.

    기본 문턱(0.8)에서는 r3 였다 — 문턱이 실제로 선택을 가른다는 증거다(받기만 하고
    안 쓰는 인자가 아니다).
    """
    _set_goals(harness, **E2E)
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(tmp_path / "a"), "--report", str(tmp_path / "a.md"),
                         "--select", "best", "--select-floor", "0.1", *MIN, *E2E_STAGES])
    assert summary["selected_round"] == 1 and summary["select_floor"] == pytest.approx(0.1)


def test_기준을_직접_주면_그것을_쓴다(rd, harness, tmp_path, monkeypatch, capsys):
    """`--select best --select-metric primary` 는 단계 ① 만 본다 — 동점이라 더 뒤인 r3."""
    _set_goals(harness, **E2E)
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(tmp_path / "b"), "--report", str(tmp_path / "b.md"),
                         "--select", "best", "--select-metric", "primary", *MIN, *E2E_STAGES])
    assert summary["select_metric"] == "primary" and summary["selected_round"] == 3


def test_잘못된_문턱은_거부한다(rd, tmp_path, monkeypatch):
    for bad in ("-0.1", "1.5"):
        monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--out", str(tmp_path / "x"),
                                          "--select-floor", bad, *MIN])
        with pytest.raises(SystemExit):
            rd.main()


def test_보고용_재생성도_고른_라운드를_쓴다(rd, harness, tmp_path, monkeypatch, capsys):
    """`--report-only` 는 로그만 보고 성적표를 다시 만든다 — 거기서도 선택이 살아 있어야 한다.

    (성적표를 다시 만드는 길이 선택을 안 타면, 같은 로그에서 서로 다른 성적표가 나온다.)
    """
    _set_goals(harness, **E2E)
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(tmp_path / "r.md"),
               "--select", "best", *MIN, *E2E_STAGES])
    again = _run_main(rd, monkeypatch, capsys,
                      ["--out", str(out), "--report", str(tmp_path / "again.md"),
                       "--report-only", "--select", "best", *MIN, *E2E_STAGES])
    assert again["selected_round"] == 3
    assert again["stage3"]["goal_rate"] == pytest.approx(0.333)
    assert harness["loaded"][-1].endswith(os.path.join("run", "policy-r3.pt"))


def test_best_체크포인트는_덮어써도_항상_고른_라운드다(rd, harness, tmp_path, monkeypatch,
                                                      capsys):
    """두 번째 실행이 다른 라운드를 고르면 `policy-best.pt` 도 따라가야 한다 —

    남아 있던 옛 복사본을 그대로 두면 뒤 단계가 **지난 실행의 그물**을 집어 간다.
    """
    _set_goals(harness, **E2E)
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(tmp_path / "r.md"),
               "--select", "best", *MIN, *E2E_STAGES])
    assert (out / rd.BEST_CKPT_NAME).read_bytes() == (out / "policy-r3.pt").read_bytes()
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(tmp_path / "r2.md"),
               "--report-only", *MIN, *E2E_STAGES])        # 이번엔 기본값(last)
    assert (out / rd.BEST_CKPT_NAME).read_bytes() == (out / "policy-r4.pt").read_bytes()


def test_저장한_체크포인트들이_서로_다르다(rd, harness, tmp_path, monkeypatch, capsys):
    """위 바이트 비교가 뜻을 가지려면 라운드별 체크포인트가 실제로 달라야 한다.

    전부 같으면 `policy-best.pt == policy-r3.pt` 는 아무것도 증명하지 않는다.
    """
    _set_goals(harness, **E2E)
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(tmp_path / "r.md"), *MIN, *E2E_STAGES])
    blobs = {(out / f"policy-r{r}.pt").read_bytes() for r in range(5)}
    assert len(blobs) == 5, "라운드별 체크포인트가 구별되지 않는다 — 바이트 비교가 공허하다"
    assert torch.load(str(out / "policy-r0.pt"), map_location="cpu",
                      weights_only=False)["state"] is not None
