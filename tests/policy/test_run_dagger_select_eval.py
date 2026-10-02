"""`scripts/run_dagger.py` 의 **선택용 평가**(`--select-variants`)와 **β 스케줄**(`--betas`) — M4s.

M4r(`docs/reports/m4r-anchor3.md`)가 찾은 두 결함:
① 라운드 선택이 **원본 판 18 판**(독립 판 6 개, 해상도 16.7 점)으로 이뤄져 잡음을 고른다 —
   원본 판 ③a 50% 대 6% 였던 두 라운드가 안 쓴 판 72 판에서는 29.2% 대 29.2% 였다.
② β 0.1·0 라운드가 실패 주행으로 데이터를 채운다.

그래서 라운드마다 **선택 창**(기본 `v{2V}~v{3V-1}`)으로 다시 재서 고르고, β 를 인자로 연다.

★ 이 파일이 잠그는 것
* **기본값은 예전과 바이트까지 같다** — 고치기 전 스크립트(`BASE_SHA`)를 git 에서 꺼내 같은
  가짜 판 위에서 나란히 돌리고 성적표·로그·`policy-best.pt` 를 바이트로 맞춘다.
* 창 셋(수집 `[0,V)`·보고 `[V,2V)`·선택 `[K,K+N)`)이 겹치면 **아무것도 하기 전에** 거부한다.
* 선택 창 숫자가 원본 판 숫자와 **다른 라운드를 가리킬 때** 선택 창을 따른다 — `pick_round()`
  의 반환값이 아니라 **`policy-best.pt`·최종 평가 체크포인트·요약·성적표까지** 따라간다.
  (`test_run_dagger_select.py` 의 역사: 도우미만 시험해서 돌연변이 두 개가 살아남았다.)
* 액터 없는 단계(①②)는 선택 창을 켜도 **원본 판 숫자**를 쓴다(다시 재지 않는다).
"""
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "run_dagger.py")

#: `--betas`·`--select-variants` 가 들어오기 **직전** 커밋(M4s 계획서). 기본값 비교의 기준선이다.
BASE_SHA = "924ef6c"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rd():
    return _load(SCRIPT, "run_dagger_select_eval_under_test")


@pytest.fixture(scope="module")
def rd_old(tmp_path_factory):
    """고치기 전 `run_dagger.py` — git 에서 꺼낸다(레포에 사본을 두지 않는다)."""
    try:
        src = subprocess.run(["git", "-C", REPO, "show", f"{BASE_SHA}:scripts/run_dagger.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git 에서 기준선 스크립트({BASE_SHA})를 못 꺼냈다: {exc}")
    path = tmp_path_factory.mktemp("old") / f"run_dagger_{BASE_SHA}.py"
    path.write_text(src, encoding="utf-8")
    module = _load(str(path), f"run_dagger_{BASE_SHA}")
    module.REPO = REPO          # 커리큘럼 경로를 레포에서 찾게(파일이 임시 폴더에 있다)
    return module


# ------------------------------------------------------------------ 도구

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
THREE = ["stage1", "stage3a", "stage3b"]
THREE_ARGS = ["--eval-stage", "stage1", "--eval-stage", "stage3a", "--eval-stage", "stage3b"]
#: M4s 본 실행과 같은 모양 — 단계 ③b 수집, 변종 4 벌.
COLLECT = ["--stage", "stage3b", "--variants", "4"]
SEL = "@sel"        # 가짜 선택 창 판 꼬리표 — 가짜 평가가 "선택 창이냐" 를 이것으로 안다


def _install(module, monkeypatch, state, fake_window=True):
    """판·수집·학습·평가만 가짜로 바꾼다. 선택·체크포인트 저장/읽기·성적표는 진짜다.

    원본 판 평가는 판 자리에 **단계 이름**, 선택 창 평가는 **`단계@sel`** 을 넣어 가짜
    평가가 둘을 구별한다. 시간은 0 으로 고정한다(성적표·로그를 바이트로 맞추려고).
    """
    monkeypatch.setattr(module, "time", types.SimpleNamespace(perf_counter=lambda: 0.0))
    monkeypatch.setattr(module, "collect_targets",
                        lambda paths, smoke, variants: [("(fake)", "course_X")])

    def fake_collect(job):
        state["betas"].append((job[6], job[3]))       # (라운드, β) — 수집에 실제로 실린 β
        return {"outcome": "goal"}

    monkeypatch.setattr(module, "_collect_job", fake_collect)
    monkeypatch.setattr(module, "_stall_start_count", lambda *args, **kw: 0)
    monkeypatch.setattr(module, "load_dir", lambda data_dir: _FakeDataset())
    monkeypatch.setattr(module, "eval_boards", lambda names, smoke: [(n, [n]) for n in names])
    if hasattr(module, "select_window_boards"):
        if fake_window:
            def fake_window_boards(names, n, offset):
                state["window_args"].append((tuple(names), n, offset))
                return [(s, [s + SEL], module.has_actors(s)) for s in names]
        else:
            def fake_window_boards(names, n, offset):
                raise AssertionError("기본 실행이 선택 창 판을 지었다")
        monkeypatch.setattr(module, "select_window_boards", fake_window_boards)

    def fake_train(net, dataset, cfg, device=None, ref=None):
        state["rnd"] += 1
        return {"epochs": cfg.epochs, "samples": len(dataset), "loss": 0.0, "control": 0.0,
                "turn": 0.0, "anchor": 0.0, "near_frac": 0.0, "seconds": 0.0}

    def fake_eval(net, boards, seeds=(0,)):
        rnd = getattr(net, "_pick_round", state["rnd"])
        state["evals"].append((rnd, boards[0], tuple(seeds)))
        return _fake_ev(state["goals"][(rnd, boards[0])])

    monkeypatch.setattr(module, "train_epochs", fake_train)
    monkeypatch.setattr(module, "evaluate_policy", fake_eval)
    monkeypatch.setattr(module, "evaluate_teacher", lambda boards, seeds=(0,): _fake_ev(1.0, 99.6))

    orig_load = module.DrivePolicy.load

    def rec_load(path, device=None):
        state["loaded"].append(str(path))
        net = orig_load(path, device=device)
        import re
        m = re.search(r"policy-r(\d+)\.pt$", str(path))
        if m:
            net._pick_round = int(m.group(1))
        return net

    monkeypatch.setattr(module.DrivePolicy, "load", staticmethod(rec_load))


def _state():
    return {"rnd": -1, "loaded": [], "goals": {}, "betas": [], "evals": [], "window_args": []}


@pytest.fixture
def harness(rd, monkeypatch):
    state = _state()
    _install(rd, monkeypatch, state)
    return state


def _set_goals(state, sel=None, **by_stage):
    """원본 판 숫자는 `stage1=[...]`, 선택 창 숫자는 `sel={"stage3b": [...]}` — 값은 라운드 순서."""
    goals = {(r, s): g for s, gs in by_stage.items() for r, g in enumerate(gs)}
    for s, gs in (sel or {}).items():
        goals.update({(r, s + SEL): g for r, g in enumerate(gs)})
    state["goals"] = goals


def _run_main(module, monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", *args])
    assert module.main() == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _rounds_arg(n):
    args = list(MIN)
    args[args.index("--rounds") + 1] = str(n)
    return args


def _log(out):
    with open(out / "log.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _init_ckpt(tmp_path):
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig
    path = tmp_path / "init.pt"
    DrivePolicy(PolicyConfig(trunk=(8, 8), squash=True)).save(str(path))
    return path


# ============================================================ ① β 스케줄

def test_기본_β_스케줄은_예전_그대로다(rd):
    """★ `--betas` 를 안 주면 **지금 값 그대로**, 목록 끝을 넘으면 0.0 — 예전 식과 하나하나 맞춘다."""
    assert rd.BETAS == [1.0, 0.5, 0.25, 0.1, 0.0]
    betas = rd.parse_betas(None)
    assert betas == [1.0, 0.5, 0.25, 0.1, 0.0]
    assert betas is not rd.BETAS, "기본 목록을 그대로 내주면 누가 고쳐 쓸 때 상수가 바뀐다"
    for rnd in range(9):
        old = rd.BETAS[rnd] if rnd < len(rd.BETAS) else 0.0          # 고치기 전 `:674` 의 식
        assert rd.beta_at(betas, rnd) == old


def test_betas_를_읽는다(rd):
    assert rd.parse_betas("1.0,0.5,0.5,0.5,0.5") == [1.0, 0.5, 0.5, 0.5, 0.5]
    assert rd.parse_betas(" 1 , 0.25 ") == [1.0, 0.25]
    assert rd.parse_betas("0") == [0.0]
    assert rd.parse_betas("0,1") == [0.0, 1.0], "경계값 0·1 은 받아야 한다"


@pytest.mark.parametrize("bad", ["1.5", "-0.1", "nan", "inf", "a", "1.0,,0.5", "", "1.0,"])
def test_betas_범위_밖이나_빈칸은_거부한다(rd, bad):
    with pytest.raises(ValueError, match="--betas"):
        rd.parse_betas(bad)


def test_betas_가_짧으면_넘친_라운드는_0이다(rd):
    """목록이 라운드보다 짧으면 **예전 `BETAS` 와 같은 규칙**(0.0)이다."""
    assert [rd.beta_at([0.7, 0.6], r) for r in range(5)] == [0.7, 0.6, 0.0, 0.0, 0.0]


def test_잘못된_betas_는_dry_run_에서_거부한다(rd, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--dry-run", "--betas", "1.0,1.5"])
    with pytest.raises(SystemExit):
        rd.main()
    assert capsys.readouterr().out == "", "거부했는데 dry-run 줄을 찍었다"


@pytest.mark.parametrize("betas,rounds,want", [
    ("1.0,0.5,0.5,0.5,0.5", 5, [1.0, 0.5, 0.5, 0.5, 0.5]),
    ("0.7,0.6", 5, [0.7, 0.6, 0.0, 0.0, 0.0]),
    (None, 7, [1.0, 0.5, 0.25, 0.1, 0.0, 0.0, 0.0]),
], ids=["m4s", "short", "default_long"])
def test_betas_가_수집과_로그와_성적표에_실제로_흘러간다(rd, harness, tmp_path, monkeypatch,
                                                      capsys, betas, rounds, want):
    """★★ 인자를 **받기만 하고 안 쓰는** 꼴을 잡는다 — 수집 작업에 실린 β, 로그의 β, 성적표의 β 칸."""
    _set_goals(harness, stage1=[1.0] * rounds, stage2=[1.0] * rounds)
    out, report = tmp_path / "run", tmp_path / "r.md"
    extra = ["--betas", betas] if betas is not None else []
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(report), *_rounds_arg(rounds), *extra])
    assert [b for _r, b in sorted(harness["betas"])] == want, "수집에 실린 β 가 스케줄과 다르다"
    assert [r["beta"] for r in _log(out)] == want, "로그의 β 가 스케줄과 다르다"
    text = report.read_text(encoding="utf-8")
    for r, b in enumerate(want):
        assert f"\n| {r} | {b} | " in text, f"성적표 표의 라운드 {r} β 칸이 {b} 가 아니다"
    if betas is None:
        assert "betas" not in summary and "β 스케줄" not in text, "기본 실행에 없던 줄이 생겼다"
    else:
        assert summary["betas"] == want
        assert f"β 스케줄(`--betas {betas}`)" in text


# ============================================ ② 기본값은 예전 스크립트와 바이트까지 같다

def _scenarios(tmp_path):
    init = _init_ckpt(tmp_path)
    m4r = {"stage1": [1.0] * 5, "stage3a": [0.0, 6 / 18, 9 / 18, 1 / 18, 1 / 18],
           "stage3b": [0.0] * 5}
    return {
        "default": ({"stage1": [1.0, 0.167, 1.0, 1.0, 0.167],
                     "stage2": [0.95, 0.2, 0.95, 0.95, 0.2]}, list(MIN)),
        "m4r_like": (m4r, [*MIN, *COLLECT, *THREE_ARGS, "--select", "best", "--init", str(init),
                           "--anchor-coef", "3", "--train-seed", "2"]),
        "past_betas": ({"stage1": [1.0] * 7, "stage2": [0.5] * 7}, _rounds_arg(7)),
    }


@pytest.mark.parametrize("name", ["default", "m4r_like", "past_betas"])
def test_기본값이면_예전_스크립트와_바이트까지_같다(rd, rd_old, tmp_path, monkeypatch, capsys,
                                                name):
    """★★★ `--betas` 없음·`--select-variants 0` 이면 성적표·로그·체크포인트·요약이 **고치기 전과 같다.**

    같은 가짜 판 위에서 옛 스크립트와 새 스크립트를 **같은 산출물 경로**로 차례로 돌린다
    (성적표가 경로를 글자로 들고 있어서). 그다음 `--report-only` 재생성도 맞춘다.
    요약 JSON 에는 새 열쇠 `select_source` 하나만 더해진다(값 `"original"`).
    """
    goals, args = _scenarios(tmp_path)[name]
    out, report = tmp_path / "run", tmp_path / "r.md"
    got = {}
    for tag, module in (("old", rd_old), ("new", rd)):
        if out.exists():
            import shutil
            shutil.rmtree(out)
        state = _state()
        with monkeypatch.context() as mp:
            _install(module, mp, state, fake_window=False)
            _set_goals(state, **goals)
            summary = _run_main(module, mp, capsys,
                                ["--out", str(out), "--report", str(report), *args])
            again = _run_main(module, mp, capsys,
                              ["--out", str(out), "--report", str(tmp_path / "again.md"),
                               "--report-only", *args])
        got[tag] = {"summary": summary, "again": again, "betas": state["betas"],
                    "report": report.read_bytes(),
                    "again_md": (tmp_path / "again.md").read_bytes(),
                    "log": (out / "log.jsonl").read_bytes(),
                    "best": (out / "policy-best.pt").read_bytes(),
                    "loaded": state["loaded"]}
    old, new = got["old"], got["new"]
    assert new["report"] == old["report"], "기본 실행의 성적표가 바뀌었다"
    assert new["again_md"] == old["again_md"], "--report-only 재생성 성적표가 바뀌었다"
    assert new["log"] == old["log"], "기본 실행의 log.jsonl 이 바뀌었다"
    assert new["best"] == old["best"], "기본 실행의 policy-best.pt 가 바뀌었다"
    assert new["betas"] == old["betas"], "수집에 실린 β 가 바뀌었다"
    assert new["loaded"] == old["loaded"], "읽은 체크포인트 순서가 바뀌었다"
    for s_new, s_old in ((new["summary"], old["summary"]), (new["again"], old["again"])):
        assert s_new.pop("select_source") == "original"
        assert s_new == s_old, "요약 JSON 의 기존 열쇠가 바뀌었다"


def test_기본_dry_run_은_예전_줄을_그대로_두고_뒤에만_더한다(rd, rd_old, monkeypatch, capsys):
    """dry-run 은 산출물이 아니지만, 예전 줄은 한 글자도 안 바뀌고 **뒤에만** 붙어야 한다."""
    args = ["--dry-run", *COLLECT, *THREE_ARGS, "--select", "best"]
    outs = {}
    for tag, module in (("old", rd_old), ("new", rd)):
        monkeypatch.setattr(sys, "argv", ["run_dagger.py", *args])
        assert module.main() == 0
        outs[tag] = capsys.readouterr().out.splitlines()
    assert outs["new"][:len(outs["old"])] == outs["old"]
    extra = outs["new"][len(outs["old"]):]
    assert extra[0] == "β 스케줄(기본): r0=1 · r1=0.5 · r2=0.25 · r3=0.1 · r4=0"
    assert "선택 없음 — 원본 판 라운드 평가로 고른다" in extra[1], extra


# ============================================================ ③ 창 셋

def test_창_셋의_기본값(rd):
    assert rd.variant_windows(4, 4, 8) == {"collect": (0, 4), "report": (4, 8), "select": (8, 12)}
    assert rd.variant_windows(4, 0, 8)["select"] is None


@pytest.mark.parametrize("variants,n,k", [(4, 4, 0), (4, 4, 3), (4, 1, 0), (2, 2, 1)])
def test_수집_창과_겹치면_거부한다(rd, variants, n, k):
    with pytest.raises(ValueError) as exc:
        rd.check_windows(variants, n, k)
    msg = str(exc.value)
    assert "수집 창" in msg and f"v0~v{variants - 1}" in msg, msg
    assert f"선택 창 {rd.window_label(n, k)}" in msg, msg


@pytest.mark.parametrize("variants,n,k", [(4, 4, 4), (4, 4, 7), (4, 1, 7), (4, 2, 5)])
def test_보고_창과_겹치면_거부한다(rd, variants, n, k):
    with pytest.raises(ValueError) as exc:
        rd.check_windows(variants, n, k)
    msg = str(exc.value)
    assert f"보고 창 v{variants}~v{2 * variants - 1}" in msg, msg
    assert f"선택 창 {rd.window_label(n, k)}" in msg, msg


@pytest.mark.parametrize("variants,n,k", [(4, 4, 8), (1, 1, 2), (4, 10, 8), (2, 4, 4), (4, 0, 0)])
def test_안_겹치는_창은_받아들인다(rd, variants, n, k):
    """기본 K=2V 는 보고 창 바로 뒤라 붙어 있지만 안 겹친다. N=0 이면 선택 창이 없다."""
    rd.check_windows(variants, n, k)


def test_기본_오프셋은_2V_다(rd, monkeypatch, capsys):
    """★ 기본 오프셋이 조용히 0(수집 창)이나 V(보고 창)가 되면 여기서 죽는다."""
    for variants, want in (("4", "선택 v8~v11"), ("2", "선택 v4~v5")):
        monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--dry-run", "--stage", "stage3b",
                                          "--variants", variants, *THREE_ARGS,
                                          "--select", "best", "--select-variants", variants])
        assert rd.main() == 0
        out = capsys.readouterr().out
        assert want in out, out[-800:]
        assert f"수집 v0~v{int(variants) - 1}" in out


@pytest.mark.parametrize("offset,window", [("0", "수집 창 v0~v3"), ("4", "보고 창 v4~v7"),
                                           ("6", "보고 창 v4~v7")])
def test_겹치면_아무것도_하기_전에_거부한다(rd, harness, tmp_path, monkeypatch, capsys, offset,
                                          window):
    """★★ dry-run 이든 본 실행이든 **돌기 전에** 막는다 — 다 돌고 나면 숫자만 멀쩡해 보인다."""
    common = [*COLLECT, *THREE_ARGS, "--select", "best", "--select-variants", "4",
              "--select-variant-offset", offset]
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--dry-run", *common])
    with pytest.raises(SystemExit):
        rd.main()
    cap = capsys.readouterr()
    assert cap.out == "", "거부했는데 dry-run 줄을 찍었다"
    assert window in cap.err and "선택 창" in cap.err, cap.err[-600:]

    out = tmp_path / "should-not-exist"
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--out", str(out), *MIN, *common])
    with pytest.raises(SystemExit):
        rd.main()
    assert not out.exists(), "거부했는데 산출물 폴더를 만들었다"
    assert harness["betas"] == [] and harness["evals"] == [], "거부했는데 일을 했다"


@pytest.mark.parametrize("extra,needle", [
    (["--select-variants", "4"], "--select best"),                         # --select last
    (["--select", "best", "--select-variant-offset", "8"], "--select-variants"),
    (["--select", "best", "--select-eval-seeds", "5"], "--select-variants"),
    (["--select", "best", "--select-variants", "-1"], "--select-variants"),
    (["--select", "best", "--select-variants", "4", "--select-eval-seeds", "0"],
     "--select-eval-seeds"),
])
def test_뜻없는_선택_창_인자는_거부한다(rd, monkeypatch, capsys, extra, needle):
    """켰다고 믿고 몇 시간 돌리는 꼴(M4e)을 막는다 — 아무 일도 안 하는 조합은 거부한다."""
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--dry-run", *COLLECT, *THREE_ARGS, *extra])
    with pytest.raises(SystemExit):
        rd.main()
    assert needle in capsys.readouterr().err


def test_액터_있는_평가_단계가_없으면_선택_창을_거부한다(rd, monkeypatch, capsys):
    """전부 접히면 선택 창 숫자가 원본 판 숫자와 같다 — 켠 것이 아무 일도 안 한다."""
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--dry-run", "--select", "best",
                                      "--select-variants", "4"])          # 평가 기본 ①②
    with pytest.raises(SystemExit):
        rd.main()
    assert "액터" in capsys.readouterr().err


def test_dry_run_이_창_셋과_단계별_선택_평가를_찍는다(rd, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--dry-run", *COLLECT, *THREE_ARGS,
                                      "--select", "best", "--select-variants", "4",
                                      "--eval-seeds", "3", "--betas", "1.0,0.5,0.5,0.5,0.5"])
    assert rd.main() == 0
    out = capsys.readouterr().out
    assert ("변종 창 — 수집 v0~v3(--variants 4) · 보고 v4~v7(eval_unseen 기본 --seen-variants 4)"
            " · 선택 v8~v11(--select-variants 4)") in out
    assert "선택 평가 stage1: 액터 0" in out
    assert "선택 평가 stage3a: 선택 창 판 24개 × 평가 시드 3개 = 72 판/라운드" in out
    assert "선택 평가 stage3b: 선택 창 판 24개 × 평가 시드 3개 = 72 판/라운드" in out
    assert "β 스케줄(--betas): r0=1 · r1=0.5 · r2=0.5 · r3=0.5 · r4=0.5" in out


# =================================== ④ 선택 창이 원본 판과 다른 라운드를 가리키면 선택 창을 따른다

#: 원본 판(18 판)은 r1(정지차 1/18)을 고르고, 선택 창(72 판)은 r3(5/72)을 고른다. 마지막 r4 는 둘 다 아니다.
DISAGREE = {"stage1": [1.0] * 5,
            "stage3a": [0.0, 6 / 18, 9 / 18, 1 / 18, 1 / 18],
            "stage3b": [0.0, 1 / 18, 0.0, 0.0, 0.0]}
DISAGREE_SEL = {"stage3a": [0.0, 20 / 72, 21 / 72, 22 / 72, 20 / 72],
                "stage3b": [0.0, 0.0, 2 / 72, 5 / 72, 1 / 72]}
SEL_ARGS = [*COLLECT, *THREE_ARGS, "--select", "best", "--select-variants", "4"]


def test_대조군_원본_판으로_고르면_r1_이다(rd, harness, tmp_path, monkeypatch, capsys):
    """아래 시험이 뜻을 가지려면 두 평가가 **정말 다른 라운드**를 가리켜야 한다."""
    _set_goals(harness, sel=DISAGREE_SEL, **DISAGREE)
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(tmp_path / "r.md"), *MIN,
                         *COLLECT, *THREE_ARGS, "--select", "best"])
    assert summary["selected_round"] == 1 and summary["select_source"] == "original"
    assert (out / rd.BEST_CKPT_NAME).read_bytes() == (out / "policy-r1.pt").read_bytes()
    assert all(SEL not in b for _r, b, _s in harness["evals"]), "기본 실행이 선택 창을 쟀다"
    assert all("select_eval" not in r for r in _log(out)), "기본 실행 로그에 선택 창 열쇠가 생겼다"


def test_선택_창_숫자로_고르고_그_라운드가_밖으로_나온다(rd, harness, tmp_path, monkeypatch,
                                                      capsys):
    """★★★ 선택 창 평가를 **해 놓고 원본 판으로 고르는** 꼴('계산은 best, 저장은 last' 의 쌍둥이)을 잡는다.

    요약의 채택 라운드·숫자, 최종 평가가 읽은 체크포인트, `policy-best.pt` 의 바이트, 성적표의
    채택 줄·'가장 좋았던 라운드' 줄·선택 창 표, 로그의 열쇠까지 따라간다.
    """
    _set_goals(harness, sel=DISAGREE_SEL, **DISAGREE)
    out, report = tmp_path / "run", tmp_path / "r.md"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(report), *MIN, *SEL_ARGS])
    # 요약 — 고른 라운드, 고른 평가, 그 라운드의 두 숫자(열쇠가 다르다)
    assert summary["selected_round"] == 3, "선택 창이 아니라 원본 판으로 골랐다"
    assert summary["select_source"] == "v8~v11"
    sel = summary["select_eval"]
    assert sel["window"] == "v8~v11" and sel["offset"] == 8 and sel["variants"] == 4
    assert sel["stages"]["stage3b"]["goal_rate"] == pytest.approx(5 / 72)
    assert summary["stage3b"]["goal_rate"] == pytest.approx(0.0), \
        "요약의 단계 열쇠는 원본 판 숫자여야 한다(선택 창 숫자는 select_eval 에)"
    # 체크포인트
    assert harness["loaded"][-1].endswith(os.path.join("run", "policy-r3.pt")), \
        f"최종 평가가 엉뚱한 체크포인트를 읽었다: {harness['loaded'][-1]}"
    best = out / rd.BEST_CKPT_NAME
    assert best.read_bytes() == (out / "policy-r3.pt").read_bytes()
    assert best.read_bytes() != (out / "policy-r1.pt").read_bytes(), \
        "policy-best.pt 가 원본 판 최선(r1) 복사본이다 — 선택 창을 안 썼다"
    assert best.read_bytes() != (out / "policy-r4.pt").read_bytes()
    # 성적표
    text = report.read_text(encoding="utf-8")
    assert "**채택: 라운드 3**" in text
    assert "고른 평가: **선택 창 `v8~v11`**" in text
    adopt = next(line for line in text.splitlines() if "채택: 라운드" in line)
    assert "선택 창 `v8~v11`" in adopt, "채택 줄이 어느 평가로 골랐는지 안 적는다"
    assert "- 가장 좋았던 라운드: 3(" in text, "'가장 좋았던 라운드' 가 원본 판으로 골랐다"
    assert summary["select_note"] in text
    assert "### 선택 창 평가(라운드별)" in text
    assert "| 3 | 0.1 | 100.0%(원본) | 98.6 | 30.6% | 98.6 | 6.9% | 98.6 |" in text
    # 로그 — 원본 판 숫자는 단계 열쇠, 선택 창 숫자는 **다른 열쇠**
    log = _log(out)
    for r, row in enumerate(log):
        assert row["stage3b"]["goal_rate"] == pytest.approx(DISAGREE["stage3b"][r])
        st = row["select_eval"]["stages"]
        assert row["select_eval"]["window"] == "v8~v11"
        assert st["stage3b"]["goal_rate"] == pytest.approx(DISAGREE_SEL["stage3b"][r])
        assert st["stage3a"]["goal_rate"] == pytest.approx(DISAGREE_SEL["stage3a"][r])
        assert st["stage3b"]["folded"] is False and st["stage1"]["folded"] is True


def test_중간_단계_동점_처리도_선택_창_숫자로_한다(rd, harness, tmp_path, monkeypatch, capsys):
    """★ `9652dfd` 의 중간 단계 동점 처리가 **선택 창 숫자**를 먹는다 — 알고리즘을 둘로 가르지 않는다.

    학습 단계(③b)가 양쪽 다 0 으로 평평하다. 원본 판 ③a 는 r2(9/18), 선택 창 ③a 는 r1(30/72).
    """
    _set_goals(harness,
               sel={"stage3a": [0.0, 30 / 72, 20 / 72, 21 / 72, 10 / 72], "stage3b": [0.0] * 5},
               stage1=[1.0] * 5, stage3a=[0.0, 6 / 18, 9 / 18, 1 / 18, 1 / 18],
               stage3b=[0.0] * 5)
    out, report = tmp_path / "run", tmp_path / "r.md"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(report), *MIN, *SEL_ARGS])
    assert summary["selected_round"] == 1
    assert summary["select_eval"]["stages"]["stage3a"]["goal_rate"] == pytest.approx(30 / 72)
    assert (out / rd.BEST_CKPT_NAME).read_bytes() == (out / "policy-r1.pt").read_bytes()
    assert "(동점이면 stage3a 완주율, 그다음 단계 ①, 그다음 더 뒤 라운드)" in summary["select_note"]
    assert "**채택: 라운드 1**" in report.read_text(encoding="utf-8")


def test_액터_없는_단계는_선택_창을_켜도_원본_판_숫자를_쓴다(rd, harness, tmp_path, monkeypatch,
                                                          capsys):
    """★★ 단계 ① 은 다시 재지 않는다 — 가짜 평가에 `stage1@sel` 을 **엉뚱한 숫자**로 심어 둔다.

    그 숫자를 쓰면 r1~r4 가 유지 문턱(80%) 밑으로 떨어져 r0 이 뽑힌다. 원본 판 숫자(전부
    100%)를 쓰면 선택 창 ③b 최대인 r3 다.
    """
    _set_goals(harness,
               sel={"stage1": [1.0, 0.0, 0.0, 0.0, 0.0], **DISAGREE_SEL}, **DISAGREE)
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(tmp_path / "r.md"), *MIN, *SEL_ARGS])
    assert not any(b == "stage1" + SEL for _r, b, _s in harness["evals"]), \
        "액터 없는 단계를 선택 창으로 다시 쟀다"
    assert summary["selected_round"] == 3
    assert (out / rd.BEST_CKPT_NAME).read_bytes() == (out / "policy-r3.pt").read_bytes()
    for row in _log(out):
        st1 = row["select_eval"]["stages"]["stage1"]
        assert st1["folded"] is True and st1["window"] == "original"
        assert st1["goal_rate"] == row["stage1"]["goal_rate"]
        assert st1["mean_score"] == row["stage1"]["mean_score"]


def test_main_이_선택_창에_K_와_N_과_시드를_넘긴다(rd, harness, tmp_path, monkeypatch, capsys):
    """기본 K=2V(=8), 직접 준 K, 그리고 `--select-eval-seeds` 가 실제 평가 시드가 되는지."""
    _set_goals(harness, sel=DISAGREE_SEL, **DISAGREE)
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "a"), "--report", str(tmp_path / "a.md"), *MIN, *SEL_ARGS])
    assert set(harness["window_args"]) == {(tuple(THREE), 4, 8)}
    sel_seeds = {s for _r, b, s in harness["evals"] if b.endswith(SEL)}
    assert sel_seeds == {(0,)}, "선택 창 평가 시드 기본은 --eval-seeds(여기선 1) 다"

    harness["window_args"].clear()
    harness["evals"].clear()
    harness["rnd"] = -1
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(tmp_path / "b"), "--report", str(tmp_path / "b.md"), *MIN,
                         *SEL_ARGS, "--select-variant-offset", "12", "--select-eval-seeds", "2"])
    assert set(harness["window_args"]) == {(tuple(THREE), 4, 12)}
    assert {s for _r, b, s in harness["evals"] if b.endswith(SEL)} == {(0, 1)}
    assert {s for _r, b, s in harness["evals"] if not b.endswith(SEL)} == {(0,)}, \
        "원본 판 평가 시드까지 바뀌었다"
    assert summary["select_source"] == "v12~v15" and summary["select_eval"]["eval_seeds"] == 2


def test_보고용_재생성도_선택_창으로_고른다(rd, harness, tmp_path, monkeypatch, capsys):
    """`--report-only` 가 같은 로그에서 **같은 라운드**를 내야 한다 — 인자가 다르면 거부한다."""
    _set_goals(harness, sel=DISAGREE_SEL, **DISAGREE)
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(tmp_path / "r.md"), *MIN, *SEL_ARGS])
    evals_before = len(harness["evals"])
    again = _run_main(rd, monkeypatch, capsys,
                      ["--out", str(out), "--report", str(tmp_path / "again.md"),
                       "--report-only", *MIN, *SEL_ARGS])
    assert again["selected_round"] == 3 and again["select_source"] == "v8~v11"
    assert (out / rd.BEST_CKPT_NAME).read_bytes() == (out / "policy-r3.pt").read_bytes()
    assert not any(b.endswith(SEL) for _r, b, _s in harness["evals"][evals_before:]), \
        "--report-only 가 선택 창을 다시 쟀다(로그의 숫자를 써야 한다)"
    for bad in (["--select", "best"],                                       # 선택 창 빠짐
                ["--select", "best", "--select-variants", "4",
                 "--select-variant-offset", "12"]):                          # 다른 창
        monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--out", str(out), "--report",
                                          str(tmp_path / "bad.md"), "--report-only", *MIN,
                                          *COLLECT, *THREE_ARGS, *bad])
        with pytest.raises(SystemExit):
            rd.main()
        assert (out / rd.BEST_CKPT_NAME).read_bytes() == (out / "policy-r3.pt").read_bytes(), \
            "거부했는데 policy-best.pt 를 덮어썼다"


def test_선택_창을_원본_판_로그에_억지로_씌우면_거부한다(rd, harness, tmp_path, monkeypatch,
                                                      capsys):
    _set_goals(harness, **DISAGREE)
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(tmp_path / "r.md"), *MIN, *COLLECT,
               *THREE_ARGS, "--select", "best"])
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--out", str(out), "--report",
                                      str(tmp_path / "x.md"), "--report-only", *MIN, *SEL_ARGS])
    with pytest.raises(SystemExit):
        rd.main()
    assert "선택 창" in capsys.readouterr().err


# ======================================== ⑤ 선택 창의 판 — 접기와 결정성(진짜 판을 짓는다)

def _pos(board):
    return tuple(sorted((a.id, round(a.motion["pos"][0], 6), round(a.motion["pos"][1], 6))
                        for a in board.scenario.actors))


def test_선택_창_판은_액터_없는_단계를_접는다(rd):
    """★ 접기는 `load_window` 가 정한다 — 단계 ① 은 원본 판, 단계 ③a·③b 는 v8~v11."""
    got = {s: (bs, varied) for s, bs, varied in rd.select_window_boards(THREE, 4, 8)}
    boards1, varied1 = got["stage1"]
    assert varied1 is False
    assert [b.name for b in boards1] == [f"course_{c}" for c in "ABDEGH"]
    for s in ("stage3a", "stage3b"):
        boards, varied = got[s]
        assert varied is True
        assert [b.name for b in boards] == [f"course_{c}@v{k}" for c in "ABDEGH"
                                            for k in range(8, 12)]


@pytest.mark.parametrize("stage", ["stage3a", "stage3b"])
def test_선택_창_판은_큰_묶음의_같은_자리_판과_같다(rd, stage, monkeypatch):
    """★★ `v8~v11` 창의 i 번째 판 = `load_curriculum(variants=K+N)` 에서 v≥K 만 고른 것의 i 번째 판.

    배치가 `(판 이름, 변종 번호)` 로만 정해져야 "선택 창" 이 실행마다 같은 판이다
    (`tests/test_stage3.py::test_오프셋_변종은_큰_묶음에서_고른_것과_같은_판이다` 와 같은 성질).
    `eval_unseen.py` 의 창 함수와도 같은 판이어야 한다(접기 규칙을 한 곳에서 쓴다).
    """
    from vtd_rl.world.board import load_curriculum
    k, n = 8, 4
    (_s, win, varied), = rd.select_window_boards([stage], n, k)
    assert varied
    _, big = load_curriculum(rd.stage_path(stage), variants=k + n)
    want = [b for b in big if int(b.name.split("@v")[1]) >= k]
    assert len(win) == len(want) == 6 * n
    for i, (x, y) in enumerate(zip(win, want)):
        assert x.name == y.name, i
        assert _pos(x) == _pos(y), x.name
    ev = _load(os.path.join(REPO, "scripts", "eval_unseen.py"), "eval_unseen_for_window")
    unseen, unseen_varied = ev.stage_boards(stage, n, k)
    assert unseen_varied and [_pos(b) for b in unseen] == [_pos(b) for b in win]


def test_선택_창은_보고_창과_판이_하나도_안_겹친다(rd):
    """이름뿐 아니라 **배치**까지 — 이름만 다르고 같은 자리면 안 쓴 판이 아니다."""
    from vtd_rl.world.board import load_curriculum
    for stage in ("stage3a", "stage3b"):
        (_s, sel, _v), = rd.select_window_boards([stage], 4, 8)
        _, rep = load_curriculum(rd.stage_path(stage), variants=4, offset=4)
        _, col = load_curriculum(rd.stage_path(stage), variants=4)
        for other in (rep, col):
            assert {b.name for b in sel} & {b.name for b in other} == set()
            for c in "ABDEGH":
                a = {_pos(b) for b in sel if b.name.startswith(f"course_{c}@")}
                o = {_pos(b) for b in other if b.name.startswith(f"course_{c}@")}
                assert a and o and not (a & o), f"{stage} course_{c}: 선택 창 배치가 겹친다"
