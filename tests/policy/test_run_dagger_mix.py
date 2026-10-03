"""`scripts/run_dagger.py --mix/--mix-len` — 수집의 β 섞는 단위를 일꾼까지 넘긴다(M4z).

판을 짓지 않는다. `collect_episode` 와 `save_shard` 를 가짜로 바꿔 일꾼이 **실제로 받은** 인자를 본다.
기본(`step`)은 예전과 **같은 인자**로 부르고, 로그 행·성적표에도 새 칸·새 줄이 없어야 한다
(예전 스크립트와 바이트까지 같은 것은 `test_run_dagger_select_eval.py` 가 잠근다).
"""
import importlib.util
import json
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "run_dagger.py")
PY = os.path.join(REPO, ".venv", "bin", "python")
#: `test_run_dagger_stage.py` 의 `MINI` 와 같은 꼴(그 커리큘럼 스키마를 그대로 쓴다).
MINI = {"name": "작은 단계 ③", "signals": "cycle", "boards": [{
    "name": "course_H", "route": "routes/HL_FMA_NEW_H.json",
    "lane": "routes/HL_FMA_NEW_H_lane.json",
    "jitter": {"s": 40.0, "lateral": 0.6},
    "actors": [
        {"id": 351, "kind": "static", "type": "obstacle", "s": 200.0, "lateral": 1.05,
         "size": [0.15, 0.46, 0.61]},
        {"id": 352, "kind": "static", "type": "vehicle", "s": 1140.0, "lateral": 0.90,
         "size": [4.5, 1.8, 1.5]}]}]}


@pytest.fixture(scope="module")
def rd():
    spec = importlib.util.spec_from_file_location("run_dagger_mix_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_작업_묶음에_섞기가_실리고_변종은_마지막이다(rd, tmp_path):
    jobs = rd.build_jobs([("c.json", "course_H@v0")], 1, 0.5, None, str(tmp_path), False, 4, 2,
                         mix="segment", mix_len=30)
    assert len(jobs) == 2
    j = jobs[1]
    assert j[2] == 1001 and j[-1] == 4
    assert (j[-3], j[-2]) == ("segment", 30)
    d = rd.build_jobs([("c.json", "course_H@v0")], 1, 0.5, None, str(tmp_path), False, 4, 1)
    assert (d[0][-3], d[0][-2], d[0][-1]) == ("step", 30, 4)


def _fake(rd, monkeypatch, seen):
    def fake_collect(board, **kw):
        seen.append(kw)
        return SimpleNamespace(meta={"outcome": "goal"})

    monkeypatch.setattr(rd, "collect_episode", fake_collect)
    monkeypatch.setattr(rd, "save_shard", lambda shard, p: None)
    monkeypatch.setattr(rd, "job_board", lambda job: SimpleNamespace(name=job[1]))


def test_step_이면_일꾼이_예전과_같은_인자로_부른다(rd, monkeypatch, tmp_path):
    seen = []
    _fake(rd, monkeypatch, seen)
    job = rd.build_jobs([("c.json", "course_H@v0")], 1, 0.5, None, str(tmp_path), False, 1, 1)[0]
    rd._collect_job(job)
    assert seen == [{"policy": None, "beta": 0.5, "seed": 1000}]


def test_segment_면_일꾼이_섞기를_넘긴다(rd, monkeypatch, tmp_path):
    seen = []
    _fake(rd, monkeypatch, seen)
    job = rd.build_jobs([("c.json", "course_H@v0")], 2, 0.5, None, str(tmp_path), False, 1, 1,
                        mix="episode", mix_len=7)[0]
    rd._collect_job(job)
    assert seen == [{"policy": None, "beta": 0.5, "seed": 2000, "mix": "episode", "mix_len": 7}]


def _dry(args):
    return subprocess.run([PY, SCRIPT, "--dry-run"] + args, capture_output=True, text=True,
                          env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})


def test_dry_run_은_섞기를_보여준다(tmp_path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(MINI), encoding="utf-8")
    p = _dry(["--stage", str(path), "--mix", "segment", "--mix-len", "30"])
    assert p.returncode == 0, p.stderr
    assert "섞기" in p.stdout and "segment" in p.stdout and "30" in p.stdout
    q = _dry(["--stage", str(path)])
    assert q.returncode == 0, q.stderr
    assert "섞기" in q.stdout and "step" in q.stdout


@pytest.mark.parametrize("bad,why", [(["--mix", "steps"], "invalid choice"),
                                     (["--mix", "segment", "--mix-len", "0"], "--mix-len 은 1 이상")])
def test_잘못된_mix_len_은_거부한다(tmp_path, bad, why):
    """거부 **이유**까지 본다 — 인자를 아예 모르는 탓(unrecognized)으로 2 가 나온 것과 가른다."""
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(MINI), encoding="utf-8")
    p = _dry(["--stage", str(path)] + bad)
    assert p.returncode == 2
    assert why in p.stderr, p.stderr[-400:]


# ----------------------------------------- 끝까지 돌리기(가짜 수집·학습·평가) — 로그·성적표

#: `test_run_dagger_select_eval.py` 의 `MIN` 과 같은 꼴, 라운드만 2.
MIN = ["--rounds", "2", "--seeds", "1", "--epochs", "1", "--eval-seeds", "1",
       "--workers", "1", "--device", "cpu"]


def _fake_ev(goal_rate=1.0, mean_score=98.6):
    ep = types.SimpleNamespace(board="course_A", seed=0, outcome="goal", steps=100,
                               reward=0.0, score=mean_score, sheet=[])
    return {"goal_rate": goal_rate, "mean_score": mean_score, "mean_score_raw": mean_score,
            "mean_score_completed": mean_score, "mean_reward": 0.0, "episodes": [ep]}


class _FakeDataset:
    def __len__(self):
        return 8


def _install(module, monkeypatch, jobs_seen):
    """판·수집·학습·평가만 가짜로 바꾼다. 로그·성적표·체크포인트 저장/읽기는 진짜다."""
    monkeypatch.setattr(module, "time", types.SimpleNamespace(perf_counter=lambda: 0.0))
    monkeypatch.setattr(module, "collect_targets",
                        lambda paths, smoke, variants: [("(fake)", "course_X")])

    def fake_collect(job):
        jobs_seen.append(job)               # 수집에 실제로 실린 작업 묶음
        return {"outcome": "goal"}

    monkeypatch.setattr(module, "_collect_job", fake_collect)
    monkeypatch.setattr(module, "_stall_start_count", lambda *args, **kw: 0)
    monkeypatch.setattr(module, "load_dir", lambda data_dir: _FakeDataset())
    monkeypatch.setattr(module, "eval_boards", lambda names, smoke: [(n, [n]) for n in names])
    monkeypatch.setattr(module, "train_epochs", lambda net, dataset, cfg, device=None, ref=None: {
        "epochs": cfg.epochs, "samples": len(dataset), "loss": 0.0, "control": 0.0,
        "turn": 0.0, "anchor": 0.0, "near_frac": 0.0, "seconds": 0.0})
    monkeypatch.setattr(module, "evaluate_policy", lambda net, boards, seeds=(0,): _fake_ev())
    monkeypatch.setattr(module, "evaluate_teacher",
                        lambda boards, seeds=(0,): _fake_ev(1.0, 99.6))


def _run_main(module, monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", *args])
    assert module.main() == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _log(out):
    with open(out / "log.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_로그와_성적표는_step_이_아닐_때만_섞기를_적는다(rd, monkeypatch, tmp_path, capsys):
    """★ 일꾼 묶음·로그 행·성적표 세 곳이 같은 값을 말해야 한다 — 인자를 받기만 하고 안 쓰는 꼴을 잡는다."""
    jobs = []
    _install(rd, monkeypatch, jobs)

    out, report = tmp_path / "mixed", tmp_path / "mixed.md"
    _run_main(rd, monkeypatch, capsys, ["--out", str(out), "--report", str(report), *MIN,
                                        "--mix", "segment", "--mix-len", "30"])
    assert jobs and all((j[-3], j[-2]) == ("segment", 30) for j in jobs), \
        "수집 작업에 섞기가 안 실렸다"
    rows = _log(out)
    assert len(rows) == 2
    assert all(r["mix"] == "segment" and r["mix_len"] == 30 for r in rows)
    text = report.read_text(encoding="utf-8")
    assert "수집 섞기" in text and "--mix segment" in text and "30" in text

    jobs.clear()
    out, report = tmp_path / "plain", tmp_path / "plain.md"
    _run_main(rd, monkeypatch, capsys, ["--out", str(out), "--report", str(report), *MIN])
    assert jobs and all((j[-3], j[-2]) == ("step", 30) for j in jobs)
    rows = _log(out)
    assert len(rows) == 2
    assert all("mix" not in r and "mix_len" not in r for r in rows), \
        "기본(step) 실행의 로그 행에 새 칸이 생겼다"
    assert "수집 섞기" not in report.read_text(encoding="utf-8"), \
        "기본(step) 실행의 성적표에 새 줄이 생겼다"


def test_성적표만_다시_만들_때는_로그의_섞기를_따른다(rd, monkeypatch, tmp_path, capsys):
    """`--report-only` 는 인자가 아니라 **로그의 값**으로 적는다 — 로그에 없으면 안 적는다."""
    _install(rd, monkeypatch, [])
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys, ["--out", str(out), "--report", str(tmp_path / "a.md"),
                                        *MIN, "--mix", "segment", "--mix-len", "12"])
    _run_main(rd, monkeypatch, capsys, ["--out", str(out), "--report", str(tmp_path / "b.md"),
                                        "--report-only", *MIN])
    text = (tmp_path / "b.md").read_text(encoding="utf-8")
    assert "수집 섞기" in text and "--mix segment" in text and "12" in text

    out2 = tmp_path / "run2"
    _run_main(rd, monkeypatch, capsys, ["--out", str(out2), "--report", str(tmp_path / "c.md"),
                                        *MIN])
    _run_main(rd, monkeypatch, capsys, ["--out", str(out2), "--report", str(tmp_path / "d.md"),
                                        "--report-only", *MIN])
    assert "수집 섞기" not in (tmp_path / "d.md").read_text(encoding="utf-8")
