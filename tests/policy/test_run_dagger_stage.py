"""`scripts/run_dagger.py` 의 단계 선택 — 무엇을 모으고 무엇을 평가하는가.

M4j 판정: 학생이 단계 ③ 에서 첫 액터에 박는다. 원인은 관측이 아니라 **데이터** 다 —
M3 의 DAgger 는 `stage1` 에서만 모았고(액터 0 개), 그래서 물체 슬롯 16 개가 학습 내내
마스크였다. 여기서 잠그는 것은 그 수집 대상을 고르는 길이다.
"""
import importlib.util
import json
import os
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PY = os.path.join(REPO, ".venv", "bin", "python")
SCRIPT = os.path.join(REPO, "scripts", "run_dagger.py")


@pytest.fixture(scope="module")
def rd():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다."""
    spec = importlib.util.spec_from_file_location("run_dagger_stage_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args, timeout=300):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([PY, SCRIPT, *args], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=timeout)


# ---------------------------------------------------------------- 단계 고르기

def test_기본_단계는_예전_그대로다(rd):
    """★ 아무것도 안 주면 M3 와 똑같이 돌아야 한다 — 기존 성적표·`--report-only` 가 걸려 있다."""
    assert rd.resolve_stages([], []) == (["stage1"], ["stage1", "stage2"])


def test_앞이_수집_뒤가_평가다(rd):
    """★★ 둘을 뒤바꾸면 단계 ③ 을 모으라고 시켜 놓고 ①② 만 모으는데, 성적표는 멀쩡해 보인다.

    이 마일스톤 전체가 "수집 대상을 ③ 으로 옮긴다" 는 한 줄이라 여기서 뒤집히면 아무것도 안 한 것이다.
    """
    collect, evals = rd.resolve_stages(["stage3"], ["stage1", "stage2", "stage3"])
    assert collect == ["stage3"]
    assert evals == ["stage1", "stage2", "stage3"]


def test_단계_이름이_커리큘럼_경로로_풀린다(rd):
    assert os.path.realpath(rd.stage_path("stage3")) == \
           os.path.join(REPO, "curricula", "stage3.json")


def test_모르는_단계는_터지고_있는_단계를_알려_준다(rd):
    with pytest.raises(ValueError, match="모르는 단계") as exc:
        rd.stage_path("stage9")
    assert "stage1" in str(exc.value) and "stage3" in str(exc.value)


# ---------------------------------------------------------------- --smoke 의 액터 소실

def test_slice_board는_액터를_통째로_버린다():
    """★ `--smoke` 가드가 필요한 **이유** 를 여기에 못 박아 둔다.

    이 사실이 바뀌면(액터를 살려서 옮기게 되면) 가드를 (b) 안으로 바꿔도 된다.
    """
    from vtd_rl.world.board import load_board, slice_board
    b = load_board({"name": "course_H", "route": "routes/HL_FMA_NEW_H.json",
                    "lane": "routes/HL_FMA_NEW_H_lane.json",
                    "actors": [{"id": 9, "kind": "static", "type": "obstacle", "s": 150.0,
                                "lateral": 1.0, "size": [0.15, 0.46, 0.61]}]})
    assert len(b.scenario.actors) == 1
    assert slice_board(b, 0.0, 250.0, "H_0_250").scenario.actors == []


def test_smoke_는_액터_있는_단계를_거부한다(rd):
    """★★ 조용히 통과하는 실패를 막는 가드. 거부(a) 를 골랐다 — `slice_board` 의 좌표계를

    건드리지 않는 쪽이다. 안 막으면 단계 ③ 스모크가 액터 0 개인 조각을 초록불로 지나가고
    "스모크 통과" 라고 말한다.
    """
    with pytest.raises(ValueError, match="stage3"):
        rd.check_smoke(["stage3"], True)
    rd.check_smoke(["stage1", "stage2"], True)    # 액터 없는 단계는 통과한다
    rd.check_smoke(["stage3"], False)             # --smoke 가 아니면 상관없다


def test_has_actors_는_판을_안_짓고_판별한다(rd):
    assert rd.has_actors("stage3") is True
    assert rd.has_actors("stage1") is False
    assert rd.has_actors("stage2") is False


# ------------------------------------------------- 실제 판으로 도는 수집 경로(스모크 대체)

MINI = {"name": "작은 단계 ③", "signals": "cycle", "boards": [{
    "name": "course_H", "route": "routes/HL_FMA_NEW_H.json",
    "lane": "routes/HL_FMA_NEW_H_lane.json",
    "jitter": {"s": 40.0, "lateral": 0.6},
    "actors": [
        {"id": 351, "kind": "static", "type": "obstacle", "s": 200.0, "lateral": 1.05,
         "size": [0.15, 0.46, 0.61]},
        {"id": 352, "kind": "static", "type": "vehicle", "s": 1140.0, "lateral": 0.90,
         "size": [4.5, 1.8, 1.5]}]}]}


def test_수집_경로는_실제_판에서_액터를_싣는다(rd, tmp_path):
    """★ `--smoke` 를 거부했으므로 **작은 진짜 판 하나**로 단계 ③ 수집 경로를 돌려 본다.

    판을 짓는 데까지가 여기 범위다(한 판을 끝까지 모는 것은 느린 테스트 몫).
    """
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(MINI), encoding="utf-8")
    targets = rd.collect_targets([str(path)], smoke=False, variants=2)
    assert [n for _p, n in targets] == ["course_H@v0", "course_H@v1"]
    seen = []
    for p, name in targets:
        board = rd._boards(p, False, name, 2)[0]
        assert board.name == name
        assert len(board.scenario.actors) == 2, "수집 판에 액터가 없다 — 단계 ② 와 같아진다"
        seen.append(tuple(tuple(a.motion["pos"]) for a in board.scenario.actors))
    assert seen[0] != seen[1], "변종인데 배치가 같다"


def test_변종은_수집에만_걸고_평가는_원본_판이다(rd):
    """성적표의 '단계 ③ 완주율' 은 출발점 0/18(판 6 × 시드 3)과 비교하는 숫자다 —

    평가 판 수가 변종에 따라 달라지면 그 비교가 깨진다.
    """
    s3 = rd.stage_path("stage3")
    assert len(rd.collect_targets([s3], False, 3)) == 18
    assert [len(bs) for _n, bs in rd.eval_boards(["stage3"], False)] == [6]
    assert all("@v" not in b.name for _n, bs in rd.eval_boards(["stage3"], False) for b in bs)


def test_평가_판은_단계_이름과_짝지어_나온다(rd):
    """★★ 이름과 판을 따로 만들어 `zip` 으로 붙이면 어긋날 수 있고, 그러면 로그의

    `"stage3"` 열쇠에 단계 ① 결과가 들어앉는다 — 숫자는 멀쩡하고 이름만 거짓말이 된다.
    그래서 `eval_boards` 가 이름과 판을 **한 곳에서 같이** 만든다.
    """
    got = rd.eval_boards(["stage1", "stage3"], False)
    assert [n for n, _bs in got] == ["stage1", "stage3"]
    by = dict(got)
    assert all(not b.scenario.actors for b in by["stage1"])
    assert all(b.scenario.actors for b in by["stage3"]), "stage3 자리에 액터 없는 판이 왔다"


def test_수집_판_이름이_단계끼리_겹치면_터진다(rd):
    """단계 ①③ 은 코스 이름이 같다 — 조각 파일 `r{라운드}-{판}-s{시드}.npz` 가 서로 덮어쓴다."""
    with pytest.raises(ValueError, match="겹친다"):
        rd.collect_targets([rd.stage_path("stage1"), rd.stage_path("stage3")], False, 1)


def test_단계를_여럿_주면_순서대로_이어_붙는다(rd, tmp_path):
    """판 이름이 갈리기만 하면 여러 단계를 한 번에 모을 수 있다 — 준 순서 그대로다.

    (기존 단계끼리는 코스 이름이 같아 못 섞는다. 위 테스트가 그것을 막는다.)
    """
    mini = json.loads(json.dumps(MINI))
    mini["boards"][0]["name"] = "mini_H"
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(mini), encoding="utf-8")
    names = [n for _p, n in rd.collect_targets([rd.stage_path("stage3"), str(path)], False, 1)]
    assert names == ["course_A", "course_B", "course_D", "course_E", "course_G", "course_H",
                     "mini_H"]


# ---------------------------------------------------------------- 명령줄 배선

def test_dry_run_이_수집과_평가를_갈라_보여_준다():
    """★★ 명령줄에서 `main` 까지의 배선을 통째로 잠근다 — 수집·평가를 뒤바꾸면 여기서 드러난다.

    (Task 3 Step 1 "개입이 무는지 먼저 확인" 이 쓰는 문이기도 하다.)
    """
    out = _run("--dry-run", "--stage", "stage3", "--eval-stage", "stage1", "--variants", "2")
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.splitlines()
    assert lines[0].startswith("수집 단계:") and "'stage3'" in lines[0]
    assert lines[1].startswith("평가 단계:") and "'stage1'" in lines[1]
    assert "stage3" not in lines[1], "평가 자리에 수집 단계가 들어갔다"
    assert "stage1" not in lines[0], "수집 자리에 평가 단계가 들어갔다"
    assert "course_A@v0:" in out.stdout and "course_A@v1:" in out.stdout
    assert "액터 없음" not in "\n".join(l for l in lines if l.startswith("  ")), \
        "수집 판에 액터가 없다"
    # 작업 묶음을 실제로 풀어 본 줄 — 변종 수가 안 실리면 여기서 드러난다.
    job_line = next(l for l in lines if l.startswith("첫 작업"))
    assert "@v" in job_line and "액터 4개" in job_line, job_line
    assert "평가 stage1: 판 6개 (액터 0개)" in out.stdout


def test_명령줄이_smoke_충돌과_모르는_단계를_거부한다(tmp_path):
    """★ 거부는 **아무 일도 하기 전에** 나야 한다 — 가드가 없으면 스모크가 끝까지 돌아

    기본 성적표 경로(`docs/reports/m3-dagger.md`)를 액터 0 개짜리 결과로 덮어쓴다.
    (2026-09-30 돌연변이 실험에서 실제로 덮어썼다.)
    """
    out_dir = tmp_path / "should-not-exist"
    report = tmp_path / "should-not-exist.md"
    bad_smoke = _run("--smoke", "--stage", "stage3", "--out", str(out_dir),
                     "--report", str(report))
    assert bad_smoke.returncode != 0
    assert "slice_board" in bad_smoke.stderr
    assert not out_dir.exists(), "거부했는데 폴더를 만들었다"
    assert not report.exists(), "거부했는데 성적표를 썼다"

    unknown = _run("--dry-run", "--stage", "stage9")
    assert unknown.returncode != 0
    assert "모르는 단계" in unknown.stderr

    bad_eval = _run("--dry-run", "--eval-stage", "stage9")
    assert bad_eval.returncode != 0, "평가 단계 이름은 안 검사했다"


def test_수집_작업은_그_변종_판을_집어_온다(rd, tmp_path, monkeypatch):
    """★ 일꾼이 받는 작업 묶음에 **변종 수**가 안 실리면 `course_H@v1` 을 못 찾는다 —

    수집은 spawn 프로세스에서 작업 묶음만 보고 판을 다시 짓는다(`_collect_job`). 여기서
    변종이 빠지면 원본 판만 계속 모으거나 판을 못 찾고 죽는다.
    """
    from types import SimpleNamespace

    path = tmp_path / "mini.json"
    path.write_text(json.dumps(MINI), encoding="utf-8")
    seen = {}

    def fake_collect(board, policy=None, beta=0.0, seed=0):
        seen["board"], seen["beta"], seen["seed"] = board, beta, seed
        return SimpleNamespace(meta={"outcome": "goal"})

    monkeypatch.setattr(rd, "collect_episode", fake_collect)
    monkeypatch.setattr(rd, "save_shard", lambda shard, p: seen.setdefault("path", p))
    jobs = rd.build_jobs([(str(path), "course_H@v0"), (str(path), "course_H@v1")],
                         1, 0.5, None, str(tmp_path), False, 2, 1)
    assert [j[1] for j in jobs] == ["course_H@v0", "course_H@v1"]
    assert all(j[-1] == 2 for j in jobs), "작업 묶음에 변종 수가 안 실렸다"
    job = jobs[1]
    assert job[2] == 1000, "시드는 1000*라운드 + s 다(라운드 1, s 0)"
    rd._collect_job(job)
    assert seen["board"].name == "course_H@v1"
    assert len(seen["board"].scenario.actors) == 2, "모으는 판에 액터가 없다"
    assert (seen["beta"], seen["seed"]) == (0.5, 1000)
    assert seen["path"].endswith("r1-course_H@v1-s1000.npz")


def test_이전_실행_표는_단계_이름을_따라가고_없는_단계는_대시다(rd, tmp_path):
    """옛 로그(`stage1`·`stage2` 만 있는)를 새 단계 목록으로 읽어도 성적표가 안 죽어야 한다."""
    row = {"round": 0, "beta": 1.0, "episodes": 6, "collect_goal": 6, "samples": 100,
           "train": {"loss": 0.5},
           "stage1": {"goal_rate": 0.98, "mean_score": 98.6},
           "stage2": {"goal_rate": 0.94, "mean_score": 94.1}}
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    text = "\n".join(rd._history_lines([str(path)], [], ["stage1", "stage3"]))
    assert "단계 ① 완주율" in text and "단계 ③ 완주율" in text
    assert "98%" in text and "98.6" in text
    assert "—% | — |" in text, "없는 단계는 대시로 비워야 한다"


def test_성적표_문구는_예전_한국어_이름을_지킨다(rd):
    """옛 성적표와 나란히 놓고 읽는다 — 기본 실행의 단계 이름이 `stage1` 로 바뀌면 안 된다."""
    assert rd.stage_label("stage1") == "단계 ①"
    assert rd.stage_label("stage3") == "단계 ③"
    assert rd.stage_label("mystage") == "mystage"      # 모르는 이름은 그대로
