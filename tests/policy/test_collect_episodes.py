"""`scripts/collect_episodes.py --variants/--seed-offset` (M5b).

판을 짓지 않는다 — `collect_episode`·`save_shard` 를 가짜로 바꾸고 작은 커리큘럼을 쓴다.
"""
import importlib.util
import json
import os
from types import SimpleNamespace

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "collect_episodes.py")


@pytest.fixture(scope="module")
def ce():
    spec = importlib.util.spec_from_file_location("collect_episodes_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mini(tmp_path):
    """`tests/policy/test_run_dagger_stage.py` 의 MINI 와 같은 꼴(액터 하나, 변종 가능)."""
    import importlib.util as u
    spec = u.spec_from_file_location("stage_mini", os.path.join(REPO, "tests", "policy",
                                                                 "test_run_dagger_stage.py"))
    m = u.module_from_spec(spec)
    spec.loader.exec_module(m)
    p = tmp_path / "mini.json"
    p.write_text(json.dumps(m.MINI), encoding="utf-8")
    return str(p)


def test_기본값은_예전_작업_목록과_같다(ce, tmp_path):
    from vtd_rl.world.board import load_curriculum
    cur = _mini(tmp_path)
    jobs = ce.build_jobs(cur, 1, 2, 0.5, None, str(tmp_path))
    _n, boards = load_curriculum(cur)                       # 예전 `main` 이 쓰던 호출 그대로
    old = [(cur, b.name, 1000 + s, 0.5, None, str(tmp_path), 1) for b in boards for s in range(2)]
    assert [j[:7] for j in jobs] == old
    assert all(j[-1] == 1 for j in jobs)


def test_변종을_주면_변종_판을_모은다(ce, tmp_path, monkeypatch):
    cur = _mini(tmp_path)
    jobs = ce.build_jobs(cur, 1, 1, 0.5, None, str(tmp_path / "o"), variants=2)
    assert any(j[1].endswith("@v1") for j in jobs) and all(j[-1] == 2 for j in jobs)
    seen = []
    monkeypatch.setattr(ce, "collect_episode",
                        lambda board, **kw: (seen.append((board.name, kw)),
                                             SimpleNamespace(meta={"steps": 1, "outcome": "goal"}))[1])
    monkeypatch.setattr(ce, "save_shard", lambda shard, p: None)
    job = next(j for j in jobs if j[1].endswith("@v1"))
    ce._one(job)
    assert seen[0][0].endswith("@v1")
    assert seen[0][1] == {"policy": None, "beta": 0.5, "seed": job[2]}


def test_seed_offset_은_시드와_파일_이름을_민다(ce, tmp_path, monkeypatch):
    cur = _mini(tmp_path)
    jobs = ce.build_jobs(cur, 2, 2, 0.5, None, str(tmp_path), variants=1, seed_offset=2)
    assert sorted({j[2] for j in jobs}) == [2002, 2003]
    paths = []
    monkeypatch.setattr(ce, "collect_episode",
                        lambda board, **kw: SimpleNamespace(meta={"steps": 1, "outcome": "goal"}))
    monkeypatch.setattr(ce, "save_shard", lambda shard, p: paths.append(p))
    ce._one(jobs[0])
    assert paths[0].endswith(f"r2-{jobs[0][1]}-s2002.npz")


@pytest.mark.parametrize("bad", [["--variants", "0"], ["--seed-offset", "-1"]])
def test_잘못된_값은_거부한다(ce, tmp_path, bad):
    cur = _mini(tmp_path)
    with pytest.raises(SystemExit) as e:
        ce.main(["--curriculum", cur, "--out", str(tmp_path / "o"), "--workers", "1"] + bad)
    assert e.value.code == 2


def test_같은_조각이_있으면_거부한다(ce, tmp_path, monkeypatch):
    cur = _mini(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    first = ce.build_jobs(cur, 1, 1, 0.5, None, str(out))[0]
    (out / f"r1-{first[1]}-s1000.npz").write_bytes(b"x")
    monkeypatch.setattr(ce, "collect_episode", lambda *a, **k: pytest.fail("돌면 안 된다"))
    with pytest.raises(SystemExit) as e:
        ce.main(["--curriculum", cur, "--out", str(out), "--round", "1", "--seeds", "1",
                 "--beta", "0.5", "--workers", "1"])
    assert e.value.code == 2
    assert (out / f"r1-{first[1]}-s1000.npz").read_bytes() == b"x"
