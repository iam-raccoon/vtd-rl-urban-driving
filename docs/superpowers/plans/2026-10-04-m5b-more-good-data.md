# M5b — 좋은 수집자로 데이터를 두 배로 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 지금 최고 그물(M4y: M4s 의 라운드 0~2 데이터 · 앵커 2 · 회피 가중 3 · EMA, 보고 창 정지차 65.8%)은 **옛 손실 학생이 모은 데이터**로 학습했다. 새 손실 학생의 데이터는 해로웠다(M4w~M4z). 그 좋은 수집자(M4s 의 r0·r1 그물)로 **새 시드의 r1·r2 데이터를 같은 양만큼 더 모아** D2 를 두 배로 만들고, 같은 설정으로 다시 학습하면 오르는지 잰다.

**Architecture:** `scripts/collect_episodes.py` 에 `--variants V`(변종 판 v0~v{V-1} 을 모은다, 기본 1 = 지금처럼 원본 판만)와 `--seed-offset K`(시드를 `1000×라운드 + K + s` 로, 기본 0 = 지금과 같다)를 넣는다. 본 실행은 그것으로 새 조각을 모으고, M4s 조각과 함께 한 폴더에 모아 `refit_round.py` 로 학습한다.

**Tech Stack:** Python 3.10, numpy 1.26.4, pytest. 본 실행은 OMEN.

**Spec:** 이 문서. 직전 증거 `docs/reports/m4y-best-data.md`·`docs/reports/m5a-near-states.md`.

## 실험 설계 (본 실행 — 계획 주관자, 구현 과제 아님)

데이터 시드 `S ∈ {0..4}` 마다:

1. 새 r1 조각: `collect_episodes.py --curriculum curricula/stage3b.json --variants 4 --round 1 --seeds 2 --seed-offset 2 --beta 0.5 --policy runs/omen/2026-10-02-m4s-s{S}/policy-r0.pt --out runs/omen/2026-10-04-m5b-s{S}/data` → 시드 1002·1003(M4s 는 1000·1001).
2. 새 r2 조각: 같은 꼴로 `--round 2 --policy …/policy-r1.pt` → 시드 2002·2003.
3. 같은 폴더에 M4s 시드 S 의 r0·r1·r2 조각을 하드링크로 넣는다(`r0-*`, `r1-*-s100[01]`, `r2-*-s200[01]`).
4. `refit_round.py --run runs/omen/2026-10-04-m5b-s{S} --upto 2 --init runs/omen/2026-10-01-m4o-s0/policy-best.pt --anchor-coef 2 --near-m 30 --near-weight 3 --batch-seed {100+S} --ema-halflife 2500 --no-eval` → EMA 그물을 선택 창 v8~v11·보고 창 v4~v7 로 잰다(CPU, `OMP_NUM_THREADS=1`).

수집도 `OMP_NUM_THREADS=1`. M4s 의 r1·r2 는 β 0.5, 걸음마다 섞기였다 — 같게 한다(기본 `step`).

### 판정 기준 (미리 적는다)

기준은 M4y 승자(M4s D2, 같은 명령)다. 선택 창 73.6 · 69.4 · 62.5 · 69.4 · 79.2(평균 70.8%), 보고 창 61.1 · 59.7 · 58.3 · 69.4 · 80.6(평균 65.8%).

- **1 차(선택 창)**: 데이터를 두 배로 한 EMA 의 평균 ≥ 74.8%(+4) **그리고** 다섯 짝 중 4 개 이상 높음 → **오른다**. 평균 ≤ 66.8%(−4) → **내린다**. 아니면 **차이 없다**.
- **2 차(보고 창, 판정과 별도로 늘 잰다)**: 평균과 짝 비교를 적는다. 1 차가 "오른다" 인데 보고 창 평균이 65.8% 이하면 "선택 창에서만 올랐다" 고 적는다.
- ① 평균 < 90% 면 1 차 결과와 상관없이 "기본 주행이 무너졌다" 고 적는다.

## Global Constraints

- numpy 는 **1.26.4** 고정 — 새 의존성을 넣지 않는다.
- `third_party/rule_stack` 은 **읽기 전용**이다.
- 커밋 금지: `runs/`, `*.pt`, `*.npz`, CSV, `*.xodr`, `*.xml`.
- 주석·독스트링·커밋 메시지·성적표는 **한국어**.
- 커밋 메시지에 **Claude 표기를 넣지 않는다**(`Co-Authored-By`, `Claude-Session`, "Generated with", 🤖, anthropic 주소 전부). 커밋 뒤 `git show -s --format=%B HEAD` 로 확인한다.
- 테스트는 `env -u PYTHONPATH .venv/bin/pytest ...` 로 돌리고 **종료 코드(rc)** 로 판정한다. "N passed" 를 grep 하지 않는다. `| tail` 로 넘기지 않고 파일로 돌린다.
- **기본값은 지금과 같다**: `--variants`·`--seed-offset` 를 안 주면 `collect_episodes.py` 의 작업 목록(판·시드·경로)이 지금과 같다.
- 스크립트끼리는 서로를 임포트하지 않는다(레포 관행).

## Review Focus

1. **변종 판 이름이 일꾼에서 안 찾아지는 것** — `_one` 이 `load_curriculum(curriculum)`(변종 없이) 로 판을 다시 지으면 `course_A@v1` 을 못 찾는다. 작업 묶음에 변종 수가 실려야 한다. → `test_변종을_주면_변종_판을_모은다`.
2. **시드가 M4s 조각과 겹치는 것** — 같은 이름의 조각이 생기면 하드링크를 덮거나 같은 판을 두 번 센다. `--seed-offset 2` 면 1002·1003 이어야 한다. → `test_seed_offset_은_시드와_파일_이름을_민다`.
3. **기본값에서 작업 목록이 바뀌는 것.** → `test_기본값은_예전_작업_목록과_같다`.
4. **음수 `--seed-offset`·0 이하 `--variants`** — 돌기 전에 거부. → `test_잘못된_값은_거부한다`.
5. **출력 폴더에 같은 이름의 조각이 이미 있을 때 덮어쓰는 것** — 본 실행은 M4s 조각을 하드링크로 넣은 폴더에 쓴다. 이미 있으면 거부(덮지 않는다). → `test_같은_조각이_있으면_거부한다`.

---

### Task 1: `collect_episodes.py --variants/--seed-offset`

**Files:**
- Modify: `scripts/collect_episodes.py`
- Test: `tests/policy/test_collect_episodes.py` (새로)

**Interfaces:**
- Produces:
  - `build_jobs(curriculum, rnd, seeds, beta, policy, out, variants=1, seed_offset=0) -> list[tuple]` — 묶음 `(curriculum, name, seed, beta, policy, out, rnd, variants)`, `seed = 1000*rnd + seed_offset + s`. 판 순서는 `load_curriculum(curriculum, variants=variants)` 순서, 시드는 판마다 `s = 0..seeds-1`.
  - `_one(job)` 은 `load_curriculum(curriculum, variants=variants)` 로 판을 다시 지어 이름으로 찾는다.
  - CLI `--variants`(기본 1), `--seed-offset`(기본 0). `--variants < 1`·`--seed-offset < 0` 은 `ap.error`(rc=2). 돌기 전에 작업 목록의 출력 파일 중 이미 있는 것이 하나라도 있으면 `ap.error`.
  - `main(argv=None) -> int`(테스트가 부를 수 있게).

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/policy/test_collect_episodes.py`:

```python
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
```

- [ ] **Step 2: 실패를 본다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/policy/test_collect_episodes.py -v > /home/user/.claude/jobs/c5216a88/tmp/b1-red.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `AttributeError: … has no attribute 'build_jobs'`.

- [ ] **Step 3: 구현한다**

`scripts/collect_episodes.py` 를 다음 모양으로 바꾼다(독스트링에 M5b 용례 한 줄을 더한다):

```python
def build_jobs(curriculum, rnd, seeds, beta, policy, out, variants=1, seed_offset=0):
    """판×시드 작업 묶음. 변종 수가 묶음에 실려야 일꾼이 `course_A@v1` 을 다시 찾는다(M5b).

    시드는 `1000×라운드 + seed_offset + s` — `run_dagger.py` 와 같은 꼴이고, `seed_offset` 으로
    이미 있는 실행의 조각(예: s1000·1001)과 안 겹치게 민다.
    """
    _name, boards = load_curriculum(curriculum, variants=variants)
    return [(curriculum, b.name, 1000 * rnd + seed_offset + s, beta, policy, out, rnd, variants)
            for b in boards for s in range(seeds)]


def shard_path(job) -> str:
    _cur, name, seed, _beta, _policy, out, rnd, _variants = job
    return os.path.join(out, f"r{rnd}-{name}-s{seed}.npz")


def _one(job):
    curriculum, name, seed, beta, policy_path, out, rnd, variants = job
    _n, boards = load_curriculum(curriculum, variants=variants)
    board = next(b for b in boards if b.name == name)
    policy = DrivePolicy.load(policy_path) if policy_path else None
    shard = collect_episode(board, policy=policy, beta=beta, seed=seed)
    path = shard_path(job)
    save_shard(shard, path)
    return {"path": path, **shard.meta}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ...  # 기존 인자 그대로
    ap.add_argument("--variants", type=int, default=1, help="변종 판 v0~v{V-1} 을 모은다(1 = 원본 판만)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="시드를 1000×라운드 + 이 값 + s 로 민다(이미 있는 조각과 안 겹치게)")
    a = ap.parse_args(argv)
    if a.variants < 1:
        ap.error(f"--variants 는 1 이상이어야 한다 — {a.variants}")
    if a.seed_offset < 0:
        ap.error(f"--seed-offset 은 0 이상이어야 한다 — {a.seed_offset}")
    jobs = build_jobs(a.curriculum, a.round, a.seeds, a.beta, a.policy, a.out,
                      variants=a.variants, seed_offset=a.seed_offset)
    clash = [shard_path(j) for j in jobs if os.path.exists(shard_path(j))]
    if clash:
        ap.error(f"이미 있는 조각을 덮게 된다({len(clash)}개, 예: {clash[0]}) — 다른 --seed-offset 이나 --out")
    os.makedirs(a.out, exist_ok=True)
    ...  # 이후 기존 그대로(pool.map(_one, jobs) → 요약 출력)
```

`load_curriculum(curriculum, variants=1)` 이 지금의 `load_curriculum(curriculum)` 과 같은 판 목록을 주는지 확인한다(`vtd_rl/world/board.py`). 다르면 `variants == 1` 일 때 예전 호출을 그대로 쓴다.

- [ ] **Step 4: 통과를 본다**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/policy/test_collect_episodes.py tests/policy/test_collect.py -v > /home/user/.claude/jobs/c5216a88/tmp/b1-green.txt 2>&1; echo rc=$?`
Expected: rc=0.

- [ ] **Step 5: 커밋한다**

```bash
git add scripts/collect_episodes.py tests/policy/test_collect_episodes.py
git commit -m "collect_episodes --variants/--seed-offset — 변종 판과 겹치지 않는 시드로 더 모은다(M5b)"
git show -s --format=%B HEAD
```
