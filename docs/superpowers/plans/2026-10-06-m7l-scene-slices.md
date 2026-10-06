# M7l — 어려운 장면 바로 앞에서 출발하는 짧은 판으로 연습량 늘리기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 남은 손실은 장애물 충돌이다(지금 배포 묶음은 큰 선택 창 384 판 중 20 판, 대부분 course_G 4 번째 정지차·course_D). 학습 판은 코스 전체(2~5 km)라, 판 하나에서 그 장면을 한 번만 겪는다. 장애물 장면의 즉시 벌(M7d·M7g·M7i)은 칸 평균을 끌어올렸지만 상위 그물은 정체했고, 선생님 모방(M7k)은 "멈춤" 을 배워 무너졌다. M7l 은 **어려운 장면 바로 앞(약 150 m)에서 출발하는 짧은 판**을 만들어 학습에 섞는다. 같은 학습 걸음으로 그 장면을 훨씬 자주 연습하게 된다.

**Architecture:**
- 판 항목에 선택 키 `slice: [s_from, s_to]`(경로 s, m)를 더한다(`BOARD_KEYS`).
  - `load_board` 는 먼저 전체 판과 액터를 예전처럼 짓는다(변종이면 흔든다). 그다음 `slice_board` 로 구간을 자르고, **흔든 뒤 s 가 구간 안인 액터만** 자른 판에 남긴다. 액터는 세계 좌표라 그대로 쓴다.
  - 자른 판의 이름은 항목 이름 그대로다(변종이면 `@v<k>` 가 붙는다). 신호 운용은 커리큘럼 값을 따른다.
  - `slice` 가 없으면 예전과 같다.
- 새 커리큘럼 `curricula/stage3s.json`(단계 3, 신호 cycle): 아래 여섯 구간. 액터 목록·흔들기는 `stage3.json` 의 같은 코스 항목을 그대로 쓰고, 구간 밖 액터는 잘릴 때 빠진다.

| 이름 | 코스 | 구간 [m] | 노리는 장면 |
|---|---|---|---|
| course_G_s2665 | G | 2515~2785 | 4 번째 정지차(가장 큰 실패) |
| course_G_s640 | G | 420~760 | 1 번째 정지차·2 번째 라바콘 |
| course_D_s2050 | D | 1900~2170 | 4 번째 정지차 |
| course_D_s730 | D | 510~920 | 라바콘 둘 |
| course_A_s1470 | A | 1305~1605 | 라바콘 줄 |
| course_E_s1740 | E | 1480~1970 | 라바콘·정지차 |

- 흔들기는 s ±40 m 라, 구간은 노리는 액터 앞 150 m, 뒤 120 m 이상 둔다(흔들어도 구간 안).

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거 `docs/reports/m7k-obstacle-imitation.md`, `docs/reports/m7j-combine.md`.

## 본 실행 (OMEN)

시드 `S ∈ {0, 1, 2, 3, 4, 5}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-06-m7l-s$S --init runs/omen/2026-10-06-m7f-s2/policy-125440.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json curricula/stage3s.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 --lat-free-range 80 --obs-profile 1.0 --block-profile 1.0 --ovl-profile 2.0 --lim-anticipate
```

- 보상·관측은 M7j (b) 와 같고 `curricula/stage3s.json` 만 더한다. **대조군은 M7j (b)**(시드 0~5)다.
- 체크포인트 48 개 모두 앞당김을 켠 채 큰 선택 창(v1100~v1131 × 시드 2, 384 판)으로 잰다. 큰 선택 창은 전체 코스(`stage3.json`)라 자른 판과 겹치지 않는다(학습 변종 v0~v3, 평가 v1100~).

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 지금 배포 묶음(M7f 시드 2 · 12.5 만 + 앞당김)은 큰 선택 창 90.90(완주 364, course_G 52)이다.

- **(가) 짧은 판이 course_G 를 고치나.** 체크포인트 48 개의 course_G 완주 평균이 대조(M7j b)보다 8 판 넘게 많거나, 충돌 평균이 20 판 넘게 적으면 "고친다" 로 본다.
- **(나) 고르기.** 후보는 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3, 앞당김 켬) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물이 큰 선택 창에서 지금 배포 묶음(90.90)보다 높을 때만, **새 시험 창 v23000~v23031 × 시드 2**(384 판, 앞당김 켬)에서 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

---

### Task 1: 판 항목 `slice` 와 커리큘럼 `stage3s.json`

**Files:**
- Modify: `vtd_rl/world/board.py`(`BOARD_KEYS` 에 `"slice"`, `load_board` 의 자르기)
- Create: `curricula/stage3s.json`
- Test: `tests/world/test_board.py`(저장소의 기존 board 테스트 파일을 따른다), `tests/test_stage3s.py`(새로)

**Interfaces:**
- Produces:
  - 판 항목 선택 키 `slice: [s_from, s_to]`.
  - `curricula/stage3s.json`.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

board 테스트 파일 끝에:

```python
def _g_entry(**kw):
    import json
    e = dict(json.load(open("curricula/stage3.json"))["boards"][4])     # course_G
    assert e["name"] == "course_G"
    e.update(kw)
    return e


def test_slice_는_구간만_자르고_구간_안_액터만_남긴다():
    from vtd_rl.world.board import load_board
    full = load_board(_g_entry())
    cut = load_board(_g_entry(name="course_G_s2665", slice=[2515.0, 2785.0]))
    assert abs(cut.route.total - 270.0) < 5.0
    assert len(cut.scenario.actors) == 1                       # 4 번째 정지차만
    car = cut.scenario.actors[0]
    full_car = [a for a in full.scenario.actors if a.id == car.id][0]
    assert car.motion == full_car.motion                       # 세계 좌표 그대로
    x0, y0, _h = full.route.point_at(2515.0)
    assert abs(cut.scenario.ego_start[0] - x0) < 2.0 and abs(cut.scenario.ego_start[1] - y0) < 2.0


def test_slice_변종은_흔든_뒤_구간_안_액터를_남긴다():
    from vtd_rl.world.board import load_board
    for v in range(4):
        cut = load_board(_g_entry(name="course_G_s2665", slice=[2515.0, 2785.0]), variant=v)
        assert cut.name == f"course_G_s2665@v{v}"
        assert len(cut.scenario.actors) == 1


def test_slice_가_없으면_예전과_같다():
    from vtd_rl.world.board import load_board
    a, b = load_board(_g_entry()), load_board(_g_entry())
    assert a.route.total == b.route.total and len(a.scenario.actors) == len(b.scenario.actors) == 4


@pytest.mark.parametrize("bad", [[2515.0], [2785.0, 2515.0], [2515.0, 2515.5], "x"])
def test_잘못된_slice_는_거부한다(bad):
    from vtd_rl.world.board import load_board
    with pytest.raises(ValueError):
        load_board(_g_entry(name="bad", slice=bad))
```

`tests/test_stage3s.py`(새로):

```python
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
    os.chdir(REPO)
    from vtd_rl.world.board import load_window
    boards, varied = load_window("curricula/stage3s.json", 4, 0)
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
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest <board 테스트 파일> tests/test_stage3s.py -q > $TMP/m7l_t1.txt 2>&1; echo rc=$?` → rc≠0.

- [ ] **Step 3: `board.py`.**

`BOARD_KEYS = {"name", "route", "lane", "actors", "jitter"}` 를 `BOARD_KEYS = {"name", "route", "lane", "actors", "jitter", "slice"}` 로 바꾼다.

`load_board` 끝의 `return board` 앞에(액터를 `sc.actors` 에 붙인 뒤):

```python
    cut = entry.get("slice")
    if cut is not None:
        # M7l — 어려운 장면 바로 앞에서 출발하는 짧은 판. 액터는 전체 판에서 (흔든 뒤) 지은 세계 좌표 그대로
        # 쓰고, 흔든 뒤 s 가 구간 안인 것만 남긴다. 구간 밖 액터는 자른 경로에서 만날 일이 없다.
        if not (isinstance(cut, (list, tuple)) and len(cut) == 2 and float(cut[1]) - float(cut[0]) >= 1.0):
            raise ValueError(f"판 항목 '{entry.get('name', '?')}' 의 slice 는 [s_from, s_to](s_to > s_from) 이어야 한다: {cut!r}")
        s0, s1 = float(cut[0]), float(cut[1])
        kept = [a for a, sp in zip(actors, specs) if s0 <= float(sp["s"]) <= s1]
        board = slice_board(board, s0, s1, name, signals)
        board.scenario.actors = kept
    return board
```

(`slice_board` 는 자른 구간이 너무 짧으면 이미 ValueError 를 낸다. 숫자가 아닌 slice 는 위 검사에서 걸린다. `"x"` 같은 문자열도 `len` 이 2 가 아니면 걸린다 — 길이 2 문자열이 들어오면 `float` 에서 ValueError 가 난다.)

- [ ] **Step 4: `curricula/stage3s.json`.**

`stage3.json` 에서 각 코스 항목(경로·차로·액터·흔들기)을 그대로 복사하고 이름과 `slice` 만 바꿔 위 표의 여섯 항목을 만든다. 머리는 `{"stage": 3, "name": "학습용 어려운 장면 짧은 판 — ③ 전체의 장애물 장면 바로 앞(약 150 m)에서 출발(평가에 안 씀)", "signals": "cycle", "boards": [...]}` 다. 생성은 작은 파이썬으로 한다(손으로 옮기지 않는다).

- [ ] **Step 5: 통과를 확인한다.**

Run: Step 2 와 같은 명령 → rc=0. 전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m7l_all.txt 2>&1; echo rc=$?` → rc=0.

- [ ] **Step 6: 커밋한다.**

```bash
git add vtd_rl/world/board.py curricula/stage3s.json <board 테스트 파일> tests/test_stage3s.py
git commit -m "M7l — 판 항목 slice(구간 자르기)와 어려운 장면 짧은 판 커리큘럼 stage3s"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `slice` 가 없는 판은 예전과 같다.
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. numpy 는 1.26.4 그대로다.

## Review Focus

- 흔든 뒤 s 로 액터를 거른다(흔들기 전 s 가 아니라).
- 자른 판의 출발점·목표·지속 시간은 `slice_board` 가 정한다. 세계 캐시·색인 캐시가 자른 판과 전체 판을 섞지 않는다(이름이 다르다).
- 평가 스크립트 기본 단계에 `stage3s` 가 없다.
