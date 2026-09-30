import json

import pytest

from vtd_rl.world.board import load_curriculum

BASE = {"name": "t", "signals": "always_green",
        "boards": [{"name": "course_H", "route": "routes/HL_FMA_NEW_H.json",
                    "lane": "routes/HL_FMA_NEW_H_lane.json"}]}


def _write(tmp_path, d):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_모르는_최상위_키는_터진다(tmp_path):
    """★ 지금은 조용히 무시된다. 오타 하나가 액터 0 개인 판을 만들고 단계 ①② 와
    구별이 안 된다 — 실험 한 팔이 통째로 무의미해진다."""
    with pytest.raises(ValueError, match="actorz"):
        load_curriculum(_write(tmp_path, {**BASE, "actorz": []}))


def test_모르는_판_항목_키는_터진다(tmp_path):
    bad = {**BASE, "boards": [{**BASE["boards"][0], "aktors": []}]}
    with pytest.raises(ValueError, match="aktors"):
        load_curriculum(_write(tmp_path, bad))


def test_기존_stage1_stage2는_그대로_읽힌다():
    for name in ("stage1", "stage2"):
        label, boards = load_curriculum(f"curricula/{name}.json")
        assert len(boards) == 6
        assert all(not b.scenario.actors for b in boards)


def test_stage키는_허용하되_안_읽는다(tmp_path):
    """기존 파일이 `"stage": 1` 을 갖고 있다 — 허용 목록에 넣되 의미는 없다."""
    label, boards = load_curriculum(_write(tmp_path, {**BASE, "stage": 3}))
    assert len(boards) == 1


def test_jitter_오타는_변종을_안_써도_터진다(tmp_path):
    """★ 변종을 켤 때까지 숨어 있으면 안 된다 — 그때는 "왜 배치가 안 흔들리지" 가 된다."""
    bad = {**BASE, "boards": [{**BASE["boards"][0], "jitter": {"s": 40.0, "lat": 0.6}}]}
    with pytest.raises(ValueError, match="lat"):
        load_curriculum(_write(tmp_path, bad))


def test_variants가_1보다_작으면_터진다():
    with pytest.raises(ValueError, match="variants"):
        load_curriculum("curricula/stage3.json", variants=0)


def test_기존_커리큘럼은_jitter가_없어도_읽힌다():
    """stage1·stage2 에는 `jitter` 키가 없다 — 그대로 읽혀야 한다."""
    for name in ("stage1", "stage2"):
        _, boards = load_curriculum(f"curricula/{name}.json", variants=2)
        assert len(boards) == 12
        assert all("@v" in b.name for b in boards)


def test_판에_액터를_실으면_scenario에_들어간다(tmp_path):
    d = {**BASE, "boards": [{**BASE["boards"][0],
         "actors": [{"id": 1, "kind": "static", "type": "obstacle",
                     "s": 150.0, "lateral": 0.0, "size": [0.15, 0.46, 0.61]}]}]}
    _, boards = load_curriculum(_write(tmp_path, d))
    assert len(boards[0].scenario.actors) == 1
    assert boards[0].scenario.actors[0].id == 1
