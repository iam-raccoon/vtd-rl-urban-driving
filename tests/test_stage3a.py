"""단계 ③a — 코스당 첫 액터 하나만 둔 쉬운 단계.

M4o 가 처음으로 단계 ③ 에서 거리를 늘렸지만(안 쓴 변종 72 판 +10~14%) 완주는 0% 였다.
코스마다 액터를 4~5 개 **연속으로** 넘겨야 끝나는데 학생은 하나를 가끔 넘기는 수준이라,
완주율 축이 늘 0 이어서 아무것도 못 가린다. 그래서 액터를 하나로 줄여 완주가 가능한 판을 둔다.

남기는 액터는 **각 코스에서 경로상 첫 번째**다 — 학생이 지금 죽는 바로 그 자리라
(M4i: H 185 걸음 ≈ s200 라바콘, D 268 걸음 ≈ s310 정지차) M4i·M4o 진단과 이어진다.
그 관계가 깨지면 "쉬운 단계" 의 결과를 단계 ③ 와 비교할 수 없게 되므로 테스트로 잠근다.
"""
from vtd_rl.world.board import load_curriculum

S3 = "curricula/stage3.json"
S3A = "curricula/stage3a.json"


def _boards(path, **kw):
    return load_curriculum(path, **kw)[1]


def test_단계3a는_단계3과_같은_여섯_코스다():
    assert sorted(b.name for b in _boards(S3A)) == sorted(b.name for b in _boards(S3))


def test_판마다_액터는_정확히_하나다():
    for b in _boards(S3A):
        assert len(b.scenario.actors) == 1, (b.name, len(b.scenario.actors))


def test_신호는_단계3과_같이_주기다():
    assert all(b.signals == "cycle" for b in _boards(S3A))


def test_남긴_액터는_단계3_같은_코스의_첫_액터다():
    """★ 단계 ③ 의 엄밀한 부분집합이어야 M4i·M4o 결과와 비교가 선다.

    "첫 번째" 는 경로상 호 길이 `s` 기준이다. 세계 좌표가 아니라 원본 기술(JSON)로 비교한다
    — 변종 흔들기·방위 계산이 끼면 같은 액터인데도 값이 달라 보일 수 있다.
    """
    import json
    s3 = {b["name"]: b for b in json.load(open(S3, encoding="utf-8"))["boards"]}
    s3a = json.load(open(S3A, encoding="utf-8"))["boards"]
    for b in s3a:
        first = min(s3[b["name"]]["actors"], key=lambda a: a["s"])
        assert b["actors"] == [first], b["name"]
        assert b.get("jitter") == s3[b["name"]].get("jitter"), b["name"]


def test_단계_전체로는_장애물과_차량이_둘_다_있다():
    """관측의 분류 채널(observation.py:60-71, 길이 2.2 m 문턱)이 둘 다 쓰이게 한다.
    판마다는 하나뿐이라 판 단위가 아니라 단계 단위로 본다."""
    lens = [b.scenario.actors[0].size[0] for b in _boards(S3A)]
    assert any(l <= 2.2 for l in lens), "장애물이 없다"
    assert any(l > 2.2 for l in lens), "차량이 없다"


def test_정지물만_쓴다():
    for b in _boards(S3A):
        assert b.scenario.actors[0].motion["kind"] == "static", b.name


def test_변종이_실제로_흩어진다():
    """jitter 가 따라왔는지 — 안 따라오면 변종 N 벌이 같은 판이라 수집 다양성이 0 이다."""
    vs = _boards(S3A, variants=4)
    for course in {b.name.split("@")[0] for b in vs}:
        pos = {tuple(round(v, 6) for v in b.scenario.actors[0].motion["pos"])
               for b in vs if b.name.startswith(course + "@")}
        assert len(pos) == 4, (course, pos)
