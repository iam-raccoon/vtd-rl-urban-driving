"""단계 ③b — 코스당 첫 **정지차** 하나만 둔 단계(차로변경 표적).

M4p 가 단계 ③a(코스당 첫 액터 하나)에서 처음으로 0% 가 아닌 완주율을 봤다(M3 16.7% →
M4o 30%). 판별로 갈라 보니 향상은 거의 전부 라바콘이었다(12/36 → 21/36). **정지차는
0/36 → 1/36** 으로 못 넘는다. 라바콘은 폭 0.46 m 라 살짝 비켜도 지나가지만 정지차는
4.5 × 1.8 m 에 횡 ±0.9 m 라 차로를 실제로 막는다 — 진짜 차로변경이 필요하다.

그리고 M4h 가 찾은 선생님의 손실(⑬ 차로변경 지시등 42 슬롯)이 **바로 그 비켜 가기**에서
나왔다. 학생이 못 하는 것과 선생님이 틀리는 것이 같은 자리라, 데이터를 그 동작에 모은다.

남기는 정지차는 **각 코스에서 경로상 첫 정지차**다. 단계 ③ 의 엄밀한 부분집합이어야
단계 ③·③a 결과와 비교가 서므로 그 관계를 잠근다.
"""
import json

from vtd_rl.world.board import load_curriculum

S3 = "curricula/stage3.json"
S3B = "curricula/stage3b.json"
VEHICLE_LEN = 2.2      # observation.py:60-71 — 이보다 길면 관측이 vehicle 로 분류한다


def _boards(path, **kw):
    return load_curriculum(path, **kw)[1]


def test_단계3b는_단계3과_같은_여섯_코스다():
    assert sorted(b.name for b in _boards(S3B)) == sorted(b.name for b in _boards(S3))


def test_판마다_액터는_정확히_하나다():
    for b in _boards(S3B):
        assert len(b.scenario.actors) == 1, (b.name, len(b.scenario.actors))


def test_전부_정지차다():
    """라바콘이 하나라도 섞이면 '정지차 차로변경' 이라는 표적이 흐려진다."""
    for b in _boards(S3B):
        a = b.scenario.actors[0]
        assert a.size[0] > VEHICLE_LEN, (b.name, a.size)
        assert a.motion["kind"] == "static", b.name


def test_신호는_단계3과_같이_주기다():
    assert all(b.signals == "cycle" for b in _boards(S3B))


def test_남긴_정지차는_단계3_같은_코스의_첫_정지차다():
    """★ 단계 ③ 의 엄밀한 부분집합. "첫" 은 정지차들 중 호 길이 `s` 최소다.

    원본 기술(JSON)로 비교한다 — 변종 흔들기·방위 계산이 끼면 같은 액터도 달라 보인다.
    """
    s3 = {b["name"]: b for b in json.load(open(S3, encoding="utf-8"))["boards"]}
    for b in json.load(open(S3B, encoding="utf-8"))["boards"]:
        veh = [a for a in s3[b["name"]]["actors"] if a["size"][0] > VEHICLE_LEN]
        assert b["actors"] == [min(veh, key=lambda a: a["s"])], b["name"]
        assert b.get("jitter") == s3[b["name"]].get("jitter"), b["name"]


def test_변종이_실제로_흩어진다():
    """jitter 가 안 걸리면 변종 N 벌이 같은 판이라 수집 다양성이 0 이다.

    ⚠ `jitter` 키를 지우는 것으로는 이 테스트를 못 깬다 — `DEFAULT_JITTER`(40 m / 0.6 m)가
    단계 ③ 값과 같아 동작이 안 바뀐다(M4p 에서 그 돌연변이가 무효였다). 끄려면 명시적 0 이다.
    """
    vs = _boards(S3B, variants=4)
    for course in {b.name.split("@")[0] for b in vs}:
        pos = {tuple(round(v, 6) for v in b.scenario.actors[0].motion["pos"])
               for b in vs if b.name.startswith(course + "@")}
        assert len(pos) == 4, (course, pos)
