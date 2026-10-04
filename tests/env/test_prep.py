import os

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.drivers.scripted import TeacherDriver
from vtd_rl.referee.core import clear_context_cache, context_for
from vtd_rl.rollout import run_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
SPEC = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "superpowers", "specs",
                    "2026-09-15-vtd-rl-urban-driving-design.md")


def test_판_입력은_판마다_한_번만_만든다():
    clear_context_cache()
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250")
    c1, c2 = context_for(b), context_for(b)
    assert c1 is not c2                      # notes 는 판마다 새로
    assert c1.secs is c2.secs and c1.tl_stops is c2.tl_stops and c1.cws is c2.cws
    c1.notes.append("x")
    assert c2.notes == []


def test_기록된_속도는_운전자가_아니라_세계_값이다():
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250")

    class LyingTeacher(TeacherDriver):
        def act(self, state, info, dt):
            cmd = super().act(state, info, dt)
            state.speed = 99.0               # 운전자가 로그를 더럽히려 해도
            return cmd

    run = run_episode(b, LyingTeacher(b))
    assert max(r["v"] for r in run.rows) < 30.0


def test_스펙_보상표는_채점기_항목_번호와_맞다():
    # 여기서는 표에 적힌 항목 번호가 score_fma.ITEMS 에 존재하는지만 본다. 그 항목이 실제로
    # 경미/중대 어느 등급까지 나오는지는 항목별 등급 능력표가 구현(심판·채점기) 어디에도 없어
    # 이 테스트로는 검증할 수 없다 — 아래 10 번 확인만 그 한 가지 예외를 좁게 잡는다.
    text = open(SPEC, encoding="utf-8").read()
    table = text.split("## 5. 보상")[1].split("## 6. 학습")[0]
    assert "11·14" in table                  # 충돌
    assert "13·15" not in table              # 옛 오기
    for line in table.splitlines():
        if line.startswith("|") and "·" in line:
            for num in line.split("|")[-2].replace("·", " ").split():
                if num.isdigit():
                    assert int(num) in rs.score_fma.ITEMS

    rows = {line.split("|")[1].strip(): line.split("|")[-2] for line in table.splitlines()
            if line.startswith("|") and "·" in line}
    assert "10" not in rows["경미 위반"].replace("·", " ").split()   # ⑩ 보행자 양보는 중대만
    assert "10" in rows["중대 위반"].replace("·", " ").split()


def test_cache_key_를_같이_쓰는_판은_심판_입력을_같이_쓰되_판은_자기_것이다():
    clear_context_cache()
    a = slice_board(load_board(H), 0.0, 250.0, "H_ctx_a")
    b = slice_board(load_board(H), 0.0, 250.0, "H_ctx_b")
    assert context_for(a).board is a and context_for(b).board is b
    assert context_for(a).secs is not context_for(b).secs   # 열쇠가 없으면 이름마다 따로
    clear_context_cache()
    a.cache_key = b.cache_key = "H|0|250"
    ca, cb = context_for(a), context_for(b)
    assert ca.secs is cb.secs and ca.tl_stops is cb.tl_stops   # 비싼 입력은 같이 쓴다
    assert ca.board is a and cb.board is b                     # 판은 자기 것
    clear_context_cache()
