import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vtd_rl import rule_stack as rs
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.env.reward import RewardConfig
from vtd_rl.world.board import load_board, load_curriculum, slice_board
from vtd_rl.world.world import WorldConfig

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def boards():
    b = load_board(H)
    return [slice_board(b, 0.0, 250.0, "H_0_250"), slice_board(b, 250.0, 500.0, "H_250_500")]


def rammer_board():
    """H 0~250 에 정지 차량 하나를 세운 판 — `referee/scenarios.py` 의 `vehicle_rammer` 와 같은 대본.

    액터 목록은 World 를 지을 때 읽히고 세계 캐시는 env 인스턴스마다 따로이므로, 이 판은 여기서
    새로 짓고(공유 `boards()` 재사용 금지) 이 판을 쓰는 env 도 매번 새로 만든다.
    """
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250_rammer")
    x, y, _h = b.route.point_at(60.0)
    b.scenario.actors = [rs.Actor(id=901, type="vehicle", size=[4.5, 1.8, 1.5],
                                  spawn={"at_time": 0.0}, motion={"kind": "static", "pos": [x, y]})]
    return b


def test_gymnasium_환경_검사():
    env = VtdDriveEnv([boards()[0]])
    check_env(env, skip_render_check=True)
    env.close()


def test_리셋은_시드로_재현된다():
    env = VtdDriveEnv(boards())
    o1, i1 = env.reset(seed=3)
    o2, i2 = env.reset(seed=3)
    assert i1["board"] == i2["board"]
    for k in o1:
        assert np.array_equal(o1[k], o2[k])
    picked = {env.reset(seed=s)[1]["board"] for s in range(12)}
    assert len(picked) == 2                      # 판을 섞어 고른다
    o3, i3 = env.reset(seed=3, options={"board": "H_250_500"})
    assert i3["board"] == "H_250_500"
    env.close()


def test_한_걸음은_두_프레임이고_심판이_따라온다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(
        {"control": np.array([0.0, 1.0], np.float32), "turn": 0})
    assert info["frames"] == 2
    assert env.world.t == pytest.approx(0.1)
    assert not terminated and not truncated
    assert info["reward_terms"]["progress"] >= 0.0
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    assert env.referee.ctx.secs is not None
    env.close()


def test_과속하면_위반_보상이_깎인다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    worst, items = 0.0, set()
    for _ in range(300):
        _o, r, term, trunc, info = env.step({"control": np.array([0.0, 1.0], np.float32), "turn": 0})
        worst = min(worst, info["reward_terms"]["violation"])
        items |= {h[2] for h in info["hits"]}
        if term or trunc:
            break
    assert worst < 0.0 and 1 in items            # 제한 50 인데 계속 가속하면 항목 1
    env.close()


def test_판이_끝나면_성적이_따라온다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    for _ in range(2000):
        _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 0.6], np.float32), "turn": 0})
        if term or trunc:
            break
    assert info["outcome"] in ("goal", "offroad", "timeout", "stalled", "collision")
    assert not (term and trunc)                  # Gymnasium 규약 — 둘이 함께 참이면 안 된다
    assert "result" in info and "sheet" in info["result"]
    assert len(info["result"]["sheet"]) == 5
    env.close()


def test_행_CSV_남기기(tmp_path):
    env = VtdDriveEnv([boards()[0]], EnvConfig(log_dir=str(tmp_path)))
    env.reset(seed=0)
    for _ in range(5):
        env.step({"control": np.array([0.0, 0.5], np.float32), "turn": 0})
    env.reset(seed=1)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files and files[0].startswith("H_0_250-")
    lines = open(tmp_path / files[0], encoding="utf-8").readlines()
    assert lines[0].startswith("t,x,y,heading,v")
    assert len(lines) == 5 * 2 + 1        # 머리글 + 시뮬 프레임마다 한 행(걸음마다 두 프레임)
    env.close()


def test_밟지_않은_리셋은_CSV_를_남기지_않는다(tmp_path):
    """행 CSV 는 첫 record 에서 연다 — 리셋만 하고 버린 판이 머리글뿐인 파일을 남기지 않게."""
    env = VtdDriveEnv([boards()[0]], EnvConfig(log_dir=str(tmp_path)))
    env.reset(seed=0)
    env.reset(seed=1)
    assert list(tmp_path.iterdir()) == []
    env.step({"control": np.array([0.0, 0.5], np.float32), "turn": 0})
    assert [p.name for p in tmp_path.iterdir()] == ["H_0_250-0002.csv"]
    env.close()


def test_종료_걸쇠는_terminated_와_truncated_를_함께_켜지_않는다():
    """충돌(terminated)과 시간초과(truncated)가 **같은 걸음**에 와도 하나만 켜진다(Gymnasium 규약)."""
    def ram(time_limit=None):
        env = VtdDriveEnv([rammer_board()])
        env.reset(seed=0)
        if time_limit is not None:
            env.world.time_limit = time_limit
        for _ in range(300):
            _o, _r, term, trunc, info = env.step(
                {"control": np.array([0.0, 1.0], np.float32), "turn": 0})
            if term or trunc:
                break
        env.close()
        return term, trunc, info

    _term, _trunc, hit_info = ram()
    term, trunc, info = ram(time_limit=hit_info["sim_time"])   # 충돌하는 그 걸음에 시간초과를 맞춘다
    assert info["outcome"] == "timeout"                        # 세계는 시간초과라고 했고
    assert 14 in {h[2] for h in info["hits"]}                  # 같은 걸음에 충돌 판정도 났다
    assert (term, trunc) == (True, False)                      # 그래도 둘이 함께 켜지지는 않는다


def test_frame_hook은_시뮬_프레임마다_불린다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    clocks = []
    env.frame_hook = lambda state, clock: clocks.append(clock)
    n_steps = 5
    for _ in range(n_steps):
        _o, _r, term, trunc, _info = env.step(
            {"control": np.array([0.0, 0.5], np.float32), "turn": 0})
        assert not term and not trunc
    assert len(clocks) == 2 * n_steps
    assert all(a <= b for a, b in zip(clocks, clocks[1:]))
    env.close()


def test_종료_후_다시_밟아도_성적표가_그대로다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    info = None
    for _ in range(60):
        _o, _r, term, trunc, info = env.step(
            {"control": np.array([1.0, 0.6], np.float32), "turn": 0})   # 최대 조향으로 도로 이탈 유도
        if term or trunc:
            break
    assert term or trunc
    assert "result" in info
    sheet_before = [dict(s) for s in env.referee.sheet.state]

    calls = []
    env.frame_hook = lambda state, clock: calls.append(clock)
    _o2, reward2, term2, trunc2, info2 = env.step(
        {"control": np.array([0.0, 0.0], np.float32), "turn": 0})

    assert [dict(s) for s in env.referee.sheet.state] == sheet_before
    assert "result" not in info2
    assert reward2 == 0.0
    assert calls == []                    # 이미 끝난 판은 프레임을 더 밟지 않는다
    assert (term2, trunc2) == (term, trunc)
    env.close()


def test_충돌하면_종료_사유가_충돌이고_다시_밟아도_성적표가_그대로다():
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    info = None
    for _ in range(300):
        _o, _r, term, trunc, info = env.step(
            {"control": np.array([0.0, 1.0], np.float32), "turn": 0})   # 똑바로 최대 가속
        if term or trunc:
            break
    assert term and not trunc                # 세계는 running 인 채 충돌로 끝난다 — 도로 이탈이 아니다
    assert info["outcome"] == "collision"
    assert 14 in {h[2] for h in info["hits"]}
    assert "result" in info
    sheet_before = [dict(s) for s in env.referee.sheet.state]

    calls = []
    env.frame_hook = lambda state, clock: calls.append(clock)
    _o2, reward2, term2, trunc2, info2 = env.step(
        {"control": np.array([0.0, 0.0], np.float32), "turn": 0})

    assert [dict(s) for s in env.referee.sheet.state] == sheet_before
    assert "result" not in info2
    assert reward2 == 0.0
    assert calls == []
    assert (term2, trunc2) == (term, trunc)
    env.close()


def test_리셋_전에_스텝하면_명확한_오류():
    env = VtdDriveEnv([boards()[0]])
    with pytest.raises(RuntimeError):
        env.step({"control": np.array([0.0, 0.0], np.float32), "turn": 0})
    env.close()


def test_없는_판_이름은_명확한_오류():
    env = VtdDriveEnv([boards()[0]])
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"board": "없는판"})
    env.close()


def test_intent가_승차감에_닿는다():
    boards = load_curriculum("curricula/stage1.json")[1]
    env = VtdDriveEnv(boards, EnvConfig(reward=RewardConfig(comfort_on_intent=True)))
    try:
        env.intent = (0.0, 0.0)          # 의도는 내내 0
        env.reset(seed=0, options={"board": boards[0].name})
        big = {"control": np.array([1.0, 1.0], np.float32), "turn": 0}
        _o, _r, _t, _tr, info = env.step(big)
        assert abs(info["reward_terms"]["comfort"]) < 1e-9
    finally:
        env.close()


@pytest.mark.slow
def test_대회_점수는_보상_설정에_영향받지_않는다():
    """보상을 바꿔도 `score_fma` 점수·위반은 그대로여야 한다 — 그게 이 실험이 성립하는 근거다.

    보상은 학습 신호고 점수는 채점기가 따로 낸다. 이 독립성이 깨지면 M4c 의 성적표를
    M4a·M4b 와 비교할 수 없다.
    """
    boards = load_curriculum("curricula/stage1.json")[1]
    results = []
    for cfg in (RewardConfig(),
                RewardConfig(comfort_steer=-99.0, comfort_accel=-99.0, comfort_on_intent=True)):
        env = VtdDriveEnv(boards, EnvConfig(reward=cfg))
        try:
            env.reset(seed=0, options={"board": boards[0].name})
            info = None
            for _ in range(20000):        # 판이 끝날 때까지 — 같은 행동이면 같은 궤적이다
                _o, _r, term, trunc, info = env.step(
                    {"control": np.array([0.05, 0.6], np.float32), "turn": 0})
                if term or trunc:
                    break
            assert info is not None and "result" in info, "판이 안 끝났다"
            results.append((info["result"]["score"], info["result"]["sheet"]))
        finally:
            env.close()
    assert results[0] == results[1]


@pytest.mark.slow
def test_구간당_한_번_모드의_위반_합계가_채점기_감점과_같다():
    """보상의 위반 항이 대회 채점기가 실제로 깎은 점수와 일치해야 한다.

    `info["result"]["sheet"]` 는 심판이 낸 구간별 감점표다 — 실측(2026-09-27, `H_0_250`
    보드·seed=0·이 판)으로 브리프가 가정한 "구간마다 `{항목: 등급}` dict 의 리스트" 모양과
    정확히 일치함을 확인했다(`vtd_rl/env/drive_env.py:_finish` 의
    `[dict(s) for s in sheet.state]`, `third_party/rule_stack` 의 `Sheet.state` 정의 그대로).
    그 표의 위반 감점 합(충돌 항목 ⑪⑭ 제외)과, 판 내내 쌓은 `reward_terms["violation"]` 의
    합이 같아야 한다 — 그게 "보상이 채점과 같은 것을 잰다" 는 이 마일스톤의 정의다.

    항목 15(리스폰) 예외(`ViolationTracker._charge_once`의 독스트링 참고)는 여기서는 안 건드린다
    — 이 판은 리스폰이 한 번도 없다(`info["result"]["respawns"] == {}`, 아래에서 확인),
    그러니 빼야 할 15 짜리 항이 애초에 없다.
    """
    from vtd_rl.env.reward import COLLISION_ITEMS

    boards = load_curriculum("curricula/stage1.json")[1]
    cfg = RewardConfig(violation_mode="once_per_section")
    env = VtdDriveEnv(boards, EnvConfig(reward=cfg))
    try:
        env.reset(seed=0, options={"board": boards[0].name})
        total, info = 0.0, None
        for _ in range(20000):
            _o, _r, term, trunc, info = env.step(
                {"control": np.array([0.05, 0.6], np.float32), "turn": 0})
            total += info["reward_terms"]["violation"]
            if term or trunc:
                break
        assert info is not None and "result" in info, "판이 안 끝났다"
        assert not info["result"]["respawns"], "이 판에 리스폰이 있다 — 항목 15 예외를 다시 따져라"
        sheet = info["result"]["sheet"]
        # sheet 는 구간별 {항목: 등급} 이다. 충돌 항목은 보상에서 따로 처리하므로 뺀다.
        expect = sum(cfg.major if lv == "major" else cfg.minor
                     for sec in sheet for item, lv in sec.items()
                     if int(item) not in COLLISION_ITEMS)
        assert total == pytest.approx(expect, abs=1e-6), (total, expect)
    finally:
        env.close()


def test_판_고르기_갈고리가_없으면_예전처럼_고른다():
    e1, e2 = VtdDriveEnv(boards()), VtdDriveEnv(boards())
    e2.board_sampler = None
    for seed in (0, 3, 7):
        o1, _ = e1.reset(seed=seed)
        o2, _ = e2.reset(seed=seed)
        assert e1.board.name == e2.board.name
        for k in o1:
            assert np.array_equal(np.asarray(o1[k]), np.asarray(o2[k])), k


def test_판_고르기_갈고리가_새로_지은_판을_쓰고_세계를_캐시하지_않는다():
    env = VtdDriveEnv(boards())
    made = []

    def sampler(rng):
        b = slice_board(load_board(H), 0.0, 250.0, f"H_fresh_{len(made)}")
        made.append(b)
        return b, True

    env.board_sampler = sampler
    for seed in range(3):
        env.reset(seed=seed)
        assert env.board is made[-1] and env.world.board is made[-1]
    assert len(env._worlds) == 0
    env.reset(seed=0, options={"board": "H_0_250"})     # 이름을 주면 예전처럼 목록에서, 캐시도 한다
    assert env.board.name == "H_0_250" and len(env._worlds) == 1


def test_판_고르기_갈고리가_원본을_주면_세계를_캐시한다():
    bs = boards()
    env = VtdDriveEnv(bs)
    env.board_sampler = lambda rng: (bs[1], False)
    env.reset(seed=0)
    env.reset(seed=1)
    assert env.board is bs[1] and list(env._worlds) == ["H_250_500"]


def _signal_board(mode):
    """H 0~250 — s≈96 m 에 신호(151)가 있다. 신호 운용만 바꾼다."""
    return slice_board(load_board(H), 0.0, 250.0, f"H_0_250_{mode}", signals=mode)


def test_red_gap_은_적색일_때만_앞범퍼_거리와_속도를_준다():
    env = VtdDriveEnv([_signal_board("always_red")])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    gap, v = env._red_gap()
    sig_s = min(sig.s for sig in env.world.reporter.signals)
    assert gap == pytest.approx(sig_s - env._info.s - rs.score_fma.FRONT)
    assert v == pytest.approx(env.world.ego.v)
    env.close()
    green = VtdDriveEnv([_signal_board("always_green")])
    green.reset(seed=0)
    green.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    assert green._red_gap() is None
    green.close()


def test_red_profile_을_켜면_적색을_달려_지날_때_red_항이_깎인다():
    def run(cfg):
        env = VtdDriveEnv([_signal_board("always_red")], EnvConfig(reward=cfg))
        env.reset(seed=0)
        reds = []
        for _ in range(400):
            _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 1.0], dtype=np.float32), "turn": 0})
            reds.append(info["reward_terms"].get("red"))
            if term or trunc or info["s"] > 110.0:
                break
        env.close()
        return reds
    on = run(RewardConfig(red_profile=0.1))
    assert min(on) < 0.0
    off = run(RewardConfig())
    assert all(r is None for r in off)


def test_lane_profile_을_켜면_오른쪽으로_치우쳐_달릴_때_lane_항이_깎인다():
    """H 0~250 을 오른쪽으로 조금씩 꺾으며 달려 길 가장자리를 문다 — 켜면 어느 걸음엔가 lane < 0."""
    def run(cfg):
        env = VtdDriveEnv([boards()[0]], EnvConfig(reward=cfg))
        env.reset(seed=0)
        lanes = []
        for k in range(300):
            steer = -0.05 if k > 40 else 0.0
            _o, _r, term, trunc, info = env.step({"control": np.array([steer, 0.3], dtype=np.float32), "turn": 0})
            lanes.append(info["reward_terms"].get("lane"))
            if term or trunc:
                break
        env.close()
        return lanes
    on = run(RewardConfig(lane_profile=5.0))
    assert min(on) < 0.0
    assert all(v is None for v in run(RewardConfig()))


def test_lane_항은_행이_풀리는_뒤_중간_걸음에도_오고_마지막_걸음에_남은_행이_들어온다():
    """살살 꺾어 길 가장자리로 천천히 나가면(조향 −0.02) 물린 행이 2.5 초 뒤 풀려 중간 걸음에도 lane < 0 이고,

    판이 끝나는 걸음(offroad)은 `finish()` 가 풀어 준 남은 행까지 받아 가장 크게 깎인다.
    (−0.05 처럼 세게 꺾으면 곧바로 길을 벗어나 마지막 걸음의 한 번만 깎인다.)
    """
    env = VtdDriveEnv([boards()[0]], EnvConfig(reward=RewardConfig(lane_profile=5.0)))
    env.reset(seed=0)
    lanes, outcome = [], None
    for k in range(400):
        steer = -0.02 if k > 40 else 0.0
        _o, _r, term, trunc, info = env.step({"control": np.array([steer, 0.3], dtype=np.float32), "turn": 0})
        lanes.append(info["reward_terms"]["lane"])
        if term or trunc:
            outcome = info["outcome"]
            break
    env.close()
    assert outcome == "offroad"
    assert any(v < 0.0 for v in lanes[:-1])           # 판이 끝나기 전에도 풀린 행이 깎는다
    assert lanes[-1] < min(lanes[:-1])                # 끝나는 걸음은 남은 행까지 받아 가장 크다
    assert all(v <= 0.0 and str(v) != "-0.0" for v in lanes)


def test_lat_gap_은_횡오차_크기와_물체_거리를_준다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    lat, near = env._lat_gap()
    assert lat == pytest.approx(abs(env._info.lateral))
    assert near == float("inf")                     # 이 판에는 물체가 없다
    env.close()
    ram = VtdDriveEnv([rammer_board()])
    ram.reset(seed=0)
    ram.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    _lat, near = ram._lat_gap()
    assert 50.0 < near < 70.0                       # 60 m 앞 정지 차량
    ram.close()


def test_lat_profile_을_켜면_경로를_벗어날_때_lat_항이_깎인다():
    def run(cfg):
        env = VtdDriveEnv([boards()[0]], EnvConfig(reward=cfg))
        env.reset(seed=0)
        lats = []
        for k in range(120):
            steer = -0.03 if k > 20 else 0.0
            _o, _r, term, trunc, info = env.step({"control": np.array([steer, 0.3], dtype=np.float32), "turn": 0})
            lats.append(info["reward_terms"].get("lat"))
            if term or trunc:
                break
        env.close()
        return lats
    on = run(RewardConfig(lat_profile=0.5))
    assert min(on) < 0.0
    assert all(v is None for v in run(RewardConfig()))


def _stall_run(cfg):
    """H 0~250 에서 가속 0 으로 서 있는다 — 세계의 정체 판정(stall_seconds)까지 간다."""
    env = VtdDriveEnv([boards()[0]], EnvConfig(world=WorldConfig(stall_seconds=3.0), reward=cfg))
    env.reset(seed=0)
    for _ in range(200):
        _o, r, term, trunc, info = env.step({"control": np.array([0.0, -1.0], dtype=np.float32), "turn": 0})
        if term or trunc:
            break
    env.close()
    return r, term, trunc, info


def test_stall_이_0이면_정체는_예전처럼_잘리기만_한다():
    r, term, trunc, info = _stall_run(RewardConfig())
    assert info["outcome"] == "stalled" and trunc and not term
    assert "stall" not in info["reward_terms"]


def test_stall_을_주면_정체는_실패로_끝나고_벌을_문다():
    r, term, trunc, info = _stall_run(RewardConfig(stall=-200.0))
    assert info["outcome"] == "stalled" and term and not trunc
    assert info["reward_terms"]["stall"] == -200.0
    assert r == pytest.approx(sum(info["reward_terms"].values()))


def test_obs_gap_은_앞길_위_물체의_거리와_다가가는_속도를_준다():
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    gap, v_close = env._obs_gap()
    # 경로 60 m 의 차(길이 4.5) — 뒷축 s≈0 에서 앞범퍼(3.808)·차 반길이(2.25)를 뺀 거리
    assert 50.0 < gap < 56.0
    assert v_close == pytest.approx(env.world.ego.v, abs=1e-6)
    env.close()
    empty = VtdDriveEnv([boards()[0]])
    empty.reset(seed=0)
    empty.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    assert empty._obs_gap() is None
    empty.close()


def test_obs_profile_을_켜면_정지차에_달려들_때_obs_항이_깎인다():
    def run(cfg):
        env = VtdDriveEnv([rammer_board()], EnvConfig(reward=cfg))
        env.reset(seed=0)
        vals = []
        for _ in range(300):
            _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 1.0], dtype=np.float32), "turn": 0})
            vals.append(info["reward_terms"].get("obs"))
            if term or trunc:
                break
        env.close()
        return vals
    on = run(RewardConfig(obs_profile=0.5))
    assert min(on) < 0.0
    assert all(v is None for v in run(RewardConfig()))


def test_block_gap_은_멈춘_차량은_보고_움직이는_물체는_안_본다():
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    gap = env._block_gap()
    assert gap is not None and abs(gap - env._obs_gap()[0]) < 1e-9     # 같은 통로·거리 계산
    env.close()
    empty = VtdDriveEnv([boards()[0]])
    empty.reset(seed=0)
    empty.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    assert empty._block_gap() is None
    empty.close()


def test_block_gap_은_사람과_움직이는_물체를_뺀다(monkeypatch):
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    car = env.state.objects[0]
    from dataclasses import replace
    moving = replace(car, speed=3.0)
    person = replace(car, length=0.5, width=0.5, height=1.7)
    for obj in (moving, person):
        monkeypatch.setattr(env.state, "objects", [obj])
        assert env._block_gap() is None
    env.close()


def test_block_profile_을_켜면_멈춘_차_뒤에서_block_항이_깎인다():
    """정지 차량 판을 천천히 달려 차 앞 15 m 안에 들어가면 `block` 항이 생기고, 끄면 키가 없다(M7g 리뷰)."""
    def run(cfg):
        env = VtdDriveEnv([rammer_board()], EnvConfig(reward=cfg))
        env.reset(seed=0)
        vals = []
        for _ in range(400):
            _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 0.3], dtype=np.float32), "turn": 0})
            vals.append(info["reward_terms"].get("block"))
            if term or trunc:
                break
        env.close()
        return vals
    on = run(RewardConfig(block_profile=0.5))
    assert set(on) == {-0.5, 0.0}
    assert all(v is None for v in run(RewardConfig()))


def test_ovl_gap_은_겹친_폭에_가까움_가중을_곱한다(monkeypatch):
    from dataclasses import replace
    env = VtdDriveEnv([rammer_board()])
    env.reset(seed=0)
    env.step({"control": np.array([0.0, 0.0], dtype=np.float32), "turn": 0})
    assert env._ovl_gap() is None                     # 60 m 앞 — ovl_range 30 m 밖
    ego = env.world.ego
    car = env.state.objects[0]
    import math
    def put(fx, fy, **kw):                            # 내 차 기준 (fx, fy) 에 물체를 놓는다
        c, s = math.cos(ego.heading), math.sin(ego.heading)
        return replace(car, x=ego.x + fx * c - fy * s, y=ego.y + fx * s + fy * c, **kw)
    gap_of = lambda fx: fx - rs.score_fma.FRONT - car.length / 2.0
    lim = car.width / 2.0 + 0.943 + 0.3
    monkeypatch.setattr(env.state, "objects", [put(20.0, -1.0)])
    assert env._ovl_gap() == pytest.approx((lim - 1.0) * (1.0 - gap_of(20.0) / 30.0))
    monkeypatch.setattr(env.state, "objects", [put(20.0, -(lim + 0.1))])      # 통로 밖
    assert env._ovl_gap() is None or env._ovl_gap() == pytest.approx(0.0)
    monkeypatch.setattr(env.state, "objects", [put(20.0, -1.0, speed=3.0)])  # 움직이는 물체
    assert env._ovl_gap() is None
    monkeypatch.setattr(env.state, "objects", [put(-5.0, -1.0)])              # 뒤
    assert env._ovl_gap() is None
    env.close()


def test_ovl_profile_을_켜면_멈춘_차에_겹친_채_다가갈_때_ovl_항이_깎인다():
    def run(cfg):
        env = VtdDriveEnv([rammer_board()], EnvConfig(reward=cfg))
        env.reset(seed=0)
        vals = []
        for _ in range(400):
            _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 0.3], dtype=np.float32), "turn": 0})
            vals.append(info["reward_terms"].get("ovl"))
            if term or trunc:
                break
        env.close()
        return vals
    on = run(RewardConfig(ovl_profile=2.0))
    assert min(on) < 0.0
    assert all(v is None for v in run(RewardConfig()))
