"""VTD 주행기 — 오프라인 세계에 물리면 환경과 같은 판을 내는가, 9910 소켓으로 끝까지 도는가."""
import importlib.util
import json
import os
import socket
import threading
import time

import pytest
import torch

from vtd_rl import rule_stack as rs
from vtd_rl.bridge.driver import VtdDriver, summarize
from vtd_rl.env.board_index import board_index
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.env.observation import ObsConfig
from vtd_rl.env.teacher_policy import run_teacher_in_env
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.world.board import load_curriculum
from vtd_rl.world.world import World

import vtd_io  # noqa: E402  rule_stack 이 경로를 잡아 둔다

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "drive_vtd.py")
CFG = EnvConfig(obs=ObsConfig(lim_anticipate=True))


def slice_board(name="course_G_s2665"):
    """M7l 의 짧은 판(정지차 앞 약 150 m, 신호 주기) — 판 하나가 2 초 안에 끝난다."""
    _, boards = load_curriculum("curricula/stage3s.json")
    return next(b for b in boards if b.name == name)


def drive_world(board, driver, seed):
    """주행기를 오프라인 세계에 물려 한 판 — VTD 대신 세계가 State 를 낸다."""
    world = World(board, CFG.world, signals=board_index(board).signals, seed=seed)
    state = world.reset(seed)
    while True:
        steer, accel, turn = driver.step(state, world.t)
        if driver.done:
            return driver.finish()
        state, _ = world.step(steer, accel, turn)


def env_episode(board, policy=None):
    """환경으로 한 판 — (마지막 info, 세계 시드, 걸음 수)."""
    env = VtdDriveEnv([board], CFG)
    if policy is None:
        out = run_teacher_in_env(env, seed=0, options={"board": board.name})
        return out["info"], env.world.seed, out["steps"]
    obs, info = env.reset(seed=0, options={"board": board.name})
    steps = 0
    while True:
        obs, _r, term, trunc, info = env.step(policy.act(obs, deterministic=True))
        steps += 1
        if term or trunc:
            return info, env.world.seed, steps


def random_policy():
    torch.manual_seed(0)
    return DrivePolicy(PolicyConfig())


@pytest.mark.parametrize("name", ["course_G_s2665", "course_A_s1470"])
def test_학생_주행기는_오프라인_세계에서_환경과_같은_판을_낸다(name):
    # 무작위 그물은 금방 길을 벗어나 판이 짧다 — 판단 수까지 같아야 판단 시각이 환경과 같은 것이다
    board, pol = slice_board(name), random_policy()
    info, seed, steps = env_episode(board, pol)
    res = drive_world(board, VtdDriver(board, policy=pol, obs_cfg=CFG.obs), seed)
    assert res["outcome"] == info["outcome"]
    assert res["s"] == pytest.approx(info["s"], abs=1e-9)
    assert res["decisions"] == steps
    assert res["score"] == info["result"]["score"]
    assert res["sheet"] == info["result"]["sheet"]


def test_선생님_주행기는_오프라인_세계에서_환경과_같은_판을_낸다():
    # 환경의 순서를 그대로 따른다: 판단에는 직전 프레임의 선생님 명령, 첫 프레임은 규칙 스택을 두 번 밟는다.
    # 정지차를 비켜 완주하는 판이라 선생님 명령이 한 프레임만 어긋나도 끝 위치가 cm 단위로 달라진다.
    board = slice_board()
    info, seed, steps = env_episode(board)
    res = drive_world(board, VtdDriver(board, teacher=True, obs_cfg=CFG.obs), seed)
    assert res["outcome"] == info["outcome"] == "goal"
    assert res["s"] == pytest.approx(info["s"], abs=1e-9)
    assert res["decisions"] == steps
    assert res["score"] == info["result"]["score"]
    assert res["sheet"] == info["result"]["sheet"]


def standing(board, t):
    x, y, h = board.start_pose
    return rs.State(x=x, y=y, heading=h, t=t)


@pytest.mark.parametrize("hz, expect", [(20, 20), (25, 20)])
def test_판단은_프레임율과_상관없이_평균_10_Hz(hz, expect):
    board = slice_board()
    driver = VtdDriver(board, policy=random_policy(), obs_cfg=CFG.obs)
    for k in range(2 * hz):                   # 2 초
        driver.step(standing(board, k / hz), k / hz)
    assert not driver.done
    assert driver.decisions == expect


def test_프레임이_끊겼다_오면_판단을_몰아서_하지_않고_다시_센다():
    board = slice_board()
    driver = VtdDriver(board, policy=random_policy(), obs_cfg=CFG.obs)
    for t in (0.0, 0.05, 0.1, 0.15, 0.9, 0.95, 1.0, 1.05):   # 0.15 → 0.9 사이 0.75 초가 비었다
        driver.step(standing(board, t), t)
    # 0.0 · 0.1 · 0.9(밀린 몫은 한 번만) · 1.0
    assert driver.decisions == 4


def test_정책과_선생님은_정확히_하나만():
    board = slice_board()
    with pytest.raises(ValueError):
        VtdDriver(board)
    with pytest.raises(ValueError):
        VtdDriver(board, policy=random_policy(), teacher=True)


def test_리스폰은_세고_경로_투영을_처음부터_다시_찾는다():
    board = slice_board()
    driver = VtdDriver(board, policy=random_policy(), obs_cfg=CFG.obs)
    x, y, h = board.start_pose
    driver.step(rs.State(x=x, y=y, heading=h), 0.0)
    far_x, far_y, far_h = board.route.point_at(200.0)
    st = rs.State(x=far_x, y=far_y, heading=far_h)
    st.respawned = True
    driver.step(st, 0.05)
    assert driver.respawns == 1
    assert driver.info.s == pytest.approx(200.0, abs=1.0)   # 창 60 점 안에서만 찾았다면 못 닿는 자리
    assert not driver.done                                    # 순간이동이 이탈·완주로 잘못 끝나지 않는다


def test_판이_끝나면_명령은_제동이고_finish_는_한_번만_닫는다():
    board, pol = slice_board(), random_policy()
    driver = VtdDriver(board, policy=pol, obs_cfg=CFG.obs)
    res = drive_world(board, driver, seed=0)
    assert driver.done
    steer, accel, turn = driver.step(rs.State(x=0.0, y=0.0), 999.0)
    assert (steer, turn) == (0.0, rs.TS_OFF) and accel < 0.0
    assert driver.finish() is res
    sm = summarize(res)
    assert sm["outcome"] == res["outcome"] and 0.0 <= sm["progress"] <= 1.01


class FakeVtd:
    """9910 흉내 — 오프라인 세계를 20 Hz 벽시계로 굴리며 DataPacket 을 보내고 CtrlPacket 을 받는다."""

    def __init__(self, board):
        self.world = World(board, CFG.world, signals=board_index(board).signals, seed=0)
        self.state = self.world.reset(0)
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.ctrls = []
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        conn, _ = self.srv.accept()
        conn.settimeout(2.0)
        buf = b""
        try:
            while True:
                s = self.state
                try:
                    conn.sendall(vtd_io.pack_data(s.x, s.y, 0.0, s.heading, 0.0, 0.0, s.objects, s.tl_id, s.tl_state))
                except OSError:
                    pass                  # 손님이 끊었다 — 남은 제어 패킷은 아래에서 마저 읽는다
                d = conn.recv(64)
                if not d:
                    return
                buf += d
                while len(buf) >= 9:
                    self.ctrls.append(vtd_io.parse_ctrl(buf[:9]))
                    buf = buf[9:]
                steer, accel, turn = self.ctrls[-1]
                self.state, _ = self.world.step(steer, accel, turn)
                time.sleep(self.world.cfg.dt)
        except OSError:
            return
        finally:
            conn.close()
            self.srv.close()


def test_drive_vtd_는_9910_소켓으로_달리고_성적을_남긴다(tmp_path):
    spec = importlib.util.spec_from_file_location("drive_vtd_under_test", SCRIPT)
    drive_vtd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drive_vtd)
    fake = FakeVtd(slice_board())
    out = tmp_path / "run"
    drive_vtd.main(["--board", "course_G_s2665", "--curriculum", "curricula/stage3s.json", "--teacher",
                    "--port", str(fake.port), "--out", str(out), "--max-seconds", "1.5"])
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert res["outcome"] == "aborted"                  # --max-seconds 로 끊었다
    assert res["frames"] >= 15 and len(fake.ctrls) >= 15
    assert res["summary"]["progress"] > 0.0              # 실제로 움직였다
    assert (out / "rows.csv").read_text(encoding="utf-8").count("\n") >= 15
    ctrl = (out / "ctrl.csv").read_text(encoding="utf-8").splitlines()
    assert ctrl[0] == "t,x,y,heading,speed,steer,accel,turn,policy_steer" and len(ctrl) - 1 == res["frames"]
    fake.thread.join(timeout=5.0)
    assert fake.ctrls[-1][1] < 0.0 and fake.ctrls[-1][0] == 0.0   # 끊기 전 마지막 명령은 제동이다


def test_drive_vtd_는_다른_시나리오가_올라와_있으면_달리지_않는다(tmp_path):
    spec = importlib.util.spec_from_file_location("drive_vtd_under_test2", SCRIPT)
    drive_vtd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drive_vtd)
    fake = FakeVtd(slice_board("course_G_s640"))        # 판은 G_s2665 인데 VTD 에는 G_s640 이 올라와 있다
    out = tmp_path / "run"
    drive_vtd.main(["--board", "course_G_s2665", "--curriculum", "curricula/stage3s.json", "--teacher",
                    "--port", str(fake.port), "--out", str(out), "--max-seconds", "1.5"])
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert res["start_mismatch"] is not None and res["start_mismatch"] > 10.0
    assert res["frames"] == 0 and res["decisions"] == 0      # 한 프레임도 몰지 않았다


def stage2_board(name):
    _, boards = load_curriculum("curricula/stage2.json")
    return next(b for b in boards if b.name == name)


def test_첫_투영은_출발점_근처에서_찾는다():
    # VTD 실측(코스 B): 자차가 출발점 4 m 옆에 놓였고, 경로가 1747 m 지점에서 출발점 곁을 다시 지난다
    board = stage2_board("course_B")
    driver = VtdDriver(board, policy=random_policy(), obs_cfg=CFG.obs)
    driver.step(rs.State(x=1015.677, y=280.246, heading=0.6661), 0.0)
    assert driver.info.s < 10.0


def test_옆_차로에서_출발하면_경로_차로에_들어올_때까지_이탈로_끝내지_않는다():
    # VTD 실측(코스 D): 자차가 경로 출발점 3.3 m 왼쪽(왼쪽 여유 1.5 m)에 놓였다
    board = stage2_board("course_D")
    driver = VtdDriver(board, policy=random_policy(), obs_cfg=CFG.obs)
    vtd_start = rs.State(x=1521.528, y=13.899, heading=1.7265)
    for k in range(20):                                   # 1 초 동안 제자리
        driver.step(vtd_start, k * 0.05)
    assert not driver.done and driver.merged_at is None
    x, y, h = board.start_pose
    driver.step(rs.State(x=x, y=y, heading=h), 1.0)       # 경로 차로에 들어왔다
    assert driver.merged_at == pytest.approx(1.0)
    driver.step(vtd_start, 1.05)                          # 이제 같은 자리는 도로 이탈이다
    assert driver.outcome == "offroad"


def test_언더스티어_보정은_모형_요레이트를_되살리고_한계에서_자른다():
    spec = importlib.util.spec_from_file_location("drive_vtd_under_test3", SCRIPT)
    drive_vtd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drive_vtd)
    import math
    L, K = drive_vtd.WHEELBASE, 0.017
    assert drive_vtd.compensate_steer(0.1, 10.0, 0.0) == 0.1                 # 끄면 그대로
    for v, d in ((5.0, 0.05), (12.0, 0.02), (8.0, -0.1)):
        sent = drive_vtd.compensate_steer(d, v, K)
        # 보정한 조향을 언더스티어 차에 넣으면 오프라인 모형의 요레이트가 나온다
        assert v * math.tan(sent) / (L + K * v * v) == pytest.approx(v * math.tan(d) / L, rel=1e-9)
    assert drive_vtd.compensate_steer(0.5, 15.0, K) == pytest.approx(math.radians(35.0))
    assert drive_vtd.compensate_steer(-0.5, 15.0, K) == pytest.approx(-math.radians(35.0))
