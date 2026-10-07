"""VTD 에서 한 판 몬다 — 9910 에 붙어 학생(정책) 또는 선생님(규칙 스택)으로 달리고 성적을 남긴다.

시나리오 로드·시작은 VTD 쪽에서 미리 해 둔다(규칙 스택 `vtd/load_scenario.py`). 이 스크립트는 9910 에 붙어
상태를 받고 명령을 보내기만 한다. 판이 끝나면(완주·이탈·충돌·정체·시간초과) 브레이크를 물고 몇 초 버틴 뒤 끊는다.

    python scripts/drive_vtd.py --board course_A --curriculum curricula/stage2.json \
        --ckpt runs/omen/2026-10-06-m7f-s2/policy-125440.pt --lim-anticipate --out runs/omen/vtd/A-student
    python scripts/drive_vtd.py --board course_A --curriculum curricula/stage2.json --teacher --out runs/omen/vtd/A-teacher

남는 것: `<out>/rows.csv`(채점기 행), `<out>/ctrl.csv`(프레임마다 보낸 명령 — VTD 동역학 보정용),
`<out>/result.json`(성적·종료 사유·요약).
"""
import argparse
import json
import math
import os
import signal
import sys
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.bridge.driver import VtdDriver, summarize  # noqa: E402
from vtd_rl.env.observation import ObsConfig  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

BRAKE_ACCEL = -4.0           # 판이 끝난 뒤 물고 있는 감속(규칙 스택 main.py 와 같다)


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", required=True, help="커리큘럼 안의 판 이름(예: course_A)")
    ap.add_argument("--curriculum", default="curricula/stage2.json")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--ckpt", help="학생 정책 체크포인트(.pt)")
    who.add_argument("--teacher", action="store_true", help="규칙 스택 선생님으로 달린다")
    ap.add_argument("--lim-anticipate", action="store_true", help="관측에 앞당긴 제한속도(배포 묶음은 켠다)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9910)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", type=float, default=4.0, help="판이 끝난 뒤 브레이크를 물고 있는 시간[s]")
    ap.add_argument("--max-seconds", type=float, default=None, help="이 시간이 지나면 끊는다(시험용)")
    ap.add_argument("--start-tolerance", type=float, default=10.0,
                    help="첫 프레임 자차가 판 출발점에서 이보다 멀면 달리지 않는다[m]")
    return ap


def find_board(curriculum: str, name: str):
    _, boards = load_curriculum(curriculum)
    for b in boards:
        if b.name == name:
            return b
    raise SystemExit(f"판을 찾을 수 없다: {name!r} (있는 판: {[b.name for b in boards]})")


def _exit_on_signal(signum, _frame):
    # SIGTERM·SIGHUP 의 기본 처리는 finally 를 건너뛰고 죽는다 — 그러면 마지막 제어값이 VTD 에 남아 차가 폭주한다
    # (규칙 스택 main.py 의 "링크가 끊기면 VTD 는 마지막 제어값을 계속 유지한다"). SystemExit 로 바꿔 finally 를 탄다.
    raise SystemExit(128 + signum)


def main(argv=None):
    a = _build_parser().parse_args(argv)
    board = find_board(a.curriculum, a.board)
    policy = None
    if a.ckpt:
        from vtd_rl.policy.net import DrivePolicy
        policy = DrivePolicy.load(a.ckpt)
    os.makedirs(a.out, exist_ok=True)
    driver = VtdDriver(board, policy=policy, teacher=a.teacher, obs_cfg=ObsConfig(lim_anticipate=a.lim_anticipate),
                       csv_path=os.path.join(a.out, "rows.csv"))
    old_handlers = {sig: signal.signal(sig, _exit_on_signal) for sig in (signal.SIGTERM, signal.SIGHUP)}
    link = rs.VTDLink(a.host, a.port)
    who = "선생님" if a.teacher else f"{os.path.basename(a.ckpt)} (앞당김 {'켬' if a.lim_anticipate else '끔'})"
    t_prev = end_t = None
    t = 0.0
    frames, last_print, link_closed, start_mismatch, error = 0, 0.0, False, None, None
    min_scale = 1.0
    ctrl = open(os.path.join(a.out, "ctrl.csv"), "w", encoding="utf-8")
    ctrl.write("t,x,y,heading,speed,steer,accel,turn\n")
    try:
        link.connect()
        print(f"[drive_vtd] {a.host}:{a.port} 연결 | 판 {board.name} {board.route.total:.0f} m | {who}", flush=True)
        while True:
            s = link.recv_state()
            if s is None:
                link_closed = True
                print("[drive_vtd] 링크가 끊겼다", flush=True)
                break
            if t_prev is None:
                # 엉뚱한 시나리오가 올라와 있으면 우리 경로와 다른 데서 출발한다 — 달리기 전에 멈춘다
                start_gap = math.hypot(s.x - board.start_pose[0], s.y - board.start_pose[1])
                if start_gap > a.start_tolerance:
                    print(f"[drive_vtd] ❌ 자차가 판 출발점에서 {start_gap:.1f} m 떨어져 있다 — 다른 시나리오가 올라와 있다",
                          flush=True)
                    start_mismatch = start_gap
                    break
            else:
                # 시뮬 시간을 쌓는다. VTD 가 실시간보다 느려지면(VTDLink 가 주변차로 잰 sim_scale) 벽시계 차를 줄인다.
                scale = s.sim_scale if s.sim_scale < rs.VTDLink.SIM_CLOCK_USE else 1.0
                min_scale = min(min_scale, s.sim_scale)
                t += max(0.0, s.t - t_prev) * scale
            t_prev = s.t
            frames += 1
            steer, accel, turn = driver.step(s, t)
            ctrl.write(f"{t:.3f},{s.x:.3f},{s.y:.3f},{s.heading:.5f},{s.speed:.3f},{steer:.5f},{accel:.3f},{turn}\n")
            if driver.done:
                if end_t is None:
                    end_t = t
                    print(f"[drive_vtd] 판 끝: {driver.outcome} (t={t:.1f}s, s={driver.info.s:.0f} m)", flush=True)
                link.send_ctrl(0.0, BRAKE_ACCEL, rs.TS_OFF)
                if t - end_t >= a.hold:
                    break
                continue
            link.send_ctrl(steer, accel, turn)
            if a.max_seconds is not None and t >= a.max_seconds:
                print(f"[drive_vtd] --max-seconds {a.max_seconds} 에서 끊는다", flush=True)
                break
            if t - last_print >= 5.0:
                last_print = t
                print(f"t={t:6.1f}s s={driver.info.s:7.1f}/{board.route.total:.0f} m v={s.speed * 3.6:5.1f} km/h "
                      f"tl={s.tl_id}/{s.tl_state} obj={len(s.objects)} 리스폰={driver.respawns}", flush=True)
    except BaseException as e:            # 무엇으로 끝나든 아래에서 제동·성적을 남기고 다시 던진다
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        try:
            link.send_ctrl(0.0, BRAKE_ACCEL, rs.TS_OFF)
        except (OSError, AttributeError):
            pass                          # 연결 전이거나 이미 끊겼다
        link.close()
        ctrl.close()
        _write_result(a, board, driver, frames, link_closed, start_mismatch, error, min_scale)
        for sig, h in old_handlers.items():
            signal.signal(sig, h)


def _write_result(a, board, driver, frames, link_closed, start_mismatch, error, min_scale):
    try:
        result = driver.finish()
        result.update({"board": board.name, "curriculum": a.curriculum, "driver": "teacher" if a.teacher else a.ckpt,
                       "lim_anticipate": bool(a.lim_anticipate), "frames": frames, "link_closed": link_closed,
                       "start_mismatch": start_mismatch, "error": error, "min_sim_scale": min_scale,
                       "wall_time": time.strftime("%Y-%m-%d %H:%M:%S")})
        result["summary"] = summarize(result)
        with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        sm = result["summary"]
        print(f"[drive_vtd] {sm['outcome']} | 점수 {sm['score']:.1f} | 진행 {sm['progress'] * 100:.1f}% | "
              f"리스폰 {sm['respawns']} | 프레임 {frames}", flush=True)
    except Exception:                     # 성적 쓰기가 실패해도 원래 오류를 가리지 않는다
        traceback.print_exc()


if __name__ == "__main__":
    main()
