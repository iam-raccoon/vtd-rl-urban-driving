"""M4j Task 1 — 단계 ③ 에서 학생이 **물체를 보고도 안 피하는지, 애초에 못 보는지** 가른다.

    env -u PYTHONPATH .venv/bin/python scripts/diag_obs.py \
        --checkpoint runs/omen/2026-09-26-m3-squash-warm/policy.pt \
        --stage stage3 --seeds 0 1 2 --device cpu --table \
        --out runs/omen/2026-09-30-diag-obs/stage3.jsonl

M4i 가 확정한 것: M3 모방 학생은 단계 ③ 18 판 전부 **첫 액터에서 충돌**로 죽는다
(`docs/reports/m4i-student-stage3.md`). 두 번째 액터까지 간 판이 없다. **왜** 인지는 안 밝혔고,
답에 따라 다음 마일스톤이 통째로 갈린다.

- **(A) 관측은 차는데 정책이 무시한다.** M3 DAgger 자료가 전부 단계 ①②(물체 0 개)에서 나와
  물체 슬롯 16 칸이 학습 내내 마스크였으니 예상되는 쪽이다. 고칠 것 = 단계 ③ DAgger 재수집.
- **(B) 관측이 안 찬다.** 파이프라인 버그 — 정책은 눈을 감고 달린다. 고칠 것 = 파이프라인,
  **자료 수집보다 먼저**. 눈감은 관측에 대고 DAgger 를 모으면 마일스톤 하나를 통째로 버린다.

그래서 이 스크립트는 걸음마다 **관측 쪽**(`object_mask`·`objects`)과 **세계 쪽**
(`State.objects`)을 **나란히** 적는다. 둘의 어긋남(`blind_steps`: 세계엔 있는데 관측엔 없는
걸음 수)이 (A)/(B) 를 가르는 유일한 숫자다.

**선생님을 함께 재는 이유(주의):** 선생님은 RL 관측을 **먹지 않는다** — 규칙 스택 상태의
`s.objects` 를 직접 읽는다(`third_party/rule_stack/.../drive.py:2402-2420`). 그래서 이 대조는
"정책 대 선생님의 행동" 이 아니라 **"세계에 물체가 있는가 대 관측에 물체가 있는가"** 다 —
(B) 가 살아 있다면 정확히 그 이음매에 있다. 선생님 궤적은 학생 궤적과 다르므로 **걸음 번호로
짝짓지 않는다**(`scripts/diag_stall.py` 가 같은 함정을 적어 뒀다).

체크포인트는 `DrivePolicy`(M3 등)와 `ActorCritic`(M4a 이후) 둘 다 받는다 — 판별법은
`scripts/diag_stall.py::load_policy` 와 같다.
"""
import argparse
import json
import math
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv  # noqa: E402
from vtd_rl.env.observation import ObsConfig  # noqa: E402
from vtd_rl.env.teacher_policy import TeacherPolicy  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.rl.actor_critic import ActorCritic  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

TAIL = 100            # 종료 직전으로 보는 마지막 걸음 수(브리프 지정값)

# ── 관측 사전의 키. 여기가 틀리면 "물체가 안 보인다" 는 **진단 쪽 거짓말**이 된다 —
#    그게 바로 이 스크립트가 가르려는 (B) 와 구별이 안 되므로 상수로 올려 테스트로 잠근다.
OBJ_KEY = "objects"          # (16, 12) 물체 행렬 (observation.py:147)
MASK_KEY = "object_mask"     # (16,) 슬롯 살아있음 (observation.py:148)
EGO_KEY = "ego"              # (9,) 자차 묶음 (observation.py:142)

# ── `objects` 12 칸의 열 순서. 근거는 **`vtd_rl/env/observation.py:135-138`**:
#    [fx, fy, cos Δh, sin Δh, speed, length, width, height, route-lateral,
#     class_vehicle, class_person, class_obstacle]
FX_COL, FY_COL = 0, 1        # 전후·좌우(자차 기준). 뒤집히면 "50 m 앞" 이 "1 m 옆" 이 된다.
SPEED_COL = 4
LENGTH_COL = 5
CLASS_COLS = (9, 10, 11)
CLASS_NAMES = ("vehicle", "person", "obstacle")

# ── `ego` 9 칸 중 0 번만 쓴다: `_clip(ego.v, cfg.v_max)`(observation.py:86). 1·2 번은
#    **직전 행동**(조향·가속)이라, 한 칸만 밀려도 "속도" 자리에 조향이 들어앉는다.
EGO_SPEED_IDX = 0

# ── 행동 벡터 축 순서. 근거는 `vtd_rl/env/action.py::to_command`(34~36 행).
#    `diag_stall.py` 와 같은 상수 — 뒤집히면 "제동을 배웠다" 가 "조향을 배웠다" 로 둔갑한다.
STEER_IDX = 0
ACCEL_IDX = 1


def load_policy(path: str, device):
    """`diag_stall.py::load_policy` 와 같은 판별법(모듈 docstring 참고)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "policy" in blob["cfg"]:                       # ActorCritic 체크포인트(가치 머리 포함)
        net = ActorCritic.load(path, device=device).policy
    else:                                              # 순수 DrivePolicy 체크포인트(M3 등)
        net = DrivePolicy.load(path, device=device)
    net.eval()
    return net


def _stage_boards(stage: str):
    path = os.path.join(REPO, "curricula", f"{stage}.json")
    _name, boards = load_curriculum(path)
    return boards


def _nearest_live(objs, mask, cfg: ObsConfig):
    """살아 있는 슬롯 중 **자차에 가장 가까운** 것의 색인. 하나도 없으면 None.

    거리는 **역정규화한 미터**로 잰다. 정규화된 채로 `hypot` 을 재면 fx 는 80 m,
    fy 는 20 m 로 나뉘어 있어 좌우가 4 배 무겁게 세어진다 — 옆 차로 물체가 바로 앞
    물체보다 "가깝다" 고 나온다.

    관측은 이미 가까운 순으로 슬롯을 채우지만(`observation.py:128`) 여기서 다시 고른다.
    슬롯 0 을 그냥 믿으면 관측 쪽 정렬이 깨져도 이 진단은 아무 말을 안 한다.
    """
    best, best_d = None, None
    for k in range(len(mask)):
        if mask[k] <= 0.0:
            continue
        d = math.hypot(float(objs[k][FX_COL]) * cfg.obj_x, float(objs[k][FY_COL]) * cfg.obj_y)
        if best_d is None or d < best_d:      # **가장 가까운** 것 — 부등호가 뒤집히면 먼 것을 고른다
            best, best_d = k, d
    return best


def _class_name(onehot) -> str:
    """3-way one-hot -> 이름. 다 0 이면 '?'(관측이 분류를 못 낸 자리)."""
    vals = [float(v) for v in onehot]
    if max(vals) <= 0.0:
        return "?"
    return CLASS_NAMES[vals.index(max(vals))]


def _obs_object(obs, cfg: ObsConfig) -> dict:
    """관측 쪽 한 걸음 — 산 슬롯 수와 **가장 가까운 슬롯**의 물리값(역정규화, 미터·m/s).

    값이 전부 `_clip(·, scale)` 로 [-1, 1] 에 눌려 있으므로(`observation.py:50-51`)
    되돌리려면 같은 scale 을 곱해야 한다. scale 을 헷갈리면(fx 에 obj_y 를 곱하는 식)
    거리가 조용히 4 배 틀린다.
    """
    objs, mask = obs[OBJ_KEY], obs[MASK_KEY]
    row = {"mask_sum": float(sum(float(m) for m in mask))}
    k = _nearest_live(objs, mask, cfg)
    if k is None:
        row.update({"slot": None, "fx": None, "fy": None, "range": None, "obj_speed": None,
                    "length": None, "cls": None, "cls_onehot": None, "fy_clipped": False})
        return row
    o = objs[k]
    fx = float(o[FX_COL]) * cfg.obj_x
    fy = float(o[FY_COL]) * cfg.obj_y
    onehot = [float(o[c]) for c in CLASS_COLS]
    # 좌우는 ±obj_y(=20 m)에서 **포화**한다(`observation.py:51`). 포화한 걸음의 `range` 는
    # 진짜 거리보다 작게 나오므로 "몇 m 앞에서 보였나" 를 이 값으로 답하면 안 된다 —
    # 그래서 포화 여부를 그 자리에서 표시하고, 요약은 세계 쪽 거리를 함께 낸다.
    row.update({"slot": int(k), "fx": fx, "fy": fy, "range": math.hypot(fx, fy),
                "fy_clipped": abs(float(o[FY_COL])) >= 1.0 - 1e-6,
                "obj_speed": float(o[SPEED_COL]) * cfg.v_max,
                "length": float(o[LENGTH_COL]) * cfg.obj_size,
                "cls": _class_name(onehot), "cls_onehot": onehot})
    return row


def _world_object(state, ego) -> dict:
    """세계 쪽 한 걸음 — `State.objects` 의 개수와 최단 수평 거리[m].

    **이 두 값이 (B) 판별의 반쪽이다.** 관측은 바로 이 `state.objects` 에서 지어지므로
    (`drive_env.py:152` -> `observation.py:128`), 세계엔 물체가 있는데 관측 마스크가 0 이면
    그 사이 어딘가가 끊어진 것이다. 선생님도 같은 `s.objects` 를 읽는다.
    """
    objs = list(state.objects)
    d = min((math.hypot(o.x - ego.x, o.y - ego.y) for o in objs), default=None)
    return {"world_n": len(objs), "world_dist": d}


def _row(step: int, obs, state, ego, act, cfg: ObsConfig) -> dict:
    """걸음 하나 — 관측 쪽 + 세계 쪽 + **그 관측에 대고 낸 행동**을 한 줄로.

    `obs`·`state`·`ego` 는 **같은 시점**이어야 한다: `env.reset()`/`env.step()` 이 돌아온
    직후에는 `env.state`·`env.world.ego` 가 방금 돌려준 `obs` 를 지은 그 상태다
    (`drive_env.py:108, 152`). `act` 는 그 `obs` 를 먹여 뽑은 행동이다.
    """
    row = {"step": int(step)}
    row.update(_obs_object(obs, cfg))
    row.update(_world_object(state, ego))
    row["steer"] = float(act["control"][STEER_IDX])
    row["accel"] = float(act["control"][ACCEL_IDX])
    row["turn"] = int(act["turn"])
    row["speed"] = float(obs[EGO_KEY][EGO_SPEED_IDX]) * cfg.v_max
    return row


def run_policy_episode_rows(env, net, board_name: str, seed: int, max_steps: int) -> tuple:
    """판 하나를 **결정적으로** 몰며 걸음마다 관측·세계·행동을 모은다.

    `vtd_rl.policy.evaluate.run_policy_episode` 를 쓰지 않는다 — 그건 종료 성적만 준다.
    """
    obs, _info = env.reset(seed=seed, options={"board": board_name})
    rows, outcome = [], "running"
    for i in range(max_steps):
        act = net.act(obs, deterministic=True)
        rows.append(_row(i, obs, env.state, env.world.ego, act, env.cfg.obs))
        obs, _r, term, trunc, info = env.step(act)
        if term or trunc:
            outcome = info["outcome"]
            break
    return rows, outcome


def run_teacher_episode_rows(env, board_name: str, seed: int, max_steps: int) -> tuple:
    """선생님이 같은 판을 몰 때의 같은 표 — 세계 쪽 물체와 관측 쪽 물체를 나란히.

    선생님은 이 관측을 **안 먹는다**(모듈 docstring). 그래도 관측은 환경이 매 걸음 지어
    돌려주므로, "선생님이 물체를 지날 때 관측 파이프라인이 채워지는가" 를 공짜로 잴 수 있다.
    """
    obs, _info = env.reset(seed=seed, options={"board": board_name})
    policy = TeacherPolicy(env)
    policy.reset()
    rows, outcome = [], "running"
    try:
        for i in range(max_steps):
            act = policy.act()
            rows.append(_row(i, obs, env.state, env.world.ego, act, env.cfg.obs))
            obs, _r, term, trunc, info = env.step(act)
            if term or trunc:
                outcome = info["outcome"]
                break
    finally:
        policy.detach()      # 갈고리를 다음 판까지 끌고 가지 않는다
    return rows, outcome


def _tail(rows, n):
    """`rows` 의 **뒤쪽** n 걸음 — '종료 직전' 이 곧 뒤쪽이다.

    앞쪽(`rows[:n]`)을 보면 출발 직후를 종료 직전이라 부르게 된다(`diag_stall.py` 가
    같은 자리에서 돌연변이를 먹었다). `n` 이 판 길이보다 크면 전부 준다.
    """
    return rows[-n:] if n > 0 else []


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(rows, board: str, seed: int, outcome: str, kind: str) -> dict:
    """판 하나의 요약 — **(B) 판별은 `blind_steps` 한 칸이 한다.**

    - `blind_steps`: 세계엔 물체가 있는데(`world_n > 0`) 관측 마스크가 0 인 걸음 수.
      **0 보다 크면 (B)** — 관측 파이프라인이 끊겼다.
    - `ghost_steps`: 반대 방향(관측엔 있는데 세계엔 없다). 정상이면 0 이다.
    - `first_seen_*`: 물체가 **관측에** 처음 들어온 걸음과 그때의 거리 — "몇 m 앞에서
      보이기 시작하나".
    - `accel_before_mean` 대 `accel_seen_mean`: 물체가 보이기 **전/후** 평균 가속.
      물체가 다가오는데도 이 둘이 같으면 "보고도 반응이 없다"(A) 는 뜻이다.
    - `accel_pre_mean`: `before` 전체가 아니라 **보이기 직전 같은 걸음 수**만의 평균.
      `accel_before_mean` 은 출발 직후 전력 가속(0 -> 순항)이 섞여 있어 "보인 뒤 가속이
      줄었다" 가 저절로 참이 된다. 순항에 든 뒤의 같은 길이 창과 견줘야 **물체 때문에**
      달라졌는지를 묻는 것이 된다.
    """
    seen = [r for r in rows if r["mask_sum"] > 0.0]
    before = [r for r in rows if r["mask_sum"] <= 0.0]
    pre = before[-len(seen):] if seen else []      # 보이기 직전 같은 길이의 창
    first = seen[0] if seen else None
    ranges = [r["range"] for r in seen if r["range"] is not None]
    return {
        "kind": kind, "board": board, "seed": seed, "outcome": outcome, "steps": len(rows),
        # ── (B) 판별
        "blind_steps": sum(1 for r in rows if r["world_n"] > 0 and r["mask_sum"] <= 0.0),
        "ghost_steps": sum(1 for r in rows if r["world_n"] == 0 and r["mask_sum"] > 0.0),
        "world_seen_steps": sum(1 for r in rows if r["world_n"] > 0),
        "obs_seen_steps": len(seen),
        "world_min_dist": min((r["world_dist"] for r in rows if r["world_dist"] is not None),
                              default=None),
        # ── 언제 보이기 시작하나
        "first_seen_step": first["step"] if first else None,
        "first_seen_fx": first["fx"] if first else None,
        "first_seen_range": first["range"] if first else None,
        # 관측 쪽 `range` 는 fy 포화(±20 m) 때문에 작게 나올 수 있다 — 진짜 거리는 이쪽이다.
        "first_seen_world_dist": first["world_dist"] if first else None,
        "first_seen_cls": first["cls"] if first else None,
        "min_obs_range": min(ranges, default=None),
        "fy_clipped_steps": sum(1 for r in seen if r.get("fy_clipped")),
        # ── 보이고 나서 행동이 달라지나
        "accel_before_mean": _mean([r["accel"] for r in before]),
        "accel_pre_mean": _mean([r["accel"] for r in pre]),
        "accel_seen_mean": _mean([r["accel"] for r in seen]),
        "accel_seen_min": min((r["accel"] for r in seen), default=None),
        "speed_pre_mean": _mean([r["speed"] for r in pre]),
        "speed_seen_mean": _mean([r["speed"] for r in seen]),
        "steer_before_absmean": _mean([abs(r["steer"]) for r in before]),
        "steer_pre_absmean": _mean([abs(r["steer"]) for r in pre]),
        "steer_seen_absmean": _mean([abs(r["steer"]) for r in seen]),
        "steer_seen_absmax": max((abs(r["steer"]) for r in seen), default=None),
        "turns_seen": sorted({r["turn"] for r in seen}),
        "final": rows[-1] if rows else None,
    }


_COLS = ("step", "mask_sum", "world_n", "fx", "fy", "range", "cls", "length", "obj_speed",
         "steer", "accel", "turn", "speed")


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def markdown_table(rows) -> str:
    head = "| " + " | ".join(_COLS) + " |"
    rule = "|" + "|".join("---" for _ in _COLS) + "|"
    body = ["| " + " | ".join(_fmt(r.get(c)) for c in _COLS) + " |" for r in rows]
    return "\n".join([head, rule] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="DrivePolicy 또는 ActorCritic 체크포인트")
    ap.add_argument("--stage", default="stage3", help="커리큘럼 이름(curricula/<stage>.json)")
    ap.add_argument("--boards", nargs="*", default=None, help="판 이름(안 주면 단계 전체)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--tail", type=int, default=TAIL, help="표에 찍을 종료 직전 걸음 수")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--teacher-seeds", type=int, nargs="*", default=None,
                    help="선생님을 돌릴 시드(기본: --seeds 의 첫 개). 빈 목록이면 안 돌린다")
    ap.add_argument("--table", action="store_true", help="판마다 종료 직전 표를 마크다운으로 찍는다")
    ap.add_argument("--out", default=None, help="주면 걸음별 원자료(JSONL) + 요약을 저장한다")
    a = ap.parse_args()

    dev = pick_device(a.device)
    net = load_policy(a.checkpoint, dev)
    boards = _stage_boards(a.stage)
    if a.boards:
        want = set(a.boards)
        boards = [b for b in boards if b.name in want]
        if not boards:
            raise SystemExit(f"판을 찾을 수 없다: {sorted(want)}")
    t_seeds = list(a.seeds[:1]) if a.teacher_seeds is None else list(a.teacher_seeds)

    env = VtdDriveEnv(list(boards), EnvConfig())
    episodes = []
    try:
        for board in boards:
            for seed in a.seeds:
                rows, outcome = run_policy_episode_rows(env, net, board.name, seed, a.max_steps)
                episodes.append({"summary": summarize(rows, board.name, seed, outcome, "policy"),
                                 "rows": rows})
        for board in boards:
            for seed in t_seeds:
                rows, outcome = run_teacher_episode_rows(env, board.name, seed, a.max_steps)
                episodes.append({"summary": summarize(rows, board.name, seed, outcome, "teacher"),
                                 "rows": rows})
    finally:
        env.close()

    summaries = [e["summary"] for e in episodes]
    total = {
        "checkpoint": a.checkpoint, "stage": a.stage, "seeds": list(a.seeds),
        "teacher_seeds": t_seeds, "episodes": len(episodes),
        # 판 전체를 통틀어 이 한 줄이 (A)/(B) 를 가른다.
        "blind_steps_total": sum(s["blind_steps"] for s in summaries),
        "ghost_steps_total": sum(s["ghost_steps"] for s in summaries),
        "verdict": ("B: 관측이 안 찬다(세계엔 물체가 있는데 마스크가 0 인 걸음이 있다)"
                    if sum(s["blind_steps"] for s in summaries) > 0
                    else "A: 관측은 찬다 — 정책이 무시한다"),
        "units": ("fx·fy·range·length·world_dist=m, obj_speed·speed=m/s,"
                  " steer·accel=[-1,1] 행동값(to_command 가 곱하기 전)"),
        "boards": summaries,
    }

    if a.table:
        for e in episodes:
            s = e["summary"]
            print(f"### {s['kind']} {s['board']} seed={s['seed']} "
                  f"outcome={s['outcome']} steps={s['steps']}")
            print(markdown_table(_tail(e["rows"], a.tail)))
            print()

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for e in episodes:
                f.write(json.dumps({"kind": "episode", **e}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"kind": "summary", **total}, ensure_ascii=False) + "\n")

    print(json.dumps(total, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
