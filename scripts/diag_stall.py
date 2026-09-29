"""M4e Task 1 — 정체 순간에 결정적 가속이 무엇에 비해 모자란지 잰다.

    env -u PYTHONPATH .venv/bin/python scripts/diag_stall.py \
        --checkpoint runs/omen/2026-09-26-m3-squash-warm/policy.pt \
        --stage stage1 --seeds 0 1 2 \
        --out runs/omen/2026-09-28-diag-stall/stage1.jsonl

M4d 까지 확정된 것: 3M 스텝까지 학습한 정책이 확률적으로는 판을 거의 다 완주하는데
결정적(`tanh(mean)`)으로는 22~48% 만 완주하고, 결정적 실패는 전부 `stalled`(시간초과 0 건)다.
"평균 가속이 모자라 멈춘다" 는 아직 가설이다 — 이 스크립트는 판을 **결정적으로** 몰면서
걸음마다 `tanh(mean)` 의 가속·조향, 같은 관측에서 뽑은 확률 표본, 실제 전진 속도를 모아
그 가설을 계측으로 좁힌다.

**선생님 값을 다루는 법(주의):** `vtd_rl.env.teacher_policy.TeacherPolicy` 는 판 하나를
**통째로** 몬다 — 즉 선생님은 **자기 궤적**을 달린다. 정책이 멈춘 지점의 관측을 선생님이
본 적이 없으므로, 걸음 번호로 정책과 선생님을 짝지으면 서로 다른 상황을 비교하는 것이다.
그래서 이 스크립트는 선생님 행동을 정체 걸음과 절대 짝짓지 않는다 — `teacher_mean` 은
**같은 판(같은 board·seed)에서 선생님이 낸 가속의 전체 걸음 분포**일 뿐이다. 정체 직전
값과 실제로 공정하게 비교되는 대조군은 `accel_sample`(같은 관측에서 뽑은 확률 표본)이다 —
결정적 값과 표본은 정확히 같은 `obs` 에서 나오므로 "잡음이 문턱을 넘겨 주는가" 를 직접 잰다.

체크포인트는 `DrivePolicy`(M3 등)와 `ActorCritic`(M4a 이후) 둘 다 받는다 — 판별법은
`scripts/eval_det_vs_stoch.py::load_policy` 와 같다(`blob["cfg"]` 에 `"policy"` 키가 있으면
`ActorCritic`).
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv  # noqa: E402
from vtd_rl.env.teacher_policy import TeacherPolicy  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.rl.actor_critic import ActorCritic  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

STALL_TAIL = 100      # "정체 직전" 으로 보는 마지막 걸음 수(브리프 지정값)

# 행동 벡터 `control` 의 축 순서. **근거는 `vtd_rl/env/action.py::to_command`(34~36 행)** —
# `steer = control[0] * max_steer`, `accel = control[1] * (accel_max | accel_min)` 이다.
# 이 둘이 뒤집히면 요약의 모양은 그대로인 채 진단만 통째로 거짓말을 한다(M4e 최종 리뷰가
# 심은 돌연변이 ④ 가 그렇게 살아남았다). 그래서 리터럴 대신 상수로 올려 테스트로 잠근다.
STEER_IDX = 0
ACCEL_IDX = 1

STALLED = "stalled"   # `info["outcome"]` 중 정체(속도 0 으로 멈춤) — 이 진단의 대상


def load_policy(path: str, device):
    """`eval_det_vs_stoch.py::load_policy` 와 같은 판별법(모듈 docstring 참고)."""
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


def run_policy_episode_rows(env, net, board_name: str, seed: int, max_steps: int, gen) -> tuple:
    """판 하나를 결정적으로 몰며 걸음마다 결정적 가속·조향·확률 표본·속도를 모은다.

    `vtd_rl.policy.evaluate.run_policy_episode` 를 쓰지 않는다 — 그건 종료 성적만 주고,
    여기는 걸음별 값(정체 직전 N 걸음)이 필요하다.
    """
    obs, _info = env.reset(seed=seed, options={"board": board_name})
    rows = []
    outcome = "running"
    for _ in range(max_steps):
        act = net.act(obs, deterministic=True)
        # 대조군은 **같은 `obs`** 에서 뽑은 확률 표본이어야 한다 — `deterministic=False`.
        # 결정적으로 뽑으면 `accel_det` 과 같은 값이 되어 "잡음이 문턱을 넘겨 주는가" 라는
        # 질문 자체가 사라진다(M4e 최종 리뷰 돌연변이 ⑥).
        smp = net.act(obs, deterministic=False, generator=gen)
        # ego 벡터 0 번 = _clip(ego.v, cfg.v_max) — v_max(기본 25 m/s)로 나눈 **정규화** 값
        # ([-1, 1] 로 clip). 실제 m/s 가 아니다(vtd_rl/env/observation.py:86).
        rows.append(_action_row(act, smp, obs["ego"][0]))
        obs, _r, term, trunc, info = env.step(act)
        if term or trunc:
            outcome = info["outcome"]
            break
    return rows, outcome


def run_teacher_episode_rows(env, board_name: str, seed: int, max_steps: int) -> list:
    """선생님이 이 판(자기 궤적)을 통째로 몰 때 걸음마다 낸 가속·조향.

    `run_teacher_in_env` 는 종료 요약만 준다 — 여기서는 걸음별 값이 필요해 같은 구조의
    루프를 직접 돈다(`vtd_rl/env/teacher_policy.py::run_teacher_in_env` 참고). **주의**: 이
    궤적은 정책이 멈춘 지점을 지나지 않을 수 있다 — 걸음 번호로 정책과 짝짓지 마라. 이
    함수가 내는 값은 오직 "이 판에서 선생님이 낸 가속·조향의 전체 분포" 로만 쓴다.
    """
    env.reset(seed=seed, options={"board": board_name})
    policy = TeacherPolicy(env)
    policy.reset()
    rows = []
    try:
        for _ in range(max_steps):
            act = policy.act()
            rows.append({"accel": float(act["control"][ACCEL_IDX]),
                         "steer": float(act["control"][STEER_IDX])})
            _obs, _r, term, trunc, _info = env.step(act)
            if term or trunc:
                break
    finally:
        policy.detach()      # 다음 판(정책이 모는 판)까지 갈고리를 끌고 가지 않는다
    return rows


def _action_row(act, smp, speed) -> dict:
    """걸음 하나의 결정적 행동·확률 표본·전진 속도를 한 줄로 묶는다.

    `act` 는 `net.act(obs, deterministic=True)`, `smp` 는 **같은 `obs`** 에서 뽑은 확률
    표본이다. 축 순서는 `STEER_IDX`·`ACCEL_IDX` 주석 참고 — 여기서 두 칸이 바뀌면
    "제동을 학습한다" 가 "조향을 학습한다" 로 조용히 둔갑한다.

    `speed` 는 정규화된 `obs["ego"][0]` 이다(m/s 아님).
    """
    return {
        "accel_det": float(act["control"][ACCEL_IDX]),
        "accel_sample": float(smp["control"][ACCEL_IDX]),
        "steer_det": float(act["control"][STEER_IDX]),
        "steer_sample": float(smp["control"][STEER_IDX]),
        "speed_norm": float(speed),
    }


def _stalled_only(episodes):
    """판 목록에서 **정체로 끝난 판만** 고른다.

    이 진단이 재려는 것은 "멈춘 판에서 무슨 일이 있었나" 다 — 필터가 반전되면 완주한
    판의 가속을 '정체 직전 가속' 이라 부르게 된다(M4e 최종 리뷰 돌연변이 ⑤).
    """
    return [e for e in episodes if e["outcome"] == STALLED]


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _tail(rows, key, n):
    """`rows` 의 **뒤쪽** n 걸음에서 `key` 값을 뽑는다 — '멈추기 직전' 이 곧 뒤쪽이다.

    앞쪽(`rows[:n]`)을 보면 출발 직후 가속을 정체 직전 값이라 부르게 된다
    (M4e 최종 리뷰 돌연변이 ③). `n` 이 판 길이보다 크면 전부 준다.
    """
    return [r[key] for r in rows[-n:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="DrivePolicy 또는 ActorCritic 체크포인트 경로")
    ap.add_argument("--stage", required=True, help="커리큘럼 이름(curricula/<stage>.json)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="판마다 도는 시드 목록")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--stall-tail", type=int, default=STALL_TAIL,
                    help="정체 판에서 '멈추기 직전' 으로 보는 마지막 걸음 수")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stoch-seed", type=int, default=0,
                    help="확률 표본 추출 시드 — 고정해야 재현 가능하다")
    ap.add_argument("--out", default=None,
                    help="주면 판마다 걸음별 원자료(JSONL, 정책·선생님 각각) + 요약 한 줄을 저장한다")
    a = ap.parse_args()

    dev = pick_device(a.device)
    net = load_policy(a.checkpoint, dev)
    boards = _stage_boards(a.stage)
    gen = torch.Generator().manual_seed(a.stoch_seed)

    env = VtdDriveEnv(list(boards), EnvConfig())
    episodes = []
    try:
        for board in boards:
            for seed in a.seeds:
                rows, outcome = run_policy_episode_rows(env, net, board.name, seed,
                                                        a.max_steps, gen)
                episodes.append({"board": board.name, "seed": seed, "outcome": outcome,
                                 "steps": len(rows), "rows": rows})
        teacher_episodes = []
        for board in boards:
            for seed in a.seeds:
                rows = run_teacher_episode_rows(env, board.name, seed, a.max_steps)
                teacher_episodes.append({"board": board.name, "seed": seed,
                                         "steps": len(rows), "rows": rows})
    finally:
        env.close()

    all_accel_det = [r["accel_det"] for e in episodes for r in e["rows"]]
    all_accel_sample = [r["accel_sample"] for e in episodes for r in e["rows"]]
    all_steer_det = [r["steer_det"] for e in episodes for r in e["rows"]]
    all_steer_sample = [r["steer_sample"] for e in episodes for r in e["rows"]]
    teacher_accel = [r["accel"] for e in teacher_episodes for r in e["rows"]]
    teacher_steer = [r["steer"] for e in teacher_episodes for r in e["rows"]]

    stalled = _stalled_only(episodes)
    tail_accel_det = [v for e in stalled for v in _tail(e["rows"], "accel_det", a.stall_tail)]
    tail_accel_sample = [v for e in stalled for v in _tail(e["rows"], "accel_sample", a.stall_tail)]
    tail_steer_det = [v for e in stalled for v in _tail(e["rows"], "steer_det", a.stall_tail)]
    tail_steer_sample = [v for e in stalled for v in _tail(e["rows"], "steer_sample", a.stall_tail)]

    summary = {
        "checkpoint": a.checkpoint,
        "stage": a.stage,
        "seeds": list(a.seeds),
        "episodes": len(episodes),
        "stalled": len(stalled),
        "stall_tail": a.stall_tail,
        # ego 벡터 0 번(v/v_max, [-1, 1] 로 clip) — m/s 가 아니라 정규화된 값이다.
        "speed_source": "obs['ego'][0] = _clip(ego.v, ObsConfig.v_max)  # 정규화, m/s 아님",
        "teacher_note": ("선생님은 자기 궤적으로 판을 통째로 몬다 — 정체 걸음과 걸음 번호로"
                         " 짝짓지 않았다. teacher_* 는 같은 판에서 선생님이 낸 값의 전체 걸음"
                         " 분포다(정체 지점 직접 대조 아님)."),
        "accel": {
            "policy_mean": _mean(all_accel_det),
            "sample_mean": _mean(all_accel_sample),
            "teacher_mean": _mean(teacher_accel),
            "stalled_tail_policy_mean": _mean(tail_accel_det),
            "stalled_tail_sample_mean": _mean(tail_accel_sample),
        },
        "steer": {
            "policy_mean": _mean(all_steer_det),
            "sample_mean": _mean(all_steer_sample),
            "teacher_mean": _mean(teacher_steer),
            "stalled_tail_policy_mean": _mean(tail_steer_det),
            "stalled_tail_sample_mean": _mean(tail_steer_sample),
        },
    }

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for e in episodes:
                f.write(json.dumps({"kind": "policy", **e}, ensure_ascii=False) + "\n")
            for e in teacher_episodes:
                f.write(json.dumps({"kind": "teacher", **e}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"kind": "summary", **summary}, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
