"""M4a — PPO 학습 스크립트(가치 워밍업 → PPO, 평가·체크포인트·로그).

    env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
        --out runs/omen/$(date +%F)-ppo --init runs/lab-main/.../policy-r4.pt \
        --dagger-data runs/lab-main/.../data --envs 30 --steps 3000000
    env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py --smoke --out /tmp/smoke

체크포인트는 `ActorCritic`(가치 머리 포함)이지 `DrivePolicy` 가 아니다 — 이름을 `ac-<스텝>.pt`
로 두는 이유가 그것이다. `DrivePolicy.load()` 로는 못 읽는다, `ActorCritic.load(path)` 를 쓰고
정책만 필요하면 `.policy` 를 꺼낸다.

평가는 단계(커리큘럼 파일)마다 따로 부른다 — `stage1.json` 과 `stage2.json` 은 판 이름을
그대로 공유하고 `signals` 만 다르므로, `evaluate_policy` 에 두 단계 판을 합쳐 넘기면
`VtdDriveEnv` 의 세계 캐시가 "AssertionError: 세계 캐시가 다른 판을 준다" 로 죽는다
(2026-09-21, `vtd_rl/rl/vec_env.py` 의 실측 기록). 평가에 넘길 판은 `load_curriculum` 으로
직접 읽는다 — `vec_env._boards()` 는 이름에 `#stage1` 접미사를 붙이므로 쓰지 않는다.

학습 중 평가(주기 평가, `--eval-seeds`)는 값싸게(기본 1시드) 자주 하고, 학습이 끝난 뒤
`--final-eval-seeds`(기본 3)로 딱 한 번 전체 평가를 해서 그 값으로 `ac-best.pt` 를 최종
선정하고 요약 JSON 에 싣는다 — 실측(OMEN) 전체 평가 한 번 ≈ 81 초인데 3M 스텝의 환경 시간은
2.7 분뿐이라, 주기 평가를 매번 전체 시드로 하면 평가가 학습보다 몇 배 더 걸린다.
"""
import argparse
import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.env.drive_env import EnvConfig  # noqa: E402
from vtd_rl.env.reward import RewardConfig  # noqa: E402
from vtd_rl.eval.verdict import completed_only, judge, major_total  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.dataset import load_dir  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy, evaluate_teacher, violation_counts  # noqa: E402
from vtd_rl.policy.net import ENTROPY_MODES  # noqa: E402
from vtd_rl.policy.train import TrainConfig  # noqa: E402
from vtd_rl.rl.actor_critic import ActorCritic  # noqa: E402
from vtd_rl.rl.buffer import RolloutBuffer  # noqa: E402
from vtd_rl.rl.diagnostics import (OutcomeCounter, ReturnTracker, RewardTermTracker,  # noqa: E402
                                   policy_drift, snapshot_policy)
from vtd_rl.rl.ppo import PPOConfig, dagger_batches, update  # noqa: E402
from vtd_rl.rl.vec_env import make_vec_env, vec_obs_to_arrays  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402
from vtd_rl.world.world import WorldConfig  # noqa: E402

PUBLIC_KEYS = ("goal_rate", "mean_score", "mean_score_raw", "mean_reward")


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def _stage_boards(curricula, smoke: bool):
    """커리큘럼 파일마다 (이름표, 판 목록) — 판 이름은 그대로 둔다(`vec_env._boards()` 와 달리

    `#stage` 접미사를 붙이지 않는다). 이름표는 파일 이름(확장자 없이)이라 기본 커리큘럼
    `stage1.json`/`stage2.json` 이면 요약 JSON 의 키가 그대로 "stage1"/"stage2" 가 된다.
    스모크에서는 각 단계 코스 A 한 판으로 줄인다.
    """
    stages = []
    for rel in curricula:
        path = _resolve(rel)
        label = os.path.splitext(os.path.basename(path))[0]
        _name, boards = load_curriculum(path)
        if smoke:
            boards = [b for b in boards if b.name == "course_A"]
        stages.append((label, boards))
    return stages


def _public(ev: dict) -> dict:
    return {k: ev[k] for k in PUBLIC_KEYS}


def _evaluate_stages(policy, stages, seeds) -> dict:
    return {label: evaluate_policy(policy, boards, seeds=seeds) for label, boards in stages}


def _major_totals(evs: dict) -> dict:
    """완주한 판만 걸러 중대 위반을 센다 — `completed_only` 를 빼면 조기 종료한 판이 중대를

    덜 잡혀 목표 4번(전 항목 중대 위반)이 부당하게 통과한다(2026-09-21 리뷰). 이름 있는
    함수로 떼어 `evaluate_policy` 서브프로세스 없이 가짜 `ev` 로 단위 테스트한다
    (`tests/rl/test_train_ppo.py`).
    """
    return {label: major_total(violation_counts(completed_only(ev)))
            for label, ev in evs.items()}


def _metric(evs: dict) -> tuple:
    """단계 평균 완주율 → 동률이면 단계 평균 점수. `ac-best.pt` 선정·주기 최고 추적에 같이 쓴다."""
    n = max(len(evs), 1)
    goal = sum(e["goal_rate"] for e in evs.values()) / n
    score = sum(e["mean_score"] for e in evs.values()) / n
    return (goal, score)


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """비평가가 실제 리턴을 얼마나 따라잡는지 — 1 이면 완벽, 0 이면 평균만 맞히는 것과 같다.

    자동 리셋 더미 행(`valid=0`)은 호출부가 미리 걸러 넘긴다 — 그 행은 학습 배치에도 안
    들어가니 진단에도 안 들어가야 한다. 리턴 분산이 0 에 가까우면(모든 보상이 같은 병적인
    경우) 정의가 안 되니 NaN 을 낸다(호출부가 JSON 에 쓰기 전에 `None` 으로 바꾼다).
    """
    var_y = float(returns.var())
    if var_y < 1e-8:
        return float("nan")
    return 1.0 - float((returns - values).var()) / var_y


def _reward_terms_per_env(info: dict, n_envs: int) -> list:
    """`venv.step()` 의 `info["reward_terms"]` 를 `RewardTermTracker.add()` 의 계약

    (환경별 dict 리스트, `vtd_rl/rl/diagnostics.py` 의 `RewardTermTracker.add()` 참고)에
    맞춰 되돌린다.

    gymnasium 의 정보 벡터화(`VectorEnv._add_info`)는 값이 dict 면 재귀적으로 파고들어
    `{항목명: 환경별 배열}` 로 만든다 — `outcome`(문자열)처럼 단순 값이 내는 "환경 수만큼의
    배열"과 달리, `reward_terms`(dict) 는 이렇게 **전치된** 모양으로 나온다(2026-09-26 실측:
    판이 실제로 끝나는 스텝에서도 이 모양이다 — 브리프의 "환경 수만큼의 dict 배열" 가정과
    다르다). 모든 환경이 리셋 직후(빈 dict)면 gymnasium 이 재귀할 것이 없어 `{}` 그대로
    나온다.

    일부 환경만 그 항목을 냈을 때(병렬 학습의 정상 상태 — 일부는 방금 리셋, 나머지는 주행
    중)는 gymnasium 이 나머지 자리를 채움값(실측: 0.0)으로 메우고 `_<항목명>` 불리언 마스크
    배열을 같이 낸다(2026-09-26 실측, `.venv/.../gymnasium/vector/vector_env.py::_add_info`
    소스로 직접 확인). 채움값이 우연히 0.0 이고 네 항목이 전부 순가산량이라 지금은 걸러내지
    않아도 수치적으로 무해하지만, 그건 gymnasium 구현 세부에 기대는 것이다 — 여기서는 마스크를
    직접 읽어 거짓인 자리를 아예 dict 에서 뺀다. 마스크 키(`_progress` 등) 자체가 없으면
    (실측: 그 항목을 하나도 낸 환경이 없어 항목 키조차 없을 때만 이렇다) 전원 유효로 본다.
    """
    rt = info.get("reward_terms") or {}
    keys = [k for k in rt if not k.startswith("_")]
    out = []
    for i in range(n_envs):
        env_terms = {}
        for k in keys:
            mask = rt.get(f"_{k}")
            if mask is None or mask[i]:
                env_terms[k] = float(rt[k][i])
        out.append(env_terms)
    return out


def _bootstrap_reward_done(reward, term, prev_done, value: torch.Tensor):
    """자동 리셋 더미 행(직전 걸음이 끝난 환경)의 `reward`·`done` 을 고쳐 GAE 사슬을 끊는다.

    gymnasium 1.3 자동 리셋(NEXT_STEP)은 판이 끝난 다음 걸음에서 행동을 무시하고 보상 0·
    `terminated=False` 를 낸다. `RolloutBuffer.compute_gae` 는 `not_done` 하나로 부트스트랩
    허용과 사슬 절단을 동시에 정한다 — 이 더미 행의 `done` 을 그대로(=`term`, 거짓) 두면
    `delta = γ·V(새 판 첫 관측) − V(끝난 판 마지막 관측)` 이 계산되고, 그 값이 `(γλ)^k` 로
    감쇠하며 이전(진짜) 걸음들의 이점에 새어 든다 — truncated(시간 초과·정체)로 끝난 판일수록
    V(새 판 시작)이 크고 V(정체 판 마지막)은 작아 체계적으로 양의 이점이 새어나간다("판을
    빨리 포기하면 이득"을 학습하게 된다, 2026-09-21 리뷰 재현).

    더미 행의 `reward` 를 제 `value` 로, `done` 을 1 로 주면 `delta = value − value = 0` 이고
    `not_done = 0` 이라 사슬이 정확히 그 자리에서 끊긴다 — 이 행 자체는 `valid=0` 이라 학습
    배치에는 어차피 안 들어간다. 진짜 종료(terminated)·계속되는 걸음(`prev_done` 이 거짓인
    행)은 손대지 않는다.
    """
    dummy = torch.as_tensor(prev_done, device=value.device)
    reward_t = torch.as_tensor(reward, dtype=torch.float32, device=value.device)
    done_t = torch.as_tensor(term, dtype=torch.float32, device=value.device)
    reward_t = torch.where(dummy, value.detach(), reward_t)
    done_t = torch.where(dummy, torch.ones_like(done_t), done_t)
    return reward_t, done_t


def _build_parser() -> argparse.ArgumentParser:
    default_cfg = PPOConfig()   # 아래 네 개 CLI 기본값의 유일한 출처 — 숫자를 여기 따로 못박지 않는다.
    default_reward_cfg = RewardConfig()   # 아래 세 개 승차감 CLI 기본값의 유일한 출처.
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None,
                    help="M3 DrivePolicy 체크포인트로 정책을 워밍스타트한다(가치 머리는 새로 시작)")
    ap.add_argument("--curricula", nargs="+",
                    default=[os.path.join(REPO, "curricula", "stage1.json"),
                            os.path.join(REPO, "curricula", "stage2.json")])
    ap.add_argument("--envs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="lab-main 14, OMEN 30 — 실측 기준(cpu_count - 2)")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--warmup-updates", type=int, default=5,
                    help="처음 이 갱신 수만큼은 정책 기울기를 끄고 가치·모방만 학습한다(스펙 §6.3)")
    ap.add_argument("--eval-every", type=int, default=500_000)
    ap.add_argument("--eval-seeds", type=int, default=1,
                    help="학습 중 주기 평가 시드 수 — 값싸게, 자주(로그·체크포인트 판단용)")
    ap.add_argument("--final-eval-seeds", type=int, default=3,
                    help="학습이 끝난 뒤 딱 한 번의 전체 평가 시드 수 — 요약 JSON·ac-best.pt 선정에 쓴다")
    ap.add_argument("--dagger-data", default=None, help="M3 가 모은 조각 폴더(runs/.../data)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    # 2026-09-21 본 학습(OMEN 3M 스텝) 진단: log_std[0](조향) 이 처음부터 끝까지 하한 -2.0 에
    # 붙어 탐색이 전혀 없었고, approx_kl 이 내내 target_kl 아래라 조기 종료가 한 번도 안 걸렸다
    # — 모방 NLL(~3.84)이 기본 엔트로피 보너스(~0.0096)를 400배 압도한다는 Task 5 리뷰의 예측
    # 그대로다. 이 네 개를 CLI 로 열어야 원인을 갈라 시험할 수 있다. 기본값은 `PPOConfig()` 자신의
    # 기본값이므로 아무 인자도 안 주면 지금 동작과 100% 같다.
    ap.add_argument("--entropy-coef", type=float, default=default_cfg.entropy_coef,
                    help="엔트로피 보너스 계수 — 키우면 log_std 하한 붙박이를 밀어낼 수 있다")
    ap.add_argument("--lr", type=float, default=default_cfg.lr, help="Adam 학습률")
    ap.add_argument("--target-kl", type=float, default=default_cfg.target_kl,
                    help="이 KL 을 넘으면 그 에폭에서 조기 종료한다 — 낮추면 정책 표류를 더 세게 막는다")
    ap.add_argument("--imitation-half-life", type=int, default=default_cfg.imitation_half_life,
                    help="모방 손실 계수가 절반으로 줄어드는 스텝 수 — 줄이면 PPO 가 더 일찍 선생님을 떠난다")
    ap.add_argument("--imitation-sigma", choices=("learn", "detach"),
                    default=default_cfg.imitation_sigma,
                    help="detach 면 모방 손실이 σ(log_std)를 안 건드린다 — 평균은 그대로 배운다."
                         " M4a 에서 모방이 σ 를 하한에 붙박아 조향 탐색이 없었다")
    ap.add_argument("--entropy-mode", choices=ENTROPY_MODES, default=None,
                    help="정책의 entropy_mode(net.py PolicyConfig 참고)를 덮어쓴다. 기본값"
                         " None 은 체크포인트에 저장된 값을 그대로 쓴다(--init 이 없으면"
                         " PolicyConfig() 기본값 'gaussian', 기존 동작 불변) — 다음 실험이"
                         " gaussian 대 squashed 를 맞대결시키려면 체크포인트를 새로 굽지 않고도"
                         " CLI 로 바꿀 수 있어야 한다")
    ap.add_argument("--smoke", action="store_true",
                    help="환경 2개·스텝 4000·롤아웃 64·동기 벡터 환경·평가는 각 단계 코스 A 한 판씩 시드 1개"
                         "·주기 평가 간격도 좁혀 로그 두 줄 이상을 남긴다")
    # M4c — 승차감(RewardConfig.comfort_*)을 CLI 로 연다. 기본값은 RewardConfig() 자신의
    # 기본값이라 아무 인자도 안 주면 지금 동작과 100% 같다(`_build_reward_cfg` 참고).
    ap.add_argument("--comfort-steer", type=float, default=default_reward_cfg.comfort_steer,
                    help="승차감(조향) 계수 — 음수, 클수록(더 음수) 급조작을 세게 벌한다")
    ap.add_argument("--comfort-accel", type=float, default=default_reward_cfg.comfort_accel,
                    help="승차감(가속) 계수")
    ap.add_argument("--comfort-on-intent", action="store_true",
                    help="승차감을 실행 행동이 아니라 정책의 의도(결정적 평균 — squash=True 면"
                         " tanh(mean), 아니면 mean.clamp(-1,1))의 변화량으로 잰다. PPO 가 매 걸음"
                         " 독립 표본을 뽑으므로 실행 행동 기준으로는 탐색 잡음이 승차감에 잡혀"
                         " 정책이 탐색을 죽인다(M4b 실측, 판당 -108.67). 걸면 벡터 환경 워커마다"
                         " `set_attr('intent', ...)` 를 매 걸음 보낸다 — 파이프 왕복이 늘어난다"
                         "(비용은 --comfort-on-intent 를 켠 채·끈 채 짧은 실행으로 실측해라).")
    return ap


def _build_cfg(a) -> PPOConfig:
    """`PPOConfig` 는 frozen dataclass 라 `dataclasses.replace` 로 CLI 로 연 네 필드만 덮어쓴다.

    인자를 하나도 안 주면 `a.lr`/`a.entropy_coef`/`a.target_kl`/`a.imitation_half_life` 가 이미
    `PPOConfig()` 자신의 기본값이므로(위 `_build_parser` 참고) 이 함수가 만드는 `cfg` 는
    `PPOConfig()` 와 완전히 같다 — 기본 동작이 안 바뀐다.
    """
    return dataclasses.replace(PPOConfig(), lr=a.lr, entropy_coef=a.entropy_coef,
                               target_kl=a.target_kl, imitation_half_life=a.imitation_half_life,
                               imitation_sigma=a.imitation_sigma)


def _build_reward_cfg(a) -> RewardConfig:
    """`RewardConfig` 는 frozen dataclass 라 `dataclasses.replace` 로 CLI 로 연 세 필드만 덮어쓴다.

    인자를 하나도 안 주면 `a.comfort_steer`/`a.comfort_accel`/`a.comfort_on_intent` 가 이미
    `RewardConfig()` 자신의 기본값이므로(위 `_build_parser` 참고) 이 함수가 만드는 `cfg` 는
    `RewardConfig()` 와 완전히 같다 — 기본 동작이 안 바뀐다(`_build_cfg` 와 같은 패턴).
    """
    return dataclasses.replace(RewardConfig(), comfort_steer=a.comfort_steer,
                               comfort_accel=a.comfort_accel, comfort_on_intent=a.comfort_on_intent)


def _build_optimizer(net, cfg: PPOConfig) -> torch.optim.Optimizer:
    """`cfg.lr` 이 실제로 옵티마이저에 닿는 자리 — 이 프로젝트에서 `gamma`/`lam` 이 `PPOConfig`

    에는 있는데 실제로는 안 쓰여 조용히 무시된 적이 있다(`compute_gae` 호출부). `--lr` 을 줬는데
    여기가 하드코딩된 값을 쓰면 같은 함정이 반복된다 — 그래서 이 자리를 별도 함수로 떼어
    `opt.param_groups[0]["lr"]` 을 직접 테스트한다(`tests/rl/test_train_ppo.py`).
    """
    return torch.optim.Adam(net.parameters(), lr=cfg.lr)


def _apply_entropy_mode(net: ActorCritic, entropy_mode: str | None) -> ActorCritic:
    """`--entropy-mode` 덮어쓰기 — `None` 이면 손대지 않고 그대로 돌려준다(체크포인트 값 유지,

    기존 동작 불변). 값이 있으면 `net.policy.cfg`(학습에 실제로 쓰이는 정책)와
    `net.cfg.policy`(`net.save()` 가 체크포인트에 쓰는 자리) 둘 다 새 값으로 바꾼다 —
    `ActorCritic.__init__` 에서 이 둘은 원래 같은 `PolicyConfig` 객체를 참조하지만
    (`self.policy = DrivePolicy(cfg.policy)`), `PolicyConfig`/`ActorCriticConfig` 가 둘 다
    frozen dataclass 라 덮어쓰려면 새 인스턴스로 바꿔 끼워야 하고, 그러면 참조가 갈라진다 —
    한쪽만 바꾸면 학습은 새 값을 쓰는데 저장된 체크포인트엔 옛 값이 실리는(또는 그 반대인)
    어긋남이 생긴다.

    `ActorCritic.from_policy(a.init)`(체크포인트 이식)와 `ActorCritic()`(새로 시작) 두 경로가
    `main()` 에서 이미 `net` 하나로 합류한 뒤 이 함수를 통과하므로, 호출부는 어느 경로로
    만들어졌는지 몰라도 된다.
    """
    if entropy_mode is None:
        return net
    new_policy_cfg = dataclasses.replace(net.policy.cfg, entropy_mode=entropy_mode)
    net.policy.cfg = new_policy_cfg
    net.cfg = dataclasses.replace(net.cfg, policy=new_policy_cfg)
    return net


def main():
    ap = _build_parser()
    a = ap.parse_args()
    # 검증을 --smoke 덮어쓰기보다 먼저 한다 — 순서가 반대면 `--smoke --steps 0`·
    # `--smoke --eval-every 0` 처럼 사용자가 준 잘못된 값이 검증 전에 스모크 기본값으로
    # 조용히 덮여 에러가 안 난다(2026-09-22 리뷰 지적).
    if a.steps < 1:
        ap.error("--steps 는 1 이상이어야 한다")
    if a.eval_every < 1:
        ap.error("--eval-every 는 1 이상이어야 한다(0 이면 while next_eval<=step 이 안 끊긴다)")
    if a.smoke:
        a.envs, a.steps, a.rollout = 2, 4000, 64
        a.eval_seeds, a.final_eval_seeds = 1, 1
        # 기본 --eval-every(500000)는 스모크 스텝(4000)보다 훨씬 커서 주기 평가가 학습 끝에
        # 딱 한 번만 걸린다 — 로그가 한 줄뿐이면 드리프트가 늘어나는지(rows[0] vs rows[-1])를
        # 볼 수가 없다(2026-09-21 실측: 이 값을 안 좁히면 test_연습_모드가_계측과_판정을_남긴다
        # 가 자기 자신과 비교하게 되어 못 통과한다). 두 번(중간·끝) 이상 걸리게 절반으로 줄인다.
        a.eval_every = max(1, a.steps // 2)

    log_path = os.path.join(a.out, "log.jsonl")
    if os.path.exists(log_path) or glob.glob(os.path.join(a.out, "ac-*.pt")):
        ap.error(f"`{a.out}` 에 이미 log.jsonl 또는 체크포인트가 있다 — 이어 쓰면 서로 다른 두"
                " 실행이 섞인다. 지우거나 다른 --out 을 써라.")
    os.makedirs(a.out, exist_ok=True)

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    stages = _stage_boards(a.curricula, a.smoke)

    # 판 하나가 2500~13100 걸음인데 --smoke 의 훈련 예산(기본 4000)은 그보다 작을 수 있어,
    # 판이 실행 내내 한 번도 안 끝나면 `rollout_return_n` 이 계측 줄마다 0 으로만 찍힌다
    # (2026-09-21 실측 — 브리프가 놓친 불일치, RED 로 잡았다). 훈련용 벡터 환경만 timeout 을
    # 크게 당겨 스모크 예산 안에서 반드시 최소 한 판은 끝나게 한다. 평가(`evaluate_policy`·
    # `evaluate_teacher`)는 이 값을 안 받고 각자 `EnvConfig()` 기본값을 새로 만들어 쓰므로,
    # 이 축소는 훈련 롤아웃에만 미치고 성적 판정(완주율·점수)에는 영향이 없다.
    reward_cfg = _build_reward_cfg(a)
    train_env_cfg = (EnvConfig(world=WorldConfig(time_limit_scale=0.1), reward=reward_cfg)
                     if a.smoke else EnvConfig(reward=reward_cfg))

    cfg = _build_cfg(a)
    # 이번 실행에 쓴 하이퍼파라미터 — log.jsonl 각 줄과 요약 JSON 에 그대로 싣는다. 성적표가
    # 여러 실행을 비교할 때 무엇이 달랐는지 산출물만 보고 알 수 있어야 한다(2026-09-21 지시).
    hparams = vars(a)
    venv = None
    try:
        # venv 는 서브프로세스(비동기 벡터 환경)를 띄운다 — `--init` 오타 등으로 그 아래 어떤
        # 준비 코드가 죽어도 finally 가 반드시 타도록 venv 생성부터 이 try 안에 둔다.
        venv = make_vec_env(a.curricula, a.envs, train_env_cfg, seed=a.seed, asynchronous=not a.smoke)
        net = (ActorCritic.from_policy(a.init, device=dev) if a.init else ActorCritic()).to(dev)
        net = _apply_entropy_mode(net, a.entropy_mode)   # 두 경로(이식/새로 시작) 모두 여기 합류한 뒤 지난다
        ref_state = snapshot_policy(net)   # 드리프트 기준점 — `--init` 로드 직후, 갱신 전
        opt = _build_optimizer(net, cfg)
        buf = RolloutBuffer(a.rollout, a.envs, dev)
        gen = torch.Generator().manual_seed(a.seed)
        dagger_iter = (dagger_batches(load_dir(a.dagger_data), TrainConfig().batch_size,
                                      generator=gen, device=dev) if a.dagger_data else None)

        t0 = time.perf_counter()
        obs, _info = venv.reset(seed=a.seed)
        updates = 0
        prev_done = np.zeros(a.envs, dtype=bool)   # 직전 걸음에 끝난 환경 = 이번 걸음은 자동 리셋 더미
        step = 0
        next_eval = a.eval_every
        best_step, best_metric = None, None
        # 롤아웃 256걸음 × 환경 30개 = 7680 환경-걸음인데 판은 2500~5000걸음이라 롤아웃마다
        # 2~3판만 끝난다 — 매 롤아웃 reset() 하면 표본 2~3개짜리 평균이 잡음을 신호로 찍는다.
        # window=30 이면 한 줄이 최근 롤아웃 약 12개 분량의 이동평균이 된다(2026-09-21 설계).
        # `reset()` 은 부르지 않는다 — 누적은 실행 내내 이어간다.
        tracker = ReturnTracker(a.envs, window=30)
        # M4b 의 "승차감 −128 대 위반 −96" 은 계산이지 측정이 아니다 — info["reward_terms"] 가
        # 걸음마다 이미 있는데 한 번도 로깅된 적이 없다. tracker 와 같은 이동창(30)을 쓴다.
        terms = RewardTermTracker(a.envs, window=30)
        outcomes = OutcomeCounter()   # 평가 구간마다(로그 한 줄마다) 새로 만든다

        while step < a.steps:
            buf.reset()
            values_log, valid_log = [], []
            # 행동 상자 진단 누적기 — 롤아웃마다 평균 내 `row` 에 싣는다(아래 `act_abs_mean` 등).
            act_abs_sum, act_sat_count, act_count = 0.0, 0, 0
            mean_abs_sum, mean_count = 0.0, 0
            for _ in range(a.rollout):
                vec, objs, mask = (torch.as_tensor(x, device=dev) for x in vec_obs_to_arrays(obs))
                with torch.no_grad():
                    out = net.act(vec, objs, mask, generator=gen)
                action = {"control": out["control"].cpu().numpy(), "turn": out["turn"].cpu().numpy()}
                if a.comfort_on_intent:
                    # 의도 = 정책의 결정적 평균(스쿼시 전 `out["mean"]`, Task 3 의 squash 를
                    # 따른다). 벡터 환경 워커는 별도 프로세스라 정책을 모른다 — 값은
                    # `set_attr()` 로 프로세스 경계를 넘길 수 있지만 콜러블(람다)은 피클이
                    # 안 되므로 `VtdDriveEnv.intent` 평범한 속성에 값만 내려보낸다
                    # (`command_tags` 와 다른 점: 저건 워커 안에서 직접 건다). 이 호출은 워커마다
                    # 파이프 왕복 + 동기 대기라 매 걸음 비용이 늘어난다 — `--comfort-on-intent`
                    # 를 켠 채·끈 채 짧은 실행으로 `elapsed_s` 를 실측해라(M4c 보고서).
                    squash = net.policy.cfg.squash
                    intent_t = torch.tanh(out["mean"]) if squash else out["mean"].clamp(-1.0, 1.0)
                    intent_np = intent_t.detach().cpu().numpy()
                    venv.set_attr("intent", [(float(r[0]), float(r[1])) for r in intent_np])
                obs, reward, term, trunc, info = venv.step(action)
                # 자동 리셋 더미 행(prev_done)은 reward=제 가치·done=1 로 줘서 GAE 사슬을 끊는다
                # (`_bootstrap_reward_done` 참고) — `done` 에 `term` 만 그대로 넣으면 truncated
                # 로 끝난 판 뒤에서 다음 판의 큰 가치가 (γλ)^k 로 새어 든다.
                reward_t, done_t = _bootstrap_reward_done(reward, term, prev_done, out["value"])
                valid_np = ~prev_done
                buf.add(vec=vec, objs=objs, mask=mask, raw=out["raw"], turn=out["turn"],
                        log_prob=out["log_prob"], value=out["value"],
                        reward=reward_t, done=done_t,
                        valid=torch.as_tensor(valid_np, dtype=torch.float32))
                values_log.append(out["value"].detach())
                valid_log.append(torch.as_tensor(valid_np, dtype=torch.bool, device=dev))
                # 행동 상자 진단 — tanh 스쿼시는 `|control| ≤ 1` 을 자명하게 통과시켜 그 검사만으론
                # 아무것도 못 증명한다(리뷰 실측: 평균을 ±7 로 밀어도 100% 통과하지만 40.4% 가
                # 정확히 ±1.0 이라 클램프와 다를 게 없다). 진짜 질문은 포화(`act_sat_frac`)와
                # 스쿼시 전 평균이 상자를 벗어나는지(`mean_abs_mean`)다. 자동 리셋 더미 행
                # (`valid_np`가 거짓인 자리)은 환경이 행동 자체를 무시하므로 `explained_variance`
                # 와 같은 이유로 뺀다.
                if valid_np.any():
                    ctrl_valid = action["control"][valid_np]
                    act_abs_sum += float(np.abs(ctrl_valid).sum())
                    act_sat_count += int((np.abs(ctrl_valid) > 0.99).sum())
                    act_count += ctrl_valid.size
                    mean_valid = out["mean"].detach().cpu().numpy()[valid_np]
                    mean_abs_sum += float(np.abs(mean_valid).sum())
                    mean_count += mean_valid.size
                done_mask = np.asarray(term) | np.asarray(trunc)
                # PPO 가 실제로 최대화하는 확률적 롤아웃 리턴 — 결정적 평가 보상과 달리 한 번도
                # 로깅된 적이 없었다(M4a 붕괴 원인 불명의 근본 원인, diagnostics.py 모듈 docstring).
                tracker.add(reward, done_mask)
                # 항목별 보상(진행·시간·위반·승차감) 누적 — gymnasium 이 reward_terms(dict)를
                # {항목명: 환경별 배열} 로 전치해 내므로 _reward_terms_per_env 로 되돌린다
                # (RewardTermTracker.add() 계약은 환경별 dict 리스트, 리셋 직후는 빈 dict).
                terms.add({"reward_terms": _reward_terms_per_env(info, a.envs)})
                terms.add_done(done_mask)
                # 종료 사유 집계 — `outcome` 은 "running" 이 아닌 스텝(그 판이 실제로 끝난
                # 스텝)에만 실린다(OutcomeCounter 가 스스로 거른다).
                outcomes.add(info)
                prev_done = done_mask
                step += a.envs
            with torch.no_grad():
                vec, objs, mask = (torch.as_tensor(x, device=dev) for x in vec_obs_to_arrays(obs))
                last_value = net.value(vec, objs, mask)
            buf.compute_gae(last_value, gamma=cfg.gamma, lam=cfg.lam)
            warming = updates < a.warmup_updates          # 가치만 먼저(스펙 §6.3)
            for p in net.policy.parameters():
                p.requires_grad_(not warming)
            stats = update(net, opt, buf, dagger_iter if not warming else None, cfg, step,
                           generator=gen)
            for p in net.policy.parameters():
                p.requires_grad_(True)
            updates += 1

            # 롤아웃마다 가벼운 진단 줄을 남긴다 — 평가(비싸다)를 기다리면 3M 실행에 약
            # 6줄뿐이라 M4a 가 무너진 1M~1.5M 구간을 점 2개로만 보게 된다(2026-09-22 리뷰
            # 지적). 여기 들어가는 값은 전부 이미 계산돼 있던 것이라 비용이 사실상 0이다.
            # 진단(explained variance)도 `valid` 로 자동 리셋 더미 행을 뺀다 — 학습 배치에
            # 안 들어가는 행이 진단에도 안 들어가야 한다.
            values_t, valid_t = torch.stack(values_log), torch.stack(valid_log)
            ev_var = _explained_variance(buf.returns[valid_t], values_t[valid_t])
            # `stats["updates"]` 는 이 롤아웃 안에서 도른 미니배치 최적화 걸음 수(ppo.update() 자체
            # 반환값)다 — 바깥 루프 반복 횟수(우리 `updates` 변수)와 이름이 겹치므로 `iter` 로 적는다.
            # `tracker.stats()`(rollout_return_mean/_n·rollout_len_mean)와
            # `terms.stats()`(term_progress_mean/_time_mean/_violation_mean/_comfort_mean/_n)와
            # `policy_drift()`(drift_l2/_rel/_log_std/_rest_rel)는 이름이 서로 겹치지 않고
            # `stats`(policy/value/entropy/approx_kl/clip_frac/imitation/imitation_coef/updates)
            # 와도 안 겹친다(M4a 에서 `**stats` 가 바깥 `updates` 를 조용히 덮어쓴 적이 있어
            # 대조해 확인했다). `act_abs_mean`/`act_sat_frac`/`mean_abs_mean`(M4c 행동 상자 진단)도
            # 위 다섯 그룹 어느 키와도 안 겹친다(대조 확인 완료).
            act_abs_mean = (act_abs_sum / act_count) if act_count else None
            act_sat_frac = (act_sat_count / act_count) if act_count else None
            mean_abs_mean = (mean_abs_sum / mean_count) if mean_count else None
            row = {"step": step, "iter": updates, "elapsed_s": time.perf_counter() - t0,
                   **stats,
                   **tracker.stats(),
                   **terms.stats(),
                   **policy_drift(net, ref_state),
                   "explained_variance": ev_var if ev_var == ev_var else None,  # NaN → None(표준 JSON)
                   "act_abs_mean": act_abs_mean, "act_sat_frac": act_sat_frac,
                   "mean_abs_mean": mean_abs_mean,
                   "log_std": net.policy.log_std.detach().cpu().tolist(),
                   "hparams": hparams}

            if step >= next_eval or step >= a.steps:
                net.eval()
                evs = _evaluate_stages(net.policy, stages, tuple(range(a.eval_seeds)))
                net.train()
                # 평가 줄에만 그 구간의 종료 사유 집계를 더 싣는다 — `outcomes.stats()` 는 매
                # 키가 `outcome_` 로 시작해 위 dict 들과 안 겹친다. 단계 결과는 "stages" 하위에
                # 감싼다 — 커리큘럼 파일 이름을 그대로 키로 쓰면(예: 누가 커리큘럼을
                # "policy.json" 처럼 지으면) 다른 통계 키와 겹칠 수 있다(`updates` 이름 충돌로
                # 한 번 데었다). 평가 줄만 고르려면 `"stages" in row` 로 거른다(Task 6 성적표가
                # 이 자리로 평가 줄을 구분한다).
                row.update(outcomes.stats())
                row["stages"] = {label: _public(ev) for label, ev in evs.items()}
                net.save(os.path.join(a.out, f"ac-{step}.pt"))
                metric = _metric(evs)
                if best_metric is None or metric > best_metric:
                    best_metric, best_step = metric, step
                outcomes = OutcomeCounter()   # 다음 평가 구간 집계를 새로 시작한다
                while next_eval <= step:
                    next_eval += a.eval_every

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if venv is not None:
            venv.close()

    # 학습이 끝난 뒤 딱 한 번 전체 평가(--final-eval-seeds) — 이 값으로 ac-best.pt 를 최종
    # 선정하고 요약 JSON 을 채운다(주기 평가는 --eval-seeds 로 값싸게, 최종 선정만 신뢰도 있게).
    net.eval()
    final_seeds = tuple(range(a.final_eval_seeds))
    final_evs = _evaluate_stages(net.policy, stages, final_seeds)
    if best_step is not None and best_step != step:
        cand = ActorCritic.load(os.path.join(a.out, f"ac-{best_step}.pt"), device=dev)
        cand.eval()
        cand_evs = _evaluate_stages(cand.policy, stages, final_seeds)
        if _metric(cand_evs) > _metric(final_evs):
            cand.save(os.path.join(a.out, "ac-best.pt"))
            chosen_evs, chosen_step = cand_evs, best_step
        else:
            net.save(os.path.join(a.out, "ac-best.pt"))
            chosen_evs, chosen_step = final_evs, step
    else:
        net.save(os.path.join(a.out, "ac-best.pt"))
        chosen_evs, chosen_step = final_evs, step

    # `iters` = 바깥 루프(롤아웃) 반복 횟수 — log.jsonl 의 `updates`(ppo.update 내부 미니배치 수)와
    # 다른 뜻이라 요약 쪽은 이름을 갈랐다.
    summary = {"steps": step, "iters": updates, "seconds": time.perf_counter() - t0,
               "best_step": chosen_step,
               "stages": {label: _public(ev) for label, ev in chosen_evs.items()},
               "hparams": hparams}

    # 판정에는 선생님 점수가 필요하다 — train_ppo.py 는 지금까지 선생님을 평가한 적이 없어
    # 여기서 단계마다 한 번씩 새로 돌린다(선생님은 규칙 스택이라 느리다, 442걸음/s). 단계①②
    # 판을 합쳐 한 번에 넘기면 판 이름이 같아 "세계 캐시가 다른 판을 준다"로 죽으므로
    # `stages`(← `load_curriculum` 로 직접 읽은, `#stage` 접미사 없는 판 목록)를 단계별로 따로
    # 넘긴다 — `--smoke` 에서는 `stages` 자체가 이미 각 단계 코스 A 한 판으로 줄어 있다.
    teacher_evs = {label: evaluate_teacher(boards, seeds=(0,)) for label, boards in stages}
    # 미완주 판은 도달 못 한 구간의 위반이 채점표에 없어 중대가 실제보다 적게 잡힌다 —
    # `completed_only` 로 완주한 판만 걸러야 조기 종료가 이득으로 보이지 않는다.
    majors = _major_totals(chosen_evs)
    verdict = judge(chosen_evs, teacher_evs, majors, eval_seeds=a.final_eval_seeds)
    summary["verdict"] = [{"name": v.name, "ok": v.ok, "line": v.line, "precondition": v.precondition}
                          for v in verdict]

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
