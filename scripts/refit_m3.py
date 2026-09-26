"""M4c Task 5 — M3 학생을 스쿼시 매개화로 재적합한다(새 출발점).

`tanh` 스쿼시는 평균의 **의미**를 바꾼다(`tanh(mean)` != `clamp(mean)`) — 기존 M3
(`squash=False`) 체크포인트를 그대로 웜스타트하면 매개화가 달라 출발점이 왜곡된다. 그래서
같은 M3 라벨로 `squash=True` `DrivePolicy` 를 **처음부터** 다시 적합한다(체크포인트 이식 없음).

    env -u PYTHONPATH .venv/bin/python scripts/refit_m3.py \
        --data runs/lab-main/2026-09-17-dagger-fix2/data \
        --out runs/omen/2026-09-23-m3-squash/policy.pt --epochs 8 --seed 0 --device auto --eval

리뷰 실측(200,511 행): 선생님 라벨의 10.48%가 정확히 ±1 이다(가속 20.77%) — `atanh_eps`
기본값은 이 실측을 바탕으로 `TrainConfig` 에서 정한다(`vtd_rl/policy/train.py` 참고).
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.dataset import load_dir  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy  # noqa: E402
from vtd_rl.policy.net import DrivePolicy, PolicyConfig  # noqa: E402
from vtd_rl.policy.train import TrainConfig, train_epochs  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

PUBLIC_KEYS = ("goal_rate", "mean_score", "mean_score_raw", "mean_reward")
CURRICULA = (os.path.join(REPO, "curricula", "stage1.json"),
             os.path.join(REPO, "curricula", "stage2.json"))


def _public(ev: dict) -> dict:
    """`evaluate_policy` 결과에서 직렬화 안 되는 `episodes` 를 거른다.

    `scripts/train_ppo.py` 에 같은 이름의 헬퍼가 있다 — 두 스크립트가 서로를 임포트하지
    않으므로(독립 실행 스크립트) 작게 다시 쓴다.
    """
    return {k: ev[k] for k in PUBLIC_KEYS}


def _stages():
    """커리큘럼 파일마다 (이름표, 판 목록) — `train_ppo.py::_stage_boards` 와 같은 규칙.

    단계마다 따로 평가해야 한다 — 판 이름이 단계끼리 같아서(`signals` 만 다르다) 합쳐
    넘기면 `VtdDriveEnv` 의 세계 캐시가 "다른 판을 준다" 로 죽는다(2026-09-21 실측).
    """
    out = []
    for path in CURRICULA:
        label = os.path.splitext(os.path.basename(path))[0]
        _name, boards = load_curriculum(path)
        out.append((label, boards))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="M3 가 모은 조각 폴더(runs/.../data)")
    ap.add_argument("--out", required=True, help="새 squash=True 체크포인트 경로(policy.pt)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--atanh-eps", type=float, default=TrainConfig().atanh_eps,
                    help="선생님 행동을 atanh 로 옮길 때의 경계 여유. 라벨의 10.5%%(가속만"
                         " 20.8%%)가 정확히 ±1 이라 이 값이 학습을 좌우한다 — 기본값은"
                         " 400 스텝 행동공간 MAE 실측으로 골랐다")
    ap.add_argument("--init", default=None,
                    help="M3 체크포인트에서 가중치를 이어받아 **미세조정**한다. 스쿼시는 파라미터를"
                         " 하나도 안 늘리므로 state_dict 모양이 같다 — 다만 tanh(mean) ≠"
                         " clamp(mean) 이라 행동은 달라진다(그래서 짧게라도 다시 적합해야 한다)")
    ap.add_argument("--freeze-sigma", action="store_true",
                    help="σ 를 초기값에 고정하고 평균만 적합한다. 스쿼시에서는 라벨이 atanh 로"
                         " 늘어나 잔차가 커지고, 학습되는 σ 가 그 잔차를 설명하려고 부풀면"
                         " 평균의 기울기가 σ² 로 나뉘어 죽는다(NLL 죽음의 나선) — 단"
                         " 2026-09-26 실측: 이 가설로 얼려 본 두 실행 모두 완주율 0.000"
                         "(점수만 99.3~99.4 로 높은데 이는 '깨끗하게 안 움직인 차' 라는 뜻이다,"
                         " docs/reports/m4c-refit.md 1 차 표) — 쓰지 마라")
    ap.add_argument("--eval", action="store_true",
                    help="단계①②를 각각 평가해 요약 JSON 의 stages 에 넣는다")
    a = ap.parse_args()

    if os.path.exists(a.out):
        ap.error(f"`{a.out}` 에 이미 정책이 있다 — 지우면 이어 쓰는 두 실행이 섞이니 지우거나"
                 " 다른 --out 을 써라.")

    dev = pick_device(a.device)
    # 그물을 짓기 **전에** 전역 씨앗을 건다. `TrainConfig(seed=)` 는 배치 순서만 정하므로
    # 이걸 안 걸면 초기 가중치가 전역 RNG 상태를 타 **같은 인자로도 실행마다 달라진다**
    # (`run_dagger.py:245` 가 같은 이유로 같은 일을 한다). 2026-09-26 실측: 씨앗을 걸기 전에는
    # 같은 설정이 완주율 0.833 과 0.333 을 오갔다 — 재적합 비교가 전부 잡음이었다.
    torch.manual_seed(a.seed)
    cfg = PolicyConfig(squash=True)
    net = DrivePolicy(cfg)
    if a.init:
        blob = torch.load(a.init, map_location="cpu", weights_only=False)
        if tuple(blob["cfg"]["trunk"]) != cfg.trunk:
            ap.error(f"--init 의 trunk {blob['cfg']['trunk']} 가 {cfg.trunk} 와 다르다")
        net.load_state_dict(blob["state"])   # 스쿼시는 파라미터를 안 늘린다 — 모양이 같다
        net.clamp_log_std()                  # 옛 체크포인트의 log_std 가 지금 하한 밖일 수 있다
    net = net.to(dev)
    ds = load_dir(a.data)
    tcfg = TrainConfig(epochs=a.epochs, seed=a.seed, squash=True, atanh_eps=a.atanh_eps,
                       sigma_grad=not a.freeze_sigma)
    log = train_epochs(net, ds, tcfg, device=dev)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    net.save(a.out)

    summary = {"epochs": a.epochs, "samples": len(ds), "seed": a.seed,
               "atanh_eps": a.atanh_eps, "init": a.init, "freeze_sigma": a.freeze_sigma,
               "log_std": net.log_std.detach().cpu().tolist(), **log}
    if a.eval:
        # 단계마다 따로 — 판 이름이 같아 합쳐 넘기면 세계 캐시 어서션으로 죽는다
        summary["stages"] = {label: _public(evaluate_policy(net, boards, seeds=(0, 1, 2)))
                             for label, boards in _stages()}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
