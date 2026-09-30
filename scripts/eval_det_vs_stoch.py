"""M4c 최종 리뷰 Critical — 결정적 평가와 확률적 평가를 같은 체크포인트로 나란히 잰다.

    env -u PYTHONPATH .venv/bin/python scripts/eval_det_vs_stoch.py \
        --checkpoint runs/omen/2026-09-23-m4c-squash/ac-best.pt \
        --checkpoint runs/omen/2026-09-23-m4c-squash-intent/ac-best.pt \
        --seeds 0 1 2 --device auto --out runs/omen/2026-09-26-det-vs-stoch/summary.jsonl

M4c 실험에서 롤아웃(확률적)은 거의 다 완주하는데 평가(결정적, `tanh(mean)`)는 22~33% 만
완주했다. tanh 스쿼시에서는 `tanh(mean)` 이 ±1 에 절대 못 닿으므로, 선생님이 자주 쓰는 '완전
가속'(가속 라벨의 20.8%가 정확히 ±1)을 결정적 정책만으로는 못 내고 탐색 잡음이 그걸 메우고
있다는 가설을 검증한다 — 성적표(`docs/reports/m4c-ppo-notes.md`)의 헤드라인 표가 이 스크립트의
출력이다.

체크포인트는 `DrivePolicy`(M3 등, 순수 정책)와 `ActorCritic`(M4a 이후, 가치 머리 포함) 둘 다
받는다 — 후자는 `.policy` 를 꺼낸다. 둘을 가르는 방법은 `blob["cfg"]` 에 `"policy"` 키가
있는지다: `ActorCritic.save()` 는 `{"cfg": {"policy": {...PolicyConfig...}, "value_hidden": ...}}`
로 중첩해 넣고, `DrivePolicy.save()` 는 `PolicyConfig` 필드를 `cfg` 최상위에 바로 둔다
(`{"cfg": {"obj_hidden": ..., ...}}`) — 실제 체크포인트로 확인함(runs/omen/2026-09-21-ppo/
ac-best.pt 대 runs/lab-main/2026-09-17-dagger-fix2/policy-r0.pt, 2026-09-26 실측).

평가는 단계(커리큘럼 파일)마다 따로 부른다 — `stage1.json`·`stage2.json` 은 판 이름을 그대로
공유하고 `signals` 만 다르므로, 합쳐 넘기면 `VtdDriveEnv` 의 세계 캐시가 죽는다(`refit_m3.py`·
`train_ppo.py` 와 같은 주의사항).

M4i — 잴 단계를 `--stage` 로 고를 수 있다(반복 가능, `scripts/measure_teacher.py::stage_path`
와 같은 규칙: 이름만 주면 `curricula/<이름>.json`, 경로나 `.json` 을 주면 그대로). **안 주면
예전 그대로 stage1 + stage2** 라 기존 호출부와 과거 성적표(JSONL 키)가 그대로다.

    ... scripts/eval_det_vs_stoch.py --checkpoint <ckpt> \
        --stage stage1 --stage stage2 --stage stage3 --seeds 0 1 2 --device cpu

확률적 평가는 `vtd_rl.policy.evaluate.evaluate_policy`/`run_policy_episode` 에 새로 생긴
`deterministic=False`(+`generator`) 를 그대로 쓴다 — 예전(임시) 버전은 `run_policy_episode` 가
`policy.act(obs, deterministic=True)` 를 박아 부르는 것을 뒤집는 얇은 프록시 클래스를 썼지만,
`DrivePolicy.act()` 자체가 이미 `deterministic`·`generator` 를 받으므로 프록시 없이 그 인자를
`evaluate_policy` 까지 관통시키는 쪽이 더 짧고 새 클래스가 없다. 기존 호출부는 인자를 안 주면
`deterministic=True` 그대로라 동작이 안 바뀐다.
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.rl.actor_critic import ActorCritic  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

CURRICULA = os.path.join(REPO, "curricula")
DEFAULT_STAGES = ("stage1", "stage2")     # `--stage` 를 안 주면 이것 — 예전 동작 그대로다


def stage_path(stage: str) -> str:
    """단계 이름 → 커리큘럼 파일 경로(`measure_teacher.py::stage_path` 와 같은 규칙).

    `"stage3"` 처럼 이름만 주면 `curricula/stage3.json`, 경로나 `.json` 을 주면 그대로 쓴다.
    """
    if stage.endswith(".json") or os.sep in stage:
        return stage
    return os.path.join(CURRICULA, f"{stage}.json")


def _stages(stages=None):
    """단계마다 (이름표, 판 목록) — `refit_m3.py::_stages()` 와 같은 규칙.

    단계마다 따로 평가해야 한다 — 판 이름이 단계끼리 같아서(`signals` 만 다르다) 합쳐 넘기면
    `VtdDriveEnv` 의 세계 캐시가 "다른 판을 준다" 로 죽는다. 독립 실행 스크립트라 `refit_m3.py`
    를 임포트하지 않고 작게 다시 쓴다(그 파일의 `_stages()` 주석과 같은 이유).

    `stages` 가 없으면 `DEFAULT_STAGES`. 이름표는 파일 basename 이다 — 기존 JSONL 키
    (`stage1`·`stage2`)를 그대로 유지해야 과거 성적표와 비교할 수 있다.

    모르는 이름·빈 커리큘럼·이름표 충돌은 전부 **터뜨린다**. 셋 다 조용히 지나가면 "아무것도
    안 잰" 결과가 진짜 측정처럼 보인다(`measure_teacher.py::load_stage` 와 같은 이유).
    """
    out, seen = [], {}
    for stage in (stages if stages else DEFAULT_STAGES):
        path = stage_path(stage)
        if not os.path.exists(path):
            have = sorted(os.path.splitext(f)[0] for f in os.listdir(CURRICULA)
                          if f.endswith(".json"))
            raise ValueError(f"그런 커리큘럼 단계가 없다: {stage!r} → {path} "
                             f"(있는 단계: {', '.join(have)})")
        label = os.path.splitext(os.path.basename(path))[0]
        if label in seen:
            raise ValueError(f"단계 이름표가 겹친다: {label!r} "
                             f"({seen[label]} 와 {path}) — JSONL 한 줄의 키라 덮어써진다")
        seen[label] = path
        _name, boards = load_curriculum(path)
        if not boards:
            raise ValueError(f"단계 '{stage}' 에 판이 하나도 없다")
        out.append((label, boards))
    return out


def load_policy(path: str, device):
    """`DrivePolicy` 체크포인트와 `ActorCritic` 체크포인트를 둘 다 받는다(판별법은 모듈 docstring)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "policy" in blob["cfg"]:                       # ActorCritic 체크포인트(가치 머리 포함)
        net = ActorCritic.load(path, device=device).policy
    else:                                              # 순수 DrivePolicy 체크포인트(M3 등)
        net = DrivePolicy.load(path, device=device)
    net.eval()
    return net


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="DrivePolicy 또는 ActorCritic 체크포인트 경로(반복 가능)")
    # 기본값을 `None` 으로 두는 것이 중요하다 — `action="append"` 에 리스트 기본값을 주면
    # `--stage stage3` 가 그 리스트를 **지우지 않고 덧붙인다**(stage1, stage2, stage3).
    ap.add_argument("--stage", action="append", default=None,
                    help="커리큘럼 이름(stage3) 또는 파일 경로. 여러 번 줄 수 있다"
                         f" (안 주면 {' + '.join(DEFAULT_STAGES)})")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="단계마다 도는 판 시드 목록(결정적·확률적 평가 둘 다 같은 시드로 잰다)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stoch-seed", type=int, default=0,
                    help="확률적 평가의 표본 추출 시드 — 고정해야 재현 가능하다(체크포인트마다"
                         " 같은 값을 새로 씨앗해 공정하게 비교한다)")
    ap.add_argument("--out", default=None, help="주면 체크포인트마다 한 줄씩 JSONL 로도 저장한다")
    a = ap.parse_args(argv)

    dev = pick_device(a.device)
    seeds = tuple(a.seeds)
    stages = _stages(a.stage)

    rows = []
    for path in a.checkpoint:
        net = load_policy(path, dev)
        row = {"checkpoint": path,
               "log_std": [round(x, 3) for x in net.log_std.detach().cpu().tolist()]}
        for label, boards in stages:
            det = evaluate_policy(net, boards, seeds=seeds)
            sto = evaluate_policy(net, boards, seeds=seeds, deterministic=False,
                                  generator=torch.Generator().manual_seed(a.stoch_seed))
            # `det_score`/`sto_score` 는 기존 키다(옛 JSONL 과 비교해야 하니 그대로 둔다) —
            # `mean_score_raw` 라 미완주 판도 그 판 점수 그대로 섞인 값이다(M4d 최종 리뷰
            # Critical 1: 멈춰 선 판일수록 이 값이 오히려 높아진다). `*_score_completed` 가
            # 완주 판만의 평균이고, `*_n_completed` 로 그 평균이 몇 판에서 나왔는지 알 수 있다.
            det_n_completed = sum(1 for e in det["episodes"] if e.outcome == "goal")
            sto_n_completed = sum(1 for e in sto["episodes"] if e.outcome == "goal")
            row[label] = {"det_goal": det["goal_rate"], "det_score": det["mean_score_raw"],
                          "det_score_completed": det["mean_score_completed"],
                          "det_n_completed": det_n_completed,
                          "sto_goal": sto["goal_rate"], "sto_score": sto["mean_score_raw"],
                          "sto_score_completed": sto["mean_score_completed"],
                          "sto_n_completed": sto_n_completed}
        print(json.dumps(row, ensure_ascii=False))
        rows.append(row)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
