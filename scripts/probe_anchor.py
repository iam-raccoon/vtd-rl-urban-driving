"""M4m Task 2 — KL 앵커가 듣는가. 가장 싼 결정적 실험. (M4n 에서 **회피 가중** 축이 붙었다.)

DAgger 를 안 돌린다. M4l 이 쓴 것과 같은 방식으로 `train_epochs` + `evaluate_policy` 를
직접 부른다: M3 체크포인트에서 출발해 **단계 ③ 데이터 1 에포크**를 `anchor_coef` 를 바꿔 가며
학습하고, 단계 ①(유지됐는가)과 단계 ③(배웠는가)을 **나란히** 잰다.

    env -u PYTHONPATH .venv/bin/python scripts/probe_anchor.py \
        --init runs/omen/2026-09-26-m3-squash-warm/policy.pt \
        --data runs/omen/2026-09-30-m4k-dagger-stage3/data \
        --out runs/omen/$(date +%F)-m4m-probe/rows.jsonl

무엇을 돌릴지만 먼저 볼 때(아무것도 안 돌린다):

    env -u PYTHONPATH .venv/bin/python scripts/probe_anchor.py --dry-run \
        --init runs/omen/2026-09-26-m3-squash-warm/policy.pt \
        --data runs/omen/2026-09-30-m4k-dagger-stage3/data

★ 숫자를 읽는 법 — M4l 이 같이 밝힌 계측 사실 둘이 여기 그대로 걸린다.

1. **학습이 비결정적이다.** 배치 순서 시드만 바꿔도 단계 ① 완주율이 16.7% ↔ 66.7% 로 흔들리는데
   손실·MAE·Δθ 는 소수 셋째 자리까지 같다. 그래서 이 스크립트는 설정마다 **학습 시드 여러 개**를
   돌리고 **평균과 범위**를 같이 찍는다. 단일 숫자로는 아무것도 못 말한다.
2. **단계 ① 은 시드가 세계를 안 바꾼다**(액터 0 개·항상 초록). 독립 판은 **6 개**이고 해상도가
   **16.7 점**이다. 그래서 `--eval-seeds` 를 줄여도 단계 ① 숫자는 안 변하고(일이 3 분의 1 로 준다),
   대신 액터가 있는 단계 ③ 의 해상도를 잃는다 — 기본값 3 을 바꿀 때는 그걸 알고 바꿔라.

3. **완주율만 보면 학습을 놓친다.** M4m 에서 단계 ③ 가 21 개 설정 전부 0% 였고, "지표가 가린
   것인가" 를 가르려면 **얼마나 멀리 갔나**(판별 걸음 합)가 필요했다 — 그때는 던져 버리는
   스크립트로 쟀다. 지금은 표가 **총걸음과 기준점 대비 비율**을 처음부터 같이 낸다.

**M4n 의 회피 가중**(`--near-m`·`--near-weights`)은 "물체가 가까운 상태" 의 손실을 올린다.
M4m 실측으로 회피는 데이터의 2% 라 손실을 98% 의 평범한 주행이 지배한다. 격자는
`계수 × 가중 배율 × 학습 시드` 다. 목록 인자는 **전부 쉼표 구분**이다(`--coefs 1 3 10` 처럼
공백으로 주면 argparse 가 rc=2 로 즉사한다):

    env -u PYTHONPATH .venv/bin/python scripts/probe_anchor.py \\
        --init runs/omen/2026-09-26-m3-squash-warm/policy.pt \\
        --data runs/omen/2026-09-30-m4k-dagger-stage3/data-r0 \\
        --coefs 10 --near-m 30 --near-weights 1,5,20,50 --seeds 0,1,2 --epochs 8 \\
        --out runs/omen/$(date +%F)-m4n-weight/rows.jsonl

**이 스크립트는 판정하지 않는다.** 합격·불합격을 찍지 않고 숫자만 낸다 — 판정 기준은 계획서에
미리 적혀 있고, 결론은 성적표가 낸다.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.dataset import load_dir  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.policy.train import (TrainConfig, make_reference, squash_aligned,  # noqa: E402
                                 train_epochs)
from vtd_rl.world.board import load_curriculum  # noqa: E402

#: 계획서가 정한 격자 — 계수 넷 × 학습 시드 셋.
DEFAULT_COEFS = "0,0.1,1,10"
DEFAULT_SEEDS = "0,1,2"

#: 회피 가중(M4n). 기본값은 **가중치 없음** 이라 예전 호출과 격자 크기가 같다.
DEFAULT_NEAR_WEIGHTS = "1"

#: 평가 단계 — 앞이 **유지**(단계 ①), 뒤가 **학습**(단계 ③)이다. 둘을 같이 봐야 한다:
#: 유지만 보면 아무것도 안 배운 정책이 만점이고, 학습만 보면 옛 능력을 버린 정책이 만점이다.
DEFAULT_EVAL_STAGES = ["stage1", "stage3"]

#: 성적표에 그대로 쓰는 사람 말 이름(`scripts/run_dagger.py` 와 같은 표기).
STAGE_LABELS = {"stage1": "단계 ①", "stage2": "단계 ②", "stage3": "단계 ③"}

#: 한 판에서 뽑아 두는 숫자. `mean_score_completed` 는 완주 판이 없으면 `None` 이다.
EVAL_KEYS = ("goal_rate", "mean_score", "mean_score_completed", "mean_reward")

#: 판별 걸음 합을 넣는 자리. **완주율이 0 이어도 배우는지**를 이 숫자로 본다 —
#: M4m 에서 그걸 따로 재느라 던져 버리는 스크립트를 썼고, 그 숫자가 성적표로 들어갔다.
STEPS_KEY = "total_steps"


def stage_label(name: str) -> str:
    return STAGE_LABELS.get(name, name)


def stage_path(name: str) -> str:
    """단계 이름 → 커리큘럼 경로. 모르는 이름은 조용히 넘기지 않는다.

    (`scripts/run_dagger.py` 에 같은 규칙의 쌍둥이가 있다 — 스크립트끼리는 서로를 임포트하지
    않는 것이 이 레포의 관행이라 작게 다시 쓴다. `scripts/refit_m3.py::_stages` 도 같은 사정이다.)
    """
    if name.endswith(".json") or os.sep in name:
        path = name if os.path.isabs(name) else os.path.join(REPO, name)
    else:
        path = os.path.join(REPO, "curricula", f"{name}.json")
    if not os.path.exists(path):
        here = os.path.join(REPO, "curricula")
        avail = sorted(os.path.splitext(f)[0] for f in os.listdir(here) if f.endswith(".json"))
        raise ValueError(f"모르는 단계 '{name}' — 커리큘럼이 없다({path}). "
                         f"있는 단계: {', '.join(avail)}")
    return path


def eval_stages(names):
    """`(단계 이름, 판 목록)` 짝 — 이름과 판을 **한 곳에서 같이** 만든다.

    따로 만들어 `zip` 으로 붙이면 이름과 판이 어긋나도 숫자는 멀쩡해 보인다(run_dagger 가 같은
    이유로 같은 모양이다). 평가는 **변종을 안 쓴다**(`variants=1`) — 과거 성적표의 "단계 ③
    완주율" 과 같은 판 집합이라야 비교가 된다.
    """
    return [(n, load_curriculum(stage_path(n))[1]) for n in names]


def parse_floats(text: str) -> list:
    out = [float(x) for x in str(text).replace(" ", "").split(",") if x != ""]
    if not out:
        raise ValueError(f"값이 없다: {text!r}")
    return out


def parse_ints(text: str) -> list:
    out = [int(x) for x in str(text).replace(" ", "").split(",") if x != ""]
    if not out:
        raise ValueError(f"값이 없다: {text!r}")
    return out


def flat_params(net) -> torch.Tensor:
    return torch.cat([p.detach().flatten().clone() for p in net.parameters()])


def evaluate_all(net, stages, eval_seeds: int) -> dict:
    """단계마다 **따로** 평가한다 — 판 이름이 단계끼리 같아서(신호·액터만 다르다) 한 번에
    넘기면 `VtdDriveEnv` 의 세계 캐시가 "다른 판을 준다" 로 죽는다(`refit_m3.py` 에 같은 주석).
    """
    seeds = tuple(range(eval_seeds))
    out = {}
    for name, boards in stages:
        ev = evaluate_policy(net, boards, seeds=seeds)
        row = {k: ev[k] for k in EVAL_KEYS}
        # ★ **합**이다(평균이 아니다). 판마다 죽는 걸음이 다르고 우리가 보려는 것은
        # "이 설정이 전체로 얼마나 멀리 갔나" 라서, M4m 성적표가 비교한 자료
        # (기준점 8149 = 18 판의 합)와 같은 자로 재야 한다. 평균을 쓰면 판 수가 바뀔 때
        # 조용히 다른 값이 되고 옛 숫자와 비교가 안 된다.
        row[STEPS_KEY] = sum(int(e.steps) for e in ev["episodes"])
        out[name] = row
    return out


def run_cell(init_path, dataset, cfg: TrainConfig, device, evaluate):
    """격자 한 칸 — 체크포인트를 **새로 읽어** 학습하고 평가한다.

    칸마다 새로 읽는 것이 핵심이다. 그물 하나를 돌려 쓰면 계수 사이에 학습이 쌓여 "계수 10 이
    제일 덜 움직였다" 같은 결과가 순서 때문에 나온다.

    참조는 `make_reference` 로 만든 **얼린 사본**이다 — 같은 객체를 넘기면 KL 이 항상 0 이라
    앵커가 조용히 꺼진다(`policy_loss` 가 그 경우를 거부한다).

    `evaluate` 는 주입받는다(`evaluate_all` 의 부분 적용) — 빠른 테스트가 판을 안 돌리고도
    이 함수 전체를 돌려 볼 수 있어야 한다.
    """
    net = DrivePolicy.load(init_path, device=device)
    net.clamp_log_std()              # 옛 체크포인트의 log_std 가 지금 범위 밖일 수 있다
    ref = make_reference(net)
    # 손실의 `squash` 를 그물에 맞춘다 — M3 체크포인트는 `squash=True` 다. 안 맞추면 tanh 정책을
    # clamp 가능도로 학습해 실험 전체가 엉뚱한 것을 재게 된다(M4l 이 H3 으로 따로 확인한 자리).
    tcfg = squash_aligned(cfg, net)
    start = flat_params(net)
    t0 = time.perf_counter()
    train = train_epochs(net, dataset, tcfg, device=device, ref=ref)
    row = {"anchor_coef": tcfg.anchor_coef, "seed": tcfg.seed, "epochs": tcfg.epochs,
           "lr": tcfg.effective_lr(), "squash": tcfg.squash, "samples": len(dataset),
           "near_m": tcfg.near_m, "near_weight": tcfg.near_weight,
           # 가중이 **실제로 몇 %에 걸렸나**. 0 이면 설정만 켜고 아무 일도 안 일어난 것이다
           # (M4m 실측 기대값: near_m=30 이면 약 11.4%).
           "near_frac": train["near_frac"],
           "loss": train["loss"], "control": train["control"], "turn": train["turn"],
           "anchor": train["anchor"],
           "delta_theta": float((flat_params(net) - start).norm()),
           "theta_norm": float(start.norm()),
           "train_s": train["seconds"]}
    row["stages"] = evaluate(net)
    row["cell_s"] = time.perf_counter() - t0
    return row


def summarize(values) -> dict:
    """시드들의 평균과 범위 — `None`(완주 판 없음)은 빼고 센다."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(vals), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}


def cell_key(row) -> tuple:
    """한 칸을 가르는 설정 = `(anchor_coef, near_weight)`. 시드만 이 안에서 묶인다.

    **계수만으로 묶으면 안 된다** — M4n 격자는 계수를 10 으로 고정하고 `near_weight` 를
    흔들기 때문에, 계수로만 묶으면 서로 다른 가중치가 한 칸에 섞여 평균이 난다.
    옛 행에는 `near_weight` 가 없다(M4m `rows.jsonl`) — 그때는 가중치가 없었으니 1.0 이다.
    """
    return (row["anchor_coef"], row.get("near_weight", 1.0))


def group_cells(rows) -> list:
    """`[((계수, 가중치), [행, ...]), ...]` — 나온 순서를 지킨다."""
    out: list = []
    for r in rows:
        key = cell_key(r)
        for seen, bucket in out:
            if seen == key:
                bucket.append(r)
                break
        else:
            out.append((key, [r]))
    return out


def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}"


def _num(x, digits=1):
    return "—" if x is None else f"{x:.{digits}f}"


def _ratio(x, base):
    """기준점 대비 비율[%] — 기준점이 없거나 0 이면 잰 적이 없다는 뜻으로 '—'."""
    if x is None or not base:
        return "—"
    return f"{x / base * 100:.1f}%"


def baseline_steps(baseline, name):
    """기준점(학습 전)의 단계별 총걸음 — 없으면 `None`(비율을 '—' 로 찍게)."""
    if baseline is None:
        return None
    return baseline.get("stages", {}).get(name, {}).get(STEPS_KEY)


def table_lines(rows, stage_names, baseline=None) -> list:
    """(계수, 가중치) × (단계 ① 유지 / 단계 ③ 학습) 표 — 칸마다 **평균과 범위**를 같이 찍는다.

    ★ **총걸음 열**이 같이 나온다(판별 걸음 합과 기준점 대비 비율). 완주율만 보면 학습을
    놓친다 — M4m 에서 단계 ③ 가 21 개 설정 전부 0% 였는데, "완주율이 가린 것인가" 를
    가르려면 "얼마나 멀리 갔나" 가 필요했다(그때는 던져 버리는 스크립트로 쟀다).

    합격·불합격을 안 찍는다. 판정 기준은 계획서에 미리 적혀 있고 결론은 성적표가 낸다 —
    스크립트가 문턱을 들고 있으면 결과를 보고 문턱을 고치고 싶어진다.
    """
    head = ["| anchor_coef | near_weight | 학습 시드 |"]
    sep = ["|---:|---:|---:|"]
    for n in stage_names:
        head.append(f" {stage_label(n)} 완주율 평균 | {stage_label(n)} 범위 |"
                    f" {stage_label(n)} 점수 평균 | {stage_label(n)} 총걸음 평균 |"
                    f" {stage_label(n)} 기준점비 |")
        sep.append("---:|---:|---:|---:|---:|")
    head.append(" Δθ 평균 | 손실 | 앵커 KL |")
    sep.append("---:|---:|---:|")
    lines = ["".join(head), "".join(sep)]

    if baseline is not None:
        cells = ["| — (학습 전) | — | — |"]
        for n in stage_names:
            st = baseline["stages"].get(n, {})
            steps = st.get(STEPS_KEY)
            cells.append(f" {_pct(st.get('goal_rate'))}% | — | {_num(st.get('mean_score'))} |"
                         f" {_num(steps, 0)} | {_ratio(steps, steps)} |")
        cells.append(" 0.000 | — | — |")
        lines.append("".join(cells))

    for (coef, weight), bucket in group_cells(rows):
        seeds = ",".join(str(r["seed"]) for r in bucket)
        cells = [f"| {coef:g} | {weight:g} | {seeds} (n={len(bucket)}) |"]
        for n in stage_names:
            goal = summarize([r["stages"].get(n, {}).get("goal_rate") for r in bucket])
            score = summarize([r["stages"].get(n, {}).get("mean_score") for r in bucket])
            # 칸 안(시드끼리)에서는 **평균**이다 — 한 칸의 총걸음은 그 시드의 판별 합이고,
            # 시드를 더 돌렸다고 숫자가 커지면 칸끼리 비교가 안 된다.
            steps = summarize([r["stages"].get(n, {}).get(STEPS_KEY) for r in bucket])
            rng = "—" if goal["n"] == 0 else f"{_pct(goal['min'])}–{_pct(goal['max'])}%"
            cells.append(f" {_pct(goal['mean'])}% | {rng} | {_num(score['mean'])} |"
                         f" {_num(steps['mean'], 0)} |"
                         f" {_ratio(steps['mean'], baseline_steps(baseline, n))} |")
        dtheta = summarize([r["delta_theta"] for r in bucket])
        loss = summarize([r["loss"] for r in bucket])
        anchor = summarize([r["anchor"] for r in bucket])
        cells.append(f" {_num(dtheta['mean'], 3)} | {_num(loss['mean'], 3)} |"
                     f" {_num(anchor['mean'], 3)} |")
        lines.append("".join(cells))
    return lines


def per_seed_lines(rows, stage_names) -> list:
    """시드별 원자료 — 평균 뒤에 무엇이 있는지 숨기지 않는다."""
    head = ["| anchor_coef | near_weight | 시드 |"]
    sep = ["|---:|---:|---:|"]
    for n in stage_names:
        head.append(f" {stage_label(n)} 완주율 | {stage_label(n)} 점수 |"
                    f" {stage_label(n)} 총걸음 |")
        sep.append("---:|---:|---:|")
    head.append(" Δθ | 손실 | 앵커 KL |")
    sep.append("---:|---:|---:|")
    lines = ["".join(head), "".join(sep)]
    for r in rows:
        coef, weight = cell_key(r)
        cells = [f"| {coef:g} | {weight:g} | {r['seed']} |"]
        for n in stage_names:
            st = r["stages"].get(n, {})
            cells.append(f" {_pct(st.get('goal_rate'))}% | {_num(st.get('mean_score'))} |"
                         f" {_num(st.get(STEPS_KEY), 0)} |")
        cells.append(f" {r['delta_theta']:.3f} | {r['loss']:.3f} | {r['anchor']:.3f} |")
        lines.append("".join(cells))
    return lines


def caveat_lines(rows, eval_seeds: int, stage_names) -> list:
    """이 표가 **주장하지 않는 것** — 성적표가 그대로 옮겨 쓸 문장들."""
    n_seeds = len({r["seed"] for r in rows})
    out = ["", "### 이 표가 주장하지 않는 것", "",
           f"- 설정마다 학습 시드 {n_seeds} 개다. M4l 실측: 배치 순서 시드만 바꿔도 단계 ① 완주율이"
           " 16.7% ↔ 66.7% 로 흔들리는데 손실·MAE·Δθ 는 소수 셋째 자리까지 같다. **범위가 겹치면"
           " 두 설정이 다르다고 말할 수 없다.**"]
    if "stage1" in stage_names:
        out.append("- 단계 ① 은 액터가 없고 신호가 항상 초록이라 **시드가 세계를 안 바꾼다** —"
                   f" 평가 시드를 {eval_seeds} 개 썼어도 독립 판은 **6 개**이고 해상도는 **16.7 점**이다."
                   " 완주율 한 칸 차이는 판 하나다.")
    out.append("- **총걸음은 판별 걸음의 합**이다(판 × 평가 시드 전부). 기준점비는 **학습 전**"
               " 체크포인트 대비다 — M4m 실측으로 설정 사이 폭이 −1.5~+2.3% 였다. 그 폭 안의"
               " 차이는 잡음과 구별되지 않는다.")
    if any(r.get("near_m", 0.0) > 0.0 for r in rows):
        out.append("- 회피 가중의 '가깝다' 는 **최근접 슬롯이 전방 `near_m` m 안**이다"
                   "(`object_mask` 로 거른 뒤 `objs[:,0,0]*80`). 라벨을 안 본다. 슬롯 0 은"
                   " 유클리드 최근접이라 뒤쪽 물체일 수 있고, 그때는 안 골라진다 —"
                   " 종방향 회피(제동)도 따로 안 센다. `near_frac` 이 실제로 걸린 비율이다.")
    out.append("- 평가 자체는 결정적이다(같은 체크포인트 → 같은 보상, M4l). 여기 보이는 흔들림은"
               " 평가 잡음이 아니라 **학습 잡음**이다.")
    out.append("- GPU 학습은 같은 시드 재실행도 조금 흔들린다(M4l: 32.13 대 32.23). 배치 순서"
               " 잡음에 비하면 작지만 0 은 아니다.")
    return out


def dry_run_lines(a, coefs, weights, seeds, stage_names) -> list:
    out = [f"체크포인트: {a.init}" + ("" if os.path.exists(a.init) else "  ← 없다!"),
           f"데이터: {a.data}",
           f"계수 {coefs} × 회피 가중 {weights} × 학습 시드 {seeds} ="
           f" {len(coefs) * len(weights) * len(seeds)} 칸"
           f" (칸마다 {a.epochs} 에포크)",
           f"회피 가중: near_m={a.near_m:g} m"
           + ("  ← 0 이라 가중치 없음(예전과 비트 동일)" if a.near_m <= 0.0 else ""),
           f"학습률: {TrainConfig(lr=a.lr, anchor_lr=a.anchor_lr).effective_lr():g}"
           f" (lr={a.lr:g}, anchor_lr={a.anchor_lr})",
           f"평가: {stage_names} · 평가 시드 {a.eval_seeds} 개 · 결정적"]
    ds = load_dir(a.data)
    out.append(f"데이터 행 {len(ds)} · 조각 {len(ds.round_counts)}")
    for name, boards in eval_stages(stage_names):
        actors = sum(len(b.scenario.actors) for b in boards)
        out.append(f"  평가 {name}: 판 {len(boards)}개 · 액터 {actors}개"
                   f" · 판당 시드 {a.eval_seeds} → {len(boards) * a.eval_seeds} 판")
    out.append(f"기준점(학습 전) 평가: {'한다' if a.baseline else '안 한다'}")
    out.append(f"원자료: {a.out or '(안 쓴다)'}")
    return out


def main():
    ap = argparse.ArgumentParser(description="KL 앵커가 듣는지 보는 가장 싼 실험")
    ap.add_argument("--init", required=True,
                    help="출발 체크포인트(M3 학생: runs/omen/2026-09-26-m3-squash-warm/policy.pt)")
    ap.add_argument("--data", required=True, help="학습에 쓸 조각 폴더(단계 ③ 데이터)")
    ap.add_argument("--coefs", default=DEFAULT_COEFS, help=f"anchor_coef 목록(기본 {DEFAULT_COEFS})")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS,
                    help=f"**학습** 시드 목록 = 배치 순서(기본 {DEFAULT_SEEDS}). 설정마다 이"
                         " 시드들의 평균과 범위를 낸다 — 단일 숫자는 뜻이 없다(M4l)")
    ap.add_argument("--near-m", type=float, default=0.0,
                    help="회피 가중의 '가깝다' 문턱[m] — 최근접 물체가 전방 이 거리 안인 표본의"
                         " 손실에 --near-weights 를 곱한다. 0(기본)이면 가중치 없음(M4n)")
    ap.add_argument("--near-weights", default=DEFAULT_NEAR_WEIGHTS,
                    help=f"회피 가중 배율 목록(**쉼표 구분**, 기본 {DEFAULT_NEAR_WEIGHTS})."
                         " --coefs 와 같은 규칙이다 — 공백으로 주면 argparse 가 거부한다")
    ap.add_argument("--epochs", type=int, default=1,
                    help="칸마다 돌릴 에포크(기본 1 — M4l 이 Δθ=0.54 에서 0%% 를 본 바로 그 설정)")
    ap.add_argument("--lr", type=float, default=TrainConfig().lr)
    ap.add_argument("--anchor-lr", type=float, default=None,
                    help="주면 --lr 대신 쓴다(웜스타트용 낮은 학습률). M4l: lr 은 원인이 아니지만"
                         " 3e-4 는 처음부터 배우는 값이다 — 앵커와 **따로** 움직일 수 있게 열어 둔다")
    ap.add_argument("--eval-stage", action="append", default=[],
                    help=f"평가 단계(반복 가능, 기본 {' '.join(DEFAULT_EVAL_STAGES)})."
                         " 앞이 유지, 뒤가 학습이다")
    ap.add_argument("--eval-seeds", type=int, default=3,
                    help="평가 시드 수(기본 3). 단계 ① 은 시드가 세계를 안 바꿔 1 로 줄여도 숫자가"
                         " 같다 — 단계 ③ 해상도만 잃는다")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None,
                    help="원자료 JSONL 경로(한 줄 = 한 칸). 이미 있으면 거부한다 — 두 실행이 섞이면"
                         " 어느 숫자가 어느 실행인지 알 수 없다")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false",
                    help="학습 전 체크포인트 평가를 건너뛴다(기본은 한다 — 비교 기준점이다)")
    ap.add_argument("--dry-run", action="store_true",
                    help="아무것도 안 돌리고 무엇을 돌릴지만 찍는다")
    a = ap.parse_args()

    try:
        coefs, seeds = parse_floats(a.coefs), parse_ints(a.seeds)
        weights = parse_floats(a.near_weights)
        stage_names = list(a.eval_stage) or list(DEFAULT_EVAL_STAGES)
        for n in stage_names:
            stage_path(n)
    except ValueError as exc:
        ap.error(str(exc))
    if any(c < 0 for c in coefs):
        ap.error("--coefs 는 0 이상이어야 한다")
    if a.near_m < 0:
        ap.error("--near-m 은 0 이상이어야 한다")
    if any(w <= 0 for w in weights):
        # 0 이면 그 표본이 사라지고, 전부 0 이면 가중치 합이 0 이라 손실이 NaN 이다 —
        # 격자 한 팔이 조용히 죽는 길을 미리 막는다.
        ap.error("--near-weights 는 0 보다 커야 한다")

    if a.dry_run:
        print("\n".join(dry_run_lines(a, coefs, weights, seeds, stage_names)))
        return 0
    if not os.path.exists(a.init):
        ap.error(f"--init 체크포인트가 없다: {a.init}")
    if a.out and os.path.exists(a.out):
        ap.error(f"`{a.out}` 가 이미 있다 — 두 실행이 섞인다. 지우거나 다른 --out 을 써라.")

    dev = pick_device(a.device)
    dataset = load_dir(a.data)
    if len(dataset) == 0:
        ap.error(f"--data 에 행이 없다: {a.data}")
    stages = eval_stages(stage_names)

    def evaluate(net):
        return evaluate_all(net, stages, a.eval_seeds)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def record(row):
        if a.out:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    t_all = time.perf_counter()
    baseline = None
    if a.baseline:
        print("[기준점] 학습 전 체크포인트 평가 …", file=sys.stderr, flush=True)
        net0 = DrivePolicy.load(a.init, device=dev)
        net0.clamp_log_std()
        baseline = {"anchor_coef": None, "seed": None, "stages": evaluate(net0)}
        record(baseline)

    rows, total = [], len(coefs) * len(weights) * len(seeds)
    for coef in coefs:
        for weight in weights:
            for seed in seeds:
                # 칸마다 전역 씨앗을 다시 건다 — 가중치는 체크포인트에서 오지만 전역 RNG 에
                # 기대는 자리가 생겨도 재현이 되게 묶어 둔다(`refit_m3.py` 가 같은 이유로
                # 같은 일을 한다).
                torch.manual_seed(seed)
                cfg = TrainConfig(lr=a.lr, anchor_lr=a.anchor_lr, epochs=a.epochs, seed=seed,
                                  anchor_coef=coef, near_m=a.near_m, near_weight=weight)
                print(f"[{len(rows) + 1}/{total}] anchor_coef={coef:g}"
                      f" near_weight={weight:g} seed={seed} …", file=sys.stderr, flush=True)
                row = run_cell(a.init, dataset, cfg, dev, evaluate)
                rows.append(row)
                record(row)
                print(f"    Δθ={row['delta_theta']:.3f} 손실={row['loss']:.3f}"
                      f" 앵커={row['anchor']:.3f} 가중비율={row['near_frac'] * 100:.1f}% "
                      + " ".join(f"{n}={row['stages'][n]['goal_rate'] * 100:.1f}%"
                                 f"/{row['stages'][n].get(STEPS_KEY, '—')}걸음"
                                 for n in stage_names)
                      + f" ({row['cell_s']:.0f}초)", file=sys.stderr, flush=True)

    lines = [f"# 앵커·회피 가중 탐침 — {datetime.date.today().isoformat()} · `{platform.node()}`"
             f" · 장치 `{dev}`", "",
             f"- 출발점 `{a.init}` · 학습 데이터 `{a.data}`({len(dataset)} 행)",
             f"- 칸마다 {a.epochs} 에포크 · 학습률 {TrainConfig(lr=a.lr, anchor_lr=a.anchor_lr).effective_lr():g}"
             f" · 계수 {len(coefs)} × 회피 가중 {len(weights)} × 학습 시드 {len(seeds)}"
             f" = {total} 칸 · 전체 {time.perf_counter() - t_all:.0f}초",
             f"- 회피 가중: `near_m`={a.near_m:g} m · 배율 {weights}"
             + ("(가중치 없음)" if a.near_m <= 0.0 else ""),
             f"- 평가: {' + '.join(stage_label(n) for n in stage_names)} · 평가 시드"
             f" {a.eval_seeds} 개 · 결정적", ""]
    lines += table_lines(rows, stage_names, baseline)
    lines += ["", "### 시드별 원자료", ""]
    lines += per_seed_lines(rows, stage_names)
    lines += caveat_lines(rows, a.eval_seeds, stage_names)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
