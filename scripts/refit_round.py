"""M4t — M4s 가 모아 둔 데이터로 **라운드 K 의 그물 하나를 다시 학습**하고 선택 창으로 잰다.

M4s 에서 β 를 0.5 로 고정해도 r4 가 다섯 시드 중 넷에서 떨어졌다. 원인 후보는 (가) 같은
8 에폭이 라운드마다 더 많은 갱신이 된다, (나) 배치 순서 운, (다) r4 에 더해진 데이터.
새로 모으지 않고 **데이터(라운드 ≤ K)·갱신 횟수·배치 시드**만 바꿔 가른다
(계획서 `docs/superpowers/plans/2026-10-03-m4t-r4-drop.md`).

★ `run_dagger.py:952-987` 과 **같은 순서**로 학습한다 — 참조를 먼저 만들고,
`torch.manual_seed(init_seed + K)`, 출발 그물, `squash_aligned(TrainConfig(...))`,
`train_epochs`. 같은 배치 시드면 원본 `policy-rK.pt` 와 가중치가 같아야 하고, 그게 이
실험의 관문이다(`--compare`).

★ `--batch-seed` 는 **그대로** `TrainConfig.seed` 가 된다. `run_dagger` 는 `--train-seed + rnd`
를 넘겼으므로 M4s 시드 s 의 r3 을 재현하려면 `--batch-seed s+3` 이다.

    env -u PYTHONPATH .venv/bin/python scripts/refit_round.py \\
        --run runs/omen/2026-10-02-m4s-s0 --upto 4 --updates-like 3 \\
        --init runs/omen/2026-10-01-m4o-s0/policy-best.pt --anchor-coef 3 \\
        --batch-seed 4 --out runs/omen/2026-10-03-m4t/C-s0-b4

평가 창은 기본이 **선택 창 v8~v11** 이다. 보고 창 v4~v7 은 성적표 헤드라인용이라 진단에
안 쓴다 — 겹치면 돌기 전에 거부한다.

**이 스크립트는 판정하지 않는다.** 숫자만 낸다 — 판정 기준은 계획서에 미리 적혀 있다.
"""
import argparse
import gc
import glob
import json
import math
import os
import platform
import re
import sys
import time

import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.dataset import DaggerDataset, load_shard  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.policy.train import (TrainConfig, make_reference, squash_aligned,  # noqa: E402
                                 train_epochs)
from vtd_rl.world.board import load_window, window_label  # noqa: E402

#: M4s 가 학습에 쓴 에폭(`run_dagger --epochs 8`). `--updates-like` 예산의 밑값이다.
DEFAULT_EPOCHS = 8
DEFAULT_STAGES = ["stage1", "stage3a", "stage3b"]
ORIGINAL = "original"
_ROUND = re.compile(r"^r(\d+)-")


def round_of(name: str) -> int:
    """조각 이름 `r{K}-…npz` 의 K. 모르는 이름은 조용히 넘기지 않는다."""
    m = _ROUND.match(os.path.basename(name))
    if not m:
        raise ValueError(f"라운드 번호가 없는 조각 이름 — {name!r} (기대: r<K>-…npz)")
    return int(m.group(1))


def shard_paths(data_dir: str, upto: int) -> list:
    """라운드 ≤ `upto` 조각 경로 — **이름순**(`load_dir` 와 같은 순서).

    라운드 0~upto 중 하나라도 조각이 없으면 거부한다. 덜 끝난 실행이나 `--upto` 오타로
    K-1 까지만 학습하면 조건 A 가 B 처럼 보인다.
    """
    allp = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    picked = [p for p in allp if round_of(p) <= upto]
    have = {round_of(p) for p in picked}
    for k in range(upto + 1):
        if k not in have:
            raise ValueError(f"라운드 {k} 조각이 `{data_dir}` 에 없다 — 라운드 0~{upto} 가 다"
                             " 있어야 그 라운드를 다시 학습할 수 있다")
    return picked


def load_upto(data_dir: str, upto: int) -> DaggerDataset:
    ds = DaggerDataset()
    for p in shard_paths(data_dir, upto):
        ds.add(load_shard(p))
    return ds


def count_upto(data_dir: str, upto: int) -> int:
    """라운드 ≤ `upto` 조각의 표본 수 — 배열을 **읽지 않고** 센다(`len(load_upto(...))` 와 같다).

    npz 는 항목마다 늦게 읽으므로 `turn` 의 길이만 본다. `--updates-like` 예산에 데이터셋을
    통째로 올리면 학습 데이터와 겹쳐 메모리 최고점이 커진다.
    """
    n = 0
    for p in shard_paths(data_dir, upto):
        with np.load(p, allow_pickle=False) as z:
            n += int(z["turn"].shape[0])
    return n


def batches_per_epoch(n: int, batch_size: int) -> int:
    return math.ceil(n / batch_size)


def budget(n: int, epochs: int, batch_size: int) -> int:
    """그 데이터를 `epochs` 에폭 돌 때의 갱신 횟수."""
    return epochs * batches_per_epoch(n, batch_size)


def effective_epochs(epochs: int, max_updates, n: int, batch_size: int) -> int:
    """예산을 채우는 데 필요한 에폭 — 예산이 `epochs` 안이면 `epochs` 그대로.

    에폭을 더 두는 것은 결과를 안 바꾼다(`train_epochs` 가 예산에서 멈추고, 같은 배치 시드면
    앞 에폭들의 배치 순서가 같다). 모자라면 예산을 못 채운 채 끝나므로 늘린다.
    """
    if max_updates is None:
        return epochs
    return max(epochs, math.ceil(max_updates / batches_per_epoch(n, batch_size)))


def check_window(collect_variants: int, window_variants: int, window_offset: int):
    """평가 창이 수집 창 `[0, V)`·보고 창 `[V, 2V)` 과 겹치면 `ValueError`.

    (`run_dagger.check_windows` 의 쌍둥이 — 스크립트끼리는 임포트하지 않는다.)
    """
    lo, hi = window_offset, window_offset + window_variants
    for name, (a, b) in (("수집", (0, collect_variants)),
                         ("보고", (collect_variants, 2 * collect_variants))):
        if lo < b and a < hi:
            raise ValueError(f"평가 창 v{lo}~v{hi - 1} 이 {name} 창 v{a}~v{b - 1} 과 겹친다 —"
                             f" --window-offset 을 {2 * collect_variants} 이상으로 둬라")


def stage_path(name: str) -> str:
    """단계 이름 → 커리큘럼 경로(`eval_unseen.stage_path` 의 쌍둥이)."""
    if name.endswith(".json") or os.sep in name:
        path = name if os.path.isabs(name) else os.path.join(REPO, name)
    else:
        path = os.path.join(REPO, "curricula", f"{name}.json")
    if not os.path.exists(path):
        raise ValueError(f"모르는 단계 '{name}' — 커리큘럼이 없다({path})")
    return path


def start_net(init: str, dev) -> DrivePolicy:
    """출발 그물(`run_dagger.round_start_net` 의 `--init` 경로 쌍둥이)."""
    net = DrivePolicy.load(init, device=dev)
    net.clamp_log_std()
    return net


def refit(init, dataset, upto, batch_seed, epochs, max_updates, anchor_coef, init_seed, dev):
    """`run_dagger.py:952-987` 과 같은 순서로 그물 하나를 학습한다 → `(net, train)`."""
    ref = make_reference(start_net(init, dev)) if anchor_coef > 0.0 else None
    torch.manual_seed(init_seed + upto)
    net = start_net(init, dev)
    tcfg = squash_aligned(TrainConfig(epochs=epochs, seed=batch_seed, anchor_coef=anchor_coef,
                                      max_updates=max_updates), net)
    train = train_epochs(net, dataset, tcfg, device=dev, ref=ref)
    return net, train


def same_weights(net, ckpt: str) -> bool:
    """가중치가 글자 그대로 같은가(파일 md5 가 아니라 텐서로 본다)."""
    other = DrivePolicy.load(ckpt, device="cpu").state_dict()
    mine = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return mine.keys() == other.keys() and all(torch.equal(mine[k], other[k]) for k in mine)


def evaluate(net, stages, window_variants, window_offset, eval_seeds) -> dict:
    """단계마다 **따로** 잰다 — 판 이름이 단계끼리 같아 한 번에 넘기면 세계 캐시가 죽는다

    (`eval_unseen.evaluate_stage` 와 같은 이유). 액터가 없는 단계는 `load_window` 가 원본 한
    벌로 접는다.
    """
    win, out = window_label(window_variants, window_offset), {}
    for name in stages:
        boards, varied = load_window(stage_path(name), window_variants, window_offset)
        ev = evaluate_policy(net, boards, seeds=tuple(range(eval_seeds)))
        out[name] = {"goal_rate": ev["goal_rate"], "mean_score": ev["mean_score"],
                     "mean_score_completed": ev["mean_score_completed"],
                     "mean_reward": ev["mean_reward"],
                     "total_steps": sum(int(e.steps) for e in ev["episodes"]),
                     "episodes": len(ev["episodes"]), "boards": len(boards),
                     "window": win if varied else ORIGINAL}
        print(f"[{name}] 완주율 {ev['goal_rate'] * 100:.1f}% · 판 {len(ev['episodes'])}",
              file=sys.stderr, flush=True)
    return out


def build_row(a, samples, max_updates, epochs, train, compare, ev, seconds, dev) -> dict:
    return {"run": a.run, "upto": a.upto, "init": a.init, "init_seed": a.init_seed,
            "anchor_coef": a.anchor_coef, "batch_seed": a.batch_seed,
            "epochs": epochs, "max_updates": max_updates, "updates_like": a.updates_like,
            "samples": samples, "train": train, "compare": compare,
            "window": window_label(a.window_variants, a.window_offset),
            "eval_seeds": a.eval_seeds, "eval": ev, "seconds": seconds,
            "device": str(dev), "host": platform.node(), "rule_stack": rs.commit()[:7]}


def parse(argv):
    ap = argparse.ArgumentParser(description="라운드 K 까지의 데이터로 그물 하나를 다시 학습한다")
    ap.add_argument("--run", required=True, help="DAgger 실행 폴더(조각은 <run>/data)")
    ap.add_argument("--upto", type=int, required=True, help="라운드 0~K 조각을 쓴다")
    ap.add_argument("--init", required=True, help="출발 체크포인트(M4s 와 같은 --init)")
    ap.add_argument("--anchor-coef", type=float, required=True)
    ap.add_argument("--batch-seed", type=int, required=True,
                    help="TrainConfig.seed 그대로(run_dagger 는 train_seed+rnd 였다)")
    ap.add_argument("--out", required=True, help="policy.pt·row.json 을 쓸 폴더")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--max-updates", type=int, default=None)
    g.add_argument("--updates-like", type=int, default=None,
                   help="라운드 0~K2 데이터를 --epochs 에폭 돌 때의 갱신 횟수를 예산으로")
    ap.add_argument("--init-seed", type=int, default=0, help="run_dagger 의 --seed(기본 0)")
    ap.add_argument("--eval-stage", action="append", default=[])
    ap.add_argument("--collect-variants", type=int, default=4)
    ap.add_argument("--window-variants", type=int, default=4)
    ap.add_argument("--window-offset", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--compare", default=None, help="가중치가 같은지 볼 체크포인트")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.eval_stage = a.eval_stage or list(DEFAULT_STAGES)
    try:
        check_window(a.collect_variants, a.window_variants, a.window_offset)
        for s in a.eval_stage:
            stage_path(s)
    except ValueError as exc:
        ap.error(str(exc))
    # 오타는 데이터를 올리기 전에, --dry-run 에서도 잡는다
    if not os.path.exists(a.init):
        ap.error(f"--init 체크포인트가 없다: {a.init}")
    if a.compare and not os.path.exists(a.compare):
        ap.error(f"--compare 체크포인트가 없다: {a.compare}")
    if not a.dry_run and os.path.exists(os.path.join(a.out, "row.json")):
        ap.error(f"`{a.out}/row.json` 이 이미 있다 — 다른 실행과 섞인다. 지우거나 다른 --out")
    return ap, a


def main(argv=None) -> int:
    ap, a = parse(argv)
    data_dir = os.path.join(a.run, "data")
    bs = TrainConfig().batch_size
    try:
        max_updates = a.max_updates
        if a.updates_like is not None:      # 길이만 센다 — 데이터셋을 둘 올리지 않는다
            max_updates = budget(count_upto(data_dir, a.updates_like), a.epochs, bs)
        dataset = load_upto(data_dir, a.upto)
    except ValueError as exc:
        ap.error(str(exc))
    epochs = effective_epochs(a.epochs, max_updates, len(dataset), bs)
    win = window_label(a.window_variants, a.window_offset)
    if a.dry_run:
        print(f"데이터: 라운드 0~{a.upto} · 조각 {len(shard_paths(data_dir, a.upto))}개"
              f" · 표본 {len(dataset)} · 에폭당 {batches_per_epoch(len(dataset), bs)} 배치")
        print(f"예산: {'없음(에폭대로)' if max_updates is None else max_updates} 걸음"
              f"{'' if a.updates_like is None else f' (라운드 0~{a.updates_like} 를 {a.epochs} 에폭)'}"
              f" · 실효 에폭 {epochs} · 배치 시드 {a.batch_seed} · 앵커 {a.anchor_coef}")
        print(f"평가: {'안 함' if a.no_eval else f'{win} · 단계 {a.eval_stage} · 평가 시드 {a.eval_seeds}'}")
        print(f"비교: {a.compare or '없음'} · 출력: {a.out}")
        return 0

    dev = pick_device(a.device)
    t0 = time.perf_counter()
    net, train = refit(a.init, dataset, a.upto, a.batch_seed, epochs, max_updates,
                       a.anchor_coef, a.init_seed, dev)
    samples = len(dataset)
    del dataset
    gc.collect()                 # 평가 전에 데이터를 놓는다 — 병렬 갈래가 메모리를 나눠 쓴다
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # GPU 쪽도 같이 — 학습 때 잡은 캐시를 평가 전에 돌려준다
    if max_updates is not None and train["updates"] != max_updates:
        print(f"★ 예산 {max_updates} 걸음을 못 채웠다(실제 {train['updates']})", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    net.save(os.path.join(a.out, "policy.pt"))
    compare = None
    if a.compare:
        compare = {"ckpt": a.compare, "identical": same_weights(net, a.compare)}
        print(f"비교 {a.compare}: {'같다' if compare['identical'] else '★ 다르다'}",
              file=sys.stderr, flush=True)
    ev = None if a.no_eval else evaluate(net, a.eval_stage, a.window_variants,
                                         a.window_offset, a.eval_seeds)
    row = build_row(a, samples, max_updates, epochs, train, compare, ev,
                    time.perf_counter() - t0, dev)
    # 병렬 실행기는 row.json 이 있으면 "끝났다" 로 본다 — 쓰다 죽어도 반쪽짜리가 남지 않게
    # 같은 폴더의 임시 파일에 쓴 뒤 한 번에 바꿔 놓는다
    row_path = os.path.join(a.out, "row.json")
    with open(row_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)
    os.replace(row_path + ".tmp", row_path)
    print(json.dumps({k: row[k] for k in ("upto", "batch_seed", "epochs", "max_updates")}
                     | {"updates": train["updates"], "anchor": train["anchor"],
                        "identical": None if compare is None else compare["identical"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
