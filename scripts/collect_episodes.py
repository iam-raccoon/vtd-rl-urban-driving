"""판×시드마다 조각 하나를 모은다(여러 프로세스).

    env -u PYTHONPATH .venv/bin/python scripts/collect_episodes.py \
        --curriculum curricula/stage1.json --out runs/lab-main/dagger/data --round 0 \
        --seeds 2 --beta 1.0 --workers 12

M5b — 변종 판(`course_A@v0~v3`)으로 더 모으되 이미 있는 조각과 안 겹치게 시드를 민다:

    env -u PYTHONPATH .venv/bin/python scripts/collect_episodes.py \
        --curriculum curricula/stage1.json --out runs/lab-main/dagger/data --round 0 \
        --seeds 2 --beta 1.0 --variants 4 --seed-offset 2 --workers 12
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.policy.collect import collect_episode  # noqa: E402
from vtd_rl.policy.dataset import save_shard  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402


def build_jobs(curriculum, rnd, seeds, beta, policy, out, variants=1, seed_offset=0):
    """판×시드 작업 묶음. 변종 수가 묶음에 실려야 일꾼이 `course_A@v1` 을 다시 찾는다(M5b).

    시드는 `1000×라운드 + seed_offset + s` — `run_dagger.py` 와 같은 꼴이고, `seed_offset` 으로
    이미 있는 실행의 조각(예: s1000·1001)과 안 겹치게 민다.
    """
    _name, boards = load_curriculum(curriculum, variants=variants)
    return [(curriculum, b.name, 1000 * rnd + seed_offset + s, beta, policy, out, rnd, variants)
            for b in boards for s in range(seeds)]


def shard_path(job) -> str:
    """작업 하나가 쓰는 조각 파일. 모으기 전에(겹침 검사)와 모은 뒤(저장) 같은 이름을 쓴다."""
    _cur, name, seed, _beta, _policy, out, rnd, _variants = job
    return os.path.join(out, f"r{rnd}-{name}-s{seed}.npz")


def _one(job):
    curriculum, name, seed, beta, policy_path, out, rnd, variants = job
    _n, boards = load_curriculum(curriculum, variants=variants)
    board = next(b for b in boards if b.name == name)
    policy = DrivePolicy.load(policy_path) if policy_path else None
    shard = collect_episode(board, policy=policy, beta=beta, seed=seed)
    path = shard_path(job)
    save_shard(shard, path)
    return {"path": path, **shard.meta}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--variants", type=int, default=1,
                    help="변종 판 v0~v{V-1} 을 모은다(1 = 원본 판만)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="시드를 1000×라운드 + 이 값 + s 로 민다(이미 있는 조각과 안 겹치게)")
    a = ap.parse_args(argv)
    if a.variants < 1:
        ap.error(f"--variants 는 1 이상이어야 한다 — {a.variants}")
    if a.seed_offset < 0:
        ap.error(f"--seed-offset 은 0 이상이어야 한다 — {a.seed_offset}")

    jobs = build_jobs(a.curriculum, a.round, a.seeds, a.beta, a.policy, a.out,
                      variants=a.variants, seed_offset=a.seed_offset)
    clash = [shard_path(j) for j in jobs if os.path.exists(shard_path(j))]
    if clash:
        ap.error(f"이미 있는 조각을 덮게 된다({len(clash)}개, 예: {clash[0]}) — "
                 "다른 --seed-offset 이나 --out")
    os.makedirs(a.out, exist_ok=True)
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            metas = pool.map(_one, jobs)
    else:
        metas = [_one(j) for j in jobs]
    print(json.dumps({"round": a.round, "episodes": len(metas),
                      "samples": sum(m["steps"] for m in metas),
                      "goal": sum(1 for m in metas if m["outcome"] == "goal"),
                      "shards": metas}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
