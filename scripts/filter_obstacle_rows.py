"""DAgger 조각에서 '장애물 앞 장면' 행만 골라 새 폴더에 쓴다(M7k).

장애물 앞 장면 = 통로(물체 반폭 + 내 차 반폭 + margin) 안, 앞범퍼에서 range 안에 **멈춘** 물체(속도 < 0.5 m/s,
높이 ≥ 0.25 m, 사람류가 아님)가 하나라도 있는 행이다. 기준은 환경의 머무는 벌(`VtdDriveEnv._block_gap`)과 같고,
물체 특징은 관측(`vtd_rl/env/observation.py`)이 정규화한 값을 되돌려 쓴다.

사용(레포 루트): python scripts/filter_obstacle_rows.py --src runs/.../data --dst runs/.../data_obs [--range 40]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.env.drive_env import EGO_HALF_W, ROAD_SURFACE_H, BLOCK_STILL_V  # noqa: E402
from vtd_rl.env.observation import ObsConfig  # noqa: E402
from vtd_rl.policy.dataset import Shard, load_shard, save_shard  # noqa: E402

_CFG = ObsConfig()


def obstacle_rows(objs: np.ndarray, mask: np.ndarray, range_m: float = 40.0, margin: float = 0.3) -> np.ndarray:
    """행마다 장애물 앞 장면인가 — `objs`[행, 물체, 12], `mask`[행, 물체] → bool[행]."""
    fx = objs[..., 0] * _CFG.obj_x
    fy = objs[..., 1] * _CFG.obj_y
    speed = objs[..., 4] * _CFG.v_max
    length = objs[..., 5] * _CFG.obj_size
    width = objs[..., 6] * _CFG.obj_size
    height = objs[..., 7] * _CFG.obj_size
    person = objs[..., 10] > 0.5                          # 관측 `_object_class` 의 사람류 칸
    gap = np.maximum(0.0, fx - rs.score_fma.FRONT - length / 2.0)
    hit = ((mask > 0.5) & (fx > 0.0) & (height >= ROAD_SURFACE_H) & (np.abs(speed) < BLOCK_STILL_V)
           & ~person & (np.abs(fy) < width / 2.0 + EGO_HALF_W + margin) & (gap < range_m))
    return hit.any(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="DAgger 조각 폴더")
    ap.add_argument("--dst", required=True, help="고른 행을 쓸 폴더")
    ap.add_argument("--range", type=float, default=40.0, help="앞범퍼에서 이 거리[m] 안의 장애물")
    ap.add_argument("--margin", type=float, default=0.3, help="통로 여유[m]")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    total = kept = 0
    for path in sorted(glob.glob(os.path.join(a.src, "*.npz"))):
        sh = load_shard(path)
        keep = obstacle_rows(sh.objs, sh.mask, a.range, a.margin)
        total += len(keep)
        kept += int(keep.sum())
        if keep.any():
            meta = dict(sh.meta, filtered="obstacle_rows", filter_range=a.range, filter_margin=a.margin,
                        rows_before=int(len(keep)))
            save_shard(Shard(sh.vec[keep], sh.objs[keep], sh.mask[keep], sh.control[keep], sh.turn[keep], meta),
                       os.path.join(a.dst, os.path.basename(path)))
    print(f"행 {total} 중 {kept} 개 고름({100.0 * kept / max(total, 1):.1f}%) → {a.dst}")


if __name__ == "__main__":
    main()
