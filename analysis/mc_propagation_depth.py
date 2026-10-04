"""Compute the propagation depth (length) of Experiment-2 cascades.

The propagation depth is the longest directed chain of removed firms
downstream of any seed; it shows whether cascades reach beyond the seeds' direct
customers. This script streams the Experiment-2 per-run JSONL files and reconstructs the
removal chains from the recorded `disruption_origin` links: depth(seed) = 0,
depth(firm) = 1 + min(depth of its origins). Reports, per seed-set size |S|,
the distribution of the per-run maximum depth and the share of cascade firms
at depth >= 2 (i.e. beyond direct customers of the seeds).

Output: outputs/figures/mc_propagation_depth.json
"""

import json
from collections import defaultdict

import numpy as np
from graphsim import paths

INPUT_PATH = paths.RUNS_DIR / "exp2"

OUTPUT_PATH = paths.FIGURES_DIR / "mc_propagation_depth.json"


def run_depths(rec: dict) -> tuple[int, int, int]:
    """Return (max depth, cascade size, firms at depth >= 2) for one run."""
    seeds = {s["id"] for s in rec.get("seeds", [])}
    removed = {r["id"]: r for r in rec["removed"]}
    cascade_ids = [i for i in removed if i not in seeds]
    if not cascade_ids:
        return 0, 0, 0

    depth: dict[int, int] = {s: 0 for s in seeds}

    def get_depth(fid: int, guard: set) -> int:
        if fid in depth:
            return depth[fid]
        if fid in guard or fid not in removed:
            return 0  # cycle guard / origin outside removal set
        guard.add(fid)
        origins = removed[fid].get("disruption_origin") or []
        known = [get_depth(o, guard) for o in origins if o in removed or o in depth]
        d = (min(known) + 1) if known else 1
        depth[fid] = d
        return d

    for fid in sorted(cascade_ids, key=lambda i: removed[i].get("removal_time") or 0):
        get_depth(fid, set())

    cas_depths = [depth[i] for i in cascade_ids]
    return max(cas_depths), len(cascade_ids), sum(1 for d in cas_depths if d >= 2)


def main() -> None:
    max_depths = defaultdict(list)
    deep_share_num = defaultdict(int)
    deep_share_den = defaultdict(int)

    files = sorted(INPUT_PATH.glob("*.jsonl"))
    print(f"Streaming {len(files)} jsonl files ...")
    for n, fp in enumerate(files, 1):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ns = rec["num_seeds"]
                dmax, csize, deep = run_depths(rec)
                max_depths[ns].append(dmax)
                deep_share_num[ns] += deep
                deep_share_den[ns] += csize
        if n % 40 == 0:
            print(f"  {n}/{len(files)}")

    out = {}
    for ns in sorted(max_depths):
        arr = np.array(max_depths[ns])
        den = deep_share_den[ns]
        out[str(ns)] = {
            "n_runs": int(arr.size),
            "max_depth_mean": float(arr.mean()),
            "max_depth_median": float(np.median(arr)),
            "max_depth_p95": float(np.percentile(arr, 95)),
            "max_depth_max": int(arr.max()),
            "share_runs_depth_ge2": float((arr >= 2).mean()),
            "share_cascade_firms_depth_ge2": (deep_share_num[ns] / den) if den else None,
        }
        print(f"|S|={ns}: runs={arr.size}, max-depth mean={arr.mean():.2f} "
              f"median={np.median(arr):.0f} max={arr.max()}, "
              f"runs with depth>=2: {(arr >= 2).mean():.1%}, "
              f"cascade firms at depth>=2: "
              f"{(deep_share_num[ns] / den if den else float('nan')):.1%}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
