"""Degree statistics by propagation depth for the Experiment-2 campaign.

Quantifies the friendship-paradox geometry of random-seed cascades: uniformly
drawn seeds are rarely hubs (heavy-tailed degree distribution), but the
depth-1 removals -- customers reached along supply edges -- are dispropor-
tionately high-degree firms whose failure then exposes many further customers.

Streams the Experiment-2 per-run JSONLs and reports, per depth bucket
(seed / 1 / 2 / >=3), the distribution of the recorded firm `size` (the
model's degree proxy) and the share of firms in the top-10% degree class
(size >= 17, cf. the top-10% range in Table A1).

Output: outputs/figures/cascade_depth_degree_stats.json
"""

import json
from collections import defaultdict

import numpy as np
from graphsim import paths

RUNS_DIR = paths.RUNS_DIR / "exp2"

OUTPUT_PATH = paths.FIGURES_DIR / "cascade_depth_degree_stats.json"

TOP_DECILE_MIN_DEGREE = 17


def run_depths(rec: dict) -> dict[int, int]:
    seeds = {s["id"] for s in rec.get("seeds", [])}
    removed = {r["id"]: r for r in rec["removed"]}
    depth: dict[int, int] = {s: 0 for s in seeds}

    def get_depth(fid: int, guard: set) -> int:
        if fid in depth:
            return depth[fid]
        if fid in guard or fid not in removed:
            return 0
        guard.add(fid)
        origins = removed[fid].get("disruption_origin") or []
        known = [get_depth(o, guard) for o in origins if o in removed or o in depth]
        d = (min(known) + 1) if known else 1
        depth[fid] = d
        return d

    for fid in removed:
        if fid not in seeds:
            get_depth(fid, set())
    return depth


def main() -> None:
    sizes: dict[str, list[int]] = defaultdict(list)

    files = sorted(RUNS_DIR.glob("*.jsonl"))
    print(f"Streaming {len(files)} jsonl files ...")
    for n, fp in enumerate(files, 1):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                seeds = {s["id"] for s in rec.get("seeds", [])}
                removed = {r["id"]: r for r in rec["removed"]}
                depth = run_depths(rec)
                for fid, r in removed.items():
                    s = r.get("size")
                    if s is None:
                        continue
                    d = 0 if fid in seeds else depth.get(fid, 1)
                    bucket = "seed" if d == 0 else ("1" if d == 1 else
                                                    ("2" if d == 2 else "3+"))
                    sizes[bucket].append(s)
        if n % 40 == 0:
            print(f"  {n}/{len(files)}")

    out = {}
    for bucket in ["seed", "1", "2", "3+"]:
        arr = np.array(sizes[bucket])
        out[bucket] = {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p90": float(np.percentile(arr, 90)),
            "share_top_decile": float((arr >= TOP_DECILE_MIN_DEGREE).mean()),
        }
        print(f"{bucket:>4}: n={arr.size}, median={np.median(arr):.0f}, "
              f"mean={arr.mean():.1f}, p90={np.percentile(arr, 90):.0f}, "
              f"top-decile share={(arr >= TOP_DECILE_MIN_DEGREE).mean():.1%}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
