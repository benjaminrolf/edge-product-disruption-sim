"""Extract example cascade runs for the disruption-graph figure.

Streams the Experiment-2 per-run JSONLs and saves two candidate runs:
the largest single-seed cascade (the tier-4 critical-cascade event of the
paper) and the run with the deepest removal chain overall. The saved records
keep the full seeds/removed structure (ids, times, disruption origins) so the
figure script does not need the full run output.

Output: outputs/figures/cascade_example_runs.json
"""

import json
from graphsim import paths

RUNS_DIR = paths.RUNS_DIR / "exp2"

OUTPUT_PATH = paths.FIGURES_DIR / "cascade_example_runs.json"


def run_stats(rec: dict) -> tuple[int, int]:
    """Return (cascade size, max depth) for one run."""
    seeds = {s["id"] for s in rec.get("seeds", [])}
    removed = {r["id"]: r for r in rec["removed"]}
    cascade_ids = [i for i in removed if i not in seeds]
    if not cascade_ids:
        return 0, 0
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

    for fid in cascade_ids:
        get_depth(fid, set())
    return len(cascade_ids), max(depth[i] for i in cascade_ids)


def main() -> None:
    best_single: tuple[int, dict] | None = None   # (cascade, rec) for |S|=1
    best_depth: tuple[int, int, dict] | None = None  # (depth, cascade, rec)

    files = sorted(RUNS_DIR.glob("*.jsonl"))
    print(f"Streaming {len(files)} jsonl files ...")
    for n, fp in enumerate(files, 1):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                csize, dmax = run_stats(rec)
                if rec["num_seeds"] == 1 and (
                    best_single is None or csize > best_single[0]
                ):
                    best_single = (csize, rec)
                if best_depth is None or (dmax, csize) > best_depth[:2]:
                    best_depth = (dmax, csize, rec)
        if n % 40 == 0:
            print(f"  {n}/{len(files)}")

    assert best_single and best_depth
    print(f"largest single-seed cascade: {best_single[0]} removals")
    print(f"deepest chain: depth {best_depth[0]} ({best_depth[1]} removals, "
          f"|S|={best_depth[2]['num_seeds']})")
    sample = best_single[1]["removed"][0]
    print(f"removed-record fields: {sorted(sample.keys())}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {"largest_single_seed": best_single[1], "deepest_chain": best_depth[2]},
            f,
        )
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
