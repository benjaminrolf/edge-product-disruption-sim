"""Extract a Tohoku (Experiment-3) example run for the disruption-graph figure.

Streams the Experiment-3 per-run JSONLs and selects the run with the most cascade
removals that have two or more DISTINCT origin suppliers (customers brought
down by several failed suppliers at once), preferring runs whose cascade stays
small enough to draw. Appends the run to cascade_example_runs.json under the
key "tohoku_multi_origin" (existing keys are preserved).
"""

import json
from graphsim import paths

RUNS_DIR = paths.RUNS_DIR / "exp3"

OUTPUT_PATH = paths.FIGURES_DIR / "cascade_example_runs.json"

MAX_CASCADE_DRAWABLE = 260


def multi_origin_stats(rec: dict) -> tuple[int, int]:
    """Return (cascade size, removals with >= 2 distinct valid origins)."""
    seeds = {s["id"] for s in rec.get("seeds", [])}
    removed = {r["id"]: r for r in rec["removed"]}
    cascade = [i for i in removed if i not in seeds]
    n_multi = 0
    for fid in cascade:
        origins = {o for o in (removed[fid].get("disruption_origin") or [])
                   if o in removed or o in seeds}
        if len(origins) >= 2:
            n_multi += 1
    return len(cascade), n_multi


def main() -> None:
    best: tuple[int, dict, str] | None = None       # drawable runs
    best_any: tuple[int, int, str] | None = None    # stats only, any size

    files = sorted(RUNS_DIR.glob("*.jsonl"))
    print(f"Streaming {len(files)} jsonl files ...")
    for n, fp in enumerate(files, 1):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                csize, n_multi = multi_origin_stats(rec)
                if best_any is None or n_multi > best_any[0]:
                    best_any = (n_multi, csize, fp.name)
                if csize <= MAX_CASCADE_DRAWABLE and (
                    best is None or n_multi > best[0]
                ):
                    best = (n_multi, rec, fp.name)
        if n % 20 == 0:
            print(f"  {n}/{len(files)}")

    assert best and best_any
    csize, _ = multi_origin_stats(best[1])
    print(f"selected: {best[0]} multi-origin removals, cascade={csize}, "
          f"seeds={best[1]['num_seeds'] if 'num_seeds' in best[1] else len(best[1]['seeds'])}, "
          f"file={best[2]}")
    print(f"best regardless of size: {best_any[0]} multi-origin removals "
          f"(cascade={best_any[1]}, file={best_any[2]})")

    with open(OUTPUT_PATH) as f:
        out = json.load(f)
    rec = best[1]
    rec["_source_file"] = best[2]
    out["tohoku_multi_origin"] = rec
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
