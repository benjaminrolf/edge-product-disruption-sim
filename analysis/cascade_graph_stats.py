"""Campaign-level statistics backing the disruption-graph figures.

Quantifies, for the Experiment-2 and Tohoku (Experiment-3)
campaigns, how interconnected the disruption graphs are and how fast cascade
firms fall:
  - share of cascade removals with >= 2 DISTINCT origin suppliers
    (customers brought down by several failed suppliers),
  - median / p90 removal week of cascade removals.
Additionally reports, for the three example runs of the figures, the Louvain
community concentration of the removed firms (canonical partition, seed 42).

Output: outputs/figures/cascade_graph_campaign_stats.json
"""

import json
from collections import Counter

import numpy as np
from graphsim import paths

CAMPAIGNS = {
    "experiment2": paths.RUNS_DIR / "exp2",
    "tohoku": paths.RUNS_DIR / "exp3",
}

FIGDIR = paths.FIGURES_DIR
LOUVAIN_PATH = paths.ANALYSIS_DIR / "louvain_partition_seed42.json"


def run_metrics(rec: dict) -> tuple[int, int, list[float]]:
    """Return (cascade size, multi-distinct-origin removals, removal weeks)."""
    seeds = {s["id"] for s in rec.get("seeds", [])}
    removed = {r["id"]: r for r in rec["removed"]}
    n_multi = 0
    weeks: list[float] = []
    n_cascade = 0
    for fid, r in removed.items():
        if fid in seeds:
            continue
        n_cascade += 1
        origins = {o for o in (r.get("disruption_origin") or [])
                   if o in removed}
        if len(origins) >= 2:
            n_multi += 1
        weeks.append((r.get("removal_time") or 0) / 2.0)
    return n_cascade, n_multi, weeks


def main() -> None:
    out: dict = {}
    for name, path in CAMPAIGNS.items():
        total = multi = 0
        weeks: list[float] = []
        files = sorted(path.glob("*.jsonl"))
        print(f"[{name}] streaming {len(files)} files ...")
        for n, fp in enumerate(files, 1):
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    c, m, w = run_metrics(json.loads(line))
                    total += c
                    multi += m
                    weeks.extend(w)
            if n % 40 == 0:
                print(f"  {n}/{len(files)}")
        arr = np.array(weeks)
        out[name] = {
            "cascade_removals": total,
            "multi_origin_removals": multi,
            "multi_origin_share": multi / total,
            "removal_week_median": float(np.median(arr)),
            "removal_week_mean": float(arr.mean()),
            "removal_week_p90": float(np.percentile(arr, 90)),
        }
        print(f"[{name}] cascade removals: {total}, multi-origin: {multi} "
              f"({multi / total:.2%}), removal week median "
              f"{np.median(arr):.1f} / mean {arr.mean():.1f} / "
              f"p90 {np.percentile(arr, 90):.1f}")

    # Community concentration of the example runs (canonical Louvain partition)
    with open(LOUVAIN_PATH) as f:
        louvain = json.load(f)
    membership = {int(k): v for k, v in louvain["membership"].items()}
    with open(FIGDIR / "cascade_example_runs.json") as f:
        runs = json.load(f)
    out["examples"] = {}
    for key, rec in runs.items():
        if not isinstance(rec, dict) or "removed" not in rec:
            continue
        seeds = {s["id"] for s in rec.get("seeds", [])}
        cascade = [r["id"] for r in rec["removed"] if r["id"] not in seeds]
        comms = Counter(membership[f] for f in cascade if f in membership)
        n_mapped = sum(comms.values())
        top_comm, top_n = comms.most_common(1)[0]
        c, m, w = run_metrics(rec)
        out["examples"][key] = {
            "cascade": c,
            "multi_origin": m,
            "multi_origin_share": m / c,
            "removal_week_median": float(np.median(w)),
            "n_in_lcc_partition": n_mapped,
            "top_community": top_comm,
            "top_community_share": top_n / n_mapped,
        }
        print(f"[{key}] cascade={c}, multi={m} ({m / c:.1%}), "
              f"median removal week {np.median(w):.1f}, "
              f"top community {top_comm}: {top_n}/{n_mapped} "
              f"({top_n / n_mapped:.1%})")

    with open(FIGDIR / "cascade_graph_campaign_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: cascade_graph_campaign_stats.json")


if __name__ == "__main__":
    main()
