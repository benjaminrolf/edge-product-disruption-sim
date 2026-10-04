"""Numbers of the Tohoku-inspired scenario (Experiment 3).

A firm counts as disrupted in a run if it is removed in that run,
as a seed or through the cascade (definition of Section 5.3). Reports
  per scenario   runs, mean seeds, mean cascade (removals beyond the seeds)
  union          firms disrupted in at least one run, inside and outside the seed region
                 (geocoded firms in prefectures with JMA intensity 5- or higher), their
                 countries and, outside the region, their Louvain community
  typical run    disrupted firms per run (median, mean), in total and outside the region,
                 so that the union is not read as the impact of a single event
  table firms    disruption probability per scenario, seed share and mean removal week
                 (seeds count as week 0) of the OEMs and Tier-1 suppliers of
                 Table 5, and their direct suppliers inside the region
  propagation    share of cascade removals that follow the loss of two or more
                 suppliers, and the median and 90th percentile cascade removal week,
                 also for Experiment 2 (--exp2) for comparison
Outputs: <out-dir>/geje_rev_summary.json and <out-dir>/geje_per_firm_disruption_probs.csv
(input of the map, Fig. 6).

Usage:
  uv run python analysis/geje_rev_summary.py --runs DIR [--exp2 DIR] --out-dir DIR
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from graphsim.networks.marklines import load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

DATA_DIR = paths.DATA_DIR
GEJE_DIR = paths.GEO_DIR
PARTITION = paths.ANALYSIS_DIR / "louvain_partition_seed42.json"
REGION_INTENSITIES = {"JMA 7", "JMA 6+", "JMA 6-", "JMA 5+", "JMA 5-"}
SCENARIOS = {0: "low", 1: "mid", 2: "high"}
COMMUNITIES = {0: "China-led", 1: "Japan-led", 2: "transatlantic", 3: "India-led",
               4: "Korea-led"}
TABLE_FIRMS = {  # firm ids as identified in geje_table_extension.json
    "Subaru": 2000870, "Mazda": 2001051, "Honda": 2000654, "Nissan": 2000132,
    "Toyota": 2000567, "Panasonic": 55435, "Denso": 93133, "Hitachi Astemo": 54705,
    "Aisin": 93125, "Marelli": 93139,
}


def removal_week(r: dict) -> float:
    if r.get("disruption_time") is None and r["removal_time"] == 1:
        return 0.0
    return r["removal_time"] / 2.0


def propagation_stats(runs_dir: Path) -> dict:
    multi = total = 0
    weeks = []
    for path in sorted(runs_dir.glob("stats_*.jsonl")):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                seeds = {s["id"] for s in rec["seeds"]}
                for r in rec["removed"]:
                    if r["id"] in seeds:
                        continue
                    total += 1
                    multi += len(set(r.get("disruption_origin") or ())) >= 2
                    weeks.append(r["removal_time"] / 2.0)
    w = np.asarray(weeks)
    return {"cascade_removals": total, "share_two_or_more_lost_suppliers": multi / total,
            "median_removal_week": float(np.median(w)),
            "p90_removal_week": float(np.percentile(w, 90))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--exp2", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    intensity = json.loads((GEJE_DIR / "tohoku_intensity.json").read_text())
    region_prefs = {p for p, i in intensity.items() if i in REGION_INTENSITIES}
    geo = json.loads((GEJE_DIR / "japanese_firms_geolocation.json").read_text())
    region = {fid for pref, fids in geo.items() if pref in region_prefs for fid in fids}

    n_runs: dict[int, int] = defaultdict(int)
    removed: dict[int, Counter] = defaultdict(Counter)
    seeded: dict[int, Counter] = defaultdict(Counter)
    seeds_per_run: dict[int, list] = defaultdict(list)
    cascade_per_run: dict[int, list] = defaultdict(list)
    per_run_total, per_run_outside = [], []
    weeks: dict[int, list] = defaultdict(list)
    table_ids = set(TABLE_FIRMS.values())
    for path in sorted(args.runs.glob("stats_*.jsonl")):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                scn = rec["scenario_id"]
                seeds = {s["id"] for s in rec["seeds"]}
                ids = {r["id"] for r in rec["removed"]}
                n_runs[scn] += 1
                removed[scn].update(ids)
                seeded[scn].update(seeds)
                seeds_per_run[scn].append(len(seeds))
                cascade_per_run[scn].append(len(ids - seeds))
                per_run_total.append(len(ids))
                per_run_outside.append(len(ids - region))
                for r in rec["removed"]:
                    if r["id"] in table_ids:
                        weeks[r["id"]].append(removal_week(r))
    total_runs = sum(n_runs.values())

    union = set().union(*(set(c) for c in removed.values()))
    outside = union - region
    firms = pl.read_csv(DATA_DIR / "Firms.csv", infer_schema_length=10000,
                        columns=["firm_id", "nation_id", "is_oem"])
    nations = pl.read_csv(DATA_DIR / "GeoNations.csv", columns=["nation_id", "nation_name"])
    firm_nation = dict(firms.join(nations, on="nation_id", how="left")
                       .select(["firm_id", "nation_name"]).iter_rows())
    countries = Counter(firm_nation.get(f) for f in union)
    missing_country = countries.pop(None, 0)
    membership = {int(k): v for k, v in
                  json.loads(PARTITION.read_text())["membership"].items()}
    blocs = Counter(COMMUNITIES.get(membership[f], "smaller community")
                    if f in membership else "not in LCC" for f in outside)

    net, _, _ = load_marklines(DATA_DIR, node_cls=ImprovedZhaoNode, search_cls=ImprovedSearch)
    table = {}
    for name, fid in TABLE_FIRMS.items():
        preds = set(net.predecessors(fid))
        table[name] = {
            "firm_id": fid,
            **{f"disruption_prob_{lab}": removed[s][fid] / n_runs[s]
               for s, lab in SCENARIOS.items()},
            "seed_share": sum(seeded[s][fid] for s in SCENARIOS) / total_runs,
            "mean_removal_week": float(np.mean(weeks[fid])) if weeks[fid] else None,
            "suppliers": len(preds), "suppliers_in_region": len(preds & region),
        }

    results = {
        "runs": dict(n_runs),
        "scenarios": {lab: {"runs": n_runs[s],
                            "mean_seeds": float(np.mean(seeds_per_run[s])),
                            "mean_cascade": float(np.mean(cascade_per_run[s]))}
                      for s, lab in SCENARIOS.items()},
        "union": {"disrupted": len(union), "inside_region": len(union & region),
                  "outside_region": len(outside),
                  "share_outside": len(outside) / len(union),
                  "countries": len(countries), "top_countries": countries.most_common(6),
                  "missing_country": missing_country,
                  "outside_by_community": dict(blocs.most_common())},
        "typical_run": {"median_disrupted": float(np.median(per_run_total)),
                        "mean_disrupted": float(np.mean(per_run_total)),
                        "median_outside_region": float(np.median(per_run_outside)),
                        "mean_outside_region": float(np.mean(per_run_outside))},
        "table_firms": table,
        "propagation": {"exp3": propagation_stats(args.runs)},
    }
    if args.exp2:
        results["propagation"]["exp2"] = propagation_stats(args.exp2)

    records = []
    for fid in sorted(union):
        rec = {"firm_id": fid}
        for s, lab in SCENARIOS.items():
            rec[f"disruption_prob_{lab}"] = removed[s][fid] / n_runs[s]
            rec[f"seed_share_{lab}"] = seeded[s][fid] / n_runs[s]
        records.append(rec)
    (pl.DataFrame(records)
     .join(firms, on="firm_id", how="left").join(nations, on="nation_id", how="left")
     .sort("disruption_prob_high", descending=True)
     .write_csv(args.out_dir / "geje_per_firm_disruption_probs.csv"))
    (args.out_dir / "geje_rev_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
