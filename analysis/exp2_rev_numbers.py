"""Experiment-2 numbers of Section 5.2 that the other scripts do not report.

  robustness     per |S|: median and mean cascade, share of runs with a cascade, the
                 ten sample means, and their spread within and across seed samples
                 (standard error over the imputations vs spread of the sample means)
  horizon        firms still Disrupted at the horizon (|S| = 100)
  volume         capacity C(t) lost per removed firm in week windows (|S| = 100)
  progress       share of the final cascade removals and of the final capacity loss
                 reached at the run's week of fastest decline and at its t90
  gamma          |S| = 100 cascades with gamma = -1 against the matched main runs

Usage:
  uv run python analysis/exp2_rev_numbers.py --runs DIR [--gamma DIR] --out FILE
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WINDOWS = ((2, 6), (6, 10), (10, 16), (16, 25), (25, 53))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--gamma", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    casc: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    horizon_disrupted, at_fast, at_t90 = [], [], []
    vol = np.zeros(len(WINDOWS))
    rem = np.zeros(len(WINDOWS))
    main_s100: dict[int, int] = {}
    for path in sorted(args.runs.glob("stats_*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                ns = r["num_seeds"]
                cascade = r["num_removed"][-1] - ns
                casc[ns][r["sample_id"]].append(cascade)
                if ns != 100:
                    continue
                main_s100[r["task_id"]] = cascade
                horizon_disrupted.append(r["num_disrupted"][-1])
                w = -np.diff(np.asarray(r["edge_cap"], dtype=float))
                n = np.diff(np.asarray(r["num_removed"], dtype=float))
                for i, (a, b) in enumerate(WINDOWS):
                    vol[i] += w[a - 1:b - 1].sum()
                    rem[i] += n[a - 1:b - 1].sum()
                cap = np.asarray(r["edge_cap"], dtype=float)
                nr = np.asarray(r["num_removed"], dtype=float)
                if cascade > 0 and cap[0] > cap[-1]:
                    loss = (cap[0] - cap) / (cap[0] - cap[-1])
                    done = (nr - ns) / cascade
                    t_fast = int(np.argmin(np.diff(cap))) + 1
                    t90 = int(np.argmax(nr >= ns + 0.9 * cascade))
                    at_fast.append((done[t_fast], loss[t_fast]))
                    at_t90.append(loss[t90])

    robustness = {}
    for ns in sorted(casc):
        samples = casc[ns]
        allv = np.concatenate([np.asarray(v, dtype=float) for v in samples.values()])
        means = np.array([np.mean(v) for v in samples.values()])
        se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) for v in samples.values()])
        robustness[str(ns)] = {
            "median": float(np.median(allv)), "mean": float(allv.mean()),
            "share_cascade": float((allv > 0).mean()),
            "sample_means": sorted(float(m) for m in means),
            "mean_se_within_sample": float(se.mean()),
            "sd_of_sample_means": float(means.std(ddof=1)),
        }
    fast = np.asarray(at_fast)
    results = {
        "robustness": robustness,
        "horizon_disrupted_s100": {"mean": float(np.mean(horizon_disrupted)),
                                   "max": int(np.max(horizon_disrupted)),
                                   "share_runs": float(np.mean(np.asarray(horizon_disrupted) > 0))},
        "volume_per_removal_s100": {f"weeks {a}-{b - 1}": float(vol[i] / rem[i])
                                    for i, (a, b) in enumerate(WINDOWS)},
        "progress_s100": {
            "at_fast_removals_median": float(np.median(fast[:, 0])),
            "at_fast_loss_median": float(np.median(fast[:, 1])),
            "at_t90_loss": {"median": float(np.median(at_t90)),
                            "q25": float(np.percentile(at_t90, 25)),
                            "q75": float(np.percentile(at_t90, 75))},
        },
    }
    if args.gamma:
        g = {}
        for path in sorted(args.gamma.glob("stats_*.jsonl")):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    g[r["task_id"]] = r["num_removed"][-1] - r["num_seeds"]
        ids = sorted(set(g) & set(main_s100))
        a = np.array([main_s100[i] for i in ids], dtype=float)
        b = np.array([g[i] for i in ids], dtype=float)
        results["gamma_minus1_s100"] = {"runs": len(ids), "main_mean": a.mean(),
                                        "gamma_mean": b.mean(), "factor": b.mean() / a.mean(),
                                        "main_median": float(np.median(a)),
                                        "gamma_median": float(np.median(b))}
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
