"""Summarise Experiment 1: baseline -> hybrid -> extended, per scenario.

Reads the JSONL stats of the three arms (runs are matched by task_id, i.e. identical
seeds and parameters), and reports per scenario and arm the mean cascade size
(firms removed beyond the seed) with a 95% bootstrap interval and the share of
runs with any cascade. The paired difference hybrid -> extended isolates the
substitution rule.

Usage:
  uv run python analysis/exp1_rev_summary.py \
      --baseline results/exp1_baseline \
      --hybrid   results/exp1_hybrid \
      --extended results/exp1_extended
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from graphsim import paths

SCENARIOS = {  # scenario_id -> (seed targeting, sigma_kappa, removal distribution)
    0: ("top-10% degree", 0.1, "normal"), 1: ("top-10% degree", 0.1, "uniform"),
    2: ("top-10% degree", 0.2, "normal"), 3: ("top-10% degree", 0.2, "uniform"),
    4: ("degree 1", 0.1, "normal"), 5: ("degree 1", 0.1, "uniform"),
    6: ("degree 1", 0.2, "normal"), 7: ("degree 1", 0.2, "uniform"),
}


def load_cascades(directory: str) -> dict[int, tuple[int, int]]:
    """Map task_id -> (scenario_id, cascade size)."""
    out = {}
    for file in sorted(glob.glob(str(Path(directory) / "stats_*.jsonl"))):
        with open(file) as f:
            for line in f:
                r = json.loads(line)
                removed = r["num_removed"]
                final = removed[-1] if isinstance(removed, list) else removed
                out[r["task_id"]] = (r["scenario_id"], final - r["num_seeds"])
    return out


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple:
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--extended", required=True)
    args = parser.parse_args()

    arms = {name: load_cascades(getattr(args, name))
            for name in ("baseline", "hybrid", "extended")}
    common = set.intersection(*(set(a) for a in arms.values()))
    print(f"runs per arm: { {k: len(v) for k, v in arms.items()} }, "
          f"matched: {len(common)}")

    rng = np.random.default_rng(42)
    rows = []
    for sid, (target, sk, dist) in SCENARIOS.items():
        tasks = sorted(t for t in common if arms["baseline"][t][0] == sid)
        row = {"scenario": sid, "targeting": target, "sigma_kappa": sk,
               "removal": dist, "n": len(tasks)}
        for name, arm in arms.items():
            x = np.array([arm[t][1] for t in tasks], dtype=float)
            lo, hi = bootstrap_ci(x, rng)
            row[name] = {"mean": x.mean(), "ci": (lo, hi),
                         "share_cascade": (x > 0).mean(), "max": x.max()}
        diff = np.array([arms["extended"][t][1] - arms["hybrid"][t][1]
                         for t in tasks], dtype=float)
        row["ext_minus_hyb"] = {"mean": diff.mean(), "ci": bootstrap_ci(diff, rng)}
        rows.append(row)

    header = (f"{'sc':>2} {'targeting':<15} {'sk':>3} {'removal':<7} "
              f"{'baseline':>16} {'hybrid':>16} {'extended':>16} {'ext-hyb':>22}")
    print(header)
    for r in rows:
        cells = [f"{r[a]['mean']:6.2f} ({100 * r[a]['share_cascade']:4.1f}%)"
                 for a in ("baseline", "hybrid", "extended")]
        d = r["ext_minus_hyb"]
        print(f"{r['scenario']:>2} {r['targeting']:<15} {r['sigma_kappa']:>3} "
              f"{r['removal']:<7} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16} "
              f"{d['mean']:7.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]")
    print("cells: mean cascade (share of runs with cascade > 0); "
          "ext-hyb: paired mean difference [95% bootstrap CI]")

    out = paths.ANALYSIS_DIR / "exp1_rev_summary.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
