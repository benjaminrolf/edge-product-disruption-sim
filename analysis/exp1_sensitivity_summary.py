"""Summarise the Experiment-1 sensitivity runs.

Imputation sensitivity: both arms re-run on scenarios 0 (top-10% degree seeds) and 4
(degree-1 seeds) with edge products drawn with probability proportional to pool^gamma.
Tau sweep: the extended arm re-run on the same scenarios with the substitution map
of each tau; tau = 0.73 is the main run. All runs keep the task ids of the main runs,
so every comparison is paired. The question is whether the main result, extended >
hybrid, holds: per setting, the mean cascade of both arms, the paired difference with
a 95% bootstrap interval, and the ratio of the means.

Usage:
  uv run python analysis/exp1_sensitivity_summary.py --results results
"""

import argparse
import json
from pathlib import Path

import numpy as np

from exp1_rev_summary import bootstrap_ci, load_cascades
from graphsim import paths

SCENARIOS = {0: "top-10% degree", 4: "degree 1"}
GAMMAS = ("-2", "-1", "+1")
TAUS = ("0.65", "0.69", "0.77", "0.81", "0.85")


def compare(hybrid: dict, extended: dict, rng: np.random.Generator) -> dict:
    out = {}
    for scn, label in SCENARIOS.items():
        ids = sorted(t for t in extended if t in hybrid and extended[t][0] == scn)
        h = np.array([hybrid[t][1] for t in ids], dtype=float)
        e = np.array([extended[t][1] for t in ids], dtype=float)
        lo, hi = bootstrap_ci(e - h, rng)
        out[label] = {"n": len(ids), "hybrid": h.mean(), "extended": e.mean(),
                      "diff": (e - h).mean(), "diff_ci": [lo, hi],
                      "ratio": e.mean() / h.mean() if h.mean() > 0 else None,
                      "share_cascade_hybrid": float((h > 0).mean()),
                      "share_cascade_extended": float((e > 0).mean())}
    return out


def show(name: str, res: dict) -> None:
    for label, r in res.items():
        ratio = f"{r['ratio']:.1f}x" if r["ratio"] else "  -  "
        print(f"{name:12s} {label:15s} n={r['n']:4d}  hybrid {r['hybrid']:6.2f}  "
              f"extended {r['extended']:6.2f}  diff {r['diff']:6.2f} "
              f"[{r['diff_ci'][0]:5.2f}, {r['diff_ci'][1]:5.2f}]  ratio {ratio}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    results = {"gamma": {}, "tau": {}}

    main_h = load_cascades(args.results / "exp1_hybrid")
    main_e = load_cascades(args.results / "exp1_extended")
    results["gamma"]["0"] = compare(main_h, main_e, rng)
    show("gamma 0", results["gamma"]["0"])
    for g in GAMMAS:
        res = compare(load_cascades(args.results / f"exp1_hybrid_gamma{g}"),
                      load_cascades(args.results / f"exp1_extended_gamma{g}"), rng)
        results["gamma"][g] = res
        show(f"gamma {g}", res)

    # The hybrid arm does not use Sigma, so every tau is compared with its main run
    results["tau"]["0.73"] = results["gamma"]["0"]
    for t in TAUS:
        res = compare(main_h, load_cascades(args.results / f"exp1_extended_tau{t}"), rng)
        results["tau"][t] = res
    for t in sorted(results["tau"]):
        show(f"tau {t}", results["tau"][t])

    out = paths.ANALYSIS_DIR / "exp1_sensitivity_summary.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
