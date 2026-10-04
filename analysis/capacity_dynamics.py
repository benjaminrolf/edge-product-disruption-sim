"""Capacity dynamics of the Experiment-2 and Experiment-3 runs.

The week with the most negative weekly change dC(t) is only where C(t) falls fastest, so
C(t) itself is analysed as well.
Per run, with C(0) the capacity before the shock:
  t_fast        week of fastest decline, argmin_t dC(t)
  max_weekly    largest weekly loss, -min_t dC(t), absolute and relative to C(0)
  loss_final    capacity lost by the horizon, 1 - C(T)/C(0)
  t50C, t90C    first week by which 50% / 90% of the final loss has occurred
  t_minC        first week in which C(t) reaches its minimum
  recovered     share of the maximum loss that is regained by the horizon
  t90           first week by which 90% of the cascade removals (seeds excluded) occurred
Figures (panels dC(t) and C(t)/C(0)):
  fig6_capacity_dynamics.pdf  Fig. 5, Experiment 2, |S| = 100, per seed sample and overall mean;
                              a third panel compares the progress of the cascade removals
                              with that of the capacity loss (both in % of the final value)
  figD2_geje_temporal.pdf     Experiment 3, mean per severity scenario with 95% CI
                              (not in the paper, which reports the numbers in Table D2)

Input: directories with the runner output (stats_*.jsonl). Output: capacity_dynamics.json and the figures in --out-dir.

Usage:
  uv run python capacity_dynamics.py --exp2 DIR [--exp3 DIR] [--out-dir DIR]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_style as st
from graphsim import paths

st.apply()

FIGDIR = paths.FIGURES_DIR
FIG_SEEDS = 100
SCENARIO_LABELS = {0: "Low Severity", 1: "Mid Severity", 2: "High Severity"}
KEYS = ("t_fast", "max_weekly", "max_weekly_rel", "loss_final", "t50C", "t90C", "t_minC", "recovered", "t90")


def first_week(series: np.ndarray, threshold: float) -> int:
    return int(np.argmax(series >= threshold))


def run_metrics(edge_cap: list, num_removed: list, num_seeds: int) -> dict:
    cap = np.asarray(edge_cap, dtype=float)
    diff = np.diff(cap)
    loss = cap[0] - cap
    final_loss, max_loss = loss[-1], loss.max()
    cascade = num_removed[-1] - num_seeds
    return {
        "t_fast": int(np.argmin(diff)) + 1,
        "max_weekly": float(-diff.min()),
        "max_weekly_rel": float(-diff.min() / cap[0]),
        "loss_final": float(final_loss / cap[0]),
        "t50C": first_week(loss, 0.5 * final_loss) if final_loss > 0 else None,
        "t90C": first_week(loss, 0.9 * final_loss) if final_loss > 0 else None,
        "t_minC": int(np.argmin(cap)),
        "recovered": float((cap[-1] - cap.min()) / max_loss) if max_loss > 0 else 0.0,
        "t90": (first_week(np.asarray(num_removed), num_seeds + 0.9 * cascade)
                if cascade > 0 else None),
        "rises": int((diff > 0).sum()),
    }


def progress(values: np.ndarray, start: float, end: float) -> np.ndarray | None:
    """Share of the way from start to end reached at each week (None without change)."""
    if end == start:
        return None
    return np.clip((values - start) / (end - start), 0.0, 1.0)


def load_runs(directory: Path, keep=lambda r: True) -> list[dict]:
    runs = []
    for path in sorted(directory.glob("stats_*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not keep(r):
                    continue
                runs.append({
                    "num_seeds": r["num_seeds"], "sample_id": r.get("sample_id"),
                    "scenario_id": r.get("scenario_id"),
                    "rel_cap": np.asarray(r["edge_cap"], dtype=float) / r["edge_cap"][0],
                    "diff_cap": np.diff(np.asarray(r["edge_cap"], dtype=float)),
                    "loss_share": progress(np.asarray(r["edge_cap"], dtype=float),
                                           r["edge_cap"][0], r["edge_cap"][-1]),
                    "removal_share": progress(np.asarray(r["num_removed"], dtype=float),
                                              r["num_seeds"], r["num_removed"][-1]),
                    "cascade": r["num_removed"][-1] - r["num_seeds"],
                    **run_metrics(r["edge_cap"], r["num_removed"], r["num_seeds"]),
                })
    if not runs:
        raise SystemExit(f"No runs found in {directory}")
    return runs


def summarise(runs: list[dict]) -> dict:
    out = {"n": len(runs),
           "mean_seeds": float(np.mean([r["num_seeds"] for r in runs])),
           "mean_cascade": float(np.mean([r["cascade"] for r in runs])),
           "share_monotone": float(np.mean([r["rises"] == 0 for r in runs])),
           "share_recovered_1pct": float(np.mean([r["recovered"] > 0.01 for r in runs])),
           "share_t90_after_fast": float(np.mean([r["t90"] is not None
                                                  and r["t90"] > r["t_fast"] for r in runs])),
           "share_t90C_after_t90": float(np.mean([r["t90"] is not None
                                                  and r["t90C"] > r["t90"] for r in runs]))}
    for key in KEYS:
        vals = np.asarray([r[key] for r in runs if r[key] is not None], dtype=float)
        out[key] = {"median": float(np.median(vals)), "mean": float(vals.mean()),
                    "q25": float(np.percentile(vals, 25)),
                    "q75": float(np.percentile(vals, 75))}
    return out


def mark_weeks(axes, weeks: dict, legend_ax) -> None:
    """Vertical reference lines in every panel, labelled in the legend of legend_ax."""
    styles = {"t_fast": (st.VERMILLION, "--"), "t90": (st.INK, ":"),
              "t90C": (st.GREEN, "-.")}
    for key, (week, label) in weeks.items():
        color, ls = styles[key]
        for ax in axes:
            ax.axvline(week, color=color, lw=1.0, ls=ls,
                       label=label if ax is legend_ax else None)
    legend_ax.legend(loc="upper right")


def fig_exp2(runs: list[dict], summary: dict, out: Path) -> None:
    by_sample = defaultdict(list)
    for r in runs:
        by_sample[r["sample_id"]].append(r)
    fig, (ax_d, ax_c, ax_p) = plt.subplots(3, 1, figsize=(6.3, 5.9), sharex=True)
    for key, ax, scale in (("diff_cap", ax_d, 1e5), ("rel_cap", ax_c, 0.01)):
        sample_means = [np.mean([r[key] for r in rs], axis=0) / scale
                        for _, rs in sorted(by_sample.items())]
        offset = 1 if key == "diff_cap" else 0
        weeks = np.arange(offset, len(sample_means[0]) + offset)
        for m in sample_means:
            ax.plot(weeks, m, color=st.BLUE, alpha=0.30, lw=0.9)
        ax.plot(weeks, np.mean(sample_means, axis=0), color=st.BLUE, lw=2.0,
                label="Mean Over Seed Samples" if ax is ax_d else None)
    for key, color, label in (("removal_share", st.ORANGE, "Cascade Removals"),
                              ("loss_share", st.BLUE, "Capacity Loss")):
        sample_means = [np.mean([r[key] for r in rs if r[key] is not None], axis=0)
                        for _, rs in sorted(by_sample.items())]
        m = 100 * np.mean(sample_means, axis=0)
        ax_p.plot(np.arange(len(m)), m, color=color, lw=2.0, label=label)
    ax_p.axhline(90, color="#999999", lw=0.8, ls="--")
    med = {key: summary[key]["median"] for key in ("t_fast", "t90", "t90C")}
    mark_weeks((ax_d, ax_c, ax_p), {
        "t_fast": (med["t_fast"], f"Fastest Decline (Median Week {med['t_fast']:.0f})"),
        "t90": (med["t90"], f"90% of Cascade Removals (Median Week {med['t90']:.0f})"),
        "t90C": (med["t90C"],
                 f"90% of Final Capacity Loss (Median Week {med['t90C']:.0f})"),
    }, legend_ax=ax_c)
    ax_d.set_ylabel("$\\Delta C(t)$ ($10^5$ Units)")
    ax_c.set_ylabel("$C(t)/C(0)$ (%)")
    ax_p.set_ylabel("Share of Final Value (%)")
    ax_p.set_ylim(0, 102)
    ax_p.set_xlabel("Week After Shock")
    ax_p.set_xlim(0, 52)
    ax_d.legend(loc="lower right")
    ax_p.legend(loc="lower right")
    for ax, tag in ((ax_d, "a"), (ax_c, "b"), (ax_p, "c")):
        ax.text(-0.1, 1.03, f"({tag})", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
    fig.align_ylabels((ax_d, ax_c, ax_p))
    fig.savefig(out / "fig6_capacity_dynamics.pdf")
    plt.close(fig)


def fig_exp3(runs: list[dict], out: Path) -> None:
    by_scn = defaultdict(list)
    for r in runs:
        by_scn[r["scenario_id"]].append(r)
    fig, (ax_d, ax_c) = plt.subplots(2, 1, figsize=(6.3, 4.9), sharex=True)
    for key, ax, scale in (("diff_cap", ax_d, 1e5), ("rel_cap", ax_c, 0.01)):
        for scn, color in zip(sorted(by_scn), st.SEQ3, strict=True):
            arrs = np.stack([r[key] for r in by_scn[scn]]) / scale
            m = arrs.mean(axis=0)
            se = arrs.std(axis=0) / np.sqrt(arrs.shape[0])
            weeks = np.arange(1, len(m) + 1) if key == "diff_cap" else np.arange(len(m))
            ax.fill_between(weeks, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.25,
                            lw=0)
            ax.plot(weeks, m, color=color, lw=1.7, label=SCENARIO_LABELS[scn])
    ax_d.set_ylabel("Weekly Capacity Change\n$\\Delta C(t)$ ($10^5$ Units)")
    ax_c.set_ylabel("Remaining Capacity\n$C(t)/C(0)$ (%)")
    ax_c.set_xlabel("Week After Shock")
    ax_c.set_xlim(0, 52)
    ax_d.legend(loc="lower right")
    for ax, tag in ((ax_d, "a"), (ax_c, "b")):
        ax.text(-0.13, 1.0, f"({tag})", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
    fig.align_ylabels((ax_d, ax_c))
    fig.savefig(out / "figD2_geje_temporal.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exp2", type=Path, required=True)
    parser.add_argument("--exp3", type=Path)
    parser.add_argument("--out-dir", type=Path, default=FIGDIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    runs = load_runs(args.exp2)
    by_size = defaultdict(list)
    for r in runs:
        by_size[r["num_seeds"]].append(r)
    results["exp2"] = {str(s): summarise(rs) for s, rs in sorted(by_size.items())}
    fig_exp2(by_size[FIG_SEEDS], results["exp2"][str(FIG_SEEDS)], args.out_dir)
    for size in sorted(by_size):
        s = results["exp2"][str(size)]
        print(f"exp2 |S|={size:>4} n={s['n']:>5}: fastest decline week "
              f"{s['t_fast']['median']:.0f}, max weekly loss {s['max_weekly']['mean']:.3g}, "
              f"final loss {100 * s['loss_final']['median']:.1f}%, t50C "
              f"{s['t50C']['median']:.0f}, t90C {s['t90C']['median']:.0f}, t90 "
              f"{s['t90']['median']:.0f}, min C week {s['t_minC']['median']:.0f}, "
              f"monotone {100 * s['share_monotone']:.0f}%, recovered>1% "
              f"{100 * s['share_recovered_1pct']:.1f}%")

    if args.exp3:
        runs = load_runs(args.exp3)
        by_scn = defaultdict(list)
        for r in runs:
            by_scn[r["scenario_id"]].append(r)
        results["exp3"] = {str(s): summarise(rs) for s, rs in sorted(by_scn.items())}
        fig_exp3(runs, args.out_dir)
        for scn in sorted(by_scn):
            s = results["exp3"][str(scn)]
            print(f"exp3 scenario {scn} n={s['n']:>4}: fastest decline week "
                  f"{s['t_fast']['mean']:.1f}, max weekly loss {s['max_weekly']['mean']:.3g}, "
                  f"final loss {100 * s['loss_final']['mean']:.1f}%, t90C "
                  f"{s['t90C']['mean']:.1f}, t90 {s['t90']['mean']:.1f}, seeds "
                  f"{s['mean_seeds']:.0f}, cascade {s['mean_cascade']:.0f}")

    out = args.out_dir / "capacity_dynamics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
