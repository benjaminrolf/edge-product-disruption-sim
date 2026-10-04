"""Regenerate all data-driven paper figures with the shared style (paper_style).

Figures (file name, figure in the paper):
  figC1_degree_ccdf.pdf    Fig. B1, degree CCDFs with power-law fit overlays
  fig4_robustness_e0.pdf   Fig. 4, Experiment-2 robustness curve (median, bootstrap CI,
                           sample means)
  fig7_geje_map.pdf        Fig. 6, per-firm disruption probability in the Tohoku-inspired
                           scenario

Data: Experiment-2 runs, MarkLines CSVs, and the per-firm disruption probabilities
of Experiment 3 (outputs/geje_per_firm_disruption_probs.csv, from geje_rev_summary.py).
Power-law overlays use the published fit parameters (alpha, xmin) of Appendix B.
The capacity dynamics (fig6_capacity_dynamics.pdf, figD2_geje_temporal.pdf) come from
capacity_dynamics.py.
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import polars as pl

import paper_style as st
from graphsim import paths

st.apply()

FIGDIR = paths.FIGURES_DIR

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- fig2: CCDF
def fig2_degree_ccdf() -> None:
    edges = pl.read_csv(paths.DATA_DIR / "SupplyRelations.csv",
                        columns=["source_firm_id", "target_firm_id"])
    out_deg: dict[int, int] = defaultdict(int)
    in_deg: dict[int, int] = defaultdict(int)
    nodes = set()
    for s, t in edges.iter_rows():
        out_deg[s] += 1
        in_deg[t] += 1
        nodes.add(s)
        nodes.add(t)
    tot = np.array([out_deg[n] + in_deg[n] for n in nodes])
    ind = np.array([in_deg[n] for n in nodes])
    outd = np.array([out_deg[n] for n in nodes])

    series = [
        ("total", tot, st.BLUE, 2.41, 9),
        ("in", ind[ind > 0], st.ORANGE, 1.89, 51),
        ("out", outd[outd > 0], st.GREEN, 2.92, 8),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for name, deg, color, alpha, xmin in series:
        k = np.sort(deg)
        ccdf = 1.0 - np.arange(len(k)) / len(k)
        ku, idx = np.unique(k, return_index=True)
        ax.loglog(ku, ccdf[idx], ".", ms=3.5, color=color, alpha=0.8, label=name)
        # fit overlay: discrete power law => CCDF slope -(alpha - 1), anchored at xmin
        anchor = ccdf[idx][np.searchsorted(ku, xmin)]
        kk = np.logspace(np.log10(xmin), np.log10(ku.max()), 50)
        ax.loglog(kk, anchor * (kk / xmin) ** (-(alpha - 1)), "--",
                  color=color, lw=1.1, alpha=0.9)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Fraction of Firms with Degree $\\geq k$")
    ax.grid(True, which="major", axis="both", color="#E5E5E5", lw=0.5)
    leg = ax.legend(title=None, loc="lower left", handletextpad=0.4)
    for t, (name, _, c, a, _x) in zip(leg.get_texts(), series):
        t.set_color(c)
    ax.text(0.98, 0.95,
            r"Dashed: Power-Law Fits ($\alpha$ = 2.41 / 1.89 / 2.92)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=st.INK)
    fig.savefig(FIGDIR / "figC1_degree_ccdf.pdf")
    plt.close(fig)
    print("fig2 done")


# ------------------------------------------- fig4: Exp-2
def figs_mc() -> None:
    mc = []
    for path in sorted((paths.RUNS_DIR / "exp2").glob("stats_*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                mc.append({k: r[k] for k in ("num_seeds", "num_removed", "sample_id")})

    # fig4: robustness curve
    sizes = sorted({r["num_seeds"] for r in mc})
    med, lo, hi = [], [], []
    sample_means = {s: defaultdict(list) for s in sizes}
    for r in mc:
        ns = r["num_seeds"]
        nr = r["num_removed"][-1]
        sample_means[ns][r["sample_id"]].append(nr - ns)
    for s in sizes:
        vals = np.concatenate([np.array(v) for v in sample_means[s].values()])
        med.append(np.median(vals))
        boots = [np.median(RNG.choice(vals, size=vals.size)) for _ in range(1000)]
        lo.append(np.percentile(boots, 2.5))
        hi.append(np.percentile(boots, 97.5))
    # Bootstrap CIs of the median are narrower than the plot markers at every
    # seed size (max width 3 removed firms on a ~190-unit axis), so they are
    # reported in the caption instead of drawn as an invisible band.
    print(f"  fig4 CI widths: max {max(h - l for h, l in zip(hi, lo)):.1f}")
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    for s in sizes:
        means = [np.mean(v) for v in sample_means[s].values()]
        ax.plot([s] * len(means), means, "o", ms=2.6, color=st.GREY, mec="none",
                alpha=0.9, zorder=2,
                label="Mean per Sample" if s == sizes[0] else None)
    ax.plot(sizes, med, "-o", color=st.BLUE, ms=4, zorder=3, label="Median Cascade Size")
    ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Seed Set Size")
    ax.set_ylabel("Cascade Size")
    ax.legend(loc="upper left")
    fig.savefig(FIGDIR / "fig4_robustness_e0.pdf")
    plt.close(fig)
    print("fig4 done")




# ---------------------------------------------------------- fig7: GEJE map
NE_GEOJSON = paths.ROOT / "assets/ne_50m_admin_0_countries.geojson"
# Source: Natural Earth 1:50m admin-0 countries via geojson.xyz
# (https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson)


def _draw_land(ax, xlim, ylim) -> None:
    """Draw Natural Earth land polygons clipped to the axis extent."""
    from matplotlib.patches import Polygon as MplPolygon

    geo = json.loads(NE_GEOJSON.read_text())
    for feat in geo["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ext = np.asarray(poly[0])
            if (ext[:, 0].max() < xlim[0] or ext[:, 0].min() > xlim[1]
                    or ext[:, 1].max() < ylim[0] or ext[:, 1].min() > ylim[1]):
                continue
            ax.add_patch(MplPolygon(ext, closed=True, facecolor="#F0EFEA",
                                    edgecolor="#B9B7AF", lw=0.5, zorder=0))


def fig7_geje_map() -> None:
    probs = pl.read_csv(paths.ANALYSIS_DIR / "geje_per_firm_disruption_probs.csv")
    probs = probs.with_columns(
        ((pl.col("disruption_prob_low") + pl.col("disruption_prob_mid")
          + pl.col("disruption_prob_high")) / 3).alias("p"),
    )
    firms = pl.read_csv(paths.DATA_DIR / "Firms.csv",
                        columns=["firm_id", "nation_id", "latitude", "longitude"])
    jp = firms.filter((pl.col("nation_id") == 1) & pl.col("latitude").is_not_null())
    df = jp.join(probs.select(["firm_id", "p"]), on="firm_id", how="left")

    xlim, ylim = (128.8, 146.0), (30.0, 45.8)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    _draw_land(ax, xlim, ylim)
    bg = df.filter(pl.col("p").is_null())
    ax.plot(bg["longitude"], bg["latitude"], ".", ms=1.1, color="#C9C7C0",
            mec="none", zorder=1)
    aff = df.filter(pl.col("p").is_not_null()).sort("p")
    sc = ax.scatter(aff["longitude"], aff["latitude"], c=aff["p"],
                    s=4 + 60 * aff["p"].to_numpy() ** 1.5, cmap=st.CMAP_SEQ,
                    vmin=0, vmax=1, lw=0, alpha=0.85, zorder=2)
    # anchors: Sendai = city at the quake epicentre; Tokyo, Toyota City, Osaka =
    # the three verified high-probability Tier-1 clusters (Hitachi Astemo/Mitsubishi
    # Electric, Denso/Aisin, Panasonic/Kawasaki Heavy; see geje_per_firm_probs lookup)
    for name, lon, lat in [("Sendai", 141.15, 38.05), ("Tokyo", 139.95, 35.45),
                           ("Toyota City", 136.55, 34.55), ("Osaka", 134.55, 34.25)]:
        ax.annotate(name, (lon, lat), fontsize=8.5, color=st.INK, style="italic",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                    zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1.25)
    ax.set_axis_off()
    # horizontal colorbar inset in the open ocean south-east of Honshu
    cax = ax.inset_axes([0.56, 0.10, 0.38, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Mean Disruption Probability", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    fig.savefig(FIGDIR / "fig7_geje_map.pdf")
    plt.close(fig)
    print(f"fig7 done ({aff.height} affected firms plotted)")


if __name__ == "__main__":
    fig2_degree_ccdf()
    figs_mc()
    fig7_geje_map()
    print("All figures regenerated.")
