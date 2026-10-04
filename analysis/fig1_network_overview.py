"""Render the network-overview figure: MarkLines supply-network LCC.

Structure package: graphviz sfdp layout, intra-community edges tinted in the community colour with
inter-community bridges in grey, degree-scaled nodes with highlighted hubs,
and a handful of well-known firms labelled as anchor points. Categorical
colours: validated Okabe-Ito subset; legend in the shared boxed style below
the panel. Nodes and edges are rasterised at 300 dpi.

Communities come from the canonical partition exported by
export_louvain_partition.py (networkx Louvain, seed 42; 18 communities,
Q = 0.5346), so the figure matches the numbers reported in the paper.

Output: outputs/figures/fig1_network_overview.pdf
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import paper_style as st
from graphsim import paths

st.apply()

DATASET = paths.DATA_DIR
OUTPUT = paths.FIGURES_DIR / "fig1_network_overview.pdf"

COLORS = [st.BLUE, st.ORANGE, st.GREEN, st.VERMILLION, st.PINK]
GREY = "#BBBBBB"
N_LABELS = 7
# canonical bloc names used throughout the paper (community rank 0-4)
BLOC_NAMES = ["China-led Bloc", "Japan-led Bloc", "Transatlantic Bloc",
              "India-led Bloc", "Korea-led Bloc"]


def main() -> None:
    edges = pl.read_csv(DATASET / "SupplyRelations.csv",
                        columns=["source_firm_id", "target_firm_id"])
    g = ig.Graph.TupleList(edges.unique().iter_rows(), directed=False)
    g.simplify()
    lcc = g.connected_components().giant()
    print(f"LCC: {lcc.vcount()} nodes, {lcc.ecount()} edges")

    print("Loading canonical Louvain partition (networkx, seed 42) ...")
    part = json.loads(
        (paths.ANALYSIS_DIR / "louvain_partition_seed42.json").read_text()
    )
    membership = {int(k): v for k, v in part["membership"].items()}
    memb = [membership.get(v["name"], 99) for v in lcc.vs]
    sizes_c = [c for _, c in Counter(memb).most_common()]
    print(f"{part['n_communities']} communities (Q = {part['Q']:.4f}), "
          f"top sizes: {sizes_c[:6]}")
    missing = sum(1 for m in memb if m == 99)
    if missing:
        print(f"WARNING: {missing} LCC nodes missing from canonical partition (drawn grey)")

    print("sfdp layout ...")
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dot = Path(td) / "lcc.dot"
        with open(dot, "w") as f:
            f.write("graph G {\n")
            for u, v in (e.tuple for e in lcc.es):
                f.write(f"{u} -- {v};\n")
            f.write("}\n")
        out = subprocess.run(["sfdp", "-Gsplines=none", "-Goverlap=true", "-Tplain", str(dot)],
                             capture_output=True, text=True, check=True)
    pos = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "node":
            pos[int(parts[1])] = (float(parts[2]), float(parts[3]))
    xy = np.array([pos[i] for i in range(lcc.vcount())])

    # legend labels: dominant nations per top-5 community
    firms = pl.read_csv(DATASET / "Firms.csv", columns=["firm_id", "nation_id"])
    nations = pl.read_csv(DATASET / "GeoNations.csv", columns=["nation_id", "nation_name"])
    nation_map = dict(firms.join(nations, on="nation_id", how="left")
                      .select(["firm_id", "nation_name"]).iter_rows())
    names_v = [v["name"] for v in lcc.vs]
    by_comm = defaultdict(list)
    for i, m in enumerate(memb):
        by_comm[m].append(i)
    labels = []
    for c in range(5):
        top = Counter(nation_map.get(names_v[i]) for i in by_comm[c]
                      if nation_map.get(names_v[i])).most_common(3)
        share = len(by_comm[c]) / lcc.vcount()
        labels.append(f"{BLOC_NAMES[c]} ({share:.0%})")
        print(f"Community {c} = {BLOC_NAMES[c]}: n={len(by_comm[c])}, "
              f"top nations: {top}")

    deg = np.array(lcc.degree())
    node_size = 0.7 * deg ** 0.55
    node_color = np.array([COLORS[m] if m < 5 else GREY for m in memb])

    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    fig.subplots_adjust(bottom=0.14)

    # edges: bridges grey below, intra-community tinted in community colour
    es = np.array([e.tuple for e in lcc.es])
    memb_arr = np.array(memb)
    same = (memb_arr[es[:, 0]] == memb_arr[es[:, 1]]) & (memb_arr[es[:, 0]] < 5)
    segs_inter = np.stack([xy[es[~same, 0]], xy[es[~same, 1]]], axis=1)
    ax.add_collection(LineCollection(segs_inter, colors="#999999", linewidths=0.18,
                                     alpha=0.05, rasterized=True, zorder=1))
    for c in range(5):
        sel = same & (memb_arr[es[:, 0]] == c)
        if not sel.any():
            continue
        segs = np.stack([xy[es[sel, 0]], xy[es[sel, 1]]], axis=1)
        ax.add_collection(LineCollection(segs, colors=COLORS[c], linewidths=0.20,
                                         alpha=0.10, rasterized=True, zorder=2))

    draw = np.argsort(node_size)
    ax.scatter(xy[draw, 0], xy[draw, 1], s=node_size[draw], c=node_color[draw],
               linewidths=0, alpha=0.9, rasterized=True, zorder=3)
    hubs = np.argsort(-deg)[:60]
    ax.scatter(xy[hubs, 0], xy[hubs, 1], s=node_size[hubs], c=node_color[hubs],
               linewidths=0.5, edgecolors="white", zorder=4)

    # anchor labels: highest-degree firm per top-5 community + global top firms
    fnames = (pl.read_csv(DATASET / "FirmNames.csv",
                          columns=["firm_id", "firm_name", "name_type"])
              .filter(pl.col("name_type") == "current")
              .unique(subset=["firm_id"], keep="first"))
    fname_map = dict(fnames.select(["firm_id", "firm_name"]).iter_rows())
    span = xy[:, 0].max() - xy[:, 0].min()
    picked = [max(by_comm[c], key=lambda i: deg[i]) for c in range(5)]
    for i in np.argsort(-deg):
        if len(picked) >= N_LABELS:
            break
        if int(i) not in picked:
            picked.append(int(i))

    placed = []
    for i in picked:
        raw = str(fname_map.get(names_v[i], ""))
        short = (raw.replace(" Corporation", "").replace(" Motor Co., Ltd.", "")
                 .replace(" Co., Ltd.", "").replace(" Company", "")
                 .replace(" Limited", "").replace(" GmbH", "").replace(" AG", "")
                 .replace(" Inc.", "").replace(" Ltd.", "").strip().rstrip(","))
        short = {"General Motors (GM)": "GM"}.get(short, short)
        if not short:
            continue
        # collision handling: alternate below the point if a nearby label exists
        near = any(np.hypot(*(xy[i] - xy[j])) < 0.06 * span for j in placed)
        va, dy = ("top", -7) if near else ("bottom", 7)
        ax.annotate(short, xy[i], fontsize=8, color="#222222", zorder=6,
                    ha="center", va=va, xytext=(0, dy), textcoords="offset points",
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
        placed.append(i)
        print(f"Label: {short} (deg {deg[i]}, community {memb[i]}, {va})")

    handles = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=c,
                      markeredgewidth=0, label=lab) for c, lab in zip(COLORS, labels)]
    handles.append(Line2D([], [], marker="o", ls="", markersize=7,
                          markerfacecolor=GREY, markeredgewidth=0,
                          label="Smaller Communities"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=True, fancybox=True, framealpha=0.95, edgecolor="#DDDDDD",
               title="Louvain Communities (Share of LCC)",
               title_fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    ax.set_axis_off()
    qx = np.percentile(xy[:, 0], [0.2, 99.8])
    qy = np.percentile(xy[:, 1], [0.2, 99.8])
    ax.set_xlim(qx[0] - 0.03 * (qx[1] - qx[0]), qx[1] + 0.03 * (qx[1] - qx[0]))
    ax.set_ylim(qy[0] - 0.03 * (qy[1] - qy[0]), qy[1] + 0.03 * (qy[1] - qy[0]))
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
