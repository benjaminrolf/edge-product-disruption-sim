"""Figure: temporal disruption graph of an example cascade.

Draws the removal chain of one Experiment-2 run as a time-resolved cascade
tree: x = removal week, y = tidy-tree order, edges = recorded
`disruption_origin` links (who triggered whom), node colour = propagation
depth, node size = number of downstream removals triggered directly. Firms
that were disrupted but recovered are not part of the record; the graph shows
removals only.

Input:  outputs/figures/cascade_example_runs.json (from extract_cascade_example.py and
        extract_tohoku_cascade_example.py)
Output: outputs/figures/fig5_cascade_graph_deep.pdf (Fig. D1), fig8_cascade_graph_tohoku.pdf
        (Fig. D2) and fig_cascade_graph.pdf (largest single-seed cascade, not in the paper)
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import paper_style as st
from graphsim import paths

st.apply()

FIGDIR = paths.FIGURES_DIR
RUN_KEY = "largest_single_seed"  # or "deepest_chain"


def build_tree(rec: dict):
    seeds = {s["id"] for s in rec.get("seeds", [])}
    removed = {r["id"]: r for r in rec["removed"]}
    cascade = [i for i in removed if i not in seeds]

    week = {}
    for fid in list(seeds) + cascade:
        if fid in seeds:
            week[fid] = 0.0
        else:
            rt = removed[fid].get("removal_time") or 0
            # seed-removal convention: removal_time 1 with no disruption_time is t=0
            week[fid] = rt / 2.0  # two half-steps per week

    depth = {s: 0 for s in seeds}

    def get_depth(fid, guard):
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

    for fid in cascade:
        get_depth(fid, set())

    # primary parent: earliest-removed valid origin (ties: smallest id);
    # origin lists repeat the same supplier once per lost edge-product -> dedupe
    parent = {}
    for fid in cascade:
        origins = {o for o in (removed[fid].get("disruption_origin") or [])
                   if o in removed or o in seeds}
        if origins:
            parent[fid] = min(origins, key=lambda o: (week.get(o, 0.0), o))
        else:
            parent[fid] = min(seeds)
    children = defaultdict(list)
    for fid, p in parent.items():
        children[p].append(fid)
    for p in children:
        children[p].sort(key=lambda c: (week[c], c))

    # tidy-tree y: leaves get sequential slots, parents sit at children's mean
    y = {}
    counter = [0]

    def assign(fid):
        kids = children.get(fid, [])
        if not kids:
            y[fid] = counter[0]
            counter[0] += 1
            return y[fid]
        vals = [assign(c) for c in kids]
        y[fid] = float(np.mean(vals))
        return y[fid]

    roots = [s for s in sorted(seeds) if children.get(s)]
    for r in roots:
        assign(r)

    # secondary edges: removals with more than one distinct origin supplier
    extra_edges = []
    for fid in cascade:
        for o in {o for o in (removed[fid].get("disruption_origin") or [])
                  if o in removed or o in seeds}:
            if o != parent[fid]:
                extra_edges.append((o, fid))

    # seeds that appear only as secondary origins have no tree slot yet;
    # place them halfway between the rows of the customers they helped fail
    sec_targets = defaultdict(list)
    for o, fid in extra_edges:
        sec_targets[o].append(fid)
    for s in seeds:
        if s not in y and s in sec_targets:
            y[s] = float(np.mean([y[t] for t in sec_targets[s]])) + 0.5

    return seeds, cascade, week, depth, parent, children, y, extra_edges


def render(rec: dict, stem: str, figsize: tuple[float, float],
           legend_ncols: int = 1, legend_loc: str = "lower right",
           headroom: float = 0.0) -> None:
    seeds, cascade, week, depth, parent, children, y, extra = build_tree(rec)
    max_depth = max(depth[i] for i in cascade)
    # hide seeds that triggered no removal (relevant for multi-seed runs);
    # seeds with a y slot are involved as primary or secondary origin
    active_seeds = {s for s in seeds if s in y}
    n_idle = len(seeds) - len(active_seeds)
    print(f"{stem}: |S|={len(seeds)} ({n_idle} without cascade, hidden), "
          f"cascade={len(cascade)}, max depth={max_depth}, "
          f"horizon={max(week.values()):.1f} weeks")

    seq = ["#94C4E4", "#4494C6", "#0B5C93", "#083D62", "#052940", "#031B2B"]
    col = {fid: st.VERMILLION if fid in seeds else seq[min(depth[fid], len(seq)) - 1]
           for fid in list(seeds) + cascade}
    n_kids = {fid: len(children.get(fid, [])) for fid in list(seeds) + cascade}

    fig, ax = plt.subplots(figsize=figsize)

    # edges: primary tree links solid, secondary origins dotted
    for fid in cascade:
        p = parent[fid]
        ax.add_patch(FancyArrowPatch(
            (week[p], y[p]), (week[fid], y[fid]),
            connectionstyle="arc3,rad=0.12", arrowstyle="-|>",
            mutation_scale=5, lw=0.7, color="#999999", alpha=0.85,
            shrinkA=3, shrinkB=3, zorder=1))
    for o, fid in extra:
        ax.add_patch(FancyArrowPatch(
            (week[o], y[o]), (week[fid], y[fid]),
            connectionstyle="arc3,rad=0.12", arrowstyle="-|>",
            mutation_scale=4, lw=0.5, color="#BBBBBB", alpha=0.6,
            linestyle=(0, (1.5, 1.5)), shrinkA=3, shrinkB=3, zorder=0))

    for fid in cascade:
        size = min(10.0 + 14.0 * np.sqrt(n_kids[fid]), 110.0)
        ax.scatter(week[fid], y[fid], s=size, marker="o", color=col[fid],
                   edgecolors="white", linewidths=0.4, zorder=3)
    for fid in active_seeds:
        ax.scatter(week[fid], y[fid], s=60, marker="s", color=col[fid],
                   edgecolors="white", linewidths=0.6, zorder=4)

    ax.set_xlabel("Removal Week")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.8, max(week.values()) + 1)
    if headroom:
        ymax = max(y.values())
        ax.set_ylim(-2, ymax * (1 + headroom))

    handles = [Line2D([], [], marker="s", ls="", color=st.VERMILLION,
                      markersize=7, label="Seed Firm")]
    for d in range(1, min(max_depth, len(seq)) + 1):
        label = (f"Depth {d}" if d < len(seq) or max_depth <= len(seq)
                 else f"Depth $\\geq$ {len(seq)}")
        handles.append(Line2D([], [], marker="o", ls="",
                              color=seq[d - 1], markersize=6, label=label))
    if extra:
        handles.append(Line2D([], [], color="#BBBBBB", lw=0.8, ls=":",
                              label="Additional lost\nsupplier"))
    ncols = len(handles) if legend_ncols == 0 else legend_ncols
    ax.legend(handles=handles, loc=legend_loc, ncols=ncols)

    fig.savefig(FIGDIR / f"{stem}.pdf")
    fig.savefig(FIGDIR / f"{stem}.png", dpi=200)
    print(f"saved {stem}.pdf/.png")


def main() -> None:
    with open(FIGDIR / "cascade_example_runs.json") as f:
        runs = json.load(f)
    render(runs["largest_single_seed"], "fig_cascade_graph", (6.3, 3.2))
    render(runs["deepest_chain"], "fig5_cascade_graph_deep", (6.3, 3.6))
    if "tohoku_multi_origin" in runs:
        render(runs["tohoku_multi_origin"], "fig8_cascade_graph_tohoku", (6.3, 4.0),
               legend_ncols=1, legend_loc="upper right")


if __name__ == "__main__":
    main()
