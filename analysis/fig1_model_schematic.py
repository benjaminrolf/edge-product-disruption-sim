"""Figure 2 (two subfigures): firm state machine (a) and substitution scene (b).

(a) fig2a_firm_states.pdf: Stable -> Searching -> Disrupted -> Removed with the
recovery arc back to Stable.
(b) fig2b_substitution.pdf: network excerpt (failed supplier u, customer v,
candidate u') embedded in a faint background supply network, plus the two
substitution rules as comparison cards with their median candidate-pool sizes
per product lost on a failed edge (Fig. 3, candidate_pools_edges.json from
fig_candidate_pools_edges.py: 30,933 vs 832).

Subfigures of Fig. 2 in the paper.
"""


import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import paper_style as st
from graphsim import paths

st.apply()
mpl.rcParams["mathtext.fontset"] = "dejavusans"

FIGDIR = paths.FIGURES_DIR

STATE_COLORS = {
    "Stable": (st.GREEN, "#E5F4EF"),
    "Searching": (st.BLUE, "#E3EEF6"),
    "Disrupted": (st.ORANGE, "#FBF1DC"),
    "Removed": (st.VERMILLION, "#F9E7DE"),
}


def rbox(ax, x, y, w, h, edge, face, lw=1.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.06",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                zorder=3))


def arrow(ax, xy_from, xy_to, color=st.INK, ls="-", rad=0.0, lw=1.1, shrink=2):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>",
                                 mutation_scale=11, linewidth=lw, linestyle=ls,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 zorder=4, shrinkA=shrink, shrinkB=shrink))


# ------------------------------------------------------------------ fig 1a
def fig_a() -> None:
    fig, ax = plt.subplots(figsize=(6.3, 1.60))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.54, 1.14)
    ax.set_axis_off()

    xs = {"Stable": 1.45, "Searching": 3.90, "Disrupted": 6.35, "Removed": 8.80}
    y, w, h = 0.28, 1.72, 0.46
    edge_dx = w / 2 + 0.02   # visual box edge incl. rounding pad
    top_y = y + h / 2 + 0.02
    bot_y = y - h / 2 - 0.02
    for name, x in xs.items():
        edge, face = STATE_COLORS[name]
        rbox(ax, x, y, w, h, edge, face)
        ax.text(x, y, name, ha="center", va="center", fontsize=9.5, color=st.INK)
    # Removed is absorbing: double border (final-state notation)
    rbox(ax, xs["Removed"], y, w - 0.13, h - 0.12,
         STATE_COLORS["Removed"][0], "none", lw=0.8)

    # unified arrow style for every transition
    A = dict(color=st.INK, lw=1.1, shrink=0)

    # initial-state entry arrows: non-seed firms -> Stable, seed firms -> Removed
    arrow(ax, (xs["Stable"] - edge_dx - 0.55, y), (xs["Stable"] - edge_dx, y),
          **A)
    ax.text(xs["Stable"] - edge_dx - 0.28, y + 0.10, "$f \\notin S$",
            ha="center", va="bottom", fontsize=7, color=st.INK)
    arrow(ax, (xs["Removed"], top_y + 0.48), (xs["Removed"], top_y), **A)
    ax.text(xs["Removed"] + 0.12, top_y + 0.26, "seeds $f \\in S$\n($t{=}0$)",
            ha="left", va="center", fontsize=6.8, color=st.INK,
            linespacing=1.3)

    # guarded forward transitions, labels attached above or below their arrows
    transitions = [
        ("Stable", "Searching", "$\\geq 1$ supply\nedge lost", "above"),
        ("Searching", "Disrupted",
         "budget exhausted\n$\\vee\\ \\mathcal{C}(p) = \\emptyset$", "above"),
        ("Disrupted", "Removed",
         "w.p. $p_\\mathrm{rem}(\\ell(f), s_f)$", "below"),
    ]
    for a, b, lab, side in transitions:
        x0, x1 = xs[a] + edge_dx, xs[b] - edge_dx
        arrow(ax, (x0, y), (x1, y), **A)
        if side == "above":
            ax.text((x0 + x1) / 2, y + h / 2 + 0.04, lab, ha="center",
                    va="bottom", fontsize=6.8, color=st.INK, linespacing=1.25)
        else:
            ax.text((x0 + x1) / 2, y - h / 2 - 0.07, lab, ha="center",
                    va="top", fontsize=6.8, color=st.INK)

    # self-loops: open circles (240 degrees) whose endpoints sit on the box
    # top edge, drawn as one arrow along the arc so the head is tangent
    def self_loop(x_c, lab):
        rx, ry = 0.26, 0.18
        c_y = top_y + 0.5 * ry  # endpoints 30 degrees below the circle centre
        th = np.deg2rad(np.linspace(-30, 210, 61))
        verts = np.column_stack([x_c + rx * np.cos(th),
                                 c_y + ry * np.sin(th)])
        ax.add_patch(FancyArrowPatch(path=mpl.path.Path(verts),
                                     arrowstyle="-|>", mutation_scale=11,
                                     lw=1.1, color=st.INK, fill=False,
                                     zorder=4, shrinkA=0, shrinkB=0))
        ax.text(x_c, c_y + ry + 0.03, lab, ha="center", va="bottom",
                fontsize=6.8, color=st.INK)

    self_loop(xs["Stable"], "no edge lost")
    self_loop(xs["Searching"], "search ongoing")
    self_loop(xs["Disrupted"], "w.p. $1 - p_\\mathrm{rem}$")

    # recovery arc: bottom edge of Searching back to bottom edge of Stable
    arrow(ax, (xs["Searching"] - 0.3, bot_y), (xs["Stable"] + 0.3, bot_y),
          rad=-0.45, **A)
    ax.text((xs["Stable"] + xs["Searching"]) / 2, -0.36,
            "all lost edges re-established", ha="center", va="top",
            fontsize=6.8, color=st.INK)
    fig.savefig(FIGDIR / "fig2a_firm_states.pdf", bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(FIGDIR / "fig2a_firm_states.png", dpi=150,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("fig1a saved")


# ------------------------------------------------------------------ fig 1b
def node(ax, x, y, r, edge, face, label, lw=1.4):
    ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=edge, lw=lw,
                        zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=st.INK,
            zorder=6)


def gauge(ax, x, y, free_frac, sub_label):
    """Small vertical capacity gauge: grey = utilised, blue = free."""
    gw, gh = 0.30, 1.10
    ax.add_patch(mpl.patches.Rectangle((x, y - gh / 2), gw, gh,
                                       facecolor="#DCDAD4", edgecolor="#999999",
                                       lw=0.8, zorder=5))
    if free_frac > 0:
        ax.add_patch(mpl.patches.Rectangle((x, y + gh * (0.5 - free_frac)), gw,
                                           gh * free_frac, facecolor=st.BLUE,
                                           lw=0, zorder=6))
        ax.text(x + gw / 2, y + gh / 2 + 0.14, "free", ha="center", va="bottom",
                fontsize=6.5, color=st.BLUE, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.text(x + gw / 2, y - gh / 2 - 0.14, sub_label, ha="center", va="top",
            fontsize=7, color="#666666", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


def cross(ax, x, y, size=0.16, color=st.VERMILLION, lw=1.6):
    for sgn in (1, -1):
        ax.plot([x - size, x + size], [y - sgn * size, y + sgn * size],
                color=color, lw=lw, zorder=7, solid_capstyle="round")


def fig_b() -> None:
    fig = plt.figure(figsize=(6.3, 2.9))
    ax = fig.add_axes([0.0, 0.0, 0.55, 1.0])       # network scene
    axc = fig.add_axes([0.565, 0.0, 0.435, 1.0])   # rule cards
    for a in (ax, axc):
        a.set_axis_off()

    # --- background supply network (faint) ---
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.3)
    ax.set_aspect("equal")
    rng = np.random.default_rng(11)
    pts = rng.uniform([0.4, 0.4], [9.6, 7.9], size=(52, 2))
    main = np.array([[2.6, 5.6], [7.2, 5.0], [2.95, 1.9],
                     [5.75, 1.30], [8.75, 2.35]])
    dist = np.linalg.norm(pts[:, None, :] - main[None, :, :], axis=2)
    pts = pts[(dist > 1.05).all(axis=1)]
    segs = set()
    for i, p in enumerate(pts):
        d = np.linalg.norm(pts - p, axis=1)
        for j in np.argsort(d)[1:3]:
            segs.add(tuple(sorted((i, int(j)))))
    for i, j in segs:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color="#E3E2DE", lw=0.6, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color="#D3D2CD", lw=0, zorder=2)
    for (mx, my), k in zip(main, (3, 3, 2, 2, 2)):
        d = np.linalg.norm(pts - (mx, my), axis=1)
        for j in np.argsort(d)[:k]:
            ax.plot([mx, pts[j, 0]], [my, pts[j, 1]], color="#E3E2DE",
                    lw=0.6, zorder=1)

    # --- failed supplier u and customer v ---
    (ux, uy), (vx, vy), c1, c2, c3 = main
    r = 0.60
    e, f = STATE_COLORS["Removed"]
    node(ax, ux, uy, r, e, f, "$u$")
    ax.plot([ux - 0.35, ux + 0.35], [uy - 0.35, uy + 0.35],
            color=st.VERMILLION, lw=1.6, zorder=6, solid_capstyle="round")
    ax.text(ux, uy + r + 0.26, "failed supplier", ha="center", va="bottom",
            fontsize=8, color=st.VERMILLION)
    node(ax, vx, vy, r, "#8C8C8C", "#F2F2F2", "$v$")
    ax.text(vx, vy + r + 0.26, "customer", ha="center", va="bottom",
            fontsize=8, color="#666666")

    # broken supply edge u -> v
    mx, my = (ux + vx) / 2, (uy + vy) / 2
    for x0, y0, x1, y1 in [(ux, uy, mx - 0.35, my + 0.05),
                           (mx + 0.35, my - 0.05, vx, vy)]:
        ax.plot([x0, x1], [y0, y1], color="#9C9C9C", lw=1.3, zorder=3)
    arrow(ax, (mx + 0.35, my - 0.05), (vx - r - 0.05, vy - 0.08),
          color="#9C9C9C", lw=1.3, shrink=0)
    cross(ax, mx, my, size=0.14, lw=1.5)
    ax.text(mx, my + 0.34, "lost: $p \\in \\pi_E(u,v)$", ha="center",
            va="bottom", fontsize=8, color=st.VERMILLION, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # --- substitute candidates ---
    e, f = STATE_COLORS["Searching"]
    for (cxn, cyn), lab in zip((c1, c2, c3), ("$u'_1$", "$u'_2$", "$u'_3$")):
        node(ax, cxn, cyn, r, e, f, lab)
    ax.text(4.05, 0.22, "substitute candidates", ha="center", va="bottom",
            fontsize=8, color=st.BLUE, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # u'_1 accepts: full dashed edge to v, green check at the delivery end
    arrow(ax, (c1[0] + r * 0.7, c1[1] + r * 0.65),
          (vx - r * 0.5, vy - r * 0.95), color=st.BLUE, ls="--", rad=-0.18,
          lw=1.4, shrink=0)
    ax.text(3.55, 3.55, "offers $q$,\n$(p, q) \\in \\Sigma$", ha="center",
            va="center", fontsize=8, color=st.BLUE, linespacing=1.35, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.add_patch(Circle((6.42, 4.02), 0.24, facecolor="white",
                        edgecolor=st.GREEN, lw=1.2, zorder=7))
    ax.plot([6.32, 6.40, 6.54], [4.02, 3.92, 4.14], color=st.GREEN, lw=1.5,
            zorder=8, solid_capstyle="round")
    ax.text(6.42, 3.68, "accepts", ha="center", va="top", fontsize=7,
            color=st.GREEN, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    gauge(ax, c1[0] - r - 0.70, c1[1], 0.25, "capacity")

    # u'_2 declines: full gauge, request edge ends in a cross
    p0 = (c2[0] + 0.15, c2[1] + r + 0.05)
    p1 = (6.28, 2.72)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=st.BLUE, ls="--", lw=1.1,
            alpha=0.75, zorder=3)
    cross(ax, *p1, size=0.14)
    ax.text(6.52, 2.55, "declines:\nno free\ncapacity", ha="left", va="center",
            fontsize=7, color="#666666", linespacing=1.3, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    gauge(ax, c2[0] + r + 0.30, c2[1] + 0.05, 0.0, "full")

    # u'_3 declines: serves existing partners first
    p0 = (c3[0] - 0.25, c3[1] + r + 0.02)
    p1 = (8.02, 4.10)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=st.BLUE, ls="--", lw=1.1,
            alpha=0.75, zorder=3)
    cross(ax, *p1, size=0.14)
    ax.text(8.85, 4.35, "declines:\nexisting\npartners first", ha="center",
            va="top", fontsize=7, color="#666666", linespacing=1.3, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    # --- rule cards ---
    axc.set_xlim(0, 10)
    axc.set_ylim(0, 8.3)
    cards = [
        (6.15, st.ORANGE, "Baseline rule (firm level)",
         "$u'$ qualifies if the portfolios overlap:\n"
         "$\\pi_V(u) \\cap \\pi_V(u') \\neq \\emptyset$",
         "median pool: 30,933 candidates per lost product"),
        (1.95, st.BLUE, "Product-specific rule (product level)",
         "$u'$ qualifies if it offers a product $q$\n"
         "$\\Sigma$-similar to a lost $p \\in \\pi_E(u,v)$",
         "median pool: 832 candidates per lost product"),
    ]
    cx, cw, ch = 5.0, 9.7, 3.55
    for cy, color, title, body, pool in cards:
        axc.add_patch(FancyBboxPatch((cx - cw / 2, cy - ch / 2), cw, ch,
                                     boxstyle="round,pad=0.02,rounding_size=0.12",
                                     linewidth=0.8, edgecolor="#DDDDDD",
                                     facecolor="white", zorder=3))
        axc.add_patch(FancyBboxPatch((cx - cw / 2 - 0.02, cy - ch / 2 - 0.02),
                                     0.22, ch + 0.04,
                                     boxstyle="round,pad=0.02,rounding_size=0.08",
                                     linewidth=0, facecolor=color, zorder=4))
        axc.text(cx - cw / 2 + 0.5, cy + ch / 2 - 0.34, title, fontsize=8.5,
                 color=color, va="top", ha="left", zorder=5)
        axc.text(cx - cw / 2 + 0.5, cy + 0.1, body, fontsize=8, color=st.INK,
                 va="center", ha="left", linespacing=1.5, zorder=5)
        axc.text(cx - cw / 2 + 0.5, cy - ch / 2 + 0.28, pool, fontsize=7.5,
                 color="#777777", va="bottom", ha="left", zorder=5)

    fig.savefig(FIGDIR / "fig2b_substitution.pdf")
    fig.savefig(FIGDIR / "fig2b_substitution.png", dpi=150)
    plt.close(fig)
    print("fig1b saved")


if __name__ == "__main__":
    fig_a()
    fig_b()
