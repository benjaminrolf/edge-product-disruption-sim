"""Shared matplotlib style for all paper figures.

Design rules (dataviz): categorical identity = fixed-order Okabe-Ito subset
(CVD-validated); ordered severity = sequential single-hue steps; magnitude on
maps = perceptual sequential colormap; thin marks, recessive grid, no top/right
spines, tight margins, >= 8 pt text at print size (text width ~ 6.3 in).
"""

import matplotlib as mpl

# Categorical palette: Okabe-Ito (fixed order, CVD-safe); use for every
# qualitative encoding. SKY and YELLOW complete the 8-colour set for figures
# that need more than five categories.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PINK = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#BBBBBB"
INK = "#333333"

# Ordered severity (low -> high): one hue, light -> dark
SEQ3 = ["#94C4E4", "#4494C6", "#0B5C93"]

CMAP_SEQ = "viridis"  # magnitude on the map


def apply() -> None:
    mpl.rcParams.update({
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.4,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#DDDDDD",
        "legend.facecolor": "white",
        "legend.borderpad": 0.6,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })
