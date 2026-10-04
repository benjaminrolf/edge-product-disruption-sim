"""Compute the exact undirected diameter of the MarkLines LCC with igraph.

Diameter reported in Section 3.3 and Table A1 of the paper (exact value: 10).

Output: outputs/figures/lcc_diameter.json
"""

import json

import igraph as ig
import polars as pl
from graphsim import paths


def main() -> None:
    edges = pl.read_csv(
        paths.DATA_DIR / "SupplyRelations.csv",
        columns=["source_firm_id", "target_firm_id"],
    )
    g = ig.Graph.TupleList(edges.iter_rows(), directed=False)
    lcc = g.connected_components().giant()
    d = lcc.diameter(directed=False)
    print(f"LCC: {lcc.vcount()} nodes; exact undirected diameter: {d}")
    out = paths.FIGURES_DIR / "lcc_diameter.json"
    json.dump(
        {"lcc_nodes": lcc.vcount(), "diameter_undirected": d},
        open(out, "w"),
        indent=2,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
