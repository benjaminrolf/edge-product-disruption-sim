"""Betweenness centrality of every firm in the directed supply network.

Predictor of Table D1 (yan_nexus_test.py, cascade_attribution.py). Exact directed node
betweenness (igraph) with the normalisation of networkx, 1 / ((n - 1)(n - 2)); firms
without relations get 0.

Output: outputs/node_betweenness.json  {firm_id: betweenness}
"""

import json

import igraph as ig
import polars as pl
from graphsim import paths


def main() -> None:
    firms = pl.read_csv(paths.DATA_DIR / "Firms.csv", columns=["firm_id"],
                        infer_schema_length=10000)["firm_id"].to_list()
    edges = pl.read_csv(paths.DATA_DIR / "SupplyRelations.csv",
                        columns=["source_firm_id", "target_firm_id"])
    index = {firm: i for i, firm in enumerate(firms)}
    for firm in set(edges["source_firm_id"]) | set(edges["target_firm_id"]):
        index.setdefault(firm, len(index))
    g = ig.Graph(n=len(index), directed=True,
                 edges=[(index[s], index[t]) for s, t in edges.iter_rows()])
    g.simplify(multiple=True, loops=True)
    n = g.vcount()
    norm = 1.0 / ((n - 1) * (n - 2))
    ids = list(index)
    betweenness = {str(ids[i]): b * norm for i, b in enumerate(g.betweenness(directed=True))}
    out = paths.ANALYSIS_DIR / "node_betweenness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(betweenness))
    print(f"{n} firms, {sum(b > 0 for b in betweenness.values())} with betweenness > 0; saved {out}")


if __name__ == "__main__":
    main()
