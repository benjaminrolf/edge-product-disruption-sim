"""Edge betweenness and edge load centrality of every supply relation.

Inputs of the edge-level cells of network_analysis.ipynb (networkx, normalised edge
betweenness; takes hours on the full network).

Outputs: outputs/betweenness_centrality.json and outputs/edge_load.json
         {relation_id: value}
"""

import json

import networkx as nx
import polars as pl

from graphsim import paths


def main() -> None:
    firms = pl.read_csv(paths.DATA_DIR / "Firms.csv", columns=["firm_id"],
                        infer_schema_length=10000)
    relations = pl.read_csv(paths.DATA_DIR / "SupplyRelations.csv")
    net = nx.from_pandas_edgelist(relations.to_pandas(), source="source_firm_id",
                                  target="target_firm_id", edge_attr="relation_id",
                                  create_using=nx.DiGraph)
    net.add_nodes_from(firms["firm_id"])
    paths.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for name, centrality in (
        ("betweenness_centrality", lambda: nx.edge_betweenness_centrality(net, normalized=True)),
        ("edge_load", lambda: nx.edge_load_centrality(net)),
    ):
        values = centrality()
        by_relation = {data["relation_id"]: values[(u, v)] for u, v, data in net.edges(data=True)}
        out = paths.ANALYSIS_DIR / f"{name}.json"
        out.write_text(json.dumps(by_relation))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
