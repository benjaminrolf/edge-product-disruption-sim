"""Clustering, path length, and small-world index of the MarkLines LCC.

Backs Appendix B.2 of the paper. Reports the average clustering coefficient
of the directed LCC (Fagiolo 2007 definition, networkx) against a directed
degree-sequence-preserving configuration-model baseline, and the same
comparison under an undirected null model, which inverts because the
network is nearly acyclic. Path lengths use the undirected LCC with the same
1000-source sampling as analysis/network_analysis.ipynb.

Output: outputs/figures/clustering_null_models.json
"""

import json
import random

import igraph as ig
import networkx as nx
import numpy as np
import polars as pl
from graphsim import paths

DATASET = paths.DATA_DIR
OUTPUT = paths.FIGURES_DIR / "clustering_null_models.json"

N_DIRECTED_NULL = 50
N_UNDIRECTED_NULL = 30
N_PATH_SOURCES = 1000


def main() -> None:
    edges = pl.read_csv(DATASET / "SupplyRelations.csv",
                        columns=["source_firm_id", "target_firm_id"]).unique().sort(
        ["source_firm_id", "target_firm_id"])  # deterministic node order for the null draws

    # directed LCC (networkx, Fagiolo directed clustering)
    g = nx.DiGraph()
    g.add_edges_from(edges.iter_rows())
    lcc = g.subgraph(max(nx.weakly_connected_components(g), key=len)).copy()
    n, m = lcc.number_of_nodes(), lcc.number_of_edges()
    c_dir = nx.average_clustering(lcc)
    indeg = [d for _, d in lcc.in_degree()]
    outdeg = [d for _, d in lcc.out_degree()]
    c_dir_null = []
    for seed in range(N_DIRECTED_NULL):
        r = nx.directed_configuration_model(indeg, outdeg, create_using=nx.DiGraph, seed=seed)
        c_dir_null.append(nx.average_clustering(r))
    c_dir_null = np.array(c_dir_null)
    print(f"directed LCC: n={n}, m={m}, C={c_dir:.4f}, null {c_dir_null.mean():.4f} "
          f"± {c_dir_null.std():.4f}, ratio {c_dir / c_dir_null.mean():.2f}, "
          f"z {(c_dir - c_dir_null.mean()) / c_dir_null.std():.1f}")

    # undirected LCC (igraph): clustering under an undirected null, path lengths
    gu = ig.Graph.TupleList(edges.iter_rows(), directed=False)
    gu.simplify()
    lu = gu.connected_components().giant()
    c_und = lu.transitivity_avglocal_undirected(mode="zero")
    t_und = lu.transitivity_undirected()
    random.seed(42)
    c_und_null, t_und_null = [], []
    for _ in range(N_UNDIRECTED_NULL):
        r = ig.Graph.Degree_Sequence(lu.degree(), method="configuration")
        r.simplify()
        c_und_null.append(r.transitivity_avglocal_undirected(mode="zero"))
        t_und_null.append(r.transitivity_undirected())
    c_und_null, t_und_null = np.array(c_und_null), np.array(t_und_null)
    print(f"undirected LCC: C={c_und:.4f}, null {c_und_null.mean():.4f}, "
          f"ratio {c_und / c_und_null.mean():.2f}; transitivity {t_und:.5f}, "
          f"null {t_und_null.mean():.5f}, ratio {t_und / t_und_null.mean():.2f}")

    random.seed(42)
    sources = random.sample(range(lu.vcount()), N_PATH_SOURCES)
    d = np.array(lu.distances(source=sources), dtype=float)
    d = d[d > 0]
    mean_path = float(d.mean())
    mean_degree = 2 * lu.ecount() / lu.vcount()
    l_random = float(np.log(lu.vcount()) / np.log(mean_degree))
    diameter = lu.diameter(directed=False)
    sigma_directed = (c_dir / c_dir_null.mean()) / (mean_path / l_random)
    sigma_undirected = (c_und / c_und_null.mean()) / (mean_path / l_random)
    print(f"L={mean_path:.3f}, L_random={l_random:.3f}, diameter={diameter}, "
          f"sigma (directed C ratio)={sigma_directed:.2f}, "
          f"sigma (undirected C ratio)={sigma_undirected:.2f}")

    OUTPUT.write_text(json.dumps({
        "lcc_nodes": n, "lcc_edges_directed": m,
        "directed": {
            "avg_clustering": c_dir,
            "null_mean": c_dir_null.mean(), "null_std": c_dir_null.std(),
            "ratio": c_dir / c_dir_null.mean(),
            "z": (c_dir - c_dir_null.mean()) / c_dir_null.std(),
            "n_null": N_DIRECTED_NULL,
        },
        "undirected": {
            "avg_clustering": c_und, "null_mean": c_und_null.mean(),
            "ratio": c_und / c_und_null.mean(),
            "transitivity": t_und, "transitivity_null_mean": t_und_null.mean(),
            "transitivity_ratio": t_und / t_und_null.mean(),
            "n_null": N_UNDIRECTED_NULL,
        },
        "mean_path_length": mean_path, "l_random": l_random, "diameter": diameter,
        "sigma_directed": sigma_directed, "sigma_undirected": sigma_undirected,
    }, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
