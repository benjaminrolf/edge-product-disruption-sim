"""Export the canonical Louvain partition of the MarkLines LCC.

Reproduces the partition reported in the paper (Section 3.3 / Appendix B):
networkx `louvain_communities` with seed 42 on the undirected projection of the
largest weakly connected component, as computed in
analysis/network_analysis.ipynb (18 communities, Q = 0.5346).

Output: outputs/louvain_partition_seed42.json
    {"Q": float, "n_communities": int,
     "membership": {firm_id: community_rank}}  # rank 0 = largest community

The figure script fig1_network_overview.py consumes this file so that the
rendered communities match the numbers reported in the text.
"""

import json

import networkx as nx

from graphsim.networks.marklines import build_network, parse
from graphsim import paths

DATASET = paths.DATA_DIR
OUTPUT = paths.ANALYSIS_DIR / "louvain_partition_seed42.json"


def main() -> None:
    data = parse(str(DATASET))
    supply_net = build_network(data)

    lcc_nodes = max(nx.weakly_connected_components(supply_net), key=len)
    lcc_undir = supply_net.subgraph(lcc_nodes).to_undirected()
    print(f"LCC: {lcc_undir.number_of_nodes()} nodes, {lcc_undir.number_of_edges()} edges")

    partition = nx.community.louvain_communities(lcc_undir, seed=42)
    q = nx.community.modularity(lcc_undir, partition)
    sizes = sorted((len(c) for c in partition), reverse=True)
    print(f"{len(partition)} communities, Q = {q:.4f}, top sizes: {sizes[:6]}")

    order = sorted(range(len(partition)), key=lambda c: -len(partition[c]))
    membership = {
        int(firm_id): rank
        for rank, c in enumerate(order)
        for firm_id in partition[c]
    }
    OUTPUT.write_text(json.dumps({
        "Q": q,
        "n_communities": len(partition),
        "membership": membership,
    }))
    print(f"Saved {len(membership)} memberships -> {OUTPUT}")


if __name__ == "__main__":
    main()
