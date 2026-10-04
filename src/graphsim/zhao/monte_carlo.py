"""Module for transforming a given supply network into a simulation-ready graph."""

import collections
import json
from math import log
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde, truncnorm

from graphsim.types import EdgeID, NodeID
from graphsim.zhao.node import ZhaoNode

if TYPE_CHECKING:
    from .config import NetworkConfig


def sample_node_config(
    net_config: "NetworkConfig",
    net: nx.DiGraph,
    network_seed: int,
    edge_config: dict[EdgeID, dict[str, Any]] | None = None,
    in_inv_ranges: dict[NodeID, int] | None = None,
    out_inv_ranges: dict[NodeID, int] | None = None,
) -> dict[NodeID, dict[str, float]]:
    """Create node and edge mappings with attributes required for the simulation.

    Prepares the Marklines dataset for the Zhao simulation model.

    Parameters
    ----------
    config : NetworkConfig
        The configuration for the Zhao-Sim model.
    net : nx.DiGraph
        The supply network to transform.
    network_seed : int
        The seed to create random request capacities and k values.
    edge_config : dict[EdgeID, dict[str, float]] | None
        The edge configuration mapping.
    in_inv_ranges : dict[NodeID, int] | None
        The inventory ranges for inbound inventory. Created in edge configuration.
    out_inv_ranges : dict[NodeID, int] | None
        The inventory ranges for outbound inventory. Created in edge configuration.

    """
    # Create random number generator (a stream independent of the edge sampling in
    # sample_edge_config, which is seeded with network_seed alone)
    rng = np.random.default_rng(seed=[network_seed, 1])

    if in_inv_ranges is None:
        in_inv_ranges = {}

    if out_inv_ranges is None:
        out_inv_ranges = {}

    # Create mapping of node IDs to sizes (use degree as size for now)
    # Isolates are assigned a size of 1
    node_size_map = {
        node: net.degree[node] if net.degree[node] > 0 else 1 for node in net.nodes
    }

    # Create mapping of node IDs to request capacities
    if net_config.dist_cap["strategy"] == "normal":
        node_cap_map = get_cap_map_normal(net_config, node_size_map, net, rng)
    elif net_config.dist_cap["strategy"] == "uniform":
        node_cap_map = get_cap_map_uniform(net_config, node_size_map, net, rng)
    elif (
        net_config.dist_cap["strategy"] == "capacity_utilization"
        and edge_config is not None
    ):
        node_cap_map = get_cap_map_util(edge_config, net_config, net, rng)
    else:
        msg = "Invalid capacity distribution strategy or missing edge config."
        raise ValueError(msg)

    # Create mapping of node IDs to k values (partner preference parameter)
    k_dist = create_truncnorm(**net_config.dist_k)
    samples = k_dist.rvs(size=len(net.nodes), random_state=rng)
    node_k_map = dict(zip(net.nodes, samples, strict=False))

    # Create mapping of identifiers to nodes
    return {
        node: {
            "k": node_k_map[node],
            "size": node_size_map[node],
            "request_cap": node_cap_map.get(node, 0),  # 0 for sinks
            "in_inv_range": in_inv_ranges.get(node, 0),
            "out_inv_range": out_inv_ranges.get(node, 0),
        }
        for node in net.nodes
    }


def create_truncnorm(
    mu: float,
    sigma: float,
    lb: float,
    ub: float,
    **kwargs: Any,
) -> Any:
    """Create a truncated normal distribution."""
    a = (lb - mu) / sigma
    b = (ub - mu) / sigma
    return truncnorm(a=a, b=b, loc=mu, scale=sigma)


def get_node_capacities(
    nodes: list,
    dist: Any,
    rng: np.random.Generator,
) -> dict[NodeID, int]:
    """Create a mapping of node IDs to request capacities."""
    samples = dist.rvs(size=len(nodes), random_state=rng)
    return {node: round(sample) for node, sample in zip(nodes, samples, strict=False)}


def get_cap_map_normal(
    config: "NetworkConfig",
    node_size_map: dict[NodeID, int],
    supply_net: nx.DiGraph,
    rng: np.random.Generator,
) -> dict[NodeID, int]:
    """Sample request capacities for nodes based on their sizes.

    First, orders the nodes by size and then splits them into three groups. Then,
    creates truncated normal distributions for each group and samples a request capacity
    for each node based on the group it belongs to.
    Based on the algorithm by Zhao et al. (2019)

    Parameters
    ----------
    config : NetworkConfig
        The configuration for the Zhao-Sim model.
    node_size_map : dict
        A mapping of node IDs to their sizes.
    supply_net : nx.DiGraph
        The supply network.
    rng : np.random.Generator
        The random number generator.

    """
    # Exclude sink nodes (OEMs have no products)
    non_sink_nodes = [
        node
        for node in supply_net.nodes
        if supply_net.out_degree[node] > 0 or supply_net.degree[node] == 0
    ]

    # Order nodes by size
    ordered_nodes = sorted(non_sink_nodes, key=lambda item: node_size_map[item])

    # Split the nodes into three groups
    split_nodes = np.array_split(ordered_nodes, 3)
    groups = ["low", "mid", "high"]

    # Create truncated normal distributions for each group
    dists = {
        group: create_truncnorm(
            mu=config.dist_cap["mu"][i],
            sigma=config.dist_cap["sigma"][i],
            lb=config.dist_cap["lb"],
            ub=config.dist_cap["ub"],
        )
        for i, group in enumerate(groups)
    }

    # Assign capacities to nodes
    node_cap_map = {}
    for i, group in enumerate(groups):
        group_nodes = [node for node in supply_net.nodes if node in split_nodes[i]]
        node_cap_map.update(get_node_capacities(group_nodes, dists[group], rng))

    return node_cap_map


def get_cap_map_uniform(
    config: "NetworkConfig",
    node_size_map: dict,
    supply_net: nx.DiGraph,
    rng: np.random.Generator,
) -> dict[NodeID, int]:
    """Sample request capacities for nodes based on their sizes.

    Creates a discrete uniform distribution for request capacities and samples a request
    capacity for each node based on the group it belongs to. Should only be used with
    actual revenue data or size indicators with high numbers.
    Based on the algorithm by Zhao et al. (2019)

    Parameters
    ----------
    config : NetworkConfig
        The configuration for the Zhao-Sim model.
    node_size_map : dict
        A mapping of node IDs to their sizes.
    supply_net : nx.DiGraph
        The supply network.
    rng : np.random.Generator
        The random number generator.

    """
    # Create discrete uniform distribution
    margin = config.dist_cap["margin"]

    # Exclude sink nodes (OEMs have no products)
    non_sink_nodes = [
        node
        for node in supply_net.nodes
        if supply_net.out_degree[node] > 0 or supply_net.degree[node] == 0
    ]

    # Create request capacities
    node_cap_map = {}
    for node in non_sink_nodes:
        # Logarithmic scaling
        high = round(log(node_size_map[node])) / 2 + margin
        low = round(log(node_size_map[node])) / 2 - margin
        low = max(low, 0)

        capacity = rng.integers(low, high + 1, dtype=int)
        node_cap_map[node] = capacity

    return node_cap_map


def get_cap_map_util(
    edge_config: dict[EdgeID, dict],
    net_config: "NetworkConfig",
    supply_net: nx.DiGraph,
    rng: np.random.Generator,
) -> dict[NodeID, int]:
    """Get request capacities based on the average capacity utilization rate."""
    # Get the weighted out-degree for every node
    node_weights = collections.defaultdict(int)
    for source, sink, edgedata in supply_net.edges(data=True):
        if edgedata["stochastic"]:
            weight = sum(edge_config[(source, sink)]["products"].values())
        else:
            weight = sum(edgedata["products"].values())
        node_weights[source] += weight

    # Add node weights for isolates
    for node in nx.isolates(supply_net):
        node_weights[node] = 1

    # Sample capacity utilization rate for every node
    dist = create_truncnorm(**net_config.dist_cap)
    node_curs = dist.rvs(size=len(node_weights), random_state=rng)

    # Get free request capacities for all isolates and source nodes
    cap_map = {
        node: np.round(node_weights[node] * (1 - node_curs[i]))
        for i, node in enumerate(node_weights)
    }

    # Firms without recorded customers (isolates and pure customers) have no observed
    # outbound volume. An optional fixed free capacity replaces the utilisation-based
    # value (~0), set after sampling to leave the random stream unchanged.
    no_customer_capacity = net_config.dist_cap.get("no_customer_capacity")
    if no_customer_capacity is not None:
        for node in supply_net.nodes:
            if supply_net.out_degree(node) == 0:
                cap_map[node] = float(no_customer_capacity)

    return cap_map


def set_node_config(
    node_config: dict[NodeID, dict[str, Any]],
    id_node_map: dict[NodeID, ZhaoNode],
    net: nx.DiGraph,
) -> nx.DiGraph:
    """Set a given node configuration to the network."""
    for node_id, attrs in node_config.items():
        # Get node object and set attributes
        node = id_node_map[node_id]
        node.set_scenario(supply_net=net, **attrs)


def sample_edge_config(
    net: nx.DiGraph,
    firms: dict[NodeID, ZhaoNode],
    edge_weight_dist: gaussian_kde,
    product_dists: dict[str, gaussian_kde | int],
    alternative_products: dict[str, dict],
    net_config: "NetworkConfig",
    inv_dists: dict[int, truncnorm] | None = None,
    product_pools: dict[str, int] | None = None,
) -> dict[EdgeID, np.ndarray]:
    """Sample a network configuration scenario.

    In the scenario, each edge is assigned a number of products sampled from the
    distribution of edge weights. Then, a count for each product is sampled from the
    distribution of models per product.
    """
    rng = np.random.default_rng(net_config.network_seed)

    stochastic_edges = [
        (source, sink)
        for source, sink, stochastic in net.edges(data="stochastic")
        if stochastic
    ]

    # Cache portfolios and their lengths for each unique source node
    source_nodes = [edge[0] for edge in stochastic_edges]
    portfolio_cache = {
        source: sorted(firms[source].products) for source in source_nodes
    }

    # Create a NumPy array of portfolio lengths, corresponding to each edge
    portfolio_lengths = np.array([len(portfolio_cache[s]) for s in source_nodes])
    # Prevent sampling from empty portfolios
    portfolio_lengths[portfolio_lengths == 0] = 1

    # Sample the number of products per edge
    num_products = get_product_counts(
        num_edges=len(stochastic_edges),
        portfolio_lengths=portfolio_lengths,
        edge_weight_dist=edge_weight_dist,
        rng=rng,
    )

    # Get specific products for each edge
    edge_products = get_edge_products(
        edges=stochastic_edges,
        product_dists=product_dists,
        alternative_products=alternative_products,
        portfolio_cache=portfolio_cache,
        num_products=num_products,
        rng=rng,
        product_pools=product_pools,
        weight_exponent=net_config.product_weight_exponent,
    )

    # Create Monte Carlo configuration
    scenario = {
        edge: {
            "stochastic": True,
            "removed": False,
            "new": False,
            "weight": sum(edge_products[edge].values()),
            "products": edge_products[edge],
        }
        for edge in stochastic_edges
    }

    # Get maximum trials parameter for each edge
    # This dictionary can contain stochastic and deterministic edges
    if inv_dists is not None:
        max_trials_map, in_inv_ranges, out_inv_ranges = get_max_trials_inventory(
            supply_net=net,
            id_firm_map=firms,
            inv_dists=inv_dists,
            rng=rng,
        )

        # Add to scenario
        for edge, max_trials in max_trials_map.items():
            if edge in scenario:
                scenario[edge]["max_trials"] = int(max_trials)
            else:
                scenario[edge] = {"max_trials": int(max_trials)}
    else:
        # Use static value for maximum trials
        for edge in net.edges:
            if edge in scenario:
                scenario[edge]["max_trials"] = int(net_config.max_trials)
            else:
                scenario[edge] = {"max_trials": int(net_config.max_trials)}

        in_inv_ranges, out_inv_ranges = {}, {}

    return scenario, in_inv_ranges, out_inv_ranges


def get_product_counts(
    num_edges: int,
    portfolio_lengths: list[int],
    edge_weight_dist: gaussian_kde,
    rng: np.random.Generator,
) -> np.ndarray:
    """Get the product count for each edge in the scenario (rejection sampling)."""
    # Sample the number of products for each edge using the KDE
    num_products = np.round(edge_weight_dist.resample(size=num_edges, seed=rng)[0])

    # Find indices where the sampled number is out of bounds
    invalid_idx = np.where((num_products < 1) | (num_products > portfolio_lengths))[0]

    while len(invalid_idx) > 0:
        # Resample only for the invalid edges
        resamples = np.round(
            edge_weight_dist.resample(size=len(invalid_idx), seed=rng)[0],
        )
        num_products[invalid_idx] = resamples

        # Check which of the resampled values are still invalid
        still_invalid = (num_products[invalid_idx] < 1) | (
            num_products[invalid_idx] > portfolio_lengths[invalid_idx]
        )

        # Update invalid_idx to only contain indices that are still invalid
        invalid_idx = invalid_idx[still_invalid]

    return num_products


def get_edge_products(
    edges: list[EdgeID],
    product_dists: dict[str, gaussian_kde | int],
    alternative_products: dict[str, dict],
    portfolio_cache: dict[NodeID, list[str]],
    num_products: np.ndarray,
    rng: np.random.Generator,
    product_pools: dict[str, int] | None = None,
    weight_exponent: float = 0.0,
) -> dict[EdgeID, dict]:
    """Get the specific products for each edge in the scenario.

    Products are drawn from the supplier's portfolio uniformly or, if weight_exponent
    is not 0, with probability proportional to pool^weight_exponent, where pool is
    the number of firms offering a Sigma-similar product (product_pools).
    """
    if weight_exponent and product_pools is None:
        msg = "product_pools are required for a non-zero weight_exponent."
        raise ValueError(msg)
    weight_cache: dict[NodeID, np.ndarray] = {}
    # Sample products for each edge based on the portfolio
    edge_products = {}
    for i, (source, target) in enumerate(edges):
        portfolio = portfolio_cache[source]
        # If the portfolio is empty, assign an empty array
        if not portfolio:
            edge_products[(source, target)] = np.array([])
            continue

        if weight_exponent:
            if source not in weight_cache:
                pools = np.array(
                    [max(product_pools.get(p, 1), 1) for p in portfolio], dtype=float,
                )
                weights = pools**weight_exponent
                weight_cache[source] = weights / weights.sum()
            products = rng.choice(
                portfolio,
                size=int(num_products[i]),
                replace=False,
                p=weight_cache[source],
            )
        else:
            products = rng.choice(
                portfolio,
                size=int(num_products[i]),
                replace=False,
            )

        # Sample counts per product
        product_counts = {}
        for product in products:
            if product in product_dists:
                kde = product_dists[product]
            else:
                # If no distribution is available, get the most similar product
                similar_product = alternative_products[product]["products"]
                kde = product_dists[similar_product]

            # Some distributions are just a single value because all products
            # have the same count, so we can just use that value
            if isinstance(kde, int):
                product_counts[str(product)] = kde
                continue

            # Ensure the count is at least 1
            count = 0
            while count < 1:
                # Sample a count from the KDE
                # (.item() for NumPy >= 2.x, which rejects int() on 1-d arrays)
                count = np.round(kde.resample(size=1, seed=rng)[0].item())

            # Save the count for the product
            product_counts[str(product)] = int(count)

        # Save sample to edge products
        edge_products[(source, target)] = product_counts

    return edge_products


def get_max_trials_inventory(
    supply_net: nx.DiGraph,
    id_firm_map: dict[NodeID, ZhaoNode],
    inv_dists: dict[int, truncnorm],
    rng: np.random.Generator,
) -> tuple[dict[NodeID, int], dict[NodeID, int], dict[NodeID, int]]:
    """Create a mapping of node IDs to maximum trials.

    The maximum trials parameter determines how many attempts a node firm can make to
    find a new supplier in case of a lost supplier. If a node exceeds its maximum trials,
    it will be permanently disrupted.
    """
    # Filter tier-1 suppliers and sinks
    t1_suppliers = [firm for firm in id_firm_map.values() if firm.tier in [0, 1]]

    # Filter tier-2 suppliers
    t2_suppliers = [firm for firm in id_firm_map.values() if firm.tier > 1]

    in_inv_ranges = {}
    out_inv_ranges = {}

    # Iterate over tier-1 suppliers and assign inventory ranges
    t1_in_samples = inv_dists["t1"].rvs(size=len(t1_suppliers), random_state=rng)
    t1_out_samples = inv_dists["t1"].rvs(size=len(t1_suppliers), random_state=rng)

    for i, firm in enumerate(t1_suppliers):
        in_inv_ranges[firm.id] = t1_in_samples[i]
        out_inv_ranges[firm.id] = t1_out_samples[i]

    # Iterate over higher-tier suppliers and assign inventory ranges
    t2_in_samples = inv_dists["t2"].rvs(size=len(t2_suppliers), random_state=rng)
    t2_out_samples = inv_dists["t2"].rvs(size=len(t2_suppliers), random_state=rng)

    for i, firm in enumerate(t2_suppliers):
        in_inv_ranges[firm.id] = t2_in_samples[i]
        out_inv_ranges[firm.id] = t2_out_samples[i]

    # Calculate maximum trials for each node
    max_trials_map = {}
    for source, target in supply_net.edges:
        # Get inventory ranges of both nodes
        in_range = in_inv_ranges[target]
        out_range = out_inv_ranges[source]

        # Calculate maximum trials
        max_trials_map[(source, target)] = np.floor(in_range + out_range)

    return max_trials_map, in_inv_ranges, out_inv_ranges


def load_edge_data(
    data: dict[str, pl.DataFrame],
    edge_product_path: str,
) -> tuple[gaussian_kde, dict[str, gaussian_kde | int], dict[str, dict]]:
    """Load all necessary data for edge sampling."""
    # Create KDE distributions for edge weights and products per edge
    edge_weight_kde, product_kdes = create_dists(data)

    # Create alternative products mapping
    # Required for sampling edge configurations if a product does not exist in the
    # relation details
    edge_alternative_products = load_edge_alternative_products(edge_product_path)

    return edge_weight_kde, product_kdes, edge_alternative_products


def create_inv_dists(net_config: "NetworkConfig") -> dict[int, truncnorm]:
    """Create normal distributions for inventory ranges.

    The distributions can be used to sample days of supply.
    """
    return {
        tier: create_truncnorm(**dist) for tier, dist in net_config.max_trials.items()
    }


def create_dists(data: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create a distribution of edge weights and product counts."""
    # Join SupplyRelations and SupplyRelationDetails
    edge_product_df = data["SupplyRelations"].join(
        data["SupplyRelationDetails"],
        on="relation_id",
        how="inner",
    )

    # Distribution of products per edge
    edge_weight = (
        edge_product_df.group_by(["source_firm_id", "target_firm_id"])
        .agg(pl.count("relation_id").alias("num_products"))
        .sort("num_products", descending=True)
    )

    # Individual distributions of models per product
    product_dists = {}
    for product in edge_product_df["product_name"].unique():
        # Must be sorted; otherwise, the order will be random and the KDE sampling too
        num_products = (
            edge_product_df.filter(pl.col("product_name") == product)
            .group_by("relation_id")
            .agg(
                pl.count("product_name").alias("num_products"),
            )
            .sort("num_products", descending=True)
        )

        # Check if all values are equal; KDE is not defined for a single value
        if num_products["num_products"].n_unique() == 1:
            product_dists[product] = num_products["num_products"][0]
        else:
            product_dists[product] = gaussian_kde(
                num_products["num_products"].to_numpy(),
            )

    return gaussian_kde(edge_weight["num_products"].to_numpy()), product_dists


def load_alternative_products(path: str) -> dict[str, dict]:
    """Load dicts of alternative products for each product and edge."""
    with open(path) as f:
        return json.load(f)


def load_edge_alternative_products(path: str) -> dict[str, dict]:
    """Load dicts of alternative products for each product."""
    with open(path) as f:
        return json.load(f)


def set_edge_config(
    edge_config: dict[EdgeID, dict[str, Any]],
    id_node_map: dict[NodeID, ZhaoNode],
    net: nx.DiGraph,
) -> None:
    """Set a given edge configuration to the network."""
    nx.set_edge_attributes(net, edge_config)

    # Get additional special products from the edge config
    for node, out_degree in net.out_degree:
        if out_degree == 0:
            continue

        add_products = [
            product
            for successor in net.successors(node)
            if not net.edges[node, successor]["stochastic"]
            for product in net.edges[node, successor]["products"]
        ]

        # Add the products to the firm portfolio
        id_node_map[node].set_additional_products(frozenset(add_products))

    return net
