"""Module for importing the Marklines dataset."""

import collections
import itertools
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl
from ordered_set import OrderedSet

from graphsim.types import EdgeID, NodeID
from graphsim.utils import timeit
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.node import ZhaoNode
from graphsim.zhao.search import ImprovedSearch, Search


def load_marklines(
    data_path: str,
    node_cls: ZhaoNode | ImprovedZhaoNode = ZhaoNode,
    search_cls: Search | ImprovedSearch = Search,
) -> tuple[nx.DiGraph, dict[str, pl.DataFrame], dict[NodeID, ZhaoNode]]:
    """Load the new Marklines dataset."""
    # Load dataset and embeddings
    data = parse(data_path)

    # Build networkx graph
    supply_net = build_network(data=data)

    # Set edge attributes
    edge_map = get_edges(data=data, weighted=issubclass(node_cls, ImprovedZhaoNode))
    nx.set_edge_attributes(supply_net, edge_map)

    # Minor fixes to the network to make it simulation-ready
    supply_net = fix_network(supply_net, data)

    # Network must be fixed before creating nodes; no changes to structure after this point
    id_node_map = get_nodes(
        data=data,
        net=supply_net,
        node_cls=node_cls,
        search_cls=search_cls,
    )

    return supply_net, data, id_node_map


def parse(path: str) -> dict[str, pl.DataFrame]:
    """Parse the Marklines dataset and return a dictionary of dataframes."""
    # Iterate over all CSV files in the directory
    data = {}
    for file in Path(path).glob("*.csv"):
        # Read the CSV file into a Polars DataFrame
        df = pl.read_csv(file, infer_schema_length=10000)
        # Store the DataFrame in the dictionary with the filename as the key
        data[file.stem] = df

    return data


def build_network(data: dict[str, pl.DataFrame]) -> nx.DiGraph:
    """Build a networkx graph from the parsed Marklines dataset."""
    # Create a directed networkx graph from supply relations
    net = nx.from_pandas_edgelist(
        df=data["SupplyRelations"].to_pandas(),
        source="source_firm_id",
        target="target_firm_id",
        edge_attr=True,
        create_using=nx.DiGraph,
    )

    # Add firms without relations to the network (given in the Firms table)
    all_firm_ids = data["Firms"]["firm_id"].to_list()
    net.add_nodes_from(all_firm_ids)

    return net


def get_products(
    data: dict[str, pl.DataFrame],
) -> tuple[
    dict[NodeID, frozenset[str]],
    dict[NodeID, frozenset[str]],
    dict[NodeID, frozenset[str]],
    dict[NodeID, frozenset[str]],
]:
    """Create a mapping of node IDs to their products, product groups and families."""
    # Create mapping of nodes to their products
    prod_capa = data["ProductionCapabilities"].join(
        data["Products"],
        on="product_id",
        how="left",
    )
    product_agg = prod_capa.group_by("firm_id").agg(pl.col("product_name"))
    node_product_map = {
        node: frozenset(products)
        for node, products in zip(
            product_agg["firm_id"],
            product_agg["product_name"],
            strict=False,
        )
    }

    # Create mapping of nodes to their product groups
    group_agg = prod_capa.group_by("firm_id").agg(pl.col("group_name"))
    node_group_map = {
        node: frozenset(groups)
        for node, groups in zip(
            group_agg["firm_id"],
            group_agg["group_name"],
            strict=False,
        )
    }

    # Create mapping of nodes to their product families
    family_agg = prod_capa.group_by("firm_id").agg(pl.col("family_name"))
    node_family_map = {
        node: frozenset(families)
        for node, families in zip(
            family_agg["firm_id"],
            family_agg["family_name"],
            strict=False,
        )
    }

    # Create mapping of nodes to their special products
    special_capa = data["SpecialProductionCapabilities"].join(
        data["SpecialProducts"],
        on="special_product_id",
        how="left",
    )
    special_agg = special_capa.group_by("firm_id").agg(pl.col("special_product_name"))
    node_special_map = {
        node: frozenset(specials)
        for node, specials in zip(
            special_agg["firm_id"],
            special_agg["special_product_name"],
            strict=False,
        )
    }

    return node_product_map, node_group_map, node_family_map, node_special_map


def get_tiers(supply_net: nx.DiGraph) -> dict[NodeID, int]:
    """Create a mapping of node IDs to tiers.

    The tiers are determined by the BFS layers of the supply network, starting from
    the sinks (nodes with no outgoing edges). Isolates are assigned to tier -1.
    """
    # Get all sinks that are not isolates: these are mostly OEMs but also suppliers
    # which results from mismatches of the Marklines database and the supply relations
    sinks = [
        node
        for node in supply_net.nodes
        if supply_net.out_degree(node) == 0 and supply_net.in_degree(node) > 0
    ]

    # Create tier dict
    layer_iter = nx.bfs_layers(supply_net.reverse(), sinks)
    tier_node_map = dict(enumerate(layer_iter))

    # Add isolates to the -1 tier
    isolates = [node for node in supply_net.nodes if supply_net.degree(node) == 0]
    tier_node_map[-1] = isolates

    # Create node dict
    return {node: tier for tier, nodes in tier_node_map.items() for node in nodes}


def get_nodes(
    data: dict[str, pl.DataFrame],
    net: nx.DiGraph,
    node_cls: ZhaoNode | ImprovedZhaoNode = ZhaoNode,
    search_cls: Search | ImprovedSearch = Search,
) -> dict[NodeID, dict[str, Any]]:
    """Create ZhaoNode objects from the Marklines dataset."""
    # Get node attributes from the Firms table
    node_base_map = data["Firms"].rows_by_key("firm_id", named=True, unique=True)

    # Create mapping of node IDs to tiers
    node_tier_map = get_tiers(net)

    # Create mapping of node IDs to products, groups, families and special products
    node_product_map, _, _, node_special_map = get_products(data)

    # Set static node attributes (independent of the scenario)
    return {
        node: node_cls(
            identifier=node,
            tier=node_tier_map[node],
            products=node_product_map.get(node, frozenset())
            | node_special_map.get(node, frozenset()),
            # product_groups=node_group_map.get(node, frozenset()),
            # product_families=node_family_map.get(node, frozenset()),
            # special_products=node_special_map.get(node, frozenset()),
            search_cls=search_cls,
            **node_base_map[node],
        )
        for node in net.nodes
    }


def get_edges(
    data: dict[str, pl.DataFrame],
    weighted: bool,
) -> dict[EdgeID, dict[str, Any]]:
    """Create edge attribute dicts from the Marklines dataset.

    Parameters
    ----------
    data : dict[str, pl.DataFrame]
        The Marklines dataset.
    weighted : bool
        Whether to use weighted edges based on product counts. Required for the improved
        model. For the baseline model, weights are 1 by default.

    """
    # Default attributes for all edges (Will be overridden for deterministic edges)
    edge_attr = {
        edge_id: {"stochastic": True, "removed": False, "new": False, "weight": 1}
        for edge_id in data["SupplyRelations"]["relation_id"].to_list()
    }

    # Add edge attributes from the SupplyRelationDetails table (keyed by tuple(source, target))
    relation_details = data["SupplyRelationDetails"].rows_by_key(
        "relation_id",
        named=True,
    )

    # Count the products per edge
    for edge_id, edge_details in relation_details.items():
        product_counts = collections.defaultdict(int)
        models = collections.defaultdict(set)
        for supply in edge_details:
            product_counts[supply["product_name"]] += 1
            models[supply["product_name"]].add(supply["model_name"])

        attr = {
            "stochastic": False,
            "removed": False,
            "new": False,
            "weight": sum(product_counts.values()) if weighted else 1,
            "products": dict(product_counts),
            "models": dict(models),
        }

        edge_attr[edge_id] = attr

    # Replace keys with tuples of node IDs
    relation_ids = data["SupplyRelations"].rows_by_key("relation_id", unique=True)
    return {relation_ids[relation_id]: attr for relation_id, attr in edge_attr.items()}


def fix_network(supply_net: nx.DiGraph, data: dict[str, pl.DataFrame]) -> nx.DiGraph:
    """Fix the network by removing certain nodes and edges.

    1. Remove nodes that are not sinks (out-degree > 0) but have no product information.
    We cannot sample a product scenario for these nodes.
        -> Removes 1 node and 2 edges
    2. Remove isolated nodes that have no products and no special products.
        -> Removes 4,260 nodes and 0 edges
    """
    products, _, _, special_products = get_products(data)

    # Remove empty portfolio nodes
    empty_portfolio_nodes = [
        node
        for node in supply_net.nodes
        if not products.get(node, None)
        and not special_products.get(node, None)
        and supply_net.out_degree(node) > 0
    ]
    num_edges = supply_net.number_of_edges()
    supply_net.remove_nodes_from(empty_portfolio_nodes)

    # Remove isolated nodes that have no products and no special products
    isolated_nodes = [
        node
        for node in supply_net.nodes
        if not products.get(node, None)
        and not special_products.get(node, None)
        and supply_net.degree(node) == 0
    ]
    supply_net.remove_nodes_from(isolated_nodes)
    num_removed_nodes = len(empty_portfolio_nodes) + len(isolated_nodes)
    num_removed_edges = num_edges - supply_net.number_of_edges()

    return supply_net


def load_product_embeddings(
    product_path: str,
    edge_product_path: str,
) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, npt.NDArray[np.float32]]]:
    """Load OpenAI embeddings of products, special products and edge products.

    Embeddings are created with the `text-embedding-3-large` model in advance and have a
    length of 3072 dimensions.
    """
    product_embeddings = {}
    for line in Path(product_path).read_text().splitlines():
        # Parse the JSON line
        json_record = json.loads(line)

        # Extract the product name and embedding
        product_name = json_record["custom_id"]
        embedding = json_record["response"]["body"]["data"][0]["embedding"]

        # Store the embedding in the dictionary
        product_embeddings[product_name] = np.array(embedding, dtype=np.float32)

    edge_product_embeddings = {}
    for line in Path(edge_product_path).read_text().splitlines():
        # Parse the JSON line
        json_record = json.loads(line)

        # Extract the product name and embedding
        product_name = json_record["custom_id"]
        embedding = json_record["response"]["body"]["data"][0]["embedding"]

        # Store the embedding in the dictionary
        edge_product_embeddings[product_name] = np.array(embedding, dtype=np.float32)

    return product_embeddings, edge_product_embeddings


@timeit
def create_comp_hash(
    nodes: dict[NodeID, ZhaoNode],
) -> tuple[dict[str, OrderedSet], dict[str, OrderedSet]]:
    """Create two hierarchical dictionaries that replace the competition network.

    The first level maps products to product combinations that contain the product. The
    second level maps product combinations to nodes that supply them. The product
    overlap as required by the original Zhao model is calculated during runtime.

    Parameters
    ----------
    nodes : dict[NodeID, ZhaoNode]
        Dictionary of node IDs with their ZhaoNode objects.

    """
    # Create a list of unique products and combinations
    combinations = OrderedSet(node.products for node in nodes.values())
    products = OrderedSet(itertools.chain.from_iterable(combinations))

    # Initialize dictionaries for the results
    product_combination_map = collections.defaultdict(OrderedSet)
    combination_node_map = collections.defaultdict(OrderedSet)

    # Create a dictionary of products and product combinations they appear in
    for product in products:
        for combination in combinations:
            if product in combination:
                product_combination_map[product].add(combination)

    # Create a dictionary of product combinations and nodes that supply them
    for node_id, node in nodes.items():
        combination_node_map[node.products].add(node_id)

    return product_combination_map, combination_node_map


def get_delivered_products(net: nx.DiGraph) -> dict[NodeID, frozenset[str]]:
    """Products each supplier is observed to deliver on annotated (non-stochastic) edges."""
    delivered = collections.defaultdict(set)
    for source, _, data in net.edges(data=True):
        if not data["stochastic"]:
            delivered[source].update(data["products"])
    return {firm: frozenset(products) for firm, products in delivered.items()}


def create_comp_hash_new(
    firms: list[ZhaoNode],
    path: str,
    delivered_products: dict[NodeID, frozenset[str]] | None = None,
) -> tuple[dict[str, OrderedSet], dict[str, OrderedSet]]:
    """Create a competition network based on product substitutability.

    A product is substitutable by another product if their embeddings are similar
    (cosine similarity > 0.73). A supplier is therefore substitutable by another
    supplier if it can supply similar products. With delivered_products, a firm can
    also supply every product it is observed to deliver, even if its portfolio does
    not list it.
    """
    # Load alternative products
    with open(path) as f:
        product_product_map = json.load(f)
    product_product_map = {
        product: OrderedSet(
            zip(alternatives["products"], alternatives["scores"], strict=True),
        )
        for product, alternatives in product_product_map.items()
    }

    # Create product to firm mapping
    product_firm_map = collections.defaultdict(OrderedSet)
    for firm in firms:
        for product in firm.products:
            product_firm_map[product].add(firm)

    if delivered_products:
        for firm in firms:
            for product in sorted(delivered_products.get(firm.id, ())):
                product_firm_map[product].add(firm)

    return product_product_map, dict(product_firm_map)
