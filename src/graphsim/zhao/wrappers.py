"""Wrappers to facilitate experiments with the Zhao simulation model."""

from typing import TYPE_CHECKING, Literal

import networkx as nx
from ordered_set import OrderedSet

from graphsim.export import write_excel
from graphsim.types import NodeID
from graphsim.zhao.config import ExportConfig
from graphsim.zhao.model import ZhaoSimulationModel
from graphsim.zhao.monte_carlo import (
    sample_edge_config,
    sample_node_config,
    set_edge_config,
    set_node_config,
)
from graphsim.zhao.node import ZhaoNode
from graphsim.zhao.testing import check_consistency

if TYPE_CHECKING:
    from .config import NetworkConfig, SimulationConfig


def configure_network(
    net: nx.DiGraph,
    id_firm_map: dict[NodeID, ZhaoNode],
    net_config: "NetworkConfig",
    edge_weight_kde,
    product_kdes,
    edge_alternative_products,
    inv_dists=None,
    unit_weights: bool = False,
    product_pools: dict[str, int] | None = None,
) -> tuple[nx.DiGraph, dict[NodeID, ZhaoNode]]:
    """Configure the network by sampling edge and node scenarios.

    With unit_weights, every supply link counts as one unit (baseline ZhaoNode, whose
    requests have amount 1) instead of its sampled product volume.
    """
    # Reset node objects
    for firm in id_firm_map.values():
        firm.reset()

    # Sample a random edge scenario
    edge_config, in_inv_ranges, out_inv_ranges = sample_edge_config(
        net=net,
        firms=id_firm_map,
        edge_weight_dist=edge_weight_kde,
        product_dists=product_kdes,
        alternative_products=edge_alternative_products,
        net_config=net_config,
        inv_dists=inv_dists,
        product_pools=product_pools,
    )
    if unit_weights:
        for edge_data in edge_config.values():
            if "weight" in edge_data:
                edge_data["weight"] = 1

    # AFTER EDGE SCENARIO
    # Sample a random scenario for the nodes
    node_config = sample_node_config(
        net_config=net_config,
        net=net,
        network_seed=net_config.network_seed,
        edge_config=edge_config,
        in_inv_ranges=in_inv_ranges,
        out_inv_ranges=out_inv_ranges,
    )

    # Set node scenario and edge scenarios
    net = set_edge_config(edge_config=edge_config, id_node_map=id_firm_map, net=net)
    # AFTER EDGE SCENARIO because node amounts depend on edge weights
    set_node_config(node_config=node_config, id_node_map=id_firm_map, net=net)

    return net, id_firm_map


def simulate_zhao(
    until: int,
    supply_net: "nx.DiGraph",
    id_firm_map: dict[NodeID, "ZhaoNode"],
    comp_hash: dict[int, OrderedSet["ZhaoNode"]],
    config: "SimulationConfig",
    model_cls: ZhaoSimulationModel = ZhaoSimulationModel,
    log: Literal["progress", "brief", "verbose", None] = None,
    export_path: str | None = None,
    track_time_series: bool = False,
    validate: bool = True,
) -> "ZhaoSimulationModel":
    """Run the Zhao model with the given configuration."""
    # Create simulation model
    sim = model_cls(
        config=config,
        supply_net=supply_net,
        comp_net=comp_hash,
        id_firm_map=id_firm_map,
        track_time_series=track_time_series,
    )

    # Run simulation model
    sim.run(until=until, log=log)

    # Check the consistency of the simulation results
    if validate:
        check_consistency(sim)

    # Export supply network and spreadsheet after simulation
    if export_path:
        export_config = ExportConfig()
        write_excel(
            graph=sim.supply_net.copy(),
            path=f"{export_path}/simulation_info.xlsx",
            config=export_config,
        )

    return sim
