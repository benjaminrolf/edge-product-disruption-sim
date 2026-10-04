"""Module containing small tests to check the consistency of the simulation model."""

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from graphsim.model import SimulationModel
    from graphsim.zhao.node import ZhaoNode


def check_consistency(sim: "SimulationModel") -> None:
    """Check the consistency of the simulation model after the simulation run.

    Parameters
    ----------
    sim : SimulationModel
        the simulation model to check

    """
    for node_id in sim.supply_net.nodes:
        node: ZhaoNode = sim.id_firm_map[node_id]

        # Test if searches and requests are empty
        # assert not node.searches, f"Searches are not empty for node {node.id}"
        # assert not node.requests, f"Requests are not empty for node {node.id}"

        # Test if request capacity is positive
        assert node.remaining_cap >= 0, (
            f"Request capacity is negative for node {node.id}"
        )

        # Test if the outgoing amount is consistent
        curr_out_amount = node.supply_net.out_degree(node.id, weight="weight")
        node2 = node.supply_net.nodes[node.id]
        node3 = node.sim.supply_net.nodes[node.id]
        pred = [sim.id_firm_map[x] for x in node.supply_net.predecessors(node.id)]
        succ = [sim.id_firm_map[x] for x in node.supply_net.successors(node.id)]
        out_edges = node.supply_net.out_edges(node.id)
        in_edges = node.supply_net.in_edges(node.id)
        out_degree = node.supply_net.out_degree(node.id)
        in_degree = node.supply_net.in_degree(node.id)
        assert (
            node.original_out_amount + node.request_cap
            == curr_out_amount + node.remaining_cap
        ), (
            f"Outgoing amount is inconsistent for node {node.id} "
            f"({node.original_out_amount} + {node.request_cap} + {node.added_cap} "
            f"!= {curr_out_amount} + {node.remaining_cap})"
        )
