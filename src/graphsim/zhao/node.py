"""Module for the Node class for the simulation model based on Zhao et al. (2019)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from ordered_set import OrderedSet

from graphsim.node import Node
from graphsim.zhao.search import ImprovedSearch, Search

if TYPE_CHECKING:
    from graphsim.types import NodeID


class ZhaoNode(Node):
    """A class to represent a node based on the paper of Zhao et al. (2019).

    Attributes
    ----------
    size : int
        the size of the node
    k : float
        the preference parameter for current partners
    num_original_suppliers : int
        The original number of suppliers at the beginning of the simulation. Each
        predecessor is counted as one supplier because all edge weights are 1.
    num_original_customers : int
        The original number of customers at the beginning of the simulation. Each
        successor is counted as one customer because all edge weights are 1.
    request_cap : int
        The maximum number of additional supply requests the node can accept.
    cluster : int
        The cluster the node belongs to. Used to identify alternatives for supplier search.
    tier : int
        The tier of the node. Based on a breadth-first layer search from all sink nodes.
        -1 indicates an isolate node.
    searches : list[Search]
        The list of ongoing supplier searches.
    requests : list[Search]
        The list of open supply requests received by potential customers.
    removed : bool
        Whether the node is removed from the network.
    num_suppliers : int
        The current number of suppliers. Changes dynamically during the simulation run.
        Can be different from the in-degree if an edge has a weight greater than 1.
    num_customers : int
        The current number of customers. Changes dynamically during the simulation run.
        Can be different from the out-degree if an edge has a weight greater than 1.
    num_lost_suppliers : int
        The number of lost suppliers. Increases when a supplier is replaced.
    num_new_suppliers : int
        The number of new suppliers. Increases when a new supplier is added.
    remaining_cap : int
        The remaining request capacity. Decreases when a request is accepted.
    added_cap : int
        The additionally added request capacity due to lost customers.
    disruption_time : int
        The time step when the node was disrupted. For stats only.
    removal_time : int
        The time step when the node was removed from the network. For stats only.
    sim : SimulationModel
        Parent simulation model

    """

    def __init__(
        self,
        identifier: NodeID,
        products: list[str],
        tier: int,
        search_cls: Search | ImprovedSearch = Search,
        **kwargs: dict,
    ) -> None:
        """Initialize a ZhaoNode instance."""
        # Base attributes
        super().__init__(identifier=identifier, products=products)

        # Static attributes
        self.search_cls = search_cls
        self.tier = tier
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Dynamic attributes
        self.searches: list[Search] = []
        self.requests: list[Search] = []
        self.removed = False
        self.affected = False
        self.added_cap = 0
        self.disruption_time: int = None
        self.removal_time: int = None
        self.disruption_origin: list[ZhaoNode] = []

        # Set in set_scenario
        self.supply_net: nx.DiGraph | None = None
        self.request_cap: int | None = None
        self.size: int | None = None
        self.k: float | None = None
        self.remaining_cap: int | None = None
        self.num_original_suppliers: int | None = None
        self.num_original_customers: int | None = None
        self.original_in_amount: int | None = None
        self.original_out_amount: int | None = None
        self.in_inv_range = None
        self.out_inv_range = None

    def set_scenario(
        self,
        supply_net: nx.DiGraph,
        request_cap: int,
        size: int,
        k: float,
        in_inv_range: int,
        out_inv_range: int,
    ) -> None:
        """Set a scenario for supply_net, request_cap, size, k and inv_range.

        This function is used to test different scenarios in the Monte Carlo simulation.
        Network must be set here too because it is copied and then modified for Monte
        Carlo simulation. in_inv_range and out_inv_range are required to determine
        max_trials.
        """
        # Scenario attributes
        self.supply_net = supply_net
        self.request_cap = request_cap
        self.size = size
        self.k = k
        self.in_inv_range = in_inv_range
        self.out_inv_range = out_inv_range

        # Some attributes can only be set when the network is fully initialized
        self.remaining_cap = request_cap
        self.num_original_suppliers = self.supply_net.in_degree(self.id)
        self.num_original_customers = self.supply_net.out_degree(self.id)
        self.original_in_amount = self.supply_net.in_degree(self.id, weight="weight")
        self.original_out_amount = self.supply_net.out_degree(self.id, weight="weight")

    def set_additional_products(self, products: frozenset[str]) -> None:
        """Extend the product portfolio of the node.

        Required to add new products resulting from edge configurations to the node.
        Sometimes, the edges have a slightly different product name space than the
        existing products and special products. However, these are usually only minor
        adjustments (e.g. LCD meter and Full LCD meter).
        """
        self.products = self.products.union(products)

    def reset(self) -> None:
        """Reset the dynamic attributes of the node to their initial state."""
        self.searches = []
        self.requests = []
        self.removed = False
        self.affected = False
        self.remaining_cap = self.request_cap
        self.added_cap = 0
        self.disruption_time = None
        self.removal_time = None
        self.disruption_origin = []

        # Reset product portfolio because the edge scenario may have added products
        self.products = self.init_products

    def select_requests(self) -> tuple[list[Search], list[Search]]:
        """Select supply requests to accept depending on the remaining request capacity.

        Current partners are preferred over other nodes. Large firms are more likely to
        be selected.

        Returns
        -------
        tuple[list[Search], list[Search]]
            The list of accepted and declined supply requests.

        """
        # Split partners and non-partners
        neighbors = OrderedSet(nx.all_neighbors(self.supply_net, self.id))
        # neighbors holds node IDs, so compare the customer's ID (not the node object)
        partners = [s for s in self.requests if s.customer.id in neighbors]
        others = [s for s in self.requests if s.customer.id not in neighbors]

        # Initialize variables for selection
        accept, decline = [], []
        remaining_cap = self.remaining_cap

        # Sequentially process partners first and then others
        for searches in (partners, others):
            # Continue if list is empty
            if not searches:
                continue

            # Decline all requests and continue if no remaining capacity
            if remaining_cap <= 0:
                decline.extend(searches)
                continue

            # Calculate selection probabilities based on firm size
            total_size = sum(search.customer.size for search in searches)
            prob = [search.customer.size / total_size for search in searches]

            # Determine how many requests can be accepted
            # (int() floors float capacities, e.g. from the capacity-utilization
            # model, and satisfies NumPy 2.x's integer `size` requirement)
            cap = int(min(remaining_cap, len(searches)))

            # Use random sampling to accept requests
            choice = self.rng.choice(searches, size=cap, p=prob, replace=False)
            accept.extend(choice)
            decline.extend([search for search in searches if search not in choice])

            # Update remaining capacity
            remaining_cap -= cap

        return accept, decline

    def get_request(self, search: Search) -> None:
        """Receive a supply request from a customer.

        Parameters
        ----------
        search : Search
            the customer node to receive the request from

        """
        self.requests.append(search)

    def reset_requests(self) -> None:
        """Reset the request list."""
        self.requests = []

    def cancel_request(self, search: Search) -> None:
        """Cancel an open request."""
        self.requests.remove(search)

    def accept_request(self, search: Search) -> None:
        """Accept a supply request from a customer.

        Parameters
        ----------
        search : Search
            the customer node to accept the request from

        """
        # Check if edge already exists; if so, increase weight, otherwise add new edge
        if search.customer.id in list(self.supply_net.successors(self.id)):
            self.supply_net.edges[self.id, search.customer.id]["weight"] += (
                search.amount
            )
            # Mark as new because the amount has changed
            self.supply_net.edges[self.id, search.customer.id]["new"] = True
        else:
            # Is this edge really stochastic if it has been added?
            self.supply_net.add_edge(
                self.id,
                search.customer.id,
                weight=search.amount,
                stochastic=True,
                new=True,
                removed=False,
                max_trials=search.max_trials,
            )

        # Decrease remaining capacity
        self.remaining_cap -= search.amount

    def start_search(self, supplier: ZhaoNode) -> None:
        """Identify alternative nodes to replace the disrupted node.

        Parameters
        ----------
        supplier : ZhaoNode | ImprovedZhaoNode
            the disrupted supplier node

        """
        # Identify alternatives (hash method)
        # Initialize as an ordered set instead of a list to enable faster membership
        # checking. Ensure that the alternative is not the current node and is still in
        # the network and will not be removed in this iteration
        alternatives = OrderedSet(
            self.sim.id_firm_map[alternative]
            for product in sorted(supplier.products)
            for combination in self.comp_net[0][product]
            for alternative in self.comp_net[1][combination]
            if self.sim.id_firm_map[alternative] != self
            and alternative in self.supply_net.nodes
            and self.sim.id_firm_map[alternative] not in self.sim.propagated
        )

        # Initialize search for alternative supplier
        search = self.search_cls(
            customer=self,
            alternatives=alternatives,
            lost_supplier=supplier,
            max_trials=self.supply_net.edges[supplier.id, self.id]["max_trials"],
            amount=1,  # Always 1 for baseline model
        )
        self.searches.append(search)

        self.affected = True

    def get_candidates(
        self,
        search: Search,
        num_candidates: int = 1,
    ) -> list[ZhaoNode] | None:
        """Select candidate nodes to approach.

        Parameters
        ----------
        search : Search
            the search object to find a candidate for

        """
        # Check whether there are alternative suppliers (edge case for unique products)
        if not search.alternatives:
            return []

        # Get edge weights (product overlap with searching node)
        num_products = len(search.lost_supplier.products)
        weights = {
            alt: len(alt.products.intersection(search.lost_supplier.products))
            / num_products
            for alt in search.alternatives
        }

        # Calculate k values for alternatives
        k_values = dict.fromkeys(nx.all_neighbors(self.supply_net, self.id), self.k)

        # Calculate total edge weight (k_values is keyed by node ID, not node object)
        total_edge_weight = sum(
            weight * k_values.get(alt.id, 1) for alt, weight in weights.items()
        )

        # Calculate probabilities to approach alternatives
        probs = [
            (weights[alt] * k_values.get(alt.id, 1)) / total_edge_weight
            for alt in search.alternatives
        ]

        # Select alternative to approach
        search.candidates = self.rng.choice(
            a=search.alternatives,
            p=probs,
            size=min(num_candidates, len(search.alternatives)),
            replace=False,
        ).tolist()

        search.alternatives.difference_update(search.candidates)
        search.trials += 1

        return search.candidates

    def terminate_search(self, search: Search) -> None:
        """Terminate supplier search process.

        Parameters
        ----------
        search : Search
            the search to terminate
        supplier : ZhaoNode
            the selected new supplier

        """
        # Remove search from active searches
        self.searches.remove(search)

    def increment_capacity(self, weight: int) -> None:
        """Increase the remaining capacity.

        Parameters
        ----------
        weight : int
            the weight to increase the capacity by. May be more than 1 if the lost
            customer had replaced a previously lost supplier with the current supplier.

        """
        # Does not increase the request capacity because to still show the original
        # request capacity in the simulation results
        self.remaining_cap += weight
        self.added_cap += weight

    def disrupt(self, time: int, disruption_origin: list[ZhaoNode]) -> None:
        """Save the time step in which the node was disrupted.

        Parameters
        ----------
        time : int
            the time step when the node was disrupted
        disruption_origin : list[ZhaoNode]
            the nodes that caused the disruption

        """
        # Keep the first disruption time; later failed searches add further origins
        if self.disruption_time is None:
            self.disruption_time = time

        self.disruption_origin = self.disruption_origin + [
            node for node in disruption_origin if node not in self.disruption_origin
        ]

    def remove(self, time: int) -> None:
        """Remove the node from the network and save the time step.

        Parameters
        ----------
        time : int
            the time step when the node was removed

        """
        # Mark as removed
        self.removed = True

        # Save removal time
        self.removal_time = time

        # Remove from network
        self.supply_net.remove_node(self.id)
