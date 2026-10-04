"""Module for creating and running a supply chain simulation model based on Zhao et al. (2019)."""

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
from ordered_set import OrderedSet
from scipy.stats import truncnorm

from graphsim.model import SimulationModel
from graphsim.types import NodeID

if TYPE_CHECKING:
    from networkx import DiGraph

    from .config import SimulationConfig
    from .node import Search, ZhaoNode


class ZhaoSimulationModel(SimulationModel):
    """Simulation model based on Zhao et al. (2019).

    Attributes
    ----------
    config : SimulationConfig
        Configuration for the Zhao simulation model.
    supply_net : nx.DiGraph
        Networkx graph that represents the supply network.
    comp_net : tuple[dict, dict]
        Competition network (represented as a two-level hash table).
    min_size : int
        Minimum size of a node in the supply network (size measure is selected in
        src.zhao.transform).
    max_size : int
        Maximum size of a node in the supply network (size measure is selected in
        src.zhao.transform).
    vis_net : nx.DiGraph
        Supply network after the simulation with all nodes and edges. Removed nodes and
        edges are marked as removed. Used for visualization purposes.
    stats : dict
        Dictionary containing the stats of the simulation run.
    event_counts : dict
        Counter of daily events. Used for logging purposes.
    events : str
        String to store daily events.
    removed_nodes : list[ZhaoNode]
        List of nodes that have been removed permananently. These are added to the
        vis_net at the end.
    removed_edges : list[(NodeID, NodeID)]
        List of edges that have been removed permanently. These are added to the vis_net
        at the end.
    propagated : OrderedSet[ZhaoNode | ImprovedZhaoNode]
        Ordered set of nodes that the disruption has hit in the current iteration
    disrupted : OrderedSet[ZhaoNode | ImprovedZhaoNode]
        Ordered set of nodes that have been disrupted permanently
    searching : OrderedSet[ZhaoNode | ImprovedZhaoNode]
        Ordered set of nodes that are currently searching for new suppliers
    time : int
        Current simulation time

    """

    def __init__(
        self,
        config: "SimulationConfig",
        supply_net: "DiGraph",
        comp_net: tuple[dict, dict],
        id_firm_map: dict[NodeID, "ZhaoNode"],
        track_time_series: bool = False,
    ) -> None:
        """Initialize the Zhao simulation model."""
        # Create random number generator
        rng = np.random.default_rng(seed=config.simulation_seed)

        # Base attributes
        super().__init__(rng, supply_net, comp_net)

        # Static attributes
        self.config = config
        self.id_firm_map = id_firm_map
        self.firms = list(id_firm_map.values())
        self.min_size = min(node.size for node in self.firms)
        self.max_size = max(node.size for node in self.firms)
        self.stats = defaultdict(list)
        self.track_time_series: bool = track_time_series
        # self.vis_net = None  # Defined in terminate

        # Dynamic attributes
        self.event_counts = defaultdict(int)
        self.events = ""
        # Use sets to make sure that nodes are not added multiple times; sets must be
        # ordered to ensure reproducibility when they are iterated
        self.removed_nodes = []
        self.removed_edges = []
        self.propagated: OrderedSet[ZhaoNode] = OrderedSet(
            node for node in self.firms if node.id in config.disruption_seeds
        )
        self.searching: OrderedSet[ZhaoNode] = OrderedSet()
        self.disrupted: OrderedSet[ZhaoNode] = OrderedSet()
        self.nodes_with_requests: OrderedSet[ZhaoNode] = OrderedSet()

        # Register parent references in nodes
        for node in self.firms:
            node.set_parents(sim=self, rng=rng, comp_net=comp_net)

        # Save initial stats
        self.update_stats()

    def step(self, log: Literal["progress", "brief", "verbose", None] = None) -> None:
        """Execute one simulation step.

        One iteration consists of two simulation steps.
        1. Odd ticks:
            Customers of disrupted nodes send supply requests to alternative suppliers.
        2. Even ticks:
            Alternative suppliers decide whether to accept supply requests.

        Parameters
        ----------
        log : Literal["progress", "brief", "verbose", None], optional
            whether to log the simulation state, by default None.
            - "progress": Log only one line per run.
            - "brief": Log a summary of the simulation state at each time step.
            - "verbose": Log a detailed summary of the simulation state at each step.

        """
        if self.time % 2 == 1:  # Odd tick
            self.handle_propagation()
            self.approach_suppliers(log)
            # Nodes must be removed at the end of the iteration because disruption
            # propagation and approach suppliers depend on the current network state
            self.remove_nodes()
        else:  # Even tick
            self.handle_requests(log)
            self.check_disruption(log)
            self.check_removal(log)

            # Track time series
            if self.track_time_series and self.time != self.until:
                self.update_stats()

    def handle_propagation(self) -> None:
        """Handle the disruption propagation logic.

        First, creates a subgraph for each node that will be removed in this iteration.
        Then, increases the request capacity of suppliers and identifies alternative
        suppliers for customers of removed nodes.
        """
        for node in self.propagated:
            self.release_supplier_capacity(node)

            # Identify alternatives to replace the lost supplier
            for customer_id in self.supply_net.successors(node.id):
                customer = self.id_firm_map[customer_id]

                # Skip nodes that will also be removed in this iteration
                if customer in self.propagated:
                    continue

                self.searching.add(customer)
                customer.start_search(node)

    def release_supplier_capacity(self, node: "ZhaoNode") -> None:
        """Return the volume a removed node took from its suppliers to their capacity."""
        for supplier_id in self.supply_net.predecessors(node.id):
            supplier = self.id_firm_map[supplier_id]

            # Skip nodes that will also be removed in this iteration
            if supplier in self.propagated:
                continue

            supplier.increment_capacity(
                self.supply_net.edges[supplier_id, node.id]["weight"],
            )

    def approach_suppliers(
        self,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Handle the search for alternative suppliers.

        First, get random candidates for each customer and then approach the candidates.

        Parameters
        ----------
        log : Literal["progress", "brief", "verbose", None], optional
            whether to log the simulation state, by default None.
            - "progress": Log only one line per run.
            - "brief": Log a summary of the simulation state at each time step.
            - "verbose": Log a detailed summary of the simulation state at each step.

        """
        # Approach random candidates for each lost supplier
        for customer in self.searching:
            for search in customer.searches:
                # Update search alternatives so that they do not contain removed nodes
                search.update_alternatives(self.propagated)

                # Candidates can be empty if there are no candidates left
                candidates = customer.get_candidates(
                    search,
                    num_candidates=self.config.num_parallel_requests,
                )

                # Add request to request list of candidates
                if candidates:
                    for candidate in candidates:
                        candidate.get_request(search)

                        # Add the candidate to the set of nodes with requests
                        self.nodes_with_requests.add(candidate)

                        match log:
                            case "brief":
                                self.event_counts["approach"] += 1
                            case "verbose":
                                self.log_event("approach", search)

    def remove_nodes(self) -> None:
        """Remove nodes from the network which are hit by a propagation."""
        # Iterate over a copy of the list to avoid modifying the original list
        # Remove nodes from the network and save them for visualization
        for node in self.propagated.copy():
            # Save removed nodes and edges for visualization
            self.removed_nodes.append(node)
            for edge in self.supply_net.in_edges(node.id):
                self.removed_edges.append(edge)
            for edge in self.supply_net.out_edges(node.id):
                self.removed_edges.append(edge)

            # Remove node from the network
            self.propagated.remove(node)
            node.remove(self.time)

    def handle_requests(
        self,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Handle the logic for suppliers accepting or declining requests.

        First, suppliers with open requests are determined. Then, suppliers select which
        requests to accept or decline.


        Parameters
        ----------
        log : Literal["progress", "brief", "verbose", None], optional
            whether to log the simulation state, by default None.
            - "progress": Log only one line per run.
            - "brief": Log a summary of the simulation state at each time step.
            - "verbose": Log a detailed summary of the simulation state at each step.

        """
        # Suppliers accept requests and customers terminate searches
        for supplier in self.nodes_with_requests:
            accept, decline = supplier.select_requests()

            # Accept requests and terminate searches
            for search in accept:
                supplier.accept_request(search)
                search.customer.terminate_search(search)

                # Cancel open requests to other candidates
                if self.config.num_parallel_requests > 1:
                    for candidate in search.candidates:
                        # Request may have been removed if it was declined previously
                        if candidate != supplier and search in candidate.requests:
                            candidate.cancel_request(search)

                # Remove customer from searching list if no more searches are open
                if not search.customer.searches:
                    self.searching.remove(search.customer)

                # Log accepted requests
                match log:
                    case "brief":
                        self.event_counts["accept"] += 1
                    case "verbose":
                        self.log_event("accept", search)

            # Log declined requests
            match log:
                case "brief":
                    self.event_counts["decline"] += len(decline)
                case "verbose":
                    for search in decline:
                        self.log_event("decline", search)

            # Reset requests for the next iteration
            supplier.reset_requests()

        # Clear the set for the next iteration
        self.nodes_with_requests.clear()

    def check_disruption(
        self,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Check disruption of searching customers.

        If searching customers did not find an alternative supplier in this
        iteration, they may become disrupted. They become disrupted if they reached the
        maximum number of trials or have no alternatives left.

        Parameters
        ----------
        log : Literal["progress", "brief", "verbose", None], optional
            whether to log the simulation state, by default None.
            - "progress": Log only one line per run.
            - "brief": Log a summary of the simulation state at each time step.
            - "verbose": Log a detailed summary of the simulation state at each step.

        """
        # Iterates over a copy of the list to avoid modifying the original list
        for customer in self.searching.copy():
            disrupt_flag = False
            disruption_origin: list[ZhaoNode] = []

            # Iterates over a copy of the list to avoid modifying the original list
            for search in customer.searches.copy():
                # Terminate search and disrupt if maximum trials are reached (>= so that
                # a trial budget of 0 ends after the first unsuccessful round)
                if search.trials >= search.max_trials:
                    disrupt_flag = True
                    disruption_origin.append(search.lost_supplier)
                    customer.terminate_search(search)

                    # Log event
                    match log:
                        case "brief":
                            self.event_counts["max_trials"] += 1
                        case "verbose":
                            self.log_event("max_trials", search)
                    continue

                # Terminate search and disrupt if no alternatives are left
                if not search.alternatives:
                    disrupt_flag = True
                    disruption_origin.append(search.lost_supplier)
                    customer.terminate_search(search)

                    # Log event
                    match log:
                        case "brief":
                            self.event_counts["no_alternatives"] += 1
                        case "verbose":
                            self.log_event("no_alternatives", search)
                    continue

            # Disrupt customer if any of the conditions are met
            if disrupt_flag:
                self.disrupted.add(customer)
                customer.disrupt(self.time, disruption_origin)

                # Remove customer from searching list if there are no more searches open
                if not customer.searches:
                    self.searching.remove(customer)

    def check_removal(
        self,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Remove disrupted nodes from the network with a certain probability.

        Parameters
        ----------
        log : Literal["progress", "brief", "verbose", None], optional
            whether to log the simulation state, by default None.
            - "progress": Log only one line per run.
            - "brief": Log a summary of the simulation state at each time step.
            - "verbose": Log a detailed summary of the simulation state at each step.

        """
        # Check if disrupted nodes are removed from the network (iterates over a copy of
        # the list to avoid modifying the original list)
        for node in self.disrupted.copy():
            # Calculate probability of removal
            prob = self.get_removal_probability(node)

            # Propagate with a certain probability
            if self.rng.random() < prob:
                self.disrupted.remove(node)
                self.propagated.add(node)

                # If there are still searches open, remove the node from searching list
                if node in self.searching:
                    self.searching.remove(node)

                # Log events
                match log:
                    case "brief":
                        self.event_counts["removed"] += 1
                    case "verbose":
                        # Event is logged here for simplicity
                        self.events += (
                            f"{node} is removed from the network "
                            f"probability: {prob}).\n"
                        )

    def get_removal_probability(self, node: "ZhaoNode") -> float:
        """Calculate the removal probability for the node.

        Parameters
        ----------
        node : ZhaoNode | ImprovedZhaoNode
            Node for which the removal probability is calculated.

        """
        # Calculate the ratio of lost in-supply
        # For the baseline model, the in amount is in most cases equal to the number of
        # suppliers, except when one edge has a weight > 1
        in_amount = self.supply_net.in_degree(node.id, weight="weight")
        lost_ratio = (node.original_in_amount - in_amount) / node.original_in_amount

        # If no suppliers are lost, the probability of removal is 0
        if lost_ratio == 0:
            return 0

        size = np.log(node.size)
        min_s = np.log(self.min_size)
        max_s = np.log(self.max_size)

        # Calculate the probability of disruption
        prob = 1 - ((size - min_s + 1) / (max_s - min_s + 1)) * (1 - lost_ratio)

        # Add random variation (uniform)
        dist = self.config.dist_disruption
        if dist["strategy"] == "uniform":
            margin = dist["interval_margin"]

            # Make sure that the probability is within the bounds [0, 1]
            low = max(prob - margin, 0)
            high = min(prob + margin, 1)

            return self.rng.uniform(low, high)

        # Add random variation (truncated normal)
        if dist["strategy"] == "normal":
            a = (dist["lb"] - prob) / dist["sigma"]
            b = (dist["ub"] - prob) / dist["sigma"]
            dist = truncnorm(a=a, b=b, loc=prob, scale=dist["sigma"])

            return dist.rvs(random_state=self.rng)
        return None

    def snapshot(
        self,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Log the current state of the simulation.

        Parameters
        ----------
        log : Literal["progress", "brief", "verbose", None], optional
            whether to log the simulation state, by default None.
            - "progress": Log only one line per run.
            - "brief": Log a summary of the simulation state at each time step.
            - "verbose": Log a detailed summary of the simulation state at each step.

        """
        if log == "brief":
            # Create the snapshot message
            snapshot_message = (
                f"Propagations: {len(self.propagated)}\n"
                f"Disrupted nodes: {len(self.disrupted)}\n"
                f"Removed nodes: {len(self.removed_nodes)}\n"
                f"Searching nodes: {len(self.searching)}\n"
                f"Open searches: {sum(len(firm.searches) for firm in self.searching)}\n\n"
            )

            if self.time % 2 == 1:
                snapshot_message += (
                    f"Number of approaches: {self.event_counts['approach']}\n"
                )
            else:
                snapshot_message += (
                    f"Number of accepted requests: {self.event_counts['accept']}\n"
                    f"Number of declined requests: {self.event_counts['decline']}\n"
                    f"Number of maximum trials reached: {self.event_counts['max_trials']}\n"
                    f"Number of no alternatives left: {self.event_counts['no_alternatives']}\n"
                    f"Number of removed nodes: {self.event_counts['removed']}\n"
                )

            # Reset event counts
            self.event_counts = defaultdict(int)

            # Log events
            self.logger.info(snapshot_message)

        elif log == "verbose":
            # Create the snapshot message
            snapshot_message = (
                f"Propagations: {self.propagated}\n"
                f"Disrupted nodes: {self.disrupted}\n"
                f"Removed nodes: {[self.removed_nodes]}\n"
                f"Searching nodes: {self.searching}\n\n"
            )

            # Add events to snapshot message
            if self.events:
                snapshot_message += f"### Events ###\n{self.events}\n"

            # Reset events
            self.events = ""

            # Log events
            self.logger.info(snapshot_message)

    def terminate(
        self,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Execute after the final simulation step and log final statistics."""
        # Firms drawn for removal in the last even tick would otherwise be neither
        # removed nor disrupted in the final statistics. Their suppliers get the freed
        # capacity as in handle_propagation; no searches start after the horizon.
        for node in self.propagated:
            self.release_supplier_capacity(node)
        self.remove_nodes()

        # Calculate final network statistics
        self.update_stats(terminate=True)

        if log == "progress":
            message = f"✅ Simulation terminated (Seed: {self.config.simulation_seed})"
            self.logger.info(message)
            return

        if log in ["brief", "verbose"]:
            # Build Final Statistics Report
            s = self.stats  # Alias for brevity

            header = (
                f"🏁 FINAL SIMULATION REPORT (Sample: {self.config.simulation_seed})"
            )

            # Overall Impact Assessment
            impact_section = (
                f"--- Impact Assessment ---\n"
                f"Initial Disruption: {s['num_seed']} seed node(s)\n"
                f"  - Total Degree: {s['degree_seed']} (In: {s['in_degree_seed']}, Out: {s['out_degree_seed']})\n"
                f"  - Weighted Total Degree: {s['w_degree_seed']} (In: {s['w_in_degree_seed']}, Out: {s['w_out_degree_seed']})\n"
                f"Cascade Effect:\n"
                f"  - Nodes Removed: {s['num_removed'][-1]} (Total Degree: {s['degree_removed'][-1]})\n"
                f"  - Weighted Total Degree: {s['w_degree_removed'][-1]} (In: {s['w_in_degree_removed'][-1]}, Out: {s['w_out_degree_removed'][-1]})\n"
                f"  - Edges Lost: {s['num_removed_edges'][-1]}\n"
                f"Final State:\n"
                f"  - Persistently Disrupted: {s['num_disrupted'][-1]} (Total Degree: {s['degree_disrupted'][-1]})\n"
                f"  - Weighted Total Degree: {s['w_degree_disrupted'][-1]} (In: {s['w_in_degree_disrupted'][-1]}, Out: {s['w_out_degree_disrupted'][-1]})\n"
            )

            # Network Health Indicators
            health_section = (
                f"--- Network Health ---\n"
                f"Connectivity:\n"
                f"  - Weakly Connected Components: {s['num_components'][0]} -> {s['num_components'][-1]}\n"
                f"  - Isolates: {s['num_isolates'][0]} -> {s['num_isolates'][-1]}\n"
                f"Capacity (Total):\n"
                f"  - Before: {s['total_cap'][0]} -> After: {s['total_cap'][-1]}\n"
                f"Capacity (Open):\n"
                f"  - Before: {s['open_cap'][0]} -> After: {s['open_cap'][-1]}\n"
            )

            stats_message = f"\n{header}\n\n{impact_section}\n{health_section}"
            self.logger.info(stats_message)

    def update_stats(self, terminate: bool = False) -> None:
        """Calculate network statistics for the given supply network."""
        # Initialize stats dictionary
        s = self.stats

        def get_node_attributes(node: "ZhaoNode") -> dict[str, Any]:
            return {
                "id": node.id,
                "size": node.size,
                "tier": node.tier,
                "k": node.k,
                # "in_degree": self.supply_net.in_degree(node.id, weight="weight"),
                # "out_degree": self.supply_net.out_degree(node.id, weight="weight"),
                "request_cap": node.request_cap,
                "disruption_time": node.disruption_time,
                "removal_time": node.removal_time,
                "disruption_origin": [n.id for n in node.disruption_origin],
            }

        # Count nodes and edges
        s["num_nodes"].append(len(self.supply_net.nodes))
        s["num_edges"].append(len(self.supply_net.edges))

        # Count searching nodes
        s["num_searching"].append(len(self.searching))
        s["num_open_searches"].append(sum(len(n.searches) for n in self.searching))
        s["cap_searching"].append(
            sum(search.amount for n in self.searching for search in n.searches),
        )

        # Count affected nodes (Nodes that lost at least one supplier)
        s["num_affected"].append(sum(n.affected for n in self.firms))

        # Count removed nodes and calculate degrees
        removed = self.removed_nodes
        s["num_removed"].append(len(removed))
        s["num_removed_edges"].append(len(self.removed_edges))
        in_degree_removed = sum(n.num_original_suppliers for n in removed)
        s["in_degree_removed"].append(in_degree_removed)
        out_degree_removed = sum(n.num_original_customers for n in removed)
        s["out_degree_removed"].append(out_degree_removed)
        s["degree_removed"].append(in_degree_removed + out_degree_removed)
        w_in_degree_removed = sum(n.original_in_amount for n in removed)
        s["w_in_degree_removed"].append(w_in_degree_removed)
        w_out_degree_removed = sum(n.original_out_amount for n in removed)
        s["w_out_degree_removed"].append(w_out_degree_removed)
        s["w_degree_removed"].append(w_in_degree_removed + w_out_degree_removed)

        # Count disrupted nodes and calculate degrees
        s["num_disrupted"].append(len(self.disrupted))
        in_degree_disrupted = sum(n.num_original_suppliers for n in self.disrupted)
        s["in_degree_disrupted"].append(in_degree_disrupted)
        out_degree_disrupted = sum(n.num_original_customers for n in self.disrupted)
        s["out_degree_disrupted"].append(out_degree_disrupted)
        s["degree_disrupted"].append(in_degree_disrupted + out_degree_disrupted)
        w_in_degree_disrupted = sum(n.original_in_amount for n in self.disrupted)
        s["w_in_degree_disrupted"].append(w_in_degree_disrupted)
        w_out_degree_disrupted = sum(n.original_out_amount for n in self.disrupted)
        s["w_out_degree_disrupted"].append(w_out_degree_disrupted)
        s["w_degree_disrupted"].append(w_in_degree_disrupted + w_out_degree_disrupted)

        # Count isolates
        s["num_isolates"].append(nx.number_of_isolates(self.supply_net))

        # Count weakly connected components
        s["num_components"].append(
            nx.number_weakly_connected_components(
                self.supply_net,
            ),
        )

        # Calculate total and open request capacity
        open_cap = sum(n.remaining_cap for n in self.firms if not n.removed)
        used_cap = sum(nx.get_edge_attributes(self.supply_net, "weight").values())
        s["open_cap"].append(open_cap)
        s["edge_cap"].append(used_cap)
        s["total_cap"].append(open_cap + used_cap)

        # These statistics are static and can be computed once at the end
        if terminate:
            t_stats = {}

            # Extract seed nodes and save attributes in stats
            seeds = self.config.disruption_seeds
            t_stats["seeds"] = [get_node_attributes(self.id_firm_map[n]) for n in seeds]

            # Count and calculate degrees for seed nodes
            t_stats["num_seed"] = len(seeds)
            in_degree_seed = sum(
                self.id_firm_map[n].num_original_suppliers for n in seeds
            )
            t_stats["in_degree_seed"] = in_degree_seed
            out_degree_seed = sum(
                self.id_firm_map[n].num_original_customers for n in seeds
            )
            t_stats["out_degree_seed"] = out_degree_seed
            t_stats["degree_seed"] = in_degree_seed + out_degree_seed
            w_in_degree_seed = sum(
                self.id_firm_map[n].original_in_amount for n in seeds
            )
            t_stats["w_in_degree_seed"] = w_in_degree_seed
            w_out_degree_seed = sum(
                self.id_firm_map[n].original_out_amount for n in seeds
            )
            t_stats["w_out_degree_seed"] = w_out_degree_seed
            t_stats["w_degree_seed"] = w_in_degree_seed + w_out_degree_seed

            # Removed nodes
            t_stats["removed"] = [get_node_attributes(n) for n in removed]
            t_stats["disrupted"] = [get_node_attributes(n) for n in self.disrupted]

            # Save sample and replication seeds
            t_stats["replication_seed"] = self.config.simulation_seed

            # Compute diff of all time series statistics
            for metric, time_series in s.items():
                t_stats[f"diff_{metric}"] = np.diff(time_series, prepend=0).tolist()

            # Concat stats and terminal stats
            self.stats = {**self.stats, **t_stats}

    def log_event(
        self,
        event: Literal[
            "approach",
            "accept",
            "decline",
            "max_trials",
            "no_alternatives",
        ],
        search: "Search",
        **kwargs,
    ) -> None:
        """Log an event that occurred during the simulation.

        Parameters
        ----------
        event : str
            Event to log
        search : Search
            Search object that contains the event

        """
        # Approach a supplier
        match event:
            case "approach":
                self.events += (
                    f"{search.customer} approaches {kwargs['candidate']} to replace "
                    f"{search.lost_supplier} for product {search.lost_product}"
                )
                self.events += f" (trial {search.trials}).\n"

            # Accept a request from a customer
            case "accept":
                self.events += (
                    f"{kwargs['candidate']} accepts the request from {search.customer} "
                    f"for product {search.lost_product}.\n"
                )

            # Decline a request from a customer
            case "decline":
                self.events += (
                    f"{kwargs['candidate']} declines the request from {search.customer} "
                    f"for product {search.lost_product}.\n"
                )
                self.events += f"Available capacity: {kwargs['candidate'].remaining_cap}/{search.amount}.\n"

            # Maximum number of trials is reached; Node becomes disrupted
            case "max_trials":
                self.events += (
                    f"{search.customer} becomes disrupted because the maximum number of "
                    f"trials ({search.trials}) for product {search.lost_product} is "
                    f"reached.\n"
                )

            # No alternatives are left; Node becomes disrupted
            case "no_alternatives":
                self.events += (
                    f"{search.customer} becomes disrupted because no alternatives are "
                    f"left for product {search.lost_product}\n"
                )
