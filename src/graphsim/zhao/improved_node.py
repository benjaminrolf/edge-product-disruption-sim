"""Module for the improved Zhao node model."""

import networkx as nx
import numpy as np
from ordered_set import OrderedSet

from .node import Search, ZhaoNode


class ImprovedZhaoNode(ZhaoNode):
    """An improved version of the Zhao node model. Overrides functions."""

    def start_search(self, supplier: "ZhaoNode") -> None:
        """Identify alternative nodes to replace the disrupted node.

        If an alternative firm supplies more than one of the lost products, the highest
        score is used.

        Parameters
        ----------
        supplier : ZhaoNode
            The disrupted supplier node.

        """
        # Identify lost products (Values are the sourced amounts)
        lost_products = self.supply_net.edges[(supplier.id, self.id)]["products"]

        # Pre-fetch node sets for faster lookups inside the loop
        valid_node_ids = self.supply_net.nodes
        propagated_nodes = self.sim.propagated

        for product, amount_lost in lost_products.items():
            # Use a dictionary to store the alternative with the highest score.
            best_alternatives = {}

            # Find all potential alternatives and their scores (a product missing from
            # the comp_net has no substitutes)
            for alternative_product, score in self.comp_net[0].get(product, ()):
                # Gracefully handle cases where the alt_product might not have alternatives
                if alternative_product not in self.comp_net[1]:
                    continue

                for alt in self.comp_net[1][alternative_product]:
                    # Filter out invalid or already processed alternatives
                    if (
                        alt == self
                        or alt.id not in valid_node_ids
                        or alt in propagated_nodes
                    ):
                        continue

                    # If the alternative is new or has a better score, update it.
                    if (
                        alt not in best_alternatives
                        or score > best_alternatives[alt][0]
                    ):
                        best_alternatives[alt] = (score, alternative_product)

            # A search is opened even without alternatives: check_disruption then
            # disrupts the customer, as in ZhaoNode (empty candidate set C(p)).

            # Convert the dictionary to an OrderedSet of (alternative, score) tuples.
            # Dictionaries preserve insertion order (Python 3.7+), so the
            # order will be based on when an alternative was first added or updated.
            final_alternatives = OrderedSet(
                (alt, score, alt_product)
                for alt, (score, alt_product) in best_alternatives.items()
            )

            # Initialize the search with the list of alternatives
            search = self.search_cls(
                customer=self,
                alternatives=final_alternatives,
                lost_supplier=supplier,
                max_trials=self.supply_net.edges[supplier.id, self.id]["max_trials"],
                amount=amount_lost,
                lost_product=product,
            )
            self.searches.append(search)

            # Mark as affected for stats
            self.affected = True

    def select_requests(self) -> tuple[list[Search], list[Search]]:
        """Select supply requests to accept depending on the remaining request capacity.

        Current partners are preferred over other nodes. Large firms are more likely to
        be selected. Requests now have varying amounts that consume capacity.

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

            # Filter out requests that exceed remaining capacity
            feasible_searches = [
                search for search in searches if search.amount <= remaining_cap
            ]
            infeasible_searches = [
                search for search in searches if search.amount > remaining_cap
            ]

            # Immediately decline infeasible requests
            decline.extend(infeasible_searches)

            # Continue if no feasible requests remain
            if not feasible_searches:
                continue

            # Calculate selection probabilities based on firm size for feasible requests
            total_size = sum(search.customer.size for search in feasible_searches)
            prob = [search.customer.size / total_size for search in feasible_searches]

            # Use iterative probabilistic selection to respect capacity constraints
            remaining_searches = feasible_searches.copy()
            current_prob = prob.copy()

            while remaining_searches and remaining_cap > 0:
                # Filter searches that still fit in remaining capacity
                fitting_searches = [
                    search
                    for search in remaining_searches
                    if search.amount <= remaining_cap
                ]

                if not fitting_searches:
                    break

                # Get probabilities for fitting searches
                fitting_indices = [
                    remaining_searches.index(search) for search in fitting_searches
                ]
                fitting_prob = [current_prob[i] for i in fitting_indices]

                # Normalize probabilities
                prob_sum = sum(fitting_prob)
                if prob_sum > 0:
                    fitting_prob = [p / prob_sum for p in fitting_prob]
                else:
                    fitting_prob = [1.0 / len(fitting_searches)] * len(fitting_searches)

                # Select one request based on probability
                selected_search = self.rng.choice(fitting_searches, p=fitting_prob)

                # Accept the selected request
                accept.append(selected_search)
                remaining_cap -= selected_search.amount

                # Remove selected search from remaining options
                selected_index = remaining_searches.index(selected_search)
                remaining_searches.pop(selected_index)
                current_prob.pop(selected_index)

            # Decline all remaining searches
            decline.extend(remaining_searches)

        return accept, decline

    def get_candidates(self, search: Search, num_candidates: int) -> ZhaoNode | None:
        """Select a candidate node to approach.

        Parameters
        ----------
        search : Search
            the search object to find a candidate for

        """
        # Check whether there are alternative suppliers (edge case for unique products)
        if not search.alternatives:
            return None

        # Get cosine similarity scores
        scores = {alt: alt[1] for alt in search.alternatives}

        # Get k values for existing customers and suppliers (partner preference)
        k_values = dict.fromkeys(nx.all_neighbors(self.supply_net, self.id), self.k)

        # Calculate total edge weight
        total_edge_weight = sum(
            weight * k_values.get(cand[0].id, 1) for cand, weight in scores.items()
        )

        # Calculate probabilities to approach alternatives
        probs = [
            (scores[alt] * k_values.get(alt[0].id, 1)) / total_edge_weight
            for alt in search.alternatives
        ]

        # Select candidates to approach
        candidates = self.rng.choice(
            a=search.alternatives,
            p=probs,
            size=min(num_candidates, len(search.alternatives)),
            replace=False,
        ).tolist()
        search.candidates = [cand[0] for cand in candidates]
        search.alt_products = [cand[2] for cand in candidates]
        for candidate in candidates:
            search.alternatives.remove(tuple(candidate))
        search.trials += 1

        return search.candidates

    def accept_request(self, search: Search) -> None:
        """Accept a supply request from a customer.

        The Function now also updates the detailed edge product information.

        Parameters
        ----------
        search : Search
            the customer node to accept the request from

        """
        # Get the alternative product
        idx = search.candidates.index(self)
        alt_product = search.alt_products[idx]

        # Check if edge already exists; if so, increase weight, otherwise add new edge
        if self.supply_net.has_edge(self.id, search.customer.id):
            edge_data = self.supply_net.edges[self.id, search.customer.id]
            edge_data["weight"] += search.amount

            # Update detailed supply information on a copy: graph.copy() is shallow, so
            # the products dict of an observed edge is shared with the base network
            edge_products = dict(edge_data["products"])
            edge_products[alt_product] = (
                edge_products.get(alt_product, 0) + search.amount
            )
            edge_data["products"] = edge_products

            # Mark as new because the amount has changed
            edge_data["new"] = True
        else:
            # Is this edge really stochastic if it has been added?
            self.supply_net.add_edge(
                self.id,
                search.customer.id,
                relation_id=None,  # New edges have no ID
                stochastic=True,
                removed=False,
                new=True,
                weight=search.amount,
                products={alt_product: search.amount},
                # floor as for the original edges in get_max_trials_inventory
                max_trials=np.floor(self.out_inv_range + search.customer.in_inv_range),
            )

        # Decrease remaining capacity
        self.remaining_cap -= search.amount
