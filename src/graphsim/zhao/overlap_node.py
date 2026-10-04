"""Module for the portfolio-overlap variant of the improved Zhao node model."""

from typing import Any

from ordered_set import OrderedSet

from .improved_node import ImprovedZhaoNode
from .node import ZhaoNode


class OverlapZhaoNode(ImprovedZhaoNode):
    """Improved Zhao node that uses the portfolio-overlap substitution rule.

    Isolates the substitution rule for the Experiment-1 comparison. Everything is
    inherited from ImprovedZhaoNode (one search per lost edge product with its volume,
    capacity model, trial budgets, parallel requests, partner inertia) except the
    candidate pool: for every lost product, the candidates are all firms sharing at
    least one portfolio product with the failed supplier, weighted by the portfolio
    overlap of Zhao et al. (2019) instead of the embedding similarity.

    Attributes
    ----------
    overlap_pool : list[tuple[ZhaoNode, float]] | None
        Firms sharing at least one portfolio product with this node and their overlap
        weight. Computed lazily once per run when the node fails.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize an OverlapZhaoNode instance."""
        super().__init__(*args, **kwargs)
        self.overlap_pool: list[tuple[ZhaoNode, float]] | None = None

    def reset(self) -> None:
        """Reset the dynamic attributes of the node to their initial state."""
        super().reset()

        # The edge scenario may extend the portfolio, so the pool is rebuilt per run
        self.overlap_pool = None

    def get_overlap_pool(self) -> list[tuple[ZhaoNode, float]]:
        """Return the firms sharing at least one portfolio product with this node.

        The weight is the share of this node's portfolio that the firm also supplies,
        as in ZhaoNode.get_candidates.
        """
        if self.overlap_pool is None:
            product_firm_map = self.comp_net[1]
            pool: dict[ZhaoNode, float] = {}

            # Sorted so that the pool order does not depend on string hashing
            for product in sorted(self.products):
                for firm in product_firm_map.get(product, ()):
                    if firm is not self and firm not in pool:
                        pool[firm] = len(firm.products & self.products) / len(
                            self.products,
                        )
            self.overlap_pool = list(pool.items())

        return self.overlap_pool

    def start_search(self, supplier: "OverlapZhaoNode") -> None:
        """Identify alternative nodes to replace the disrupted node.

        Creates the same searches as ImprovedZhaoNode.start_search (one per lost
        product, also when no alternative exists) so that only the candidate pools
        differ.

        Parameters
        ----------
        supplier : OverlapZhaoNode
            The disrupted supplier node.

        """
        # Identify lost products (Values are the sourced amounts)
        lost_products = self.supply_net.edges[(supplier.id, self.id)]["products"]

        valid_node_ids = self.supply_net.nodes
        propagated_nodes = self.sim.propagated
        pool = supplier.get_overlap_pool()

        for product, amount_lost in lost_products.items():
            # The substitute takes over the lost product, so it is kept as edge product
            alternatives = OrderedSet(
                (alt, weight, product)
                for alt, weight in pool
                if alt is not self
                and alt.id in valid_node_ids
                and alt not in propagated_nodes
            )

            search = self.search_cls(
                customer=self,
                alternatives=alternatives,
                lost_supplier=supplier,
                max_trials=self.supply_net.edges[supplier.id, self.id]["max_trials"],
                amount=amount_lost,
                lost_product=product,
            )
            self.searches.append(search)

            # Mark as affected for stats
            self.affected = True
