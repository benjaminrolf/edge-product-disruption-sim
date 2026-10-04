"""Module to define the Search class."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ordered_set import OrderedSet

if TYPE_CHECKING:
    from .node import ZhaoNode


# eq=False: searches are compared by identity (list.remove, `in`), not field by field
@dataclass(eq=False)
class Search:
    """A class to represent the search for alternative suppliers.

    Attributes
    ----------
    customer : ZhaoNode | ImprovedZhaoNode
        the customer node requesting a new supplier
    candidates : list[ZhaoNode | ImprovedZhaoNode]
        the candidate nodes selected to approach
    alternatives : OrderedSet[ZhaoNode | ImprovedZhaoNode]
        the candidate nodes to replace a disrupted supplier
    lost_supplier : ZhaoNode | ImprovedZhaoNode
        the disrupted supplier to replace
    amount : int
        the amount of goods to be sourced from the supplier
    max_trials : int
        the maximum number of trials to replace the lost supplier. Depends on the
        supplier for improved model
    lost_product : str | None
        the product that was lost from the disrupted supplier (one product per search)
    alt_product : str | None
        the alternative product being sourced from the selected supplier
    trials : int
        the number of candidates approached without success

    """

    # Static attributes
    customer: "ZhaoNode"
    alternatives: OrderedSet["ZhaoNode"]
    lost_supplier: "ZhaoNode"
    max_trials: int
    amount: int
    lost_product: str | None = None  # Only for improved model
    alt_products: list[str] | None = None  # Only for improved model

    # Dynamic attributes
    candidates: list["ZhaoNode"] | None = None
    trials: int = 0

    def update_alternatives(self, propagated: OrderedSet["ZhaoNode"]) -> None:
        """Update the list of alternative nodes to replace the disrupted supplier."""
        # Check whether all alternatives are still in the network and do not propagate
        # in this iteration (because nodes are removed from the network after
        # approaching alternatives)
        nodes = self.customer.supply_net.nodes
        self.alternatives = OrderedSet(
            alt
            for alt in self.alternatives
            if alt.id in nodes and alt not in propagated
        )


@dataclass(eq=False)
class ImprovedSearch(Search):
    """An improved version of the Search class with additional features."""

    def update_alternatives(self, propagated: OrderedSet["ZhaoNode"]) -> None:
        """Update the list of alternative nodes to replace the disrupted supplier.

        Alternative is now a tuple with (node, score, alternative_product).
        """
        # Check whether all alternatives are still in the network and do not propagate
        # in this iteration (because nodes are removed from the network after
        # approaching alternatives)
        nodes = self.customer.supply_net.nodes

        self.alternatives = OrderedSet(
            alt
            for alt in self.alternatives
            if alt[0].id in nodes and alt[0] not in propagated
        )
