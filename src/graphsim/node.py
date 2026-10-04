"""Node class to represent nodes in the supply chain."""

from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from networkx import DiGraph
    from numpy.random import Generator

    from .model import SimulationModel


class Node(ABC):
    """Base class to represent nodes in the supply chain.

    Attributes
    ----------
    id : Any
        ID of the entity in the supply chain
    output : str
        Output that the node produces
    sim : SimulationModel
        Parent simulation model
    rng : np.Generator
        Random number generator
    supply_net : nx.DiGraph
        Supply network
    comp_net : nx.MultiDiGraph | CompNet
        Competition network (represented as a directed graph or a two-level hash table)

    """

    def __init__(
        self,
        identifier: int,
        products: list[str] | None = None,
        sim: "SimulationModel" = None,
        rng: "Generator" = None,
        supply_net: "DiGraph" = None,
        comp_net: tuple[dict, dict] | None = None,
    ) -> None:
        """Initialize the Node with its identifier and optional attributes."""
        # Static attributes
        self.id = identifier
        self.sim = sim
        self.rng = rng
        self.supply_net = supply_net
        self.comp_net = comp_net
        self.products = products
        self.init_products = products

    def __repr__(self) -> str:
        """Return a string representation of the Node."""
        return f"Node({self.id})"

    def set_parents(self, **kwargs: dict[str, Any]) -> None:
        """Set the parent objects.

        These are not set during the initialization because the model must be created
        first.
        """
        for arg, value in kwargs.items():
            setattr(self, arg, value)
