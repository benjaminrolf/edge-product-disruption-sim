"""Configuration for the Zhao-Sim model."""

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from graphsim.types import NodeID


class DistCap(TypedDict):
    """Parameters for the truncated normal distribution for request capacities."""

    strategy: Literal["normal"]
    mu: list[float]
    sigma: list[float]
    lb: float
    ub: float


class DistDisruption(TypedDict):
    """Parameters for the trunc norm for the mean probability of disruption propagation."""

    strategy: Literal["normal"]
    sigma: float
    lb: float
    ub: float


class DistUniform(TypedDict):
    """Parameters for the uniform distribution for the mean probability of disruption propagation."""

    strategy: Literal["uniform"]
    interval_margin: float


class DistTruncNorm(TypedDict):
    """Parameters for the truncated normal distribution for k and inventory."""

    mu: float
    sigma: float
    lb: float
    ub: float


@dataclass
class NetworkConfig:
    """Configuration for the supply network generation."""

    # Request capacity parameters for the truncated normal distribution
    dist_cap: DistCap | DistUniform

    # k parameter for the truncated normal distribution
    dist_k: DistTruncNorm

    # Seed for the random number generator
    network_seed: int = 42

    # Length of a period in days
    period_len: int = 7

    # Inventory parameters (Either distribution or max_trials parameter)
    dist_inventory: dict[int, DistTruncNorm] = None
    max_trials: int = None

    # Edge-product imputation: products are drawn from the supplier's portfolio with
    # probability proportional to pool^gamma (pool: firms offering a Sigma-similar
    # product). 0 draws uniformly; < 0 favours products with fewer substitutes.
    product_weight_exponent: float = 0.0


@dataclass
class SimulationConfig:
    """Configuration for the Zhao simulation models.

    Attributes
    ----------
    disruption_seeds : list[NodeID]
        The seed nodes for the disruption.
    strategy : Literal["none", "reactive", "proactive"]
        The strategy for the disruption propagation. Currently, only "reactive" is
        implemented.
    dist_disruption : DistDisruption | DistUniform
        The distribution for the mean probability of disruption propagation. If
        `strategy` is "uniform", the distribution is uniform; otherwise, it is a
        truncated normal distribution.
    simulation_seed : int
        The seed for the random number generator.
    sample_id : int, optional
        The ID of the experiment sample. It resembles a specific configuration of the
        simulation.
    rep_id : int, optional
        The ID of the replication. It resembles a specific run of the a configuration.
    num_disruption_seeds : int, optional
        The number of disruption seeds. Used to identify the experiment instance.
    tier : int, optional
        The tier of the disruption seeds. Used to identify the experiment instance.
    bin : int, optional
        The bin in the cluster size distribution of the disruption seeds. Used to
        identify the experiment instance.

    """

    # Scenario parameters
    disruption_seeds: list[NodeID]
    strategy: Literal["none", "reactive", "proactive"]

    # Disruption probability parameters for the truncated normal distribution
    dist_disruption: DistDisruption | DistUniform

    # Simulation parameters
    simulation_seed: int = 42

    # Number of parallel requests to alternative suppliers
    num_parallel_requests: int = 1

    # Network config
    network_config: NetworkConfig = None

    # Parameters to identify experiment instances
    network_id: int = None
    sample_id: int = None
    scenario_id: int = None
    rep_id: int = None

    num_disruption_seeds: int = None
    tier: int = None
    bin: int = None


@dataclass
class ExportConfig:
    """Configuration for exporting the simulation results to Excel files."""

    # Node keys to export
    node_keys: list[str] = field(
        default_factory=lambda: [
            "output",
            "size",
            "k",
            "num_original_suppliers",
            "num_suppliers",
            "num_new_suppliers",
            "num_lost_suppliers",
            "num_original_customers",
            "num_customers",
            "num_new_customers",
            "request_cap",
            "remaining_cap",
            "added_cap",
            "removed",
        ],
    )

    # Edge keys to export
    edge_keys: list[str] = field(default_factory=lambda: ["removed", "new", "weight"])

    # Column widths for nodes
    node_column_widths: dict[str, int] = field(
        default_factory=lambda: {
            "id": 150,
            "output": 350,
            "size": 60,
            "k": 60,
            "num_original_suppliers": 60,
            "num_suppliers": 60,
            "num_new_suppliers": 60,
            "num_lost_suppliers": 60,
            "num_original_customers": 60,
            "num_customers": 60,
            "num_new_customers": 60,
            "request_cap": 60,
            "remaining_cap": 60,
            "added_cap": 60,
            "removed": 60,
            "label": 150,
            "country": 150,
            "city": 150,
        },
    )

    # Column widths for edges
    edge_column_widths: dict[str, int] = field(
        default_factory=lambda: {
            "origin": 350,
            "target": 350,
            "removed": 60,
            "new": 60,
            "weight": 60,
        },
    )
