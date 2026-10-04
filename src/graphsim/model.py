"""Module for creating and running a supply chain simulation model."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from networkx import DiGraph
    from numpy.random import Generator


class SimulationTimeFormatter(logging.Formatter):
    """Custom formatter for logging simulation time.

    Attributes
    ----------
    time_getter : function
        Function to get the current simulation time.

    """

    def __init__(
        self,
        fmt: str = None,
        datefmt: str = None,
        style: str = "%",
        time_getter=None,
    ) -> None:
        """Initialize the formatter with a function to get the simulation time."""
        super().__init__(fmt, datefmt, style)
        self.time_getter = time_getter

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with the current simulation time."""
        if self.time_getter:
            record.time = self.time_getter()
        else:
            record.time = "Unknown Time"
        return super().format(record)


class SimulationModel(ABC):
    """Abstract base class for a supply chain simulation model.

    Attributes
    ----------
    rng : RandomNumberGenerator
        numpy random number generator for managing random states
    supply_net : nx.DiGraph
        networkx graph that represents the supply network
    comp_net : tuple[dict, dict]
        Two-level dict that represents supplier substitutability
    logger : Logger
        event logger for the simulation model
    time : int
        current simulation time

    """

    def __init__(
        self,
        rng: "Generator",
        supply_net: "DiGraph",
        comp_net: tuple[dict, dict],
    ) -> None:
        """Initialize the simulation model."""
        # Static attributes
        self.supply_net = supply_net
        self.comp_net = comp_net
        self.rng = rng

        # Dynamic attributes
        self.time = 0
        self.until = None

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Define the formatter with a function to get the simulation time
        formatter = SimulationTimeFormatter(
            fmt="##### t=%(time)s - %(asctime)s - %(name)s ##### \n%(message)s",
            time_getter=self.get_sim_time,
        )

        # Create a console handler and set the formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(console_handler)

    def run(
        self,
        until: int,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Run the simulation until the specified time.

        Parameters
        ----------
        until : int
            time until the simulation is run
        log : Literal["progress", "brief", "verbose", None], optional
            whether to log the simulation state, by default None.
            - "progress": Log only one line per run.
            - "brief": Log a summary of the simulation state at each time step.
            - "verbose": Log a detailed summary of the simulation state at each step.

        """
        # Run simulation until specified time
        self.until = until
        while self.time < until:
            # Increment simulation time
            self.time += 1

            # Main simulation logic
            self.step(log)

            # Monitoring
            self.snapshot(log)

        # Endsim
        self.terminate(log)

    def get_sim_time(self) -> int:
        """Get the current simulation time."""
        return self.time

    @abstractmethod
    def step(self, log: Literal["progress", "brief", "verbose", None] = None) -> None:
        """Execute one simulation step."""

    @abstractmethod
    def snapshot(
        self,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Save the current state of the simulation."""

    @abstractmethod
    def terminate(
        self,
        log: Literal["progress", "brief", "verbose", None] = None,
    ) -> None:
        """Terminate the simulation.

        Executed when the max simulation time is reached.
        """
