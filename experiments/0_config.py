"""Script to generate configuration files for Monte Carlo simulations."""

import itertools
import json

import numpy as np

from graphsim.networks.marklines import load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

# This list defines the main seeds for the experiment runs.
SEED_RANGE = [1, 2, 3, 4, 5, 10, 20, 50, 75, 100]
NUM_SAMPLES = 10
NUM_NETWORKS = 500
NUM_REPLICATIONS = 1

# Create iterator of all runs
# Use this order to split the high disruption seeds between the parallel runs to save
# memory
runs = itertools.product(
    range(NUM_NETWORKS),
    SEED_RANGE,
    range(NUM_SAMPLES),
    range(NUM_REPLICATIONS),
)

# Set paths
DATA_DIR = paths.DATA_DIR
EMBEDDINGS_DIR = paths.EMBEDDINGS_DIR
RESULTS_DIR = paths.CONFIGS_DIR


# Load seed files
with open(paths.SEEDS_DIR / "rep_seeds.json", encoding="utf-8") as fp:
    rep_seeds = json.load(fp)
with open(paths.SEEDS_DIR / "sample_seeds.json", encoding="utf-8") as fp:
    sample_seeds = json.load(fp)
with open(paths.SEEDS_DIR / "network_seeds.json", encoding="utf-8") as fp:
    network_seeds = json.load(fp)

# Load Marklines network and data
base_net, data, id_firm_map = load_marklines(
    DATA_DIR,
    node_cls=ImprovedZhaoNode,
    search_cls=ImprovedSearch,
)

# Filter possible seeds (no isolates and no sinks)
possible_seeds = [
    node
    for node in base_net.nodes
    if base_net.out_degree(node) > 0 and base_net.degree(node) > 0
]

# Initialize config list
configs = []

# Disruption seed samples. The seed sets of the paper were drawn without a fixed random
# seed and are shipped in seeds/exp2_disruption_seeds.json; set REDRAW = True to draw new ones.
REDRAW = False
if REDRAW:
    seed_rng = np.random.default_rng()
    disruption_seeds = {
        (num_seeds, sample_id): seed_rng.choice(
            possible_seeds,
            size=num_seeds,
            replace=False,
        ).tolist()
        for num_seeds, sample_id in itertools.product(SEED_RANGE, range(NUM_SAMPLES))
    }
else:
    with open(paths.SEEDS_DIR / "exp2_disruption_seeds.json", encoding="utf-8") as fp:
        disruption_seeds = {(s["num_seeds"], s["sample_id"]): s["disruption_seeds"]
                            for s in json.load(fp)}

# Generate configurations for all runs
for i, (network_id, num_seeds, sample_id, rep_id) in enumerate(runs):
    sim_config = {
        "simulation_seed": rep_seeds[rep_id],
        "disruption_seeds": disruption_seeds[(num_seeds, sample_id)],
        "strategy": "reactive",
        "dist_disruption": {
            "strategy": "uniform",
            "interval_margin": 0.1,
        },
        "num_parallel_requests": 3,
        "network_id": network_id,
        "sample_id": sample_id,
        "rep_id": rep_id,
        "num_disruption_seeds": num_seeds,
    }

    PERIOD_LEN = 7
    net_config = {
        "network_seed": network_seeds[network_id],
        "dist_cap": {
            "strategy": "capacity_utilization",
            "mu": 0.75,
            "sigma": 0.1,
            "lb": 0,
            "ub": 1,
        },
        "dist_k": {
            "mu": 1.5,
            "sigma": 0.1,
            "lb": 1,
            "ub": 10000,
        },
        "max_trials": {
            "t1": {
                "mu": 12 / PERIOD_LEN,
                "sigma": 5 / PERIOD_LEN,
                "lb": 0,
                "ub": 10_000,
            },  # Tier-1 suppliers
            "t2": {
                "mu": 35 / PERIOD_LEN,
                "sigma": 5 / PERIOD_LEN,
                "lb": 0,
                "ub": 10_000,
            },  # Tier-2 suppliers
        },
    }

    configs.append(
        {
            "task_id": i,
            "network_config": net_config,
            "simulation_config": sim_config,
        },
    )

# Export to json
with open(RESULTS_DIR / "0_monte_carlo_configs_v2.json", "w", encoding="utf-8") as fp:
    fp.write(json.dumps(configs))

print(f"Successfully generated {len(configs)} configurations.")
