"""Script to generate configuration files for Monte Carlo simulations.

Scenarios:
----------
0: Baseline, High disruption (top 10% high-degree nodes), k=N(1.5, 0.1), rem=N
1: Baseline, High disruption (top 10% high-degree nodes), k=N(1.5, 0.1), rem=Unif
2: Baseline, High disruption (top 10% high-degree nodes), k=N(1.5, 0.2), rem=N
3: Baseline, High disruption (top 10% high-degree nodes), k=N(1.5, 0.2), rem=Unif
4: Baseline, Low disruption (1-degree nodes), k=N(1.5, 0.1), rem=N
5: Baseline, Low disruption (1-degree nodes), k=N(1.5, 0.1), rem=Unif
6: Baseline, Low disruption (1-degree nodes), k=N(1.5, 0.2), rem=N
7: Baseline, Low disruption (1-degree nodes), k=N(1.5, 0.2), rem=Unif

"""

import itertools
import json

import numpy as np

from graphsim.networks.marklines import load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

# This list defines the main seeds for the experiment runs.
NUM_SAMPLES = 1000
NUM_NETWORKS = 1
NUM_SCENARIOS = 8
NUM_REPLICATIONS = 1

# Create iterator of all runs
# Use this order to split the high disruption seeds between the parallel runs to save
# memory
runs = itertools.product(
    range(NUM_NETWORKS),
    range(NUM_SAMPLES),
    range(NUM_SCENARIOS),
    range(NUM_REPLICATIONS),
)

# Set paths
DATA_DIR = paths.DATA_DIR
EMBEDDINGS_DIR = paths.EMBEDDINGS_DIR
RESULTS_DIR = paths.CONFIGS_DIR
GEJE_DIR = paths.GEO_DIR


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

# Use replication seeds because they are not needed in this experiment
seed_rng = np.random.default_rng(42)

# Filter possible seeds (no isolates and no sinks)
possible_seeds = [
    node
    for node in base_net.nodes
    if base_net.out_degree(node) > 0 and base_net.degree(node) > 0
]

# Filter top 10% high-degree nodes
percentile = np.percentile([base_net.degree(n) for n in possible_seeds], 90)
high_degree_nodes = [
    node for node in possible_seeds if base_net.degree(node) > percentile
]
high_degree_seeds = seed_rng.choice(
    high_degree_nodes,
    size=NUM_SAMPLES,
    replace=False,
)

# Filter low-degree nodes with degree 1
low_degree_nodes = [node for node in possible_seeds if base_net.degree(node) == 1]
low_degree_seeds = seed_rng.choice(low_degree_nodes, size=NUM_SAMPLES, replace=False)

disruption_seeds = {
    (scenario_id, sample_id): seed_rng.choice(
        high_degree_seeds if scenario_id in [0, 1, 2, 3] else low_degree_seeds,
        size=1,
        replace=False,
    ).tolist()
    for scenario_id, sample_id in itertools.product(
        range(NUM_SCENARIOS),
        range(NUM_SAMPLES),
    )
}

# Initialize config list
configs = []

# Generate configurations for all runs
for i, (network_id, sample_id, scenario_id, rep_id) in enumerate(runs):
    if scenario_id in [0, 2, 4, 6]:
        dist_disruption = {
            "strategy": "normal",
            "sigma": 0.1,
            "lb": 0,
            "ub": 10_000,
        }
    else:
        dist_disruption = {
            "strategy": "uniform",
            "interval_margin": 0.1,
        }

    sim_config = {
        "simulation_seed": rep_seeds[rep_id],
        "disruption_seeds": disruption_seeds[(scenario_id, sample_id)],
        "strategy": "reactive",
        "dist_disruption": dist_disruption,
        "max_trials": 10,
        "network_id": network_id,
        "sample_id": sample_id,
        "scenario_id": scenario_id,
        "rep_id": rep_id,
        "num_disruption_seeds": 1,
    }

    if scenario_id in [0, 1, 4, 5]:  # Low k
        dist_k = {
            "mu": 1.5,
            "sigma": 0.1,
            "lb": 1,
            "ub": 10_000,
        }
    else:  # High k
        dist_k = {
            "mu": 1.5,
            "sigma": 0.2,
            "lb": 1,
            "ub": 10_000,
        }

    # Always use normal distribution for the baseline model
    dist_cap = {
        "strategy": "normal",
        "mu": [2, 4, 6],
        "sigma": [0.5, 1, 2],
        "lb": 0,
        "ub": 10_000,
    }

    net_config = {
        "network_seed": network_seeds[network_id],
        "dist_cap": dist_cap,
        "dist_k": dist_k,
    }

    configs.append(
        {
            "task_id": i,
            "network_config": net_config,
            "simulation_config": sim_config,
        },
    )

# Export to json
with open(RESULTS_DIR / "1_comp_configs_baseline.json", "w", encoding="utf-8") as fp:
    fp.write(json.dumps(configs))

print(f"Successfully generated {len(configs)} configurations.")
