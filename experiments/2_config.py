"""Script to generate configuration files for Monte Carlo simulations."""

import itertools
import json

import numpy as np

from graphsim.networks.marklines import load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

# This list defines the main seeds for the experiment runs.
NUM_SAMPLES = 30
NUM_NETWORKS = 250
NUM_SCENARIOS = 3
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

# Load earthquake intensity data and firm-prefecture mapping
with open(GEJE_DIR / "tohoku_intensity.json") as f:
    intensity_data = json.load(f)
with open(GEJE_DIR / "japanese_firms_geolocation.json") as f:
    firm_prefecture_map = json.load(f)

# Three scenarios
DISRUPTION_PROBS = {
    "JMA 7": [0.4, 0.5, 0.6],
    "JMA 6+": [0.2, 0.3, 0.4],
    "JMA 6-": [0.2, 0.3, 0.4],
    "JMA 5+": [0.05, 0.1, 0.15],
    "JMA 5-": [0.05, 0.1, 0.15],
}

scenarios = itertools.product(range(NUM_SCENARIOS), range(NUM_SAMPLES))
disruption_seeds = {}
for i, (scenario, sample) in enumerate(scenarios):
    # Use replication seeds because they are not needed in this experiment
    rng = np.random.default_rng(rep_seeds[i])

    disruption_issued = False
    scenario_seeds = []

    for prefecture, intensity in intensity_data.items():
        if intensity not in DISRUPTION_PROBS:
            continue

        prob = DISRUPTION_PROBS[intensity][scenario]

        # Sample disruption seeds
        seeds = rng.choice(
            firm_prefecture_map[prefecture],
            size=int(len(firm_prefecture_map[prefecture]) * prob),
            replace=False,
        ).tolist()
        scenario_seeds.extend(seeds)
        disruption_issued = True

    if disruption_issued:
        disruption_seeds[(scenario, sample)] = scenario_seeds

# Initialize config list
configs = []

# Generate configurations for all runs
for i, (network_id, sample_id, scenario_id, rep_id) in enumerate(runs):
    sim_config = {
        "simulation_seed": rep_seeds[rep_id],
        "disruption_seeds": disruption_seeds[(scenario_id, sample_id)],
        "strategy": "reactive",
        "dist_disruption": {
            "strategy": "uniform",
            "interval_margin": 0.1,
        },
        "num_parallel_requests": 3,
        "network_id": network_id,
        "sample_id": sample_id,
        "scenario_id": scenario_id,
        "rep_id": rep_id,
        "num_disruption_seeds": len(disruption_seeds[(scenario_id, sample_id)]),
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
with open(RESULTS_DIR / "2_geje_configs.json", "w", encoding="utf-8") as fp:
    fp.write(json.dumps(configs))

print(f"Successfully generated {len(configs)} configurations.")
