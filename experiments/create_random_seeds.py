"""Script to create random seeds for replications and samples.

Pre-created seeds ensure that the experiments are reproducible.
"""

import json
import secrets
from graphsim import paths

SEED_RANGE = 1000000

# Get paths
DATA_DIR = paths.DATA_DIR
RESULTS_DIR = paths.CONFIGS_DIR


# Create random seeds for network generation
network_seeds = [secrets.randbits(132) for _ in range(SEED_RANGE)]
with open(paths.SEEDS_DIR / "network_seeds.json", "w", encoding="utf-8") as f:
    json.dump(network_seeds, f)
