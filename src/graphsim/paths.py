"""Locations of inputs and outputs, configurable via environment variables.

All defaults are folders below the repository root, which is GRAPHSIM_ROOT or, if it is
not set, the directory that contains src/ (editable install or PYTHONPATH=src):

  GRAPHSIM_DATA_DIR        data/        MarkLines tables (CSV), licensed from MarkLines
  GRAPHSIM_EMBEDDINGS_DIR  embeddings/  product-name embeddings and derived substitution maps
  GRAPHSIM_GEO_DIR         geo/         JMA intensities and geocoding of Japanese firms
  GRAPHSIM_SEEDS_DIR       seeds/       random seeds of all experiments (shipped)
  GRAPHSIM_CONFIGS_DIR     configs/     experiment configurations
  GRAPHSIM_RUNS_DIR        results/     simulation outputs (stats_*.jsonl) per experiment
  GRAPHSIM_ANALYSIS_DIR    outputs/     derived network statistics and analysis results
  GRAPHSIM_FIGURES_DIR     outputs/figures/  figures and the statistics reported in the paper
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("GRAPHSIM_ROOT", Path(__file__).resolve().parents[2]))


def _path(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, ROOT / default))


DATA_DIR = _path("GRAPHSIM_DATA_DIR", "data")
EMBEDDINGS_DIR = _path("GRAPHSIM_EMBEDDINGS_DIR", "embeddings")
GEO_DIR = _path("GRAPHSIM_GEO_DIR", "geo")
SEEDS_DIR = _path("GRAPHSIM_SEEDS_DIR", "seeds")
CONFIGS_DIR = _path("GRAPHSIM_CONFIGS_DIR", "configs")
RUNS_DIR = _path("GRAPHSIM_RUNS_DIR", "results")
ANALYSIS_DIR = _path("GRAPHSIM_ANALYSIS_DIR", "outputs")
FIGURES_DIR = _path("GRAPHSIM_FIGURES_DIR", "outputs/figures")
