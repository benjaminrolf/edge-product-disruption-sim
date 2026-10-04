#!/bin/bash
# Submits the Experiment-2 runs as SLURM arrays (revision_array.sh):
#   main runs         50,000 configs in 20 slices, at most 9 nodes at once
#   gamma = -1        500 configs (|S| = 100, one seed sample per network), 1 node
# Together at most 10 exclusive nodes. 20 workers per node follow the memory
# benchmark of the |S| = 75/100 runs (peak 146 of about 226 GB).
# Usage (on the login node): bash submit_exp2_rev.sh

set -euo pipefail
D=${GRAPHSIM_BASE:?Set GRAPHSIM_BASE to the repository directory on the cluster}
OUT=$D/results
cd $D/experiments
mkdir -p $OUT

m=$(sbatch -J rev_exp2 --array=0-19%9 --export=ALL,WORKERS=20 --parsable \
    revision_array.sh 0_monte_carlo.py 50000 --configs 0_monte_carlo_configs_rev.json \
    --delivered-products --output-dir $OUT/exp2)
g=$(sbatch -J rev_exp2_gamma-1 --array=0-0 --export=ALL,WORKERS=20 --parsable \
    revision_array.sh 0_monte_carlo.py 500 --configs 0_monte_carlo_configs_rev_gamma-1.json \
    --delivered-products --output-dir $OUT/exp2_gamma-1)
echo "exp2 main $m, gamma -1 $g"
