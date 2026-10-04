#!/bin/bash
# Submits the Experiment-3 runs (Tohoku-inspired scenario) as a SLURM array
# (revision_array.sh): 22,500 configs in 40 slices, at most 10 exclusive nodes at once.
# 16 workers per node follow the memory benchmark (peak 170 of about 226 GB); one
# slice of about 560 runs takes about 11 hours.
# Usage (on the login node): bash submit_exp3_rev.sh

set -euo pipefail
D=${GRAPHSIM_BASE:?Set GRAPHSIM_BASE to the repository directory on the cluster}
OUT=$D/results
cd $D/experiments
mkdir -p $OUT

j=$(sbatch -J rev_exp3 --array=0-39%10 --export=ALL,WORKERS=16 --parsable \
    revision_array.sh 2_geje.py 22500 --configs 2_geje_configs_rev.json \
    --delivered-products --output-dir $OUT/exp3)
echo "exp3 $j"
