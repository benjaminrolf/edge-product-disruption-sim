#!/bin/bash

#SBATCH -J rev_test                # Test of all runners
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --partition=short
#SBATCH --time=00:55:00
#SBATCH --output=logs/%j_test.out
#SBATCH --error=logs/%j_test.err

# Runs every runner on a few configurations in parallel (3 workers each)
# and writes to results/test. Each Experiment-1 arm runs twice with different
# PYTHONHASHSEED values, so identical results show that the runs are deterministic
# and independent of string hashing. The configurations are also covered by
# memory_benchmark.sh, so both can be compared across environments.

D=${GRAPHSIM_BASE:?Set GRAPHSIM_BASE to the repository directory on the cluster}
OUT=$D/results/test
EXP1="--configs $D/configs/1_comp_configs_rev.json --config-start 0 --config-end 4 --delivered-products"
BASE="--configs $D/configs/1_comp_configs_baseline.json --config-start 0 --config-end 4"

# Python environment built from uv.lock (see README)
source "${GRAPHSIM_VENV:-$D/.venv}/bin/activate"
# The environment holds only the dependencies; the code on scratch comes via PYTHONPATH
export PYTHONPATH=$D/src
cd $D/experiments

run() {
    local name=$1 hashseed=$2
    shift 2
    PYTHONHASHSEED=$hashseed python "$@" --base-dir $D --workers 3 --chunk-size 200 \
        > "$D/logs/${SLURM_JOB_ID}_test_${name}.log" 2>&1
    echo "$name exit $?"
}

echo "Test job on $(hostname), python: $(which python)"
run baseline 1 1_comparison_baseline.py $BASE --output-dir $OUT/exp1_baseline &
run baseline_rep 2 1_comparison_baseline.py $BASE --output-dir $OUT/exp1_baseline_rep &
run extended 1 1_comparison_rev.py --arm extended $EXP1 --output-dir $OUT/exp1_extended &
run extended_rep 2 1_comparison_rev.py --arm extended $EXP1 --output-dir $OUT/exp1_extended_rep &
run hybrid 1 1_comparison_rev.py --arm hybrid $EXP1 --output-dir $OUT/exp1_hybrid &
run hybrid_rep 2 1_comparison_rev.py --arm hybrid $EXP1 --output-dir $OUT/exp1_hybrid_rep &
run exp2 1 0_monte_carlo.py --configs $D/configs/0_monte_carlo_configs_rev.json --delivered-products \
    --output-dir $OUT/exp2 --config-start 90 --config-end 93 &
run exp3 1 2_geje.py --configs $D/configs/2_geje_configs_rev.json --delivered-products \
    --output-dir $OUT/exp3 --config-start 0 --config-end 2 &
wait
echo "Test job finished."
