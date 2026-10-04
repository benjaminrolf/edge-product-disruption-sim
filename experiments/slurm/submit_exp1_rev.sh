#!/bin/bash
# Submits all Experiment-1 runs as SLURM arrays (revision_array.sh), in three
# waves so that at most 10 exclusive nodes run at once:
#   wave 1  main runs of the three arms (8 nodes)
#   wave 2  imputation sensitivity, gamma in {-2, -1, +1}, both arms (9 nodes)
#   wave 3  tau sweep of the extended arm (5 nodes)
# Each wave starts when the previous one has ended (afterany). Worker counts follow
# memory_benchmark.sh: baseline 32, extended 28, hybrid 20.
# Usage (on the login node): bash submit_exp1_rev.sh

set -euo pipefail
D=${GRAPHSIM_BASE:?Set GRAPHSIM_BASE to the repository directory on the cluster}
OUT=$D/results
cd $D/experiments
mkdir -p $OUT

submit() {  # submit <name> <array> <workers> <dependency or -> <runner> <total> [args]
    local name=$1 array=$2 workers=$3 dep=$4
    shift 4
    local opts=(-J "$name" --array="$array" --export=ALL,WORKERS="$workers" --parsable)
    [ "$dep" != "-" ] && opts+=(--dependency="afterany:$dep")
    sbatch "${opts[@]}" revision_array.sh "$@"
}

EXP1="--configs 1_comp_configs_rev.json --delivered-products"
b=$(submit rev_exp1_baseline 0-1 32 - 1_comparison_baseline.py 8000 \
    --configs 1_comp_configs_baseline.json --output-dir $OUT/exp1_baseline)
e=$(submit rev_exp1_extended 0-1 28 - 1_comparison_rev.py 8000 --arm extended $EXP1 \
    --output-dir $OUT/exp1_extended)
h=$(submit rev_exp1_hybrid 0-3 20 - 1_comparison_rev.py 8000 --arm hybrid $EXP1 \
    --output-dir $OUT/exp1_hybrid)
wave1="$b:$e:$h"
echo "wave 1: baseline $b, extended $e, hybrid $h"

wave2=()
for g in -2 -1 +1; do
    cfg="--configs 1_comp_configs_rev_gamma$g.json --delivered-products"
    wave2+=("$(submit rev_exp1_ext_gamma$g 0-0 28 "$wave1" 1_comparison_rev.py 2000 \
        --arm extended $cfg --output-dir $OUT/exp1_extended_gamma$g)")
    wave2+=("$(submit rev_exp1_hyb_gamma$g 0-1 20 "$wave1" 1_comparison_rev.py 2000 \
        --arm hybrid $cfg --output-dir $OUT/exp1_hybrid_gamma$g)")
done
wave2_dep=$(IFS=:; echo "${wave2[*]}")
echo "wave 2 (gamma): ${wave2[*]}"

wave3=()
for t in 0.65 0.69 0.77 0.81 0.85; do
    wave3+=("$(submit rev_exp1_ext_tau$t 0-0 28 "$wave2_dep" 1_comparison_rev.py 2000 \
        --arm extended --configs 1_comp_configs_rev_tau$t.json --delivered-products \
        --output-dir $OUT/exp1_extended_tau$t)")
done
echo "wave 3 (tau): ${wave3[*]}"
