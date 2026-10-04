#!/bin/bash

#SBATCH -J rev                     # Overridden per experiment with -J
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # Whole node (32 physical cores)
#SBATCH --exclusive                # No other job shares the node's commit limit
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --partition=big
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Runs one slice of an experiment as a SLURM array task. The configurations
# are split into N contiguous slices of (almost) equal size, N = array size.
#
# Usage: sbatch -J <name> --array=0-<N-1>%<max parallel> [--export=ALL,WORKERS=<n>] \
#            revision_array.sh <runner.py> <total configs> [runner arguments]
# WORKERS defaults to 32. The nodes use strict overcommit (vm.overcommit_memory = 2),
# so memory-heavy runs (hybrid arm, large seed sets) need fewer workers per node.
# Example:
#   sbatch -J rev_exp2 --array=0-19%10 revision_array.sh 0_monte_carlo.py 50000 \
#       --configs 0_monte_carlo_configs_rev.json --output-dir results/exp2

D=${GRAPHSIM_BASE:?Set GRAPHSIM_BASE to the repository directory on the cluster}
RUNNER=$1
TOTAL=$2
shift 2

WORKERS=${WORKERS:-$SLURM_CPUS_PER_TASK}
N=$SLURM_ARRAY_TASK_COUNT
PER=$(( (TOTAL + N - 1) / N ))
START=$(( SLURM_ARRAY_TASK_ID * PER ))
END=$(( START + PER < TOTAL ? START + PER : TOTAL ))

# Python environment built from uv.lock (see README)
source "${GRAPHSIM_VENV:-$D/.venv}/bin/activate"
# The environment holds only the dependencies; the code on scratch comes via PYTHONPATH
export PYTHONPATH=$D/src
# The runs do not depend on string hashing (checked in revision_test.sh); fixed anyway
export PYTHONHASHSEED=0
cd $D/experiments

# Accounting does not record memory on this cluster, so log node memory every 10 min
( while true; do echo "[mem $(date +%H:%M)] committed $(awk '/Committed_AS/ {printf "%.0f", $2/1048576}' /proc/meminfo) GB, $(free -g | awk '/Mem:/ {print $3 " GB used of " $2}')"; sleep 600; done ) &
MEMLOG=$!

echo "Task ${SLURM_ARRAY_TASK_ID}/${N} on $(hostname): ${RUNNER} configs ${START}-${END}, ${WORKERS} workers"
python "$RUNNER" --base-dir $D --config-start $START --config-end $END \
    --workers $WORKERS --chunk-size 200 "$@"
STATUS=$?
kill $MEMLOG
echo "Task ${SLURM_ARRAY_TASK_ID} finished with exit code ${STATUS}."
exit $STATUS
