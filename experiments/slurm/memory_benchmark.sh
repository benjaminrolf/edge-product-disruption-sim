#!/bin/bash

#SBATCH -J mem_bench
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --partition=medium
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Memory benchmark for the worker counts of the runs of the article (strict overcommit):
# runs configs [start, end) of one runner with the given workers and logs the node's
# committed memory every minute.
# Usage: sbatch -J bench_<name> memory_benchmark.sh <runner.py> <start> <end> <workers> [args]

D=${GRAPHSIM_BASE:?Set GRAPHSIM_BASE to the repository directory on the cluster}
RUNNER=$1
START=$2
END=$3
WORKERS=$4
shift 4

# Python environment built from uv.lock (see README)
source "${GRAPHSIM_VENV:-$D/.venv}/bin/activate"
export PYTHONPATH=$D/src
export PYTHONHASHSEED=0
cd $D/experiments

( while true; do
    echo "[mem $(date +%H:%M)] committed $(awk '/Committed_AS/ {printf "%.0f", $2/1048576}' /proc/meminfo) GB of $(awk '/CommitLimit/ {printf "%.0f", $2/1048576}' /proc/meminfo), used $(free -g | awk '/Mem:/ {print $3}') GB"
    sleep 60
done ) &
MEMLOG=$!

python "$RUNNER" --base-dir $D --config-start $START --config-end $END \
    --workers $WORKERS --chunk-size 200 "$@"
STATUS=$?
kill $MEMLOG
echo "Benchmark finished with exit code ${STATUS}."
