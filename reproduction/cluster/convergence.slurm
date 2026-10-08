#!/bin/bash
# The account to charge is not written here: #SBATCH directives are read before any shell runs,
# so a variable would never expand. Export SBATCH_ACCOUNT before submitting -- sbatch reads it
# natively -- or pass `sbatch -A <account> ...` on the command line.
#SBATCH -p icelake
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=76
#SBATCH --mem=240G
#SBATCH -J es-convergence
#SBATCH -o logs/convergence_%j.out
#SBATCH -e logs/convergence_%j.err
set -euo pipefail

# The timing exhibit, and the only one that needs a machine to itself.
#
# It reports wall-clock seconds for the two phases, and a second measured while sharing a node with other
# jobs is not the quantity the table claims. The exhibit knows this and refuses above a load average of
# 0.5, sampled twice a minute apart -- so on a shared node it would decline to run rather than report a
# wrong number. Hence --exclusive: the whole node, nothing else on it.
#
# No time budget. Timing must not be interrupted mid-cell, so a cell either completes or is not recorded;
# the wall kills the job, finished cells are already in the sink, and resubmitting continues with the
# ones that are left.

cd "${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" || {
    echo "cannot enter ${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" >&2; exit 1; }

echo "== inputs =="
echo "  cwd $(pwd)"
for required in reproduction/cluster/env.sh reproduce scripts src; do
    [ -e "$required" ] || { echo "  MISSING $required" >&2; exit 1; }
    echo "  ok $required"
done
source reproduction/cluster/env.sh
[ -d "$ENDOGENOUS_SUE_DATA" ] || { echo "  MISSING data at $ENDOGENOUS_SUE_DATA" >&2; exit 1; }
echo "  ok data $(ls "$ENDOGENOUS_SUE_DATA" | wc -l) network directories"
python -c "import numpy, scipy; print('  ok python', numpy.__version__, scipy.__version__)" || exit 1
python -c "import numba; print('  ok numba', numba.__version__)" \
    || { echo "  REFUSING: without numba every time below is one to two orders out" >&2; exit 1; }

CORES=$(nproc)
echo "host $(hostname)  exclusive node, $CORES cores  load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "started $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=$CORES NUMBA_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CELL_SECONDS=${CELL_SECONDS:-}
set +e
python -u reproduction/exhibits/convergence.py --out results/reproduce/convergence.jsonl \
    ${CELL_SECONDS:+--cell-seconds "$CELL_SECONDS"}
status=$?
echo "finished $(date -u +%FT%TZ)  elapsed ${SECONDS}s  exit $status"
echo "exit 1 means it refused to time because the node was not idle"

sleep 20
sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,AllocCPUS%9,ReqMem%9,MaxRSS%10,Elapsed%12,State%15
exit $status
