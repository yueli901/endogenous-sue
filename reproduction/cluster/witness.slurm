#!/bin/bash
# The account to charge is not written here: #SBATCH directives are read before any shell runs,
# so a variable would never expand. Export SBATCH_ACCOUNT before submitting -- sbatch reads it
# natively -- or pass `sbatch -A <account> ...` on the command line.
#SBATCH -p icelake
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -J es-witness
#SBATCH -o logs/witness_%j.out
#SBATCH -e logs/witness_%j.err
set -euo pipefail

# The five-node witness. One job: it is a fixed synthetic network with its own dispersion grid, so it
# declares no corpus cells and takes no --networks. It enumerates the potential orderings and, separately,
# every non-empty link subset, and fails if the two enumerations disagree.

cd "${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" || {
    echo "cannot enter ${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" >&2; exit 1; }

echo "== inputs =="
echo "  cwd $(pwd)"
for required in reproduction/cluster/env.sh reproduce scripts src; do
    [ -e "$required" ] || { echo "  MISSING $required" >&2; exit 1; }
    echo "  ok $required"
done
source reproduction/cluster/env.sh
python -c "import numpy, scipy; print('  ok python', numpy.__version__, scipy.__version__)" || exit 1
python -c "import numba; print('  ok numba', numba.__version__)" \
    || echo "  WARNING: no numba, the compiled kernels degrade to Python"

echo "host $(hostname)  cores $SLURM_CPUS_PER_TASK"
echo "started $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

set +e
python -u reproduction/exhibits/witness.py --out results/reproduce/witness.jsonl
status=$?
echo "finished $(date -u +%FT%TZ)  elapsed ${SECONDS}s  exit $status"
echo "exit 1 means the ordering and subset enumerations disagreed, which cannot happen if both are right"

sleep 20
sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,AllocCPUS%9,ReqMem%9,MaxRSS%10,Elapsed%12,State%15
exit $status
