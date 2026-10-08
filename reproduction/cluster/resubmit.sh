#!/bin/bash
# Chain rounds of an array so a sweep continues across the wall-clock limit.
#
# A task killed at the wall leaves the checkpoint its last periodic write put there, ten minutes old at
# worst. Because a finished cell is recorded in the shard and skipped by any later run, resubmitting the
# identical array continues the sweep and never repeats finished work. A round with nothing left to do
# therefore costs seconds, not a wall,
# which is why the rounds are chained unconditionally rather than gated on the previous exit codes: the
# alternative is a process sitting on a login node for days polling sacct.
#
# The chain is a ceiling, not a plan. Stop it early with `scancel` on the job ids printed below, and read
# the state with the sacct line at the end.
set -euo pipefail
SCRIPT=${1:?usage: resubmit.sh <slurm script> [rounds]}
ROUNDS=${2:-6}

previous=""
for round in $(seq 1 "$ROUNDS"); do
    if [ -z "$previous" ]; then
        job=$(sbatch --parsable "$SCRIPT")
    else
        job=$(sbatch --parsable --dependency=afterany:"$previous" "$SCRIPT")
    fi
    echo "round $round: submitted $job"
    previous=$job
done
echo
echo "Progress:  sacct -j <id> --format=JobID,State,ExitCode,Elapsed"
echo "A round whose tasks all exit 0 has finished the sweep; the rounds after it find nothing to do."
