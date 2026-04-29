#!/usr/bin/env bash
# GPU cleanup epilog — runs on the compute node after every job ends (as root).
# Logs final GPU state and kills stragglers that escaped the job's cgroup
# (e.g. processes that reparented to PID 1). SLURM's normal proctrack/cgroup
# teardown already kills in-cgroup tasks before this runs.
#
# IMPORTANT: this only kills PIDs whose /proc/<pid>/cgroup still references
# job_${SLURM_JOB_ID}. With shards, two jobs can share one physical GPU; a
# blanket kill of every PID on the GPU would take out the surviving job.

set -uo pipefail

job_id="${SLURM_JOB_ID:-}"
gpus="${SLURM_JOB_GPUS:-}"
[[ -z "$gpus" || -z "$job_id" ]] && exit 0

for gpu in ${gpus//,/ }; do
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if grep -q "job_${job_id}\b" "/proc/$pid/cgroup" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
                        --id="$gpu" 2>/dev/null)
    nvidia-smi -i "$gpu" -q -d MEMORY >> /var/log/slurm/epilog.log 2>&1 || true
done

exit 0
