#!/usr/bin/env bash
# GPU cleanup epilog — runs on the compute node after every job ends (as root).
# Kills any processes left behind on allocated GPUs and logs final GPU state.
# SLURM sets SLURM_JOB_GPUS to the comma-separated device indices for the job.

set -uo pipefail

gpus="${SLURM_JOB_GPUS:-}"
[[ -z "$gpus" ]] && exit 0

for gpu in ${gpus//,/ }; do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$gpu" \
        2>/dev/null | xargs -r kill -9 2>/dev/null || true
    nvidia-smi -i "$gpu" -q -d MEMORY >> /var/log/slurm/epilog.log 2>&1 || true
done

exit 0
