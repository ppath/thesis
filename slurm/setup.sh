#!/usr/bin/env bash
set -euo pipefail

# 0. Preflight: NVIDIA stack must already be installed.
#    We deliberately do NOT install drivers here — driver/CUDA installs are
#    too varied (DKMS, runfile, datacenter repo) for a generic apt line. Fail
#    fast with a hint instead.
preflight_fail() {
    echo "ERROR: $1" >&2
    echo "       fix:    $2" >&2
    exit 1
}
command -v nvidia-smi >/dev/null \
    || preflight_fail "nvidia-smi not found" \
       "sudo apt install nvidia-utils-550-server (or matching driver series)"
nvidia-smi -L 2>/dev/null | grep -q . \
    || preflight_fail "nvidia-smi -L lists no GPUs (driver not loaded?)" \
       "check 'sudo dmesg | grep -i nvidia' and reboot after driver install"
ldconfig -p | grep -q libnvidia-ml \
    || preflight_fail "libnvidia-ml not found (NVML; needed for AutoDetect=nvml)" \
       "sudo apt install libnvidia-ml1 (or nvidia-utils-XXX-server)"
systemctl is-active --quiet nvidia-persistenced \
    || echo "WARN: nvidia-persistenced is not active — first job per GPU will" \
            "pay a driver-init cost. enable with 'sudo systemctl enable --now" \
            "nvidia-persistenced'." >&2

# 1. Install packages (Ubuntu 22.04+; for RHEL use dnf equivalents)
sudo apt update
sudo apt install -y slurm-wlm slurmdbd munge slurm-wlm-basic-plugins \
                    mariadb-server libpmix2

# 2. Create required directories with correct ownership
sudo mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm \
              /var/spool/slurm/archive
sudo chown -R slurm:slurm /var/spool/slurmctld /var/log/slurm \
                           /var/spool/slurm/archive
sudo chown -R root:root   /var/spool/slurmd
sudo chmod 755 /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
sudo chmod 750 /var/spool/slurm/archive

# 3. Copy the config files into place.
#    Read the DB password once here; it is substituted into slurmdbd.conf via
#    envsubst (never exposed in argv) and used in the CREATE USER statement
#    in step 6.
if [ -z "${SLURM_DB_PASS:-}" ]; then
    read -r -s -p "Enter slurm DB password: " SLURM_DB_PASS
    echo
fi
export SLURM_DB_PASS

sudo cp ./etc/slurm/slurm.conf   /etc/slurm/slurm.conf
sudo cp ./etc/slurm/gres.conf    /etc/slurm/gres.conf
sudo cp ./etc/slurm/cgroup.conf  /etc/slurm/cgroup.conf

# Substitute ${SLURM_DB_PASS} in slurmdbd.conf without exposing the password
# on the command line (safe for passwords containing /, &, or \).
tmp=$(mktemp)
envsubst '$SLURM_DB_PASS' < ./etc/slurm/slurmdbd.conf > "$tmp"
sudo install -m 0600 -o slurm -g slurm "$tmp" /etc/slurm/slurmdbd.conf
rm -f "$tmp"

# GPU epilog: runs after every job to kill stray compute processes
sudo install -m 0755 -o root -g root \
    ./etc/slurm/epilog.sh /etc/slurm/epilog.sh

# Logrotate for /var/log/slurm/*.log (slurmctld, slurmd, slurmdbd, epilog)
sudo install -m 0644 -o root -g root \
    ./etc/logrotate.d/slurm /etc/logrotate.d/slurm

# 4. Verify hardware as SLURM sees it, and compare to slurm.conf.
#    (slurmd -C reads /etc/slurm/slurm.conf, so this MUST run after the
#    config copy above.) Adjust RealMemory/CPUs in slurm.conf to match if
#    the values differ.
echo "---- slurmd -C output (compare with NodeName= line in slurm.conf) ----"
sudo slurmd -C
echo "---- end slurmd -C ----"

# 5. Provision the munge auth key (keep a backup — needed on every future node)
#    Guard: do not overwrite an existing key on re-runs (would break auth).
if [ ! -s /etc/munge/munge.key ]; then
    sudo /usr/sbin/create-munge-key -f
fi
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400         /etc/munge/munge.key
sudo systemctl enable --now munge

# 6. Create the accounting database (idempotent — safe to re-run)
sudo systemctl enable --now mariadb
sudo mysql <<EOF
CREATE DATABASE IF NOT EXISTS slurm_acct_db;
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY '${SLURM_DB_PASS}';
ALTER  USER               'slurm'@'localhost' IDENTIFIED BY '${SLURM_DB_PASS}';
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
FLUSH PRIVILEGES;
EOF

# 7. Make sure the node can resolve its own hostname
getent hosts enif \
    || echo "WARNING: 'enif' not in DNS or /etc/hosts — add it before starting daemons."

# 8. Firewall: no ports need to be opened on a single-node cluster — all
#    daemons talk over localhost. When adding a second node, open:
#      6817/tcp (slurmctld) — all nodes must reach the controller
#      6818/tcp (slurmd)    — restricted to cluster subnet
#      6819/tcp (slurmdbd)  — restricted to controller only

# 9. Start the daemons IN ORDER
sudo systemctl enable --now slurmdbd     # accounting first

# Wait for slurmdbd to actually be listening (up to 30 s) before slurmctld
slurmdbd_ready=false
for _ in $(seq 1 30); do
    if ss -ltn 'sport = :6819' | grep -q LISTEN; then
        slurmdbd_ready=true
        break
    fi
    sleep 1
done
if [ "$slurmdbd_ready" = false ]; then
    echo "ERROR: slurmdbd did not start within 30 s — check journalctl -u slurmdbd" >&2
    exit 1
fi

sudo systemctl enable --now slurmctld    # then the controller
sudo systemctl enable --now slurmd       # then the worker

# 10. Register the cluster + a default account in the accounting DB
#     '|| true' makes these idempotent: safe to re-run if already registered.
sudo sacctmgr -i add cluster plast-hpc                                         || true
sudo sacctmgr -i add account default Description="default" Organization="lab"  || true
sudo sacctmgr -i add user "${SUDO_USER:-$USER}" DefaultAccount=default          || true

# 11. Verify
sinfo                                          # node should be 'idle'
scontrol show node enif                        # check CPU/GPU/memory match

# Wait for the node to reach 'idle' before submitting the smoke test;
# slurmd registration with slurmctld can take a few seconds after boot.
for _ in $(seq 1 30); do
    state=$(sinfo -h -o '%T' -n enif | head -n1 || true)
    [[ "$state" == "idle" ]] && break
    sleep 1
done
if [[ "${state:-}" != "idle" ]]; then
    echo "WARN: node did not reach 'idle' within 30 s (state=$state)." \
         "Check 'sinfo -R' and 'journalctl -u slurmd'. Skipping smoke test." >&2
    exit 0
fi

srun -p main --gres=gpu:h100:1 nvidia-smi -L  # smoke test
