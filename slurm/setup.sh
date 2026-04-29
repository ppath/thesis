#!/usr/bin/env bash
set -euo pipefail

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

# 3. Verify hardware as SLURM sees it, and compare to slurm.conf
sudo slurmd -C
#    -> adjust RealMemory in slurm.conf if it differs from `slurmd -C` output.

# 4. Copy the config files into place.
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

# 8. Open firewall ports (only slurmctld needs external access on a single-node
#    cluster; add 6818/6819 restricted to cluster subnet when a second node is
#    added).
sudo ufw allow 6817/tcp 2>/dev/null || true

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
srun -p gpu --gres=gpu:h100:1 nvidia-smi -L   # smoke test
