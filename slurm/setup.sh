# 1. Install packages (Ubuntu 22.04+; for RHEL use dnf equivalents)
sudo apt update
sudo apt install -y slurm-wlm slurmdbd munge \
                    mariadb-server libpmix-dev

# 2. Create required directories with correct ownership
sudo mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
sudo chown -R slurm:slurm /var/spool/slurmctld /var/log/slurm
sudo chown -R root:root  /var/spool/slurmd
sudo chmod 755 /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

# 3. Verify hardware as SLURM sees it, and compare to slurm.conf
sudo slurmd -C
#    -> adjust RealMemory in slurm.conf if it differs from `slurmd -C` output.

# 4. Copy the four config files into place
sudo cp ./etc/slurm/slurm.conf      /etc/slurm/slurm.conf
sudo cp ./etc/slurm/gres.conf       /etc/slurm/gres.conf
sudo cp ./etc/slurm/cgroup.conf     /etc/slurm/cgroup.conf
sudo cp ./etc/slurm/slurmdbd.conf   /etc/slurm/slurmdbd.conf
sudo chown slurm:slurm  /etc/slurm/slurmdbd.conf
sudo chmod 600          /etc/slurm/slurmdbd.conf   # required by slurmdbd

# 5. Provision the munge auth key (keep a backup — needed on every future node)
sudo /usr/sbin/create-munge-key -f
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400         /etc/munge/munge.key
sudo systemctl enable --now munge

# 6. Create the accounting database
sudo systemctl enable --now mariadb
sudo mysql <<EOF
CREATE DATABASE slurm_acct_db;
CREATE USER 'slurm'@'localhost' IDENTIFIED BY 'CHANGE_ME_STRONG_PASSWORD';
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
FLUSH PRIVILEGES;
EOF

# 7. Make sure the node can resolve its own hostname
getent hosts enif    # must return an IP; if not, add it to /etc/hosts

# 8. Open firewall ports (only needed if a firewall is active)
#    6817/tcp slurmctld   6818/tcp slurmd   6819/tcp slurmdbd
sudo ufw allow 6817,6818,6819/tcp 2>/dev/null || true

# 9. Start the daemons IN ORDER
sudo systemctl enable --now slurmdbd     # accounting first
sleep 2
sudo systemctl enable --now slurmctld    # then the controller
sudo systemctl enable --now slurmd       # then the worker

# 10. Register the cluster + a default account in the accounting DB
sudo sacctmgr -i add cluster plast-hpc
sudo sacctmgr -i add account default Description="default" Organization="lab"
sudo sacctmgr -i add user "$USER" DefaultAccount=default

# 11. Verify
sinfo                       # node should be 'idle'
scontrol show node enif    # check CPU/GPU/memory match
srun -p gpu --gres=gpu:h100:1 nvidia-smi -L   # smoke test