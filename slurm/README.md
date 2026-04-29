# SLURM Cluster — Setup README

## Quick setup guide

All commands below run on **`enif`** as a user with `sudo`. Replace `CHANGE_ME_STRONG_PASSWORD` everywhere before starting.

```bash
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
```

If `sinfo` shows the node as `down` or `drain`, run `sudo scontrol update nodename=enif state=resume` after fixing the cause (`scontrol show node enif` reports the reason).

---

## What this cluster is

A single-node SLURM cluster on **`enif`**: dual-socket server (2 × 16 physical cores, hyper-threaded → 64 logical CPUs), ~512 GB RAM, and **4 NVIDIA H100 GPUs** in two NVLink pairs (GPU0↔GPU1 on socket 0, GPU2↔GPU3 on socket 1). Built so a second node can be added later with minimal changes.

## What we want SLURM to do

- **Schedule at fine granularity**: per physical core, per GB of RAM, per whole GPU, *and* per fractional GPU (shards) so small jobs can share an H100.
- **Respect hardware topology**: a job asking for 1 GPU gets CPU cores on the same socket; a 2-GPU job prefers the NVLink-connected pair.
- **Enforce limits strictly**: jobs exceeding requested memory or touching unallocated GPUs are killed via Linux cgroups.
- **Account everything**: every job's CPU/RAM/GPU usage is logged to MariaDB for reporting and future quota enforcement.
- **Reserve OS headroom**: 2 cores and 12 GB RAM are kept off-limits to jobs.

## Config files at a glance

| File | Role |
|---|---|
| `/etc/slurm/slurm.conf` | Main config: cluster identity, controller host, node hardware, the `gpu` partition, scheduler (`backfill` + `cons_tres`), cgroup process tracking, and slurmdbd accounting. |
| `/etc/slurm/gres.conf` | Describes the 4 H100s and their 40 shards, binds each GPU to its NUMA-local cores, and declares NVLink topology. |
| `/etc/slurm/cgroup.conf` | Enforces per-job limits on cores, RAM, swap, and GPU devices via Linux cgroups. |
| `/etc/slurm/slurmdbd.conf` | Accounting daemon + MariaDB backend. Must be mode `0600`, owned by `slurm`. |

## Daemons running on `enif`

- **`munge`** — authenticates messages between all SLURM daemons.
- **`slurmdbd`** — writes job accounting records to MariaDB.
- **`slurmctld`** — the brain: receives `sbatch`/`srun`, schedules jobs, tracks state.
- **`slurmd`** — the worker: launches jobs on the node, sets up cgroups, reports status.

## How users submit jobs

```bash
sbatch -p gpu --gres=gpu:h100:1  -c 8  --mem=64G   job.sh   # 1 whole GPU
sbatch -p gpu --gres=gpu:h100:2  -c 16 --mem=128G  job.sh   # NVLink pair
sbatch -p gpu --gres=shard:h100:3 -c 4 --mem=32G   job.sh   # share a GPU
squeue          # current queue
sacct -X        # job history (from accounting DB)
```

## Useful admin commands

```bash
sudo scontrol reconfigure              # reload most config changes
sudo systemctl restart slurmctld slurmd # required after node/partition edits
sudo journalctl -u slurmctld -f         # live controller log
sudo tail -f /var/log/slurm/slurmd.log  # node-side log
sinfo -R                                # why is a node down/drained?
```

## Adding a second node later

1. Install `slurm-wlm` + `munge` on the new host.
2. Copy `/etc/munge/munge.key` from `enif` (same permissions: `munge:munge 400`).
3. Copy `/etc/slurm/slurm.conf` and `/etc/slurm/cgroup.conf` (must be **identical** on every node).
4. Place a `gres.conf` matching the new node's own GPU topology.
5. Ensure both hosts resolve each other (`/etc/hosts` or DNS).
6. Append a `NodeName=` line and add the new node to `Nodes=` in the `gpu` partition of `slurm.conf` on **all** nodes.
7. On the controller: `sudo scontrol reconfigure`. On the new node: `sudo systemctl enable --now slurmd`.

## Troubleshooting checklist

- **Node `down`/`drain`**: `scontrol show node enif` shows the reason; common causes are `RealMemory` mismatch with `slurmd -C`, or `gres.conf` listing a `/dev/nvidiaN` that doesn't exist.
- **Jobs stuck in `PD` (pending)**: `squeue --start` and `scontrol show job <id>` give the reason (resources, priority, association missing).
- **`Invalid account`**: the user wasn't added with `sacctmgr` — required because we set `AccountingStorageEnforce=associations`.
- **`slurmdbd` won't start**: check that `slurmdbd.conf` is mode `0600`, owned by `slurm`, and that MariaDB is reachable with the configured password.
