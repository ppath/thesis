# SLURM Cluster — Setup README

## Setup guide

Run `setup.sh` on **`enif`** as a user with `sudo`:

```bash
# Optional: set the DB password in the environment to avoid the interactive prompt
export SLURM_DB_PASS="<strong-password>"

bash setup.sh
```

`setup.sh` is idempotent — safe to re-run. It handles package installation, directory creation, config deployment (with password substitution), munge key provisioning, MariaDB setup, daemon startup, and cluster registration in the correct order.

After it completes:

```bash
sinfo                                        # node should be 'idle'
scontrol show node enif                      # verify CPU/GPU/memory match slurm.conf
srun -p gpu --gres=gpu:h100:1 nvidia-smi -L  # smoke test
```

If `sinfo` shows the node as `down` or `drain`, run `sudo scontrol update nodename=enif state=resume` after fixing the cause (`scontrol show node enif` reports the reason).

> **Note on `RealMemory`**: before running `setup.sh`, verify that the value in `slurm.conf` matches `sudo slurmd -C`. If `RealMemory` exceeds what slurmd reports, the node will be marked DOWN immediately.

> **Firewall**: no ports need to be opened on a single-node cluster — all daemons communicate over localhost. When adding a second node, open `6817/tcp` (slurmctld, all nodes), `6818/tcp` (slurmd, cluster subnet only), and `6819/tcp` (slurmdbd, controller only).

---

## What this cluster is

A single-node SLURM cluster on **`enif`**: dual-socket server (2 × 16 physical cores, hyper-threaded → 64 logical CPUs), ~512 GB RAM, and **4 NVIDIA H100 GPUs** in two NVLink pairs (GPU0↔GPU1 on socket 0, GPU2↔GPU3 on socket 1). Built so a second node can be added later with minimal changes.

## What we want SLURM to do

- **Schedule at fine granularity**: per physical core, per GB of RAM, per whole GPU, *and* per fractional GPU (shards) so small jobs can share an H100.
- **Respect hardware topology**: a job asking for 1 GPU gets CPU cores on the same socket; a 2-GPU job prefers the NVLink-connected pair.
- **Enforce limits strictly**: jobs exceeding requested memory or touching unallocated GPUs are killed via Linux cgroups.
- **Account everything**: every job's CPU/RAM/GPU usage is logged to MariaDB for reporting and future quota enforcement.
- **Reserve OS headroom**: 12 GB RAM is kept off-limits to jobs via `MemSpecLimit`.

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
