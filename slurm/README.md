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
sinfo                                         # node should be 'idle'
scontrol show node enif                       # verify CPU/GPU/memory match slurm.conf
srun -p main --gres=gpu:h100:1 nvidia-smi -L  # smoke test
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

## Direct SSH access (important caveat)

`enif` remains a normal multi-user host. Users may continue to `ssh enif` and
work outside SLURM as before. **SLURM's cgroup limits only apply to processes
started through `srun`/`sbatch`** — anything you launch from a plain SSH shell
runs outside any job cgroup, with no CPU, memory, or GPU isolation, and is
not visible to `squeue`/`sacct`. We are intentionally not deploying
`pam_slurm_adopt` (which would adopt SSH sessions into a running job's
cgroup, or refuse login if the user has no job).

If you need fair-share or accounting for a piece of work, submit it as a
SLURM job. If you need a quick interactive shell that *is* sandboxed, use
`srun -p main --pty bash`.

## Config files at a glance

| File | Role |
|---|---|
| `/etc/slurm/slurm.conf` | Main config: cluster identity, controller host, node hardware, the single `main` partition, scheduler (`backfill` + `cons_tres`), cgroup process tracking, and slurmdbd accounting. |
| `/etc/slurm/gres.conf` | Describes the 4 H100s and their 40 shards, binds each GPU to its NUMA-local cores, and declares NVLink topology. |
| `/etc/slurm/cgroup.conf` | Enforces per-job limits on cores, RAM, swap, and GPU devices via Linux cgroups. |
| `/etc/slurm/slurmdbd.conf` | Accounting daemon + MariaDB backend. Must be mode `0600`, owned by `slurm`. |

## Daemons running on `enif`

- **`munge`** — authenticates messages between all SLURM daemons.
- **`slurmdbd`** — writes job accounting records to MariaDB.
- **`slurmctld`** — the brain: receives `sbatch`/`srun`, schedules jobs, tracks state.
- **`slurmd`** — the worker: launches jobs on the node, sets up cgroups, reports status.

## How users submit jobs

`main` is the only partition and is the default, so `-p main` is optional. A
bare `--gres=gpu:h100:1` automatically gets 8 cores and 64 GB RAM
(`DefCpuPerGPU` / `DefMemPerGPU` in `slurm.conf`); pass `-c` and `--mem`
explicitly if you want different sizing.

```bash
sbatch --gres=gpu:h100:1                    job.sh   # 1 whole GPU, 8 c / 64G defaults
sbatch --gres=gpu:h100:2  -c 16 --mem=128G  job.sh   # NVLink pair, explicit sizes
sbatch --gres=shard:h100:3 -c 4 --mem=32G   job.sh   # share a GPU
sbatch -c 8 --mem=32G                       job.sh   # CPU-only (no --gres)
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

`slurm.conf` already sets `SlurmctldParameters=enable_configless`, so the new
node's `slurmd` can fetch `slurm.conf` and `cgroup.conf` from the controller
at startup — you don't need to copy them.

1. Install `slurm-wlm` + `munge` on the new host (NVIDIA stack already in place).
2. Copy `/etc/munge/munge.key` from `enif` (same permissions: `munge:munge 400`).
3. Place a node-local `/etc/slurm/gres.conf` matching the new node's own GPU topology.
   `cgroup.conf` is per-node and `slurm.conf` is fetched from the controller — neither needs copying.
4. Ensure both hosts resolve each other (`/etc/hosts` or DNS).
5. On the **controller**: append a `NodeName=<new>` line and add the new node to `Nodes=` in the `main` partition of `slurm.conf`, then `sudo scontrol reconfigure`.
6. On the **new node**: `sudo systemctl enable --now slurmd` with the unit configured to start with `--conf-server=enif:6817` (Ubuntu's default unit reads `/etc/default/slurmd`; add `SLURMD_OPTIONS="--conf-server=enif:6817"`).
7. Open the firewall on the new node: 6817/tcp outbound to `enif`, 6818/tcp inbound from the cluster subnet.

## Troubleshooting checklist

- **Node `down`/`drain`**: `scontrol show node enif` shows the reason; common causes are `RealMemory` mismatch with `slurmd -C`, or `gres.conf` listing a `/dev/nvidiaN` that doesn't exist.
- **Jobs stuck in `PD` (pending)**: `squeue --start` and `scontrol show job <id>` give the reason (resources, priority, association missing).
- **`Invalid account`**: the user wasn't added with `sacctmgr` — required because we set `AccountingStorageEnforce=associations,limits,qos` (strict from day one). Add with `sudo sacctmgr -i add user <name> DefaultAccount=default`.
- **`slurmdbd` won't start**: check that `slurmdbd.conf` is mode `0600`, owned by `slurm`, and that MariaDB is reachable with the configured password.

> **Note on MariaDB**: `setup.sh` leaves MariaDB at Ubuntu's default — root via unix_socket, slurm user reachable only on `localhost`, no network listener exposed externally. Review and harden (`bind-address`, root password, TLS) before pointing a future second node at this DBD over the network.
