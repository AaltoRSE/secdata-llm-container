# gpu-queue — a minimal single-GPU job queue

A tiny **file-based** queue for sharing **one GPU** among a small, trusted team (~6 people) on SECDATA. One worker process on the GPU node runs jobs one at a time.

Each job carries **its own `.sif`**, so different users/tasks can use different container images.

This is a cooperative tool, not a production scheduler. Everyone who uses it is also
responsible for keeping it healthy — starting the worker, cancelling stale jobs, and
fixing stuck state by hand when something goes wrong.

## How it works

```
VDI / any host              NFS  $GPU_QUEUE_ROOT                 GPU node
----------------            ---------------------                --------
gpu-queue-submit  ──▶  pending/  running/  done/  failed/  ◀── gpu-queue-worker
gpu-queue-status  ◀──  logs/   worker.pid   .gpu.lock          (one job at a time,
manual rm/cancel  ──▶  pending/                               flock on .gpu.lock)
                       gpu-queue-start / gpu-queue-stop
```

### Lifecycle of a job

1. **Submit** (`gpu-queue-submit`): writes a `<id>.job` descriptor into `pending/`.
   The write is atomic (temp file + `mv`).
2. **Worker poll** (`gpu-queue-worker`): every `GPU_QUEUE_POLL` seconds (default 10),
   lists `pending/*.job` and picks the **first** file returned by `ls` (see
   [Ordering](#ordering) below).
3. **Claim**: the worker `mv`s the descriptor from `pending/` to `running/` **before**
   validating it. This prevents a broken descriptor from being retried forever.
4. **Run**: the worker `source`s the descriptor, then runs:
   ```bash
   singularity exec --nv -B /nfs --no-home <sif> bash <job-script>
   ```
   Optional wall-clock limit via `timeout` when `--time` was set at submit time.
5. **Finish**: on success, `mv` to `done/`; on any failure, `mv` to `failed/`.
   stdout/stderr append to `logs/<id>.log`.

### Mutual exclusion

Only one job may use the GPU at a time. The worker opens `.gpu.lock` and acquires an
exclusive `flock` before each job. If the lock is held, it sleeps 5 seconds and retries.
When the worker process dies, the kernel releases the lock automatically — there is no
separate “stale lock file” to clear.

### Job descriptor format

Each `<id>.job` file is a small bash snippet created by `gpu-queue-submit`:

```bash
USER=alice
SIF=/path/to/your_image.sif
JOB_SCRIPT=/path/to/your_job_script.sh
TIME_LIMIT=4h          # empty if not set
SUBMITTED=2026-06-05T14:30:00+00:00
```

Job ids look like `YYYYMMDD-HHMMSS-<user>-<name>` (name is sanitized to
`[A-Za-z0-9._-]`).

### NFS layout

```
$GPU_QUEUE_ROOT/
├── pending/          # waiting jobs (*.job)
├── running/          # job currently claimed by the worker
├── done/             # finished successfully
├── failed/           # exited non-zero or failed validation
├── logs/
│   ├── worker.log    # worker stdout/stderr
│   └── <job-id>.log  # per-job output
├── bin/              # installed scripts (see setup)
├── worker.pid        # pid written by gpu-queue-start
└── .gpu.lock         # flock file (created when worker runs)
```



## Start the worker (on the GPU node)

```bash
ssh <gpu-node>
bash /nfs/data/gpu-queue/gpu-queue-start          
bash /nfs/data/gpu/queue/gpu-queue-status        
```

`gpu-queue-start` is idempotent: if `worker.pid` points to a live process, it exits
without starting a second worker. Stale pid files are removed automatically.

After a GPU-node reboot, someone runs `gpu-queue-start` again. 

### Stop the worker

```bash
bash /nfs/data/gpu-queue/gpu-queue-stop                    # signal worker to exit; running container may continue
```

## Submit a job (any user, from VDI or GPU node)

A job script is bash script that runs **inside** the container:

```bash
#!/bin/bash
python /nfs/home/alice/project/train.py
```

Submit it with the container image (.sif file) you want:

```bash
gpu-queue-submit \
  --sif /path/to/your_image.sif \
  --job /path/to/your_job_script.sh \
  --name alice-train \
  --time 4h            # optional wall-clock limit (timeout syntax: 3600, 2h, 30m)
```

Paths should be **absolute paths** at submit time so the GPU-node worker can find them.

## Monitor, cancel, and inspect

```bash
gpu-queue-status                                   # worker state + queue counts
tail -f "/nfs/data/gpu-queue/logs/<job-id>.log"        # live job output
tail -f "/nfs/data/gpu-queue/logs/worker.log"        # worker errors / crash loop
```

### Cancel a pending job

There is no `gpu-queue-cancel` script — cancel by removing the descriptor:

```bash
rm "/nfs/data/gpu-queue/pending/<job-id>.job"
```

Double-check the id with `gpu-queue-status` first. Only remove files in `pending/`;
never delete a file from `running/` while the worker is alive (race with the worker).

### Stop a running job

On the GPU node:

```bash
gpu-queue-stop --kill-running
```

Then clean up `running/` manually (see below). If a Singularity process is still alive,
find and kill it:

```bash
ps aux | grep -E 'singularity'
kill <pid>    # or kill -9 if needed
nvidia-smi    # confirm GPU is free
```

## Ordering

The worker picks `ls -1 pending/*.job | head -1` — **filesystem order**, not necessarily
submission time. In practice, ids begin with `YYYYMMDD-HHMMSS`, so `ls` order usually
matches FIFO within a day. `gpu-queue-status` lists pending jobs sorted for readability;
that order may differ from what the worker runs next.

## Scripts

| Script | Where | Purpose |
|--------|-------|---------|
| `gpu-queue-submit` | VDI / GPU node | Enqueue a job (`--sif`, `--job`, `--name`, `--time`) |
| `gpu-queue-worker` | GPU node | Run pending jobs one at a time (invoked by `gpu-queue-start`) |
| `gpu-queue-start`  | GPU node | Start the worker detached (`setsid` + `nohup`, writes `worker.pid`) |
| `gpu-queue-stop`   | GPU node | Stop the worker (`--kill-running` to also stop the active container) |
| `gpu-queue-status` | VDI / GPU node | Show worker state, counts, pending and running job ids |


## Troubleshooting & manual fixes

Because state is just files on NFS, most problems are visible and fixable by hand.
**When in doubt:** check `gpu-queue-status`, read the relevant log, then move or remove
the `.job` file in the right directory.

### Worker not running, jobs pile up in `pending/`

```bash
ssh <gpu-node>
gpu-queue-start
gpu-queue-status
```

### `gpu-queue-status` says NOT running but `worker.pid` exists

Stale pid file (worker crashed without cleanup). Safe to remove and restart:

```bash
rm "/nfs/data/gpu-queue/worker.pid"
ssh <gpu-node> gpu-queue-start
```

`gpu-queue-start` and `gpu-queue-stop` also clean stale pid files automatically.

### Job stuck in `running/`

The worker died, was killed, or the node rebooted mid-job.

1. Read `logs/<job-id>.log` to see how far it got.
2. Confirm nothing is still on the GPU (`nvidia-smi`, `ps`).
3. Move the descriptor:
   - **Retry:** `mv running/<id>.job pending/<id>.job`
   - **Abandon:** `mv running/<id>.job failed/<id>.job`
4. Ensure the worker is running: `gpu-queue-start` on the GPU node.

### Job in `failed/` — should it run again?

Inspect the log, fix the script/image/path, then resubmit with `gpu-queue-submit` (new id).

### Job keeps failing immediately

Open the descriptor and the log. Common causes:

| Log message | Fix |
|-------------|-----|
| `Bad job descriptor` | Descriptor is corrupt or not valid bash — fix or delete it |
| `Missing SIF` / `Missing JOB_SCRIPT` | Edit the `.job` file or resubmit |
| `Missing .sif` / `Missing job script` | Path wrong or file deleted; fix path and resubmit |
| `timeout: ...` | Job exceeded `--time`; resubmit with a longer limit or no limit |

### Queue blocked by a bad pending job

If the **first** pending job (per `ls` order) is broken, nothing behind it runs until
that descriptor is moved to `failed/` or fixed:

```bash
mv "/nfs/data/gpu-queue/pending/<bad-id>.job" "/nfs/data/gpu-queue/failed/<bad-id>.job"
```

### GPU still in use after stop

`gpu-queue-stop` without `--kill-running` may leave Singularity running (see above).
Use `--kill-running`, then kill any leftover processes manually and verify with
`nvidia-smi`.

### Two workers accidentally started

`flock` on `.gpu.lock` still ensures only one job runs at a time, but extra workers waste
CPU and add log noise. Find duplicate worker processes on the GPU node, kill the extras,
and keep a single `gpu-queue-start`.

### NFS glitches (`mv` fails, stale handles)

Symptoms: jobs not moving between directories, worker log shows intermittent errors.
Wait for NFS to recover, restart the worker, and retry the `mv` by hand if a descriptor
is in an unexpected directory.

### Cleaning up old jobs

`done/` and `failed/` are not pruned automatically. Periodically archive or delete old
`.job` files and matching logs:

```bash
ls "/done/"
rm "/nfs/data/gpu-queue/done/<old-id>.job" "/nfs/data/gpu-queue/logs/<old-id>.log"
```

### Quick health checklist

```bash
bash /nfs/data/gpu-queue/bin/gpu-queue-status
tail -20 "/nfs/data/gpu-queue/logs/worker.log"
ssh <gpu-node> nvidia-smi
```

## Limitations

- **Worker must be running** or jobs wait in `pending/` forever.
- **Single GPU**, one job at a time — no priorities, fair-share, or reservations.
- **Cooperative security** — shared write access to the queue dir means shared responsibility.
- All jobs run with `singularity exec --nv -B /nfs --no-home`; extra bind mounts require
  changing the worker script or working within paths already under `/nfs`.
- **No automatic retry** — failed jobs stay in `failed/` until someone moves or resubmits them.
- **Stop semantics are coarse** — stopping the worker does not always stop the container;
  killing and manual cleanup are sometimes required.
