# Jobs stuck in `pending/`, `running/` empty

Worker is alive but never claims (`mv pending → running` never happens). Usually a stuck NFS `flock` on `.gpu.lock`.

Do this on the **GPU node**. `QUEUE=/nfs/data/gpu-queue`.

## Is the worker blocked on the lock?

```bash
pid=$(cat "$QUEUE/worker.pid")
pstree -ap "$pid"
readlink /proc/$pid/fd/9          # should be $QUEUE/.gpu.lock
```

| `pstree` | Meaning |
|---|---|
| `sleep 5` | `flock -n` failed — lock busy |
| `sleep 10` | idle (no pending job, or wrong queue root) |
| `ls` / `singularity` | NFS hang, or a job is actually running |

```bash
flock -n "$QUEUE/.gpu.lock" -c 'echo GOT_LOCK'; echo exit:$?
```

No `GOT_LOCK` / exit 1 = exclusive lock denied.

`fuser -v "$QUEUE/.gpu.lock"` showing worker + `sleep` is one process tree (fd 9 inherited), not two holders.

## Local lock vs stale NFS lock

`lslocks | grep .gpu.lock` is unreliable (NFS locks often have no path). Match inode:

```bash
pgrep -af gpu-queue-worker
fuser -v "$QUEUE/.gpu.lock"
stat -c 'dev=%d inode=%i' "$QUEUE/.gpu.lock"
cat /proc/locks
```

If that inode is **not** in `/proc/locks`, and `pgrep`/`fuser` are empty on **both** GPU and login nodes, the lock is **stale on the NFS server**. `rm` of `.gpu.lock` does not clear it. Locks are per inode.

`strace -p` may fail (`ptrace_scope`); the `flock -n` test is enough.

## Replace the poisoned inode

Stop the worker first. `fuser` on `.gpu.lock` must be empty.

```bash
flock -n "$QUEUE/.gpu.lock.new" -c 'echo NEW_OK'; echo new:$?
mv "$QUEUE/.gpu.lock" "$QUEUE/.gpu.lock.stale"
touch "$QUEUE/.gpu.lock"
flock -n "$QUEUE/.gpu.lock" -c 'echo GOT_LOCK'; echo lock:$?
```

Leave `.gpu.lock.stale` in place. Then start one worker. Healthy idle is `sleep 10`, not `sleep 5`.

If even `.gpu.lock.new` fails, NFSv4 locking on the mount is broken — give the NFS admin the inode from `stat`.
