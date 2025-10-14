import random
import time
import numpy as np
from mpi4py import MPI

from pshmem import MPIBatch

comm = MPI.COMM_WORLD

def fake_task_work(wrk_comm):
    """A function which emulates the work for a single task.
    """
    # All processes in the worker group so something.
    slp = 0.2 + 0.2 * random.random()
    time.sleep(slp)
    # Wait until everyone in the group is done.
    if wrk_comm is not None:
        wrk_comm.barrier()

ntask = 10

# The top-level directory
task_dir = "test"

# The "names" (subdirectories) of each task
task_names = [f"task_{x:03d}" for x in range(ntask)]

# Two workers
worker_size = 1
if comm.size > 1:
    worker_size = comm.size // 2

# Create the batch system to track the state of tasks.
batch = MPIBatch(
    comm,
    worker_size,
    ntask,
    task_fs_root=task_dir,
    task_fs_names=task_names,
)

# Track the tasks executed by each worker to so we can
# display that at the end.  This variable is only for
# purposes of printing.
proc_tasks = batch.INVALID * np.ones(ntask, dtype=np.int32)

# Workers loop over tasks until there are no more left.
task = -1
while task is not None:
    task = batch.next_task()
    if task is None:
        # Nothing left for this worker
        break
    try:
        proc_tasks[task] = batch.RUNNING
        fake_task_work(batch.worker_comm)
        if batch.worker_rank == 0:
            # Only one process in the worker group needs
            # to update the state.
            batch.set_task_state(task, batch.DONE)
        proc_tasks[task] = batch.DONE
    except Exception:
        # The task raised an exception, mark this task
        # as failed.
        if batch.worker_rank == 0:
            # Only one process in the worker group needs
            # to update the state.
            batch.set_task_state(task, batch.FAILED)
        proc_tasks[task] = batch.FAILED

# Wait for all workers to finish
comm.barrier()

# Each worker reports on their status
for iwork in range(batch.n_worker):
    if iwork == batch.worker:
        if batch.worker_rank == 0:
            proc_stat = [MPIBatch.state_to_string(x) for x in proc_tasks]
            msg = f"Worker {batch.worker} tasks = {proc_stat}"
            print(msg, flush=True)
    batch.comm.barrier()

# Cleanup
del batch
