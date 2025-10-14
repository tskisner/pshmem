# MPI design patterns with shared memory

This is a small package that implements parallel design patterns using MPI one-sided and
shared memory constructs.

## Installation and Requirements

This package needs a recent version of the `mpi4py` package in order to be useful.
However, the classes also accept a value of `None` for the communicator, in which case a
trivial local implementation is used.  The code uses other widely available packages
(like numpy) and requires a recent Python3 installation.

### Binary Packages

Wheels are available on PyPI:

    pip install pshmem

Or you can install packages from conda-forge:

    conda install -c conda-forge pshmem

### Installing from Source

You can install the code from a git checkout with:

    pip install .


## MPIShared Class

This class implements a pattern where a shared array is allocated on each node.
Processes can update pieces of the shared array with the synchronous "set()" method.
During this call, the data from the desired process is first replicated to all nodes,
and then one process on each node copies that piece into the shared array.

All processes on all nodes can freely read data from the node-local copy of the shared
array.

### Example

You can use `MPIShared` as a context manager or by explicitly creating and freeing
memory.  Here is an example of creating a shared memory object that is replicated across
nodes:

```python
import numpy as np
from mpi4py import MPI

from pshmem import MPIShared

comm = MPI.COMM_WORLD

with MPIShared((3, 5), np.float64, comm) as shm:
    # A copy of the data exists on every node and is initialized to zero.
    # There is a numpy array "view" of that memory available with slice notation
    # or by accessing the "data" member:
    if comm.rank == 0:
        # You can get a summary of the data by printing it:
        print("String representation:\n")
        print(shm)
        print("\n===== Initialized Data =====")
    for p in range(comm.size):
        if p == comm.rank:
            print("rank {}:\n".format(p), shm.data, flush=True)
        comm.barrier()

    set_data = None
    set_offset = None
    if comm.rank == 0:
        set_data = np.arange(6, dtype=np.float64).reshape((2, 3))
        set_offset = (1, 1)

    # The set() method is collective, but the inputs only matter on one rank
    shm.set(set_data, offset=set_offset, fromrank=0)

    # You can also use the usual '[]' notation.  However, this call must do an
    # additional pre-communication to detect which process the data is coming from.
    # And this line is still collective and must be called on all processes:
    shm[set_offset] = set_data

    # This updated data has now been replicated to the shared memory on all nodes.
    if comm.rank == 0:
        print("======= Updated Data =======")
    for p in range(comm.size):
        if p == comm.rank:
            print("rank {}:\n".format(p), shm.data, flush=True)
        comm.barrier()

    # You can read from the node-local copy of the data from all processes,
    # using either the "data" member or slice access:
    if comm.rank == comm.size - 1:
        print("==== Read-only access ======")
        print("rank {}: shm[2, 3] = {}".format(comm.rank, shm[2, 3]), flush=True)
        print("rank {}: shm.data = \n{}".format(comm.rank, shm.data), flush=True)

```

Putting the above code into a file `test.py` and running this on 4 processes gives:

```
mpirun -np 4 python3 test.py

String representation:

<MPIShared
  replicated on 1 nodes, each with 4 processes (4 total)
  shape = (3, 5), dtype = float64
  [ [0. 0. 0. 0. 0.] [0. 0. 0. 0. 0.] [0. 0. 0. 0. 0.] ]
>

===== Initialized Data =====
rank 0:
 [[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
rank 1:
 [[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
rank 2:
 [[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
rank 3:
 [[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
======= Updated Data =======
rank 0:
 [[0. 0. 0. 0. 0.]
 [0. 0. 1. 2. 0.]
 [0. 3. 4. 5. 0.]]
rank 1:
 [[0. 0. 0. 0. 0.]
 [0. 0. 1. 2. 0.]
 [0. 3. 4. 5. 0.]]
rank 2:
 [[0. 0. 0. 0. 0.]
 [0. 0. 1. 2. 0.]
 [0. 3. 4. 5. 0.]]
rank 3:
 [[0. 0. 0. 0. 0.]
 [0. 0. 1. 2. 0.]
 [0. 3. 4. 5. 0.]]
==== Read-only access ======
rank 3: shm[2, 3] = 5.0
rank 3: shm.data =
[[0. 0. 0. 0. 0.]
 [0. 0. 1. 2. 0.]
 [0. 3. 4. 5. 0.]]
 ```

Note that if you are not using a context manager, then you should be careful to close
and delete the object like this:

```python
shm = MPIShared((3, 5), np.float64, comm=comm)
# Do stuff
shm.close()
del shm
```

## MPILock Class

This implements a MUTEX lock across an arbitrary communicator.  A memory buffer on a
single process acts as a waiting list where processes can add themselves (using
one-sided calls).  The processes pass a token to transfer ownership of the lock.  The
token is passed in order of request.

### Example

A typical use case is where we want to serialize some operation across a large number of
processes that reside on different nodes.  For example, perhaps we are making requests
to the external network from a computing center and we do not want to saturate that with
all processes simultaneously.  Or perhaps we are writing to a shared data file which
does not support parallel writes and we have a sub-communicator of writing processes
which take turns updating the filesystem.  We can instantiate a lock on any
communicator, so it is possible to split the world communicator into groups and have
some operation serialized just within that group:

```python
with MPILock(MPI.COMM_WORLD) as mpilock:
    mpilock.lock()
    # Do something here.  Only one process at a time will do this.
    mpilock.unlock()
```

## MPIBatch Class

This class is useful for a common pattern where a pool of "workers", each
consisting of multiple processes, needs to complete "tasks" of varying size. The
fact that each worker is a group of processes means we cannot use the standard
MPIPoolExecutor from mpi4py. This scenario also means that it is difficult to
recover from errors like a segmentation fault (since there is no "nanny"
process). Still, there are many applications where groups of processes need to
work on a fixed number of tasks and dynamically assign those tasks to a smaller
number of workers.

By default, the state of tasks are tracked purely in MPI shared memory. The
calling code is responsible for initializing the state of each task from
external information. There is also support for simple use of the filesystem for
state tracking in addition to the in-memory copy. This assumes a top-level
directory with a subdirectory for each task. A special "state" file is created
in each task directory and the `MPIBatch` class ensures that only one process at
a time modifies that state file and the in-memory copy. Using the filesystem can
also help when running multiple batch instances that are working on the same set
of tasks.


### Example

Here is an example using `MPIBatch` to track the state of tasks using the
filesystem (not just in memory).  For that use case, the tasks must have
a "name" which is used as a subdirectory.  Note that if you run this script
twice, make sure to remove the output directory- otherwise nothing will
happen since all tasks are done.

```python
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
```

Putting this code into a script called `test_batch.py` and running it
produces:
```
mpirun -np 4 python3 test.py

Worker 0 tasks = ['DONE', 'INVALID', 'INVALID', 'DONE', 'INVALID', 'DONE', 'INVALID', 'DONE', 'DONE', 'INVALID']
Worker 1 tasks = ['INVALID', 'DONE', 'DONE', 'INVALID', 'DONE', 'INVALID', 'DONE', 'INVALID', 'INVALID', 'DONE']
```

So you can see that tasks are assigned to different worker groups as those workers
complete previous tasks.  The state is tracked on the filesystem with a `state` file
in each task directory.  After running the script above we can look at the contents
of those:
```
cat test/*/state
DONE
DONE
DONE
DONE
DONE
DONE
DONE
DONE
DONE
DONE
```

## Tests

After installation, you can run some tests with:

    mpirun -np 4 python3 -c 'import pshmem.test; pshmem.test.run()'

If you have mpi4py available but would like to explicitly disable the use of MPI in the
tests, you can set an environment variable:

    MPI_DISABLE=1 python3 -c 'import pshmem.test; pshmem.test.run()'
