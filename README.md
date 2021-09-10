# MPI design patterns with shared memory

This is a small package that implements parallel design patterns using MPI one-sided and
shared memory constructs.

## Installation and Requirements

This package needs a recent version of the `mpi4py` package in order to be useful.
However, the classes also accept a value of `None` for the communicator, in which case a
trivial local implementation is used.  The code uses other widely available packages
(like numpy) and requires a recent Python3 installation.  You can install the code from
a git checkout with:

    pip install .

Or:

    python3 setup.py install

Or directly from github.

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

## Tests

After installation, you can run some tests with:

    mpirun -np 4 python3 -c 'import pshmem.test; pshmem.test.run()'

If you have mpi4py available but would like to explicitly disable the use of MPI in the
tests, you can set an environment variable:

    MPI_DISABLE=1 python3 -c 'import pshmem.test; pshmem.test.run()'
