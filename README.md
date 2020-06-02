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

## MPILock Class

This implements a MUTEX lock across an arbitrary communicator.  A memory buffer on a
single process acts as a waiting list where processes can add themselves (using
one-sided calls).  The processes pass a token to transfer ownership of the lock.

## Tests

After installation, you can run some tests with:

    mpirun -np 4 python3 'import pshmem; pshmem.test()'

If you have mpi4py available but would like to explicitly disable the use of MPI in the
tests, you can set an environment variable:

    PSHMEM_MPI_DISABLE=1 python3 -c 'import pshmem; pshmem.test()'
