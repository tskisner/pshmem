##
# Copyright (c) 2017-2019, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

from mpi4py import MPI

import sys
import time

import numpy as np
import numpy.testing as nt

from shmem import MPIShared
from locking import MPILock


def run_test_lock(comm, sleepsec):
    """
    Test mutex locking using MPI one-sided operations.
    """
    rank = 0
    procs = 1
    if comm is not None:
        rank = comm.rank
        procs = comm.size

    lock = MPILock(comm, root=0, debug=True)

    msg = "test lock:  process {} got the lock".format(rank)

    lock.lock()
    print(msg)
    sys.stdout.flush()
    time.sleep(sleepsec)
    lock.unlock()

    if comm is not None:
        comm.barrier()

    return


def run_test_shmem(comm, datatype):
    """
    Test shared memory replication.
    """
    rank = 0
    procs = 1
    if comm is not None:
        rank = comm.rank
        procs = comm.size

    # Dimensions of our shared memory array
    datadims = (2, 5, 10)

    # Dimensions of the incremental slab that we will
    # copy during each set() call.
    updatedims = (1, 1, 5)

    # How many updates are there to cover the whole
    # data array?
    nupdate = 1
    for d in range(len(datadims)):
        nupdate *= datadims[d] // updatedims[d]

    # For testing the "set()" method, every process is going to
    # create a full-sized data buffer and fill it with its process rank.
    local = np.ones(datadims, dtype=datatype)
    local *= rank

    # A context manager is the pythonic way to make sure that the
    # object has no dangling reference counts after leaving the context,
    # and will ensure that the shared memory is freed properly.

    with MPIShared(local.shape, local.dtype, comm) as shm:

        for p in range(procs):
            # Every process takes turns writing to the buffer.
            setdata = None
            setoffset = (0, 0, 0)

            # Write to the whole data volume, but in small blocks
            for upd in range(nupdate):
                if p == rank:
                    # My turn!  Write my process rank to the buffer slab.
                    setdata = local[setoffset[0]:setoffset[0] + updatedims[0],
                                    setoffset[1]:setoffset[1] + updatedims[1],
                                    setoffset[2]:setoffset[2] + updatedims[2]]
                try:
                    # All processes call set(), but only data on rank p matters.
                    shm.set(setdata, setoffset, fromrank=p)
                except:
                    print("proc {} threw exception during set()".format(rank))
                    sys.stdout.flush()
                    if comm is not None:
                        comm.Abort()
                    else:
                        sys.exit(1)

                # Increment the write offset within the array

                x = setoffset[0]
                y = setoffset[1]
                z = setoffset[2]

                z += updatedims[2]
                if z >= datadims[2]:
                    z = 0
                    y += updatedims[1]
                if y >= datadims[1]:
                    y = 0
                    x += updatedims[0]

                setoffset = (x, y, z)

            # Every process is now going to read a copy from the shared memory
            # and make sure that they see the data written by the current process.
            check = np.zeros_like(local)
            check[:, :, :] = shm[:, :, :]

            truth = np.ones_like(local)
            truth *= p

            # This should be bitwise identical, even for floats
            nt.assert_equal(check[:, :, :], truth[:, :, :])

        # Ensure that we can reference the memory buffer from numpy without
        # a memory copy.  The intention is that a slice of the shared memory
        # buffer should appear as a C-contiguous ndarray whenever we slice
        # along the last dimension.

        for p in range(procs):
            if p == rank:
                slc = shm[1, 2]
                print("proc {} slice has dims {}, dtype {}, C = {}".format(
                    p, slc.shape, slc.dtype.str, slc.flags["C_CONTIGUOUS"]),
                    flush=True)
            if comm is not None:
                comm.barrier()

    return

# Run all tests with COMM_WORLD


comm = MPI.COMM_WORLD

# Locking

run_test_lock(comm, 2)

# Run shared memory test with a few different datatypes

run_test_shmem(comm, np.dtype(np.float64))
run_test_shmem(comm, np.dtype(np.float32))
run_test_shmem(comm, np.dtype(np.int64))
run_test_shmem(comm, np.dtype(np.int32))

# Run all tests with no communicator

comm = None

# Locking

run_test_lock(comm, 2)

# Run shared memory test with a few different datatypes

run_test_shmem(comm, np.dtype(np.float64))
run_test_shmem(comm, np.dtype(np.float32))
run_test_shmem(comm, np.dtype(np.int64))
run_test_shmem(comm, np.dtype(np.int32))
