##
##  Copyright (c) 2017, all rights reserved.  Use of this source code 
##  is governed by a BSD license that can be found in the top-level
##  LICENSE file.
##

from mpi4py import MPI

import sys
import numpy as np
import numpy.testing as nt

from shmem import MPIShared
from locking import MPILock


comm = MPI.COMM_WORLD
rank = comm.rank
procs = comm.size

# Test simple locking
#========================

lock = MPILock(comm)

msg = "process {} got the lock".format(rank)

lock.lock()
print(msg)
sys.stdout.flush()
lock.unlock()


# Test shared memory
#========================

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

# Data type of our shared memory array
datatype = np.float64

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
                setdata = local[setoffset[0]:setoffset[0]+updatedims[0],
                                setoffset[1]:setoffset[1]+updatedims[1],
                                setoffset[2]:setoffset[2]+updatedims[2]]
            try:
                # All processes call set(), but only data on rank p matters.
                shm.set(setdata, setoffset, p)
            except:
                print("proc {} threw exception during set()".format(rank))
                sys.stdout.flush()
                comm.Abort()
            
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
        check[:,:,:] = shm[:,:,:]

        truth = np.ones_like(local)
        truth *= p

        # This should be bitwise identical, even for floats
        nt.assert_equal(check[:,:,:], truth[:,:,:])
