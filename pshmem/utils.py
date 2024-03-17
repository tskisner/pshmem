##
# Copyright (c) 2017-2024, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import random
import sys
# Import for monkey patching resource tracker
from multiprocessing import resource_tracker

import numpy as np


def mpi_data_type(comm, dt):
    """Helper function to return the byte size and MPI datatype.

    Args:
        comm (mpi4py.Comm): The communicator, or None.
        dt (np.dtype): The datatype.

    Returns:
        (tuple):  The (bytesize, MPI type) of the input dtype.

    """
    dtyp = np.dtype(dt)
    dsize = None
    mpitype = None
    if comm is None:
        dsize = dtyp.itemsize
    else:
        from mpi4py import MPI

        # We are actually using MPI, so we need to ensure that
        # our specified numpy dtype has a corresponding MPI datatype.
        try:
            # Technically this is an internal variable, but online
            # forum posts from the developers indicate this is stable
            # at least until a public interface is created.
            mpitype = MPI._typedict[dtyp.char]
        except Exception:
            msg = "Process {} failed to get MPI data type for numpy dtype ".format(
                comm.rank
            )
            msg += "{}, char '{}'".format(dtyp, dtyp.char)
            print(msg, flush=True)
            raise
        dsize = mpitype.Get_size()
    return (dsize, mpitype)


def random_shm_key():
    """Get a random positive integer for using in shared memory naming.

    The python random library is used, and seeded with the default source
    (either system time or os.urandom).

    Returns:
        (int):  The random integer.

    """
    min_val = 0
    max_val = sys.maxsize
    # Seed with default source of randomness
    random.seed(a=None)
    return random.randint(min_val, max_val)


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]
