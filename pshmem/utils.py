##
# Copyright (c) 2017-2025, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import random
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm

import numpy as np


"""Backport the new `track` option from python 3.13 to older versions
More details at: https://github.com/python/cpython/issues/82300
"""
if sys.version_info >= (3, 13):
    SharedMemory = _mpshm.SharedMemory
else:
    class SharedMemory(_mpshm.SharedMemory):
        __lock = threading.Lock()

        def __init__(
            self, name=None, create=False, size=0, *, track=True
        ) -> None:
            self._track = track

            # if tracking, normal init will suffice
            if track:
                return super().__init__(name=name, create=create, size=size)

            # lock so that other threads don't attempt to use the
            # register function during this time
            with self.__lock:
                # temporarily disable registration during initialization
                orig_register = _mprt.register
                _mprt.register = self.__tmp_register

                # initialize; ensure original register function is
                # re-instated
                try:
                    super().__init__(name=name, create=create, size=size)
                finally:
                    _mprt.register = orig_register

        @staticmethod
        def __tmp_register(*args, **kwargs) -> None:
            return

        def unlink(self) -> None:
            if _mpshm._USE_POSIX and self._name:
                _mpshm._posixshmem.shm_unlink(self._name)
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")


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


@contextmanager
def exception_guard(comm=None, timeout=5):
    """Ensure if one MPI process raises an un-caught exception, the program shuts down.

    Args:
        comm (mpi4py.MPI.Comm): The MPI communicator or None.
        timeout (int): The number of seconds to wait before aborting all processes

    """
    rank = 0 if comm is None else comm.rank
    try:
        yield
    except Exception:
        # Note that the intention of this function is to handle *any* exception.
        # The typical use case is to wrap main() and ensure that the job exits
        # cleanly.
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [f"Proc {rank}: {x}" for x in lines]
        msg = "".join(lines)
        print(msg, flush=True)
        # kills the job
        if comm is None or comm.size == 1:
            # Raising the exception allows for debugging
            raise
        else:
            if comm.size > 1:
                # gives other processes a bit of time to see whether
                # they encounter the same error
                time.sleep(timeout)
            comm.Abort(1)
