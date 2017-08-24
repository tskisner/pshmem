##
##  Copyright (c) 2017, all rights reserved.  Use of this source code 
##  is governed by a BSD license that can be found in the top-level
##  LICENSE file.
##

from mpi4py import MPI

import sys
import numpy as np


class MPILock(object):
    """
    Manage a lock using MPI shared memory.

    The lock is created across the given communicator.  This simply uses
    a single-element integer buffer as a way of controlling process flow.
    Nothing is actually written to or read from the buffer.

    Args:
        comm (MPI.Comm): the full communicator to use.
    """
    def __init__(self, comm):
        self._comm = comm
        self._rank = 0
        self._procs = 1
        if self._comm is not None:
            self._rank = self._comm.rank
            self._procs = self._comm.size

        if self._rank == 0:
            self._nlocal = 1
        else:
            self._nlocal = 0

        # Allocate the shared memory buffer.

        self._mpitype = None
        self._win = None

        if self._comm is not None:
            # We are actually using MPI, so we need to ensure that
            # our specified numpy dtype has a corresponding MPI datatype.
            status = 0
            try:
                # Technically this is an internal variable, but online
                # forum posts from the developers indicate this is stable
                # at least until a public interface is created.
                self._mpitype = MPI._typedict[np.dtype("int32").char]
            except:
                status = 1
            self._checkabort(self._comm, status,
                "numpy to MPI type conversion")

            # Number of bytes in our buffer
            dsize = self._mpitype.Get_size()
            nbytes = self._nlocal * dsize

            # Rank zero allocates the buffer
            status = 0
            try:
                self._win = MPI.Win.Allocate_shared(nbytes, dsize, 
                    comm=self._comm)
            except:
                status = 1
            self._checkabort(self._comm, status, 
                "shared memory allocation")


    @property
    def comm(self):
        """
        The communicator.
        """
        return self._comm


    def _checkabort(self, comm, status, msg):
        failed = comm.allreduce(status, op=MPI.SUM)
        if failed > 0:
            if comm.rank == 0:
                print("MPIShared: one or more processes failed: {}".format(
                    msg))
            comm.Abort()
        return


    def lock(self):
        """
        Request the lock and wait.

        This call blocks until lock is available.
        """
        self._win.Lock(self._rank, MPI.LOCK_EXCLUSIVE)
        return


    def unlock(self):
        """
        Unlock and return.
        """
        self._win.Unlock(self._rank)
        return

