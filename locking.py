##
##  Copyright (c) 2017, all rights reserved.  Use of this source code 
##  is governed by a BSD license that can be found in the top-level
##  LICENSE file.
##

import sys
import itertools

import numpy as np


class MPILock(object):
    """
    Implement a MUTEX lock with MPI one-sided operations.

    The lock is created across the given communicator.  This uses an array
    of bytes (one per process) to track which processes have requested the
    lock.  When a given process releases the lock, it passes it to the next
    process in line.

    Args:
        comm (MPI.Comm): the full communicator to use.
        root (int): the rank which stores the list of waiting processes.
        debug (bool): if True, print debugging info about the lock status.
    """
    # This creates a new integer for each time the class is instantiated.
    newid = next(itertools.count())

    def __init__(self, comm, root=0, debug=False):
        self._comm = comm
        self._root = root
        self._debug = debug

        # A unique tag for each instance of the class
        self._tag = MPILock.newid

        self._rank = 0
        self._procs = 1
        if self._comm is not None:
            self._rank = self._comm.rank
            self._procs = self._comm.size

        if self._rank == self._root:
            self._nlocal = self._procs
        else:
            self._nlocal = 0

        # Allocate the shared memory buffer.

        self._mpitype = None
        self._win = None
        self._have_lock = False
        self._waiting = None
        if self._rank == 0:
            self._waiting = np.zeros((self._procs,), dtype=np.uint8)

        if self._comm is not None:
            from mpi4py import MPI
            # Root allocates the buffer
            status = 0
            try:
                self._win = MPI.Win.Create(self._waiting, comm=self._comm)
            except:
                if self._debug:
                    print("rank {} win create raised exception".format(self._rank), 
                        flush=True)
                status = 1
            self._checkabort(self._comm, status, 
                "shared memory allocation")

            self._comm.barrier()


    def __del__(self):
        self.close()
        

    def __enter__(self):
        return self


    def __exit__(self, type, value, tb):
        self.close()
        return False


    def close(self):
        # The shared memory window is automatically freed
        # when the class instance is garbage collected.
        # This function is for any other clean up on destruction.
        return


    @property
    def comm(self):
        """
        The communicator.
        """
        return self._comm


    def _checkabort(self, comm, status, msg):
        from mpi4py import MPI
        failed = comm.allreduce(status, op=MPI.SUM)
        if failed > 0:
            if comm.rank == self._root:
                print("MPIShared: one or more processes failed: {}".format(
                    msg))
            comm.Abort()
        return


    def lock(self):
        """
        Request the lock and wait.

        This call blocks until lock is available.  Then it acquires
        the lock and returns.
        """
        # Do we already have the lock?
        if self._have_lock:
            return

        if self._comm is not None:
            from mpi4py import MPI
            waiting = np.zeros((self._procs,), dtype=np.uint8)
            lock = np.zeros((1,), dtype=np.uint8)
            lock[0] = 1
            
            # lock the window
            if self._debug:
                print("lock:  rank {}, instance {} locking shared window".format(
                    self._rank, self._tag), flush=True)
            self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)

            # add ourselves to the list of waiting ranks
            if self._debug:
                print("lock:  rank {}, instance {} putting rank".format(
                    self._rank, self._tag), flush=True)
            self._win.Put([lock, 1, MPI.UNSIGNED_CHAR], self._root, target=self._rank)
            
            # get the full list of current processes waiting or running
            if self._debug:
                print("lock:  rank {}, instance {} getting waitlist".format(
                    self._rank, self._tag), flush=True)
            self._win.Get([waiting, self._procs, MPI.UNSIGNED_CHAR], self._root)
            if self._debug:
                print("lock:  rank {}, instance {} list = {}".format(self._rank,
                    self._tag, waiting), flush=True)

            self._win.Flush(self._root)
            
            # unlock the window
            if self._debug:
                print("lock:  rank {}, instance {} unlocking shared window".format(
                    self._rank, self._tag), flush=True)
            self._win.Unlock(self._root)

            # Go through the list of waiting processes.  If any one is
            # active or waiting, then wait for a signal that we can have 
            # the lock.
            for p in range(self._procs):
                if (waiting[p] == 1) and (p != self._rank):
                    # we have to wait...
                    if self._debug:
                        print("lock:  rank {} waiting for the lock".format(self._rank),
                         flush=True)
                    self._comm.Recv(lock, source=MPI.ANY_SOURCE, tag=self._tag)
                    if self._debug:
                        print("lock:  rank {} got the lock".format(self._rank),
                         flush=True)
                    break

        # We have the lock now!
        self._have_lock = True
        return


    def unlock(self):
        """
        Unlock and return.
        """
        # Do we even have the lock?
        if not self._have_lock:
            return

        if self._comm is not None:
            from mpi4py import MPI
            waiting = np.zeros((self._procs,), dtype=np.uint8)
            lock = np.zeros((1,), dtype=np.uint8)
            
            # lock the window
            if self._debug:
                print("unlock:  rank {}, instance {} locking shared window"\
                    .format(self._rank, self._tag), flush=True)
            self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)

            # remove ourselves to the list of waiting ranks
            if self._debug:
                print("unlock:  rank {}, instance {} putting rank".format(
                    self._rank, self._tag), flush=True)
            self._win.Put([lock, 1, MPI.UNSIGNED_CHAR], self._root, 
                target=self._rank)
            
            # get the full list of current processes waiting or running
            if self._debug:
                print("unlock:  rank {}, instance {} getting waitlist".format(
                    self._rank, self._tag), flush=True)
            self._win.Get([waiting, self._procs, MPI.UNSIGNED_CHAR], 
                self._root)
            if self._debug:
                print("unlock:  rank {}, instance {} list = {}"\
                    .format(self._rank, self._tag, waiting), flush=True)

            self._win.Flush(self._root)
            
            # unlock the window
            if self._debug:
                print("unlock:  rank {}, instance {} unlocking shared window"\
                    .format(self._rank, self._tag), flush=True)
            self._win.Unlock(self._root)

            # Go through the list of waiting processes.  Pass the lock
            # to the next process.
            next = self._rank + 1
            for p in range(self._procs):
                nextrank = next % self._procs
                if waiting[nextrank] == 1:
                    if self._debug:
                        print("unlock:  rank {} passing lock to {}"\
                            .format(self._rank, nextrank), flush=True)
                    self._comm.Send(lock, nextrank, tag=self._tag)
                    self._have_lock = False
                    break
                next += 1
        else:
            self._have_lock = False
        
        return
