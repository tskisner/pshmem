##
##  Copyright (c) 2017, all rights reserved.  Use of this source code 
##  is governed by a BSD license that can be found in the top-level
##  LICENSE file.
##

from mpi4py import MPI

import sys
import numpy as np


class MPIShared(object):
    """
    Create a shared memory buffer that is replicated across nodes.

    For the given array dimensions and datatype, the original communicator
    is split into groups of processes that can share memory (i.e. that are
    on the same node).

    The values of the memory buffer can be set by one process at a time.
    When the set() method is called the data passed by the specified
    process is replicated to all nodes and then copied into the desired
    place in the shared memory buffer on each node.  This way the shared
    buffer on each node is identical.

    All processes across all nodes may do read-only access to their node-
    local copy of the buffer, simply by using the standard array indexing
    notation ("[]") on the object itself.

    Args:
        shape (tuple): the dimensions of the array.
        dtype (np.dtype): the data type of the array.
        comm (MPI.Comm): the full communicator to use.  This may span
            multiple nodes, and each node will have a copy.
    """
    def __init__(self, shape, dtype, comm):
        self._shape = shape
        self._dtype = dtype

        # Global communicator.
        
        self._comm = comm
        self._rank = 0
        self._procs = 1
        if self._comm is not None:
            self._rank = self._comm.rank
            self._procs = self._comm.size
        
        # Compute the flat-packed buffer size.

        self._n = 1
        for d in self._shape:
            self._n *= d

        # Split our communicator into groups on the same node.  Also
        # create an inter-node communicator between corresponding
        # processes on all nodes (for use in "setting" slices of the
        # buffer.
        
        self._nodecomm = None
        self._rankcomm = None
        self._noderank = 0
        self._nodeprocs = 1
        self._nodes = 1
        self._mynode = 0
        if self._comm is not None:
            self._nodecomm = self._comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
            self._noderank = self._nodecomm.rank
            self._nodeprocs = self._nodecomm.size
            self._nodes = self._procs // self._nodeprocs
            if self._nodes * self._nodeprocs < self._procs:
                self._nodes += 1
            self._mynode = self._rank // self._nodeprocs
            self._rankcomm = self._comm.Split(self._noderank, self._mynode)

        # Consider a corner case of the previous calculation.  Imagine that
        # the number of processes is not evenly divisible by the number of
        # processes per node.  In that case, when we later use the set()
        # method, the rank-wise communicator may not have a member on the
        # final node.  Here we compute the "highest" rank within a node which
        # is present on all nodes.  That sets the possible allowed processes
        # which may call the set() method.

        dist = self._disthelper(self._procs, self._nodes)
        self._maxsetrank = dist[-1][1] - 1

        # Divide up the total memory size among the processes on each
        # node.  For reasonable NUMA settings, this should spread the
        # allocated memory to locations across the node.

        # FIXME: the above statement works fine for allocating the window,
        # and it is also great in C/C++ where the pointer to the start of 
        # the buffer is all you need.  In mpi4py, querying the rank-0 buffer
        # returns a buffer-interface-compatible object, not just a pointer.
        # And this "buffer object" has the size of just the rank-0 allocated
        # data.  SO, for now, disable this and have rank 0 allocate the whole
        # thing.  We should change this back once we figure out how to
        # take the raw pointer from rank zero and present it to numpy as the
        # the full buffer.

        # dist = self._disthelper(self._n, self._nodeprocs)
        # self._localoffset, self._nlocal = dist[self._noderank]
        if self._noderank == 0:
            self._localoffset = 0
            self._nlocal = self._n
        else:
            self._localoffset = 0
            self._nlocal = 0

        # Allocate the shared memory buffer and wrap it in a 
        # numpy array.  If the communicator is None, just make
        # a normal numpy array.

        self._mpitype = None
        self._win = None
        self._buffer = None
        self._dbuf = None
        self._flat = None
        self._data = None

        if self._comm is not None:
            # We are actually using MPI, so we need to ensure that
            # our specified numpy dtype has a corresponding MPI datatype.
            status = 0
            try:
                # Technically this is an internal variable, but online
                # forum posts from the developers indicate this is stable
                # at least until a public interface is created.
                self._mpitype = MPI._typedict[self._dtype.char]
            except:
                status = 1
            self._checkabort(self._comm, status, 
                "numpy to MPI type conversion")

            # Number of bytes in our buffer
            dsize = self._mpitype.Get_size()
            nbytes = self._nlocal * dsize

            # Every process allocates a piece of the buffer.  The per-
            # process pieces are guaranteed to be contiguous.
            status = 0
            try:
                self._win = MPI.Win.Allocate_shared(nbytes, dsize, 
                    comm=self._nodecomm)
            except:
                status = 1
            self._checkabort(self._nodecomm, status, 
                "shared memory allocation")

            # Every process looks up the memory address of rank zero's piece,
            # which is the start of the contiguous shared buffer.
            status = 0
            try:
                self._buffer, dsize = self._win.Shared_query(0)
            except:
                status = 1
            self._checkabort(self._nodecomm, status, "shared memory query")

            # Create a numpy array which acts as a "view" of the buffer.
            self._dbuf = np.array(self._buffer, dtype="B", copy=False)
            self._flat = self._dbuf.view(self._dtype)
            self._data = self._flat.reshape(self._shape)

            # Initialize to zero.  Any of the processes could do this to the
            # whole buffer, but it is safe and easy for each process to just
            # initialize its local piece.

            # FIXME: change this back once every process is allocating a 
            # piece of the buffer.            
            # self._flat[self._localoffset:self._localoffset + self._nlocal] = 0
            if self._noderank == 0:
                self._flat[:] = 0

        else:
            self._data = np.zeros(_n, dtype=_dtype).reshape(_shape)


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
    def shape(self):
        """
        The tuple of dimensions of the shared array.
        """
        return self._shape


    @property
    def dtype(self):
        """
        The numpy datatype of the shared array.
        """
        return self._dtype


    @property
    def comm(self):
        """
        The full communicator.
        """
        return self._comm


    @property
    def nodecomm(self):
        """
        The node-local communicator.
        """
        return self._nodecomm


    def _disthelper(self, n, groups):
        dist = []
        for i in range(groups):
            myn = n // groups
            first = 0
            leftover = n % groups
            if i < leftover:
                myn += 1
                first = i * myn
            else:
                first = ((myn + 1) * leftover) + (myn * (i - leftover))
            dist.append( (first, myn) )
        return dist


    def _checkabort(self, comm, status, msg):
        failed = comm.allreduce(status, op=MPI.SUM)
        if failed > 0:
            if comm.rank == 0:
                print("MPIShared: one or more processes failed: {}".format(
                    msg))
            comm.Abort()
        return


    def set(self, data, offset, fromrank=0):
        """
        Set the values of a slice of the shared array.

        This call is collective across the full communicator, but only the
        data input from process "fromrank" is meaningful.  The offset 
        specifies the starting element along each dimension when copying
        the data into the shared array.  Regardless of which node the 
        "fromrank" process is on, the data will be replicated to the
        shared memory buffer on all nodes.

        Args:
            data (array): a numpy array with the same number of dimensions
                as the full array.
            offset (tuple): the starting offset along each dimension, which
                determines where the input data should be inserted into the
                shared array.
            fromrank (int): the process rank of the full communicator which
                is passing in the data.

        Returns:
            Nothing
        """
        # First check that the dimensions of the data and the offset tuple
        # match the shape of the data.

        if self._rank == fromrank:
            if len(data.shape) != len(self._shape):
                msg = "data has incompatible number of dimensions"
                if self._comm is not None:
                    print(msg)
                    self._comm.Abort()
                else:
                    raise RuntimeError(msg)
            if len(offset) != len(self._shape):
                msg = "offset tuple has incompatible number of dimensions"
                if self._comm is not None:
                    print(msg)
                    self._comm.Abort()
                else:
                    raise RuntimeError(msg)

        # The input data is coming from exactly one process on one node.
        # First, we broadcast the data from this process to the same node-rank
        # process on each of the nodes.

        if self._comm is not None:
            target_noderank = self._comm.bcast(self._noderank, root=fromrank)
            fromnode = self._comm.bcast(self._mynode, root=fromrank)
            
            # Verify that the node rank with the data actually has a member on
            # every node (see notes in the constructor).
            if target_noderank > self._maxsetrank:
                if self._rank == 0:
                    print("set() called with data from a node rank which does"
                        " not exist on all nodes")
                    self._comm.Abort()

            nodedata = None
            if self._noderank == target_noderank:
                # We are the lucky process on this node that gets to write
                # the data into shared memory!

                nodedata = self._rankcomm.bcast(data, root=fromnode)

                # Now one process on every node has a copy of the data, and
                # can copy it into the shared memory buffer.

                dslice = []
                ndims = len(data.shape)
                for d in range(ndims):
                    dslice.append( slice(offset[d], offset[d]+data.shape[d]) )
                slc = tuple(dslice)

                # Get a write-lock on the shared memory
                self._win.Lock(self._noderank, MPI.LOCK_EXCLUSIVE)

                # Copy data slice
                self._data[slc] = data

                # Release the write-lock
                self._win.Unlock(self._noderank)

        else:
            # We are just copying to a numpy array...
            dslice = []
            ndims = len(data.shape)
            for d in range(ndims):
                dslice.append( slice(offset[d], offset[d]+data.shape[d]) )
            slc = tuple(dslice)

            self._data[slc] = data

        return


    def __getitem__(self, key):
        return self._data[key]


    def __setitem__(self, key, value):
        raise NotImplementedError("Setting individual array elements not"
            " supported.  Use the set() method instead.")

