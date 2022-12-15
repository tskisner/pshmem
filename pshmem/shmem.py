##
# Copyright (c) 2017-2020, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import sys
import mmap
import uuid

import numpy as np
import posix_ipc

from .utils import mpi_data_type


class MPIShared(object):
    """Create a shared memory buffer that is replicated across nodes.

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

    If comm is None, a simple local numpy array is used.

    Args:
        shape (tuple):  The dimensions of the array.
        dtype (np.dtype):  The data type of the array.
        comm (MPI.Comm):  The full communicator to use.  This may span
            multiple nodes, and each node will have a copy of the data.
        comm_node (MPI.Comm):  The communicator of processes within the
            same node.  If None, the node communicator will be created.
        comm_node_rank (MPI.Comm):  The communicator of processes with
            the same rank across all nodes.  If None, this will be
            created.

    """

    def __init__(self, shape, dtype, comm, comm_node=None, comm_node_rank=None):
        # Copy the datatype in order to support arguments that are aliases,
        # like "numpy.float64".
        self._dtype = np.dtype(dtype)

        # Verify that our shape contains only integral values
        self._n = 1
        for d in shape:
            if not isinstance(d, (int, np.integer)):
                msg = "input shape '{}' contains non-integer values".format(shape)
                raise ValueError(msg)
            if d < 0:
                msg = "input shape '{}' contains negative values".format(shape)
                raise ValueError(msg)
            self._n *= d

        self._shape = tuple(shape)

        # Global communicator.

        self._comm = comm
        self._rank = 0
        self._procs = 1
        if self._comm is not None:
            self._rank = self._comm.rank
            self._procs = self._comm.size

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
            from mpi4py import MPI

            self._free_comm_node = False
            if comm_node is None:
                # Create it
                self._nodecomm = self._comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
                self._free_comm_node = True
            else:
                # Check it
                if self._procs % comm_node.size != 0:
                    msg = "Node communicator size ({}) does not divide ".format(
                        comm_node.size
                    )
                    msg += "evenly into the total number of processes ({})".format(
                        self._procs
                    )
                    raise ValueError(msg)
                self._nodecomm = comm_node
            self._noderank = self._nodecomm.rank
            self._nodeprocs = self._nodecomm.size
            self._nodes = self._procs // self._nodeprocs
            if self._nodes * self._nodeprocs < self._procs:
                self._nodes += 1
            self._mynode = self._rank // self._nodeprocs

            self._free_comm_node_rank = False
            if comm_node_rank is None:
                # Create it
                self._rankcomm = self._comm.Split(self._noderank, self._mynode)
                self._free_comm_node_rank = True
            else:
                # Check it
                if comm_node_rank.size != self._nodes:
                    msg = "Node rank communicator size ({}) does not match ".format(
                        comm_node_rank.size
                    )
                    msg += "the number of nodes ({})".format(self._nodes)
                    raise ValueError(msg)
                self._rankcomm = comm_node_rank

        # Consider a corner case of the previous calculation.  Imagine that
        # the number of processes is not evenly divisible by the number of
        # processes per node.  In that case, when we later use the set()
        # method, the rank-wise communicator may not have a member on the
        # final node.  Here we compute the "highest" rank within a node which
        # is present on all nodes.  That sets the possible allowed processes
        # which may call the set() method.

        dist = self._disthelper(self._procs, self._nodes)
        self._maxsetrank = dist[-1][1] - 1

        # Compute the data sizes
        self._dsize, self._mpitype = mpi_data_type(self._comm, self._dtype)

        # Number of bytes used on our node
        nbytes = self._n * self._dsize

        # Every process will open a shared memory segment.  We use the same
        # unique name for this segment, based on the properties of the array
        # and a unique random ID.

        self._name = None
        if self._rank == 0:
            rng_str = uuid.uuid4().hex[:12]
            self._name = f"MPIShared_{rng_str}"
        if self._comm is not None:
            self._name = self._comm.bcast(self._name, root=0)

        # Only allocate our buffers if the total number of elements is > 0

        self._shmem = None
        self._shmap = None
        self._flat = None
        self.data = None

        if self._n > 0:
            if self._nodeprocs == 1:
                # There is only one process on a node, so we use a standard
                # numpy array rather than wrapped shared memory.  This helps
                # reduce the total number of open files, which is limited by
                # the kernel.
                self._flat = np.zeros(
                    self._n,
                    dtype=self._dtype,
                )
                # Wrap
                self.data = self._flat.reshape(self._shape)
            else:
                # First rank on each node creates the buffer
                if self._noderank == 0:
                    try:
                        self._shmem = posix_ipc.SharedMemory(
                            self._name,
                            posix_ipc.O_CREX,
                            size=int(nbytes),
                        )
                    except Exception as e:
                        msg = "Process {}: {}".format(self._rank, self._name)
                        msg += " failed allocation of {} bytes".format(nbytes)
                        msg += " ({} elements of {} bytes each)".format(
                            self._n, self._dsize
                        )
                        msg += ": {}".format(e)
                        print(msg, flush=True)
                        raise
                    try:
                        # MMap the shared memory
                        self._shmap = mmap.mmap(
                            self._shmem.fd,
                            self._shmem.size,
                        )
                    except Exception as e:
                        msg = "Process {}: {}".format(self._rank, self._name)
                        msg += " failed MMap of {} bytes".format(nbytes)
                        msg += " ({} elements of {} bytes each)".format(
                            self._n, self._dsize
                        )
                        msg += ": {}".format(e)
                        print(msg, flush=True)
                        # Try to free the shared memory object
                        try:
                            self._shmem.close_fd()
                            self._shmem.unlink()
                        except Exception as eclose:
                            pass
                        raise

                # Wait for that to be created
                if self._nodecomm is not None:
                    self._nodecomm.barrier()

                # Other ranks on the node attach
                if self._noderank != 0:
                    try:
                        self._shmem = posix_ipc.SharedMemory(self._name)
                        # MMap the shared memory
                        self._shmap = mmap.mmap(
                            self._shmem.fd,
                            self._shmem.size,
                        )
                    except Exception as e:
                        msg = "Process {}: {}".format(self._rank, self._name)
                        msg += " failed to attach buffer of {} bytes".format(nbytes)
                        msg += " ({} elements of {} bytes each)".format(
                            self._n, self._dsize
                        )
                        msg += ": {}".format(e)
                        print(msg, flush=True)
                        raise

                # Wait for other processes to attach
                if self._nodecomm is not None:
                    self._nodecomm.barrier()

                # Now that all processes have mmap'ed the shared memory we can
                # close the shared memory handle
                self._shmem.close_fd()

                # Wait for all processes to close file handle
                if self._nodecomm is not None:
                    self._nodecomm.barrier()

                # One process requests the file to be deleted, but this will not
                # actually happen until all processes release their mmap.
                if self._noderank == 0:
                    try:
                        self._shmem.unlink()
                    except posix_ipc.ExistentialError:
                        msg = "Process {}: {}".format(self._rank, self._name)
                        msg += " failed to unlink shared memory"
                        msg += ": {}".format(e)
                        print(msg, flush=True)
                        raise

                # Create a numpy array which acts as a view of the buffer.
                self._flat = np.ndarray(
                    self._n,
                    dtype=self._dtype,
                    buffer=self._shmap,
                )
                # Initialize to zero.
                if self._noderank == 0:
                    self._flat[:] = 0

                # Wrap
                self.data = self._flat.reshape(self._shape)



    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
        return False

    def __len__(self):
        if self.data is None:
            return 0
        else:
            return len(self.data)

    def __getitem__(self, key):
        if self.data is None:
            return None
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        if self.data is None:
            raise RuntimeError("Data size is zero- cannot assign elements")
        if self._comm is None:
            # shortcut for the serial case
            self.data[key] = value
            return
        # WARNING: Using this function will have a performance penalty over using
        # the explicit 'set()' method, since this function must first communicate to
        # find which process has the input data.
        import mpi4py.MPI as MPI

        check_rank = np.zeros((self._procs,), dtype=np.int32)
        check_result = np.zeros((self._procs,), dtype=np.int32)
        if value is not None:
            check_rank[self._rank] = 1
        self._comm.Allreduce([check_rank, MPI.INT], [check_result, MPI.INT], op=MPI.SUM)
        tot = np.sum(check_result)
        if tot > 1:
            msg = "When setting data with [] notation, there were "
            msg += "{} processes with a non-None value for the data".format(tot)
            msg += " instead of one process."
            raise RuntimeError(msg)

        from_rank = np.where(check_result == 1)[0][0]

        # compute the offset from the slice keys
        offset = None
        if self._rank == from_rank:
            offset = list()
            if isinstance(key, slice):
                # Just one dimension
                if key.start is None:
                    offset.append(0)
                else:
                    offset.append(key.start)
            else:
                # Is it iterable?
                try:
                    for k in key:
                        if isinstance(k, slice):
                            if k.start is None:
                                offset.append(0)
                            else:
                                offset.append(k.start)
                        else:
                            # Must be an index
                            if not isinstance(k, (int, np.integer)):
                                msg = "[] key elements must be a slice or integer"
                                raise ValueError(msg)
                            offset.append(k)
                except TypeError:
                    # No- must be an index
                    if not isinstance(key, (int, np.integer)):
                        msg = "[] key must be scalar or tuple of slices or integers"
                        raise ValueError(msg)
                    offset.append(key)
        self.set(value, offset=offset, fromrank=from_rank)

    def __iter__(self):
        """This provides sample-by-sample access."""
        if self.data is None:
            return iter(list())
        else:
            return iter(self.data)

    def __array__(self, *args, **kwargs):
        """Provide the underlying numpy data view as the array."""
        return self.data.__array__(*args, **kwargs)

    @property
    def __array_interface__(self):
        """Provide the underlying numpy data view as the array interface."""
        return self.data.__array_interface__

    def __repr__(self):
        val = "<MPIShared"
        val += "\n  replicated on {} nodes, each with {} processes ({} total)".format(
            self._nodes, self._nodeprocs, self._procs
        )
        val += "\n  shape = {}, dtype = {}".format(self._shape, self._dtype)

        if self.data is None:
            val += "\n No Data"
        else:
            if self._shape[0] <= 4:
                val += "\n  [ "
                for i in range(self._shape[0]):
                    val += "{} ".format(self.data[i])
                val += "]"
            else:
                val += "\n  [ {} {} ... {} {} ]".format(
                    self.data[0], self.data[1], self.data[-2], self.data[-1]
                )
        val += "\n>"
        return val

    def close(self):
        # Other processes close their access to the shared memory
        if hasattr(self, "data"):
            del self.data
        if hasattr(self, "_flat"):
            del self._flat
        if hasattr(self, "_shmap"):
            # Close the mmap'ed memory
            if self._shmap is not None:
                self._shmap.close()
                del self._shmap
                self._shmap = None
        if hasattr(self, "_shmem"):
            if self._shmem is not None:
                del self._shmem
                self._shmem = None

        self._flat = None
        self.data = None

        # Free other communicators if needed
        if (
            hasattr(self, "_rankcomm")
            and (self._rankcomm is not None)
            and self._free_comm_node_rank
        ):
            self._rankcomm.Free()
            self._rankcomm = None
        if (
            hasattr(self, "_nodecomm")
            and (self._nodecomm is not None)
            and self._free_comm_node
        ):
            self._nodecomm.Free()
            self._nodecomm = None

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
            dist.append((first, myn))
        return dist

    def set(self, data, offset=None, fromrank=0):
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
        # Explicit barrier here, to ensure that we don't try to update
        # data while other processes are reading.
        if self._comm is not None:
            self._comm.barrier()

        if self.data is None:
            raise RuntimeError("Data size is zero- cannot assign elements")

        # First check that the dimensions of the data and the offset tuple
        # match the shape of the data.

        if self._rank == fromrank:
            if len(data.shape) != len(self._shape):
                if len(data.shape) != len(self._shape):
                    msg = (
                        "input data dimensions {} incompatible with "
                        "buffer ({})".format(len(data.shape), len(self._shape))
                    )
                    raise RuntimeError(msg)
            if offset is None:
                offset = tuple([0 for x in self._shape])
            if len(offset) != len(self._shape):
                msg = (
                    "input offset dimensions {} incompatible with "
                    "buffer ({})".format(len(offset), len(self._shape))
                )
                raise RuntimeError(msg)
            if data.dtype != self._dtype:
                msg = (
                    "input data type ({}, {}) incompatible with "
                    "buffer ({}, {})".format(
                        data.dtype.str, data.dtype.num, self._dtype.str, self._dtype.num
                    )
                )
                raise RuntimeError(msg)

        # The input data is coming from exactly one process on one node.
        # First, we broadcast the data from this process to the same node-rank
        # process on each of the nodes.

        if self._comm is not None:
            import mpi4py.MPI as MPI

            target_noderank = self._comm.bcast(self._noderank, root=fromrank)
            fromnode = self._comm.bcast(self._mynode, root=fromrank)

            # Verify that the node rank with the data actually has a member on
            # every node (see notes in the constructor).
            if target_noderank > self._maxsetrank:
                msg = "set() called with data from a node rank ({}) which does".format(
                    target_noderank
                )
                msg += " not exist on all nodes"
                raise RuntimeError(msg)

            if self._noderank == target_noderank:
                # We are the lucky process on this node that gets to write
                # the data into shared memory!

                # Broadcast the offsets of the input slice
                copyoffset = None
                if self._mynode == fromnode:
                    copyoffset = offset
                copyoffset = self._rankcomm.bcast(copyoffset, root=fromnode)

                # Pre-allocate buffer, so that we can use the low-level
                # (and faster) Bcast method.
                datashape = None
                if self._mynode == fromnode:
                    datashape = data.shape
                datashape = self._rankcomm.bcast(datashape, root=fromnode)

                nodedata = None
                if self._mynode == fromnode:
                    # nodedata = np.array(data, copy=False).astype(self._dtype)
                    nodedata = data
                else:
                    nodedata = np.zeros(datashape, dtype=self._dtype)

                # Broadcast the data buffer
                self._rankcomm.Bcast([nodedata, self._mpitype], root=fromnode)

                # Now one process on every node has a copy of the data, and
                # can copy it into the shared memory buffer.

                dslice = []
                ndims = len(nodedata.shape)
                for d in range(ndims):
                    dslice.append(
                        slice(copyoffset[d], copyoffset[d] + nodedata.shape[d], 1)
                    )
                slc = tuple(dslice)

                # Copy data slice
                self.data[slc] = nodedata

                # Delete the temporary copy
                del nodedata
        else:
            # We are just copying to the array view
            dslice = []
            ndims = len(data.shape)
            for d in range(ndims):
                dslice.append(slice(offset[d], offset[d] + data.shape[d], 1))
            slc = tuple(dslice)
            self.data[slc] = data

        # Explicit barrier here, to ensure that other processes do not try
        # reading data before the writing processes have finished.
        if self._comm is not None:
            self._comm.barrier()
