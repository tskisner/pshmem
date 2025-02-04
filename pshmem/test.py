##
# Copyright (c) 2017-2025, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import os
import sys
import unittest

import numpy as np
import numpy.testing as nt

from .shmem import MPIShared
from .locking import MPILock
from .utils import exception_guard

MPI = None
use_mpi = True

if "MPI_DISABLE" in os.environ:
    use_mpi = False

if use_mpi and (MPI is None):
    try:
        import mpi4py.MPI as MPI
    except ImportError:
        print("Cannot import mpi4py, will only test serial functionality.", flush=True)


class ShmemTest(unittest.TestCase):
    def setUp(self):
        self.comm = None
        if MPI is not None:
            self.comm = MPI.COMM_WORLD
        self.rank = 0
        self.procs = 1
        if self.comm is not None:
            self.rank = self.comm.rank
            self.procs = self.comm.size

    def tearDown(self):
        pass

    def _write_read_buffer(self, local, shm):
        rank = 0
        procs = 1
        if shm.comm is not None:
            rank = shm.comm.rank
            procs = shm.comm.size
        datadims = local.shape
        # Dimensions of the incremental slab that we will
        # copy during each set() call.
        updatedims = (1, 1, datadims[2] // 2)

        # How many updates are there to cover the whole
        # data array?
        nupdate = 1
        for d in range(len(datadims)):
            nupdate *= datadims[d] // updatedims[d]

        for p in range(procs):
            # Every process takes turns writing to the buffer.
            setdata = None
            setoffset = (0, 0, 0)

            # Write to the whole data volume, but in small blocks
            for upd in range(nupdate):
                if p == rank:
                    # My turn!  Write my process rank to the buffer slab.
                    setdata = local[
                        setoffset[0] : setoffset[0] + updatedims[0],
                        setoffset[1] : setoffset[1] + updatedims[1],
                        setoffset[2] : setoffset[2] + updatedims[2],
                    ]
                try:
                    # All processes call set(), but only data on rank p matters.
                    shm.set(setdata, setoffset, fromrank=p)
                except (RuntimeError, ValueError):
                    print(
                        "proc {} threw exception during set()".format(rank),
                        flush=True,
                    )
                    raise
                try:
                    # Same as set(), but using __setitem__ with an
                    # allreduce to find which process is setting.

                    # key as a tuple slices
                    if setdata is None:
                        shm[None] = setdata
                    else:
                        shm[
                            setoffset[0] : setoffset[0] + setdata.shape[0],
                            setoffset[1] : setoffset[1] + setdata.shape[1],
                            setoffset[2] : setoffset[2] + setdata.shape[2],
                        ] = setdata
                except (RuntimeError, ValueError):
                    print(
                        "proc {} threw exception during __setitem__".format(rank),
                        flush=True,
                    )
                    raise

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

            # Try full array assignment with slices containing None start
            # values
            if p != rank:
                shm[None] = None
            else:
                shm[:, :, :] = local

            check[:, :, :] = shm[:, :, :]
            nt.assert_equal(check[:, :, :], truth[:, :, :])

        # Ensure that we can reference the memory buffer from numpy without
        # a memory copy.  The intention is that a slice of the shared memory
        # buffer should appear as a C-contiguous ndarray whenever we slice
        # along the last dimension.

        for p in range(procs):
            if p == rank:
                slc = shm[1, 2]
                print(
                    "proc {} slice has dims {}, dtype {}, C = {}".format(
                        p, slc.shape, slc.dtype.str, slc.flags["C_CONTIGUOUS"]
                    ),
                    flush=True,
                )
            if shm.comm is not None:
                shm.comm.barrier()

    def context_write_read(self, comm, comm_node=None, comm_node_rank=None):
        """Run a sequence of various access tests."""
        rank = 0
        if comm is not None:
            rank = comm.rank

        # Dimensions of our shared memory array
        datadims = (2, 5, 10)

        for datatype in [np.int32, np.int64, np.float32, np.float64]:
            # For testing the "set()" method, every process is going to
            # create a full-sized data buffer and fill it with its process rank.
            local = np.ones(datadims, dtype=datatype)
            local *= rank

            # A context manager is the pythonic way to make sure that the
            # object has no dangling reference counts after leaving the context,
            # and will ensure that the shared memory is freed properly.
            with MPIShared(
                local.shape,
                local.dtype,
                comm,
                comm_node=comm_node,
                comm_node_rank=comm_node_rank,
            ) as shm:
                self._write_read_buffer(local, shm)

    def create_separated(self, comm):
        # Test creation of shared memory objects outside of a context manager
        data = dict()
        rank = 0
        if comm is not None:
            rank = comm.rank

        # Dimensions of our shared memory array
        datadims = (2, 5, 10)

        for datatype in [np.int32, np.int64, np.float32, np.float64]:
            local = np.ones(datadims, dtype=datatype)
            local *= rank
            data[datatype] = MPIShared(
                local.shape,
                local.dtype,
                comm,
            )
            self._write_read_buffer(local, data[datatype])
        return data

    def close_separated(self, data):
        for dt, shm in data.items():
            shm.close()
        del data

    def test_world(self):
        if self.comm is None:
            print("Testing MPIShared without MPI...", flush=True)
        elif self.comm.rank == 0:
            print("Testing MPIShared with world communicator...", flush=True)
        self.context_write_read(self.comm)

    def test_separated(self):
        if self.comm is None:
            print("Testing separated create/close without MPI...", flush=True)
        elif self.comm.rank == 0:
            print(
                "Testing separated create/close with world communicator...", flush=True
            )
        data = self.create_separated(self.comm)
        self.close_separated(data)

    def test_split(self):
        if self.comm is not None:
            if self.comm.rank == 0:
                print("Testing MPIShared with split communicators...", flush=True)

            # Take the world comm and create intra-node and inter-node comms
            wcomm = self.comm
            wrank = wcomm.rank
            wsize = wcomm.size
            nodecomm = wcomm.Split_type(MPI.COMM_TYPE_SHARED, 0)
            nodeprocs = nodecomm.size
            myworldnode = wrank // nodeprocs
            noderankcomm = wcomm.Split(nodecomm.rank, myworldnode)

            # Now split the world comm into groups
            gsize = 1
            if wsize >= 2:
                gsize = wsize // 2
            ngroups = wsize // gsize
            group = wrank // gsize
            grank = wrank % gsize

            if ngroups == 1:
                # We just have one group with all processes.  The group comm is the same
                # as the world, and same with the intra-node and inter-node comms.
                gcomm = wcomm
                gnodecomm = nodecomm
                gnodeprocs = gnodecomm.size
                mygroupnode = grank // gnodeprocs
                gnoderankcomm = noderankcomm
                # In the case of one group, the group-wise rank communicator
                # is just each individual process.
                rcomm = MPI.COMM_SELF
            else:
                # We need to split the world communicator into groups, and then create
                # inter-node and intra-node comms for each group.
                gcomm = wcomm.Split(group, grank)
                rcomm = wcomm.Split(grank, group)
                gnodecomm = gcomm.Split_type(MPI.COMM_TYPE_SHARED, 0)
                gnodeprocs = gnodecomm.size
                mygroupnode = grank // gnodeprocs
                gnoderankcomm = gcomm.Split(gnodecomm.rank, mygroupnode)

            # For each process group, test several grid configurations.  Create a
            # communicator along each row and column of this grid, as well as inter-node
            # and intra-node communicators along each row and column.

            sq_rows = int(np.sqrt(gsize))
            if sq_rows == 0:
                sq_rows = 1
            for grid_rows in [sq_rows, 1, gsize]:
                grid_cols = gcomm.size // grid_rows

                col_rank = gcomm.rank // grid_cols
                row_rank = gcomm.rank % grid_cols

                if grid_cols == 1:
                    comm_row = MPI.Comm.Dup(MPI.COMM_SELF)
                else:
                    comm_row = gcomm.Split(col_rank, row_rank)

                if grid_rows == 1:
                    comm_col = MPI.Comm.Dup(MPI.COMM_SELF)
                else:
                    comm_col = gcomm.Split(row_rank, col_rank)

                # Node and node-rank comms for each row and col.
                comm_row_node = comm_row.Split_type(MPI.COMM_TYPE_SHARED, 0)
                row_nodeprocs = comm_row_node.size
                row_node = comm_row.rank // row_nodeprocs
                comm_row_rank_node = comm_row.Split(comm_row_node.rank, row_node)

                comm_col_node = comm_col.Split_type(MPI.COMM_TYPE_SHARED, 0)
                col_nodeprocs = comm_col_node.size
                col_node = comm_col.rank // col_nodeprocs
                comm_col_rank_node = comm_col.Split(comm_col_node.rank, col_node)

                # Test the access and creation of shared memory objects across all
                # these different communicators.

                self.context_write_read(
                    wcomm, comm_node=nodecomm, comm_node_rank=noderankcomm
                )
                wcomm.barrier()

                self.context_write_read(
                    gcomm, comm_node=gnodecomm, comm_node_rank=gnoderankcomm
                )
                wcomm.barrier()

                self.context_write_read(
                    comm_row, comm_node=comm_row_node, comm_node_rank=comm_row_rank_node
                )
                wcomm.barrier()

                self.context_write_read(
                    comm_col, comm_node=comm_col_node, comm_node_rank=comm_col_rank_node
                )
                wcomm.barrier()

                # Clean up row / column communicators
                comm_col_rank_node.Free()
                comm_row_rank_node.Free()
                comm_col_node.Free()
                comm_row_node.Free()
                comm_col.Free()
                comm_row.Free()

            # Clean up group communicators
            if ngroups > 1:
                gnoderankcomm.Free()
                gnodecomm.Free()
                rcomm.Free()
                gcomm.Free()
            noderankcomm.Free()
            nodecomm.Free()

    def test_comm_self(self):
        if self.comm is not None:
            if self.comm.rank == 0:
                print("Testing MPIShared with COMM_SELF...", flush=True)
            # Every process does the operations on COMM_SELF
            self.context_write_read(MPI.COMM_SELF)

    def test_comm_reuse(self):
        if self.comm is not None:
            if self.comm.rank == 0:
                print("Testing MPIShared with re-used node comm...", flush=True)
            nodecomm = self.comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
            noderank = nodecomm.rank
            nodeprocs = nodecomm.size
            nodes = self.comm.size // nodeprocs
            mynode = self.comm.rank // nodeprocs
            rankcomm = self.comm.Split(noderank, mynode)

            self.context_write_read(
                self.comm, comm_node=nodecomm, comm_node_rank=rankcomm
            )

            if nodes > 1 and nodeprocs > 2:
                # We have at least one node, test passing in an incorrect
                # communicator for the node comm.
                evenoddcomm = self.comm.Split(self.comm.rank % 2, self.comm.rank // 2)
                try:
                    test_shared = MPIShared(
                        (10, 5),
                        np.float64,
                        self.comm,
                        comm_node=evenoddcomm,
                        comm_node_rank=evenoddcomm,
                    )
                    print("Failed to catch construction with bad node comm", flush=True)
                    self.assertTrue(False)
                except ValueError:
                    print(
                        "Successfully caught construction with bad node comm",
                        flush=True,
                    )

    def test_shape(self):
        good_dims = [
            (2, 5, 10),
            np.array([10, 2], dtype=np.int32),
            np.array([5, 2], dtype=np.int64),
            np.array([10, 2], dtype=np.int_),
        ]
        bad_dims = [
            (-1,),
            (2, 5.5, 10),
            np.array([10, 2], dtype=np.float32),
            np.array([5, 2], dtype=np.float64),
            np.array([10, 2.5], dtype=np.float32),
        ]

        dt = np.float64

        for dims in good_dims:
            try:
                shm = MPIShared(dims, dt, self.comm)
                if self.rank == 0:
                    print("successful creation with shape {}".format(dims), flush=True)
                shm.close()
                del shm
            except (RuntimeError, ValueError):
                if self.rank == 0:
                    print(
                        "unsuccessful creation with shape {}".format(dims), flush=True
                    )
                self.assertTrue(False)
        for dims in bad_dims:
            try:
                shm = MPIShared(dims, dt, self.comm)
                if self.rank == 0:
                    print("unsuccessful rejection of shape {}".format(dims), flush=True)
                shm.close()
                del shm
                self.assertTrue(False)
            except (RuntimeError, ValueError):
                if self.rank == 0:
                    print("successful rejection of shape {}".format(dims), flush=True)

    def test_array(self):
        dims = (2, 5, 10)
        dt = np.float64
        with MPIShared(dims, dt, self.comm) as shm:
            view = np.array(shm, copy=False)
            vptr, vflag = view.__array_interface__["data"]
            sptr, sflag = shm._flat.__array_interface__["data"]
            print(f"numpy view address = {vptr}", flush=True)
            print(f"original address = {sptr}", flush=True)
            self.assertTrue(vptr == sptr)

    def test_zero(self):
        with MPIShared((0,), np.float64, self.comm) as shm:
            self.assertTrue(len(shm) == 0)
            self.assertTrue(shm[5] is None)
            try:
                shm[0] = 1.0
                if self.rank == 0:
                    print(
                        "unsuccessful raise with no data during assignment", flush=True
                    )
                self.assertTrue(False)
            except RuntimeError:
                print("successful raise with no data during assignment", flush=True)
            try:
                if self.rank == 0:
                    shm.set(1.0, fromrank=0)
                else:
                    shm.set(None, fromrank=0)
                if self.rank == 0:
                    print("unsuccessful raise with no data during set()", flush=True)
                self.assertTrue(False)
            except RuntimeError:
                print("successful raise with no data during set()", flush=True)

    def test_max_shmem_segments(self):
        handles = list()
        n_seg = 0
        failed = False
        while not failed and n_seg < 10000:
            try:
                shm = MPIShared((5, 5), np.float64, self.comm)
                handles.append(shm)
                n_seg += 1
            except Exception:
                failed = True
        print(f"Allocated {n_seg} shared memory segments without OS error", flush=True)
        for h in handles:
            h.close()
        handles.clear()

    # def test_hang(self):
    #     # Run this while monitoring memory usage (e.g. with htop) and then
    #     # do kill -15 (SIGTERM) on one of the processes to verify that the signal
    #     # handler is executed to cleanup memory.
    #     dims = (200, 1000000)
    #     dt = np.float64
    #     shm = MPIShared(dims, dt, self.comm)
    #     if self.comm is None or self.comm.rank == 0:
    #         temp = np.ones(dims, dtype=dt)
    #     else:
    #         temp = None
    #     shm.set(temp, fromrank=0)
    #     del temp
    #     import time
    #     time.sleep(60)
    #     shm.close()
    #     del shm
    #     return


class LockTest(unittest.TestCase):
    def setUp(self):
        self.comm = None
        if MPI is not None:
            self.comm = MPI.COMM_WORLD
        self.rank = 0
        self.procs = 1
        if self.comm is not None:
            self.rank = self.comm.rank
            self.procs = self.comm.size
        self.sleepsec = 0.2

    def tearDown(self):
        pass

    def test_lock(self):
        with MPILock(self.comm, root=0, debug=True) as lock:
            for lk in range(5):
                msg = "test_lock:  process {} got lock {}".format(self.rank, lk)
                lock.lock()
                print(msg, flush=True)
                # time.sleep(self.sleepsec)
                lock.unlock()
        if self.comm is not None:
            self.comm.barrier()


def run():
    comm = None
    if MPI is not None:
        comm = MPI.COMM_WORLD

    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(LockTest))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(ShmemTest))
    runner = unittest.TextTestRunner()

    ret = 0
    with exception_guard(comm=comm):
        _ret = runner.run(suite)
        if not _ret.wasSuccessful():
            ret += 1

    if comm is not None:
        ret = comm.allreduce(ret, op=MPI.SUM)

    if ret > 0:
        print(f"{ret} Processes had failures")
        sys.exit(6)

    return
