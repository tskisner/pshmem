import numpy as np
from mpi4py import MPI

from pshmem import MPIShared

comm = MPI.COMM_WORLD

with MPIShared((3, 5), np.float64, comm) as shm:
    # A copy of the data exists on every node and is initialized to zero.
    # There is a numpy array "view" of that memory available with slice notation
    # or by accessing the "data" member:
    if comm.rank == 0:
        # You can get a summary of the data by printing it:
        print("String representation:\n")
        print(shm)
        print("\n===== Initialized Data =====")
    for p in range(comm.size):
        if p == comm.rank:
            print("rank {}:\n".format(p), shm.data, flush=True)
        comm.barrier()

    set_data = None
    set_offset = None
    if comm.rank == 0:
        set_data = np.arange(6, dtype=np.float64).reshape((2, 3))
        set_offset = (1, 1)

    # The set() method is collective, but the inputs only matter on one rank
    shm.set(set_data, offset=set_offset, fromrank=0)

    # You can also use the usual '[]' notation.  However, this call must do an
    # additional pre-communication to detect which process the data is coming from.
    # And this line is still collective and must be called on all processes:
    shm[set_offset] = set_data

    # This updated data has now been replicated to the shared memory on all nodes.
    if comm.rank == 0:
        print("======= Updated Data =======")
    for p in range(comm.size):
        if p == comm.rank:
            print("rank {}:\n".format(p), shm.data, flush=True)
        comm.barrier()

    # You can read from the node-local copy of the data from all processes,
    # using either the "data" member or slice access:
    if comm.rank == comm.size - 1:
        print("==== Read-only access ======")
        print("rank {}: shm[2, 3] = {}".format(comm.rank, shm[2, 3]), flush=True)
        print("rank {}: shm.data = \n{}".format(comm.rank, shm.data), flush=True)
