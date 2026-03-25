#!/usr/bin/env python3
import time
from mpi4py import MPI

import subprocess as sp
import numpy as np

from pshmem import MPIShared


def shmem_stat(msg, limits=False):
    spalloc = sp.check_output(["ipcs", "-m"], universal_newlines=True)
    n_segment = len(spalloc.split("\n")) - 5
    if limits:
        spout = sp.check_output(["ipcs", "-lm"], universal_newlines=True)
        msg += ":\n"
        for line in spout.split("\n")[2:-2]:
            msg += f"  {line}\n"
        msg += f"  {n_segment} allocated segments"
        print(f"{msg}")
    else:
        print(f"{msg}:  {n_segment} allocated segments")


def main():
    comm = MPI.COMM_WORLD
    procs = comm.size
    rank = comm.rank

    # Dimensions / type of our shared memory array
    n_elem = np.iinfo(np.int32).max - 10
    #n_elem = np.iinfo(np.int32).max + 10
    datadims = (n_elem,)
    datatype = np.dtype(np.uint8)

    shmem_stat(f"Proc {rank} Starting state", limits=True)

    # Create local data on one process
    if rank == 0:
        print(f"Creating large array of {n_elem} bytes", flush=True)
        local = np.ones(datadims, dtype=datatype)
    else:
        local = None

    with MPIShared(datadims, datatype, comm) as shm:
        sptr = id(shm._shmem.buf.obj)
        print(f"Proc {rank} address = {sptr}", flush=True)

        shm.set(local, fromrank=0)
        del local

        sptr = id(shm._shmem.buf.obj)
        print(f"Proc {rank} after set, address = {sptr}", flush=True)

        shmem_stat(f"Proc {rank} inside context", limits=True)
        time.sleep(10)

        # Check results on all processes.
        count = np.count_nonzero(shm[:])
        if count != n_elem:
            print(f"Rank {rank} got {count} non-zeros, not {n_elem}", flush=True)

        comm.barrier()

    shmem_stat(f"Proc {rank} Ending state", limits=True)


if __name__ == "__main__":
    main()
