#!/usr/bin/env python3

from mpi4py import MPI

import sys
import os

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

    shmem_stat(f"Proc {rank} Starting state", limits=True)

    # Create the split communicators that we will re-use.
    nodecomm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
    noderank = nodecomm.rank
    nodeprocs = nodecomm.size
    nodes = procs // nodeprocs
    if nodes * nodeprocs < procs:
        nodes += 1
    mynode = rank // nodeprocs
    rankcomm = comm.Split(noderank, mynode)

    shmem_stat(f"Proc {rank} After comm split")

    test = list()
    for imem in range(100):
        test.append(
            MPIShared(
                (10000,),
                np.float64,
                comm,
                comm_node=nodecomm,
                comm_node_rank=rankcomm,
            )
        )
        shmem_stat(f"Proc {rank} After allocation {imem}")


if __name__ == "__main__":
    main()
