# Utilities for MPI design patterns with shared memory

This is simply an informal place to put code snippets that make use of
MPI-3 shared memory concepts.

## Python MPIShared Class

This class implements a pattern where a shared array is allocated on
each node.  Processes can update pieces of the shared array with the
synchronous "set()" method.  During this call, the data from the desired
process is first replicated to all nodes, and then one process on each
node copies that piece into the shared array.

All processes on all nodes can freely read data from the node-local
copy of the shared array.

### Simple Test

You can run the simple test with:

    $> mpirun -np 4 python3 test.py

## Note on OS Packages

Some Linux distributions (like Ubuntu) default to OpenMPI for their implementation.
Some installations of OpenMPI do not support shared memory in their default
configurations.  I have had much better luck using MPICH instead.  On ubuntu you can switch to MPICH with:

    sudo apt install libmpich-dev
    sudo update-alternatives --set mpi /usr/bin/mpicc.mpich
    sudo update-alternatives --set mpirun /usr/bin/mpirun.mpich

After doing this, you should make sure that your mpi4py installation is built against
MPICH.  If you are using a virtualenv then you can do:

    pip install --no-binary=:all: mpi4py

If you are using conda packages of mpi4py, then this uses conda-provided MPI libraries.
There are both OpenMPI and MPICH variants of the mpi4py conda package.
