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

