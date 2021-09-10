##
# Copyright (c) 2017-2020, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import numpy as np


def mpi_data_type(comm, dt):
    """Helper function to return the byte size and MPI datatype.

    Args:
        comm (mpi4py.Comm): The communicator, or None.
        dt (np.dtype): The datatype.

    Returns:
        (tuple):  The (bytesize, MPI type) of the input dtype.

    """
    dtyp = np.dtype(dt)
    dsize = None
    mpitype = None
    if comm is None:
        dsize = dtyp.itemsize
    else:
        from mpi4py import MPI

        # We are actually using MPI, so we need to ensure that
        # our specified numpy dtype has a corresponding MPI datatype.
        try:
            # Technically this is an internal variable, but online
            # forum posts from the developers indicate this is stable
            # at least until a public interface is created.
            mpitype = MPI._typedict[dtyp.char]
        except Exception:
            msg = "Process {} failed to get MPI data type for numpy dtype ".format(
                comm.rank
            )
            msg += "{}, char '{}'".format(dtyp, dtyp.char)
            print(msg, flush=True)
            raise
        dsize = mpitype.Get_size()
    return (dsize, mpitype)
