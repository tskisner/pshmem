##
# Copyright (c) 2017-2020, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##
"""Parallel shared memory tools.

This package contains tools for using synchronized shared memory across nodes
and implementing communicator-wide MUTEX locks.

"""

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# Namespace imports

from .shmem import MPIShared
from .locking import MPILock
