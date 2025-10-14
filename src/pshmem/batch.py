##
# Copyright (c) 2025-2025, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import itertools
import os

import numpy as np

from .utils import mpi_data_type


class MPIBatch(object):
    """Implement a simple batch execution of tasks by workers.

    Each "worker" consists of a group of MPI processes.  These fixed-size
    groups are created at instantiation by using Comm.Split().  The number
    of tasks is also fixed at instantiation.

    Each worker group can use methods to request a task and mark a task as
    completed or failed.

    No other metadata is stored about each task- only the global index and
    state of each task is tracked.  The calling code is responsible for
    ensuring that all processes have information about the work associated
    with each task index.

    Args:
        comm (MPI.Comm):  The full communicator to use.
        worker_size (int):  The fixed size of each worker.
        n_task (int):  The total number of tasks.
        init_state (array):  Only accessed on rank `root`.  The initial
            state to set for all tasks.
        task_fs_root (str):  Optionally sync state to the filesystem, using
            per-task directories in this parent location.
        task_fs_names (list):  If syncing to the filesystem, the "name"
            (subdirectory) of every task.
        root (int):  The rank which stores the task state.
        debug (bool):  If True, print debugging info.

    """

    # Task states
    INVALID = -1
    OPEN = 0
    RUNNING = 1
    DONE = 2
    FAILED = 3

    # Name of the state file
    state_file_name = "state"

    # This creates a new integer for each time the class is instantiated.
    newid = next(itertools.count())

    def __init__(
        self,
        comm,
        worker_size,
        n_task,
        init_state=None,
        task_fs_root=None,
        task_fs_names=None,
        root=0,
        debug=False,
    ):
        self._comm = comm
        self._n_task = n_task
        self._root = root
        self._debug = debug
        self._task_fs_root = task_fs_root
        self._task_fs_names = task_fs_names

        if task_fs_root is not None and task_fs_names is None:
            raise RuntimeError(
                "If tracking state with the filesystem, task names are required"
            )
        if task_fs_names is not None and n_task != len(task_fs_names):
            raise RuntimeError(
                "If using filesystem, n_task should match length of task_fs_names"
            )

        self._global_rank = 0
        self._global_procs = 1
        if self._comm is not None:
            self._global_rank = self._comm.rank
            self._global_procs = self._comm.size

        if comm is None:
            if worker_size != 1:
                raise RuntimeError("If comm is None, worker_size should be 1")
            self._n_worker = 1
            self._worker_size = 1
            self._worker = 0
            self._worker_rank = 0
            self._worker_comm = None
        else:
            n_proc = comm.size
            self._n_worker = n_proc // worker_size
            if self._n_worker * worker_size != n_proc:
                raise RuntimeError(
                    "Total number of MPI processes must be divisible by worker_size"
                )
            self._worker_size = worker_size

            # Compute group and group_rank for this process
            self._worker = comm.rank // worker_size
            self._worker_rank = comm.rank % worker_size

            # Split the communicator
            self._worker_comm = comm.Split(self._worker, self._worker_rank)

        # A unique tag for each instance of the class
        self._tag = MPIBatch.newid

        if self._global_rank == self._root:
            self._nlocal = self._n_task
        else:
            self._nlocal = 0

        self._dtype = np.dtype(np.int32)

        # Local memory buffers.  The root process will also use the full state
        # array to build the initial state.
        self._all_task_state = np.zeros(self._n_task, dtype=self._dtype)
        self._one_task_state = np.zeros(1, dtype=self._dtype)

        # Data type sizes
        self._dsize, self._mpitype = mpi_data_type(self._comm, self._dtype)

        # If using the filesystem, one process creates the per-task directories
        # and checks initial state.
        if self._global_rank == self._root:
            if self._task_fs_root is None:
                # Just set any initial state from calling code
                if init_state is not None:
                    self._all_task_state[:] = init_state
            else:
                if os.path.exists(self._task_fs_root) and init_state is not None:
                    msg = f"Filesystem root {self._task_fs_root} exists.  Initial"
                    msg += " state will be read from that and init_state should be None"
                    raise RuntimeError(msg)
                os.makedirs(self._task_fs_root, exist_ok=True)
                # Read initial state from the filesystem
                for task in range(self._n_task):
                    task_dir = os.path.join(
                        self._task_fs_root, self._task_fs_names[task]
                    )
                    os.makedirs(task_dir, exist_ok=True)
                    istate = self.read_task_state(task)
                    if istate == self.INVALID:
                        istate = self.OPEN
                        self.write_task_state(task, istate)
                    self._all_task_state[task] = istate

        # Allocate the shared memory buffer.

        self._win = None
        nbytes = self._nlocal * self._dsize

        if self._comm is not None:
            from mpi4py import MPI

            # Root allocates the buffer
            try:
                self._win = MPI.Win.Allocate(
                    nbytes, disp_unit=self._dsize, info=MPI.INFO_NULL, comm=self._comm
                )
            except Exception:
                msg = "Process {} failed Win.Allocate_shared of {} bytes".format(
                    self._comm.rank, nbytes
                )
                msg += " ({} elements of {} bytes each".format(
                    self._nlocal, self._dsize
                )
                print(msg, flush=True)
                raise

            self._win.Fence()

            if self._global_rank == self._root:
                # Root initializes
                self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)
                self._win.Put(
                    [self._all_task_state, self._n_task, self._mpitype],
                    self._root,
                    target=[0, self._n_task, self._mpitype],
                )
                self._win.Flush(self._root)
                self._win.Unlock(self._root)

            self._win.Fence()

        if self._comm is not None:
            self._comm.barrier()

        return

    @classmethod
    def state_to_string(cls, state):
        if state == cls.OPEN:
            return "OPEN"
        elif state == cls.RUNNING:
            return "RUNNING"
        elif state == cls.FAILED:
            return "FAILED"
        elif state == cls.DONE:
            return "DONE"
        elif state == cls.INVALID:
            return "INVALID"
        else:
            msg = f"Unknown state '{state}'"
            raise RuntimeError(msg)

    @classmethod
    def string_to_state(cls, state_str):
        if state_str == "OPEN":
            return cls.OPEN
        elif state_str == "RUNNING":
            return cls.RUNNING
        elif state_str == "FAILED":
            return cls.FAILED
        elif state_str == "DONE":
            return cls.DONE
        elif state_str == "INVALID":
            return cls.INVALID
        else:
            msg = f"Unknown state string '{state_str}'"
            raise RuntimeError(msg)

    def write_task_state(self, task, state):
        if self._task_fs_root is None:
            return
        task_dir = os.path.join(self._task_fs_root, self._task_fs_names[task])
        state_file = os.path.join(task_dir, self.state_file_name)
        temp_state = f"{state_file}.temp"
        state_str = self.state_to_string(state)
        with open(temp_state, "w") as f:
            f.write(f"{state_str}\n")
        os.rename(temp_state, state_file)

    def read_task_state(self, task):
        state_file = os.path.join(
            self._task_fs_root, self._task_fs_names[task], self.state_file_name
        )
        if os.path.isfile(state_file):
            with open(state_file, "r") as f:
                state_str = f.readline().rstrip()
            state = self.string_to_state(state_str)
        else:
            state = self.INVALID
        return state

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
        return False

    def close(self):
        # Explicitly free the shared window
        if hasattr(self, "_win") and (self._win is not None):
            self._win.Fence()
            self._win.Free()
            self._win = None
        return

    @property
    def comm(self):
        """The global communicator."""
        return self._comm

    @property
    def n_task(self):
        """The total number of tasks."""
        return self._n_task

    @property
    def n_worker(self):
        """The total number of workers."""
        return self._n_worker

    @property
    def worker_size(self):
        """The number of processes in each worker."""
        return self._worker_size

    @property
    def worker(self):
        """The worker containing this process."""
        return self._worker

    @property
    def worker_rank(self):
        """The rank of this process within the worker."""
        return self._worker_rank

    @property
    def worker_comm(self):
        """The communicator spanning this worker's processes."""
        return self._worker_comm

    def _worker_str(self):
        wstr = f"{self._worker}:{self._worker_rank}({self._global_rank})"
        return wstr

    def next_task(self, ignore_running=False, ignore_failed=False, ignore_done=False):
        """Get the next task available to this worker.

        This method is collective across the worker communicator.  The first
        rank in the worker will query the task list, find the next task to
        run, and broadcast that result across the worker processes.

        The task index returned will be globally marked as RUNNING.

        Args:
            ignore_running (bool):  If True, consider tasks marked as
                RUNNING to be available.
            ignore_failed (bool):  If True, consider tasks marked as
                FAILED to be available.
            ignore_done (bool):  If True, consider tasks marked as
                DONE to be available.

        Returns:
            (int):  The next task index to process.

        """
        if self._debug and self._worker_rank == 0:
            msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
            msg += "getting next task"
            print(msg, flush=True)

        def _available(tstate):
            if tstate == self.OPEN:
                return True
            elif tstate == self.RUNNING and ignore_running:
                return True
            elif tstate == self.FAILED and ignore_failed:
                return True
            elif tstate == self.DONE and ignore_done:
                return True
            else:
                return False

        oldstate = None
        result = None
        if self._comm is None:
            for task in range(self._n_task):
                if self._task_fs_root is not None:
                    # The filesystem is the source of truth, since another job might
                    # have modified state.
                    state = self.read_task_state(task)
                    self._all_task_state[task] = state
                else:
                    state = self._all_task_state[task]
                if _available(state):
                    oldstate = state
                    result = task
                    break
        else:
            if self._worker_rank == 0:
                from mpi4py import MPI

                # Lock the window
                self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)
                if self._debug:
                    msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                    msg += "vvvvvv  Acquired Lock   vvvvvv"
                    print(msg, flush=True)

                for task in range(self._n_task):
                    # Get the current state in memory
                    self._win.Get(
                        [self._one_task_state, 1, self._mpitype],
                        self._root,
                        target=[task, 1, self._mpitype],
                    )

                    # Get the state from the filesystem
                    if self._task_fs_root is not None:
                        # The filesystem is the source of truth, since another job might
                        # have modified state.
                        state = self.read_task_state(task)
                    else:
                        state = self._one_task_state[0]

                    # If the two are different, update the memory copy
                    if state != self._one_task_state[0]:
                        if self._debug:
                            msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                            msg += f"task {task} has filesystem state {state} and mem state"
                            msg += f" {self._one_task_state[0]}.  Updating memory."
                            print(msg, flush=True)
                        self._one_task_state[0] = state
                        self._win.Put(
                            [self._one_task_state, 1, self._mpitype],
                            self._root,
                            target=[task, 1, self._mpitype],
                        )

                    if _available(state):
                        oldstate = state
                        result = task
                        # Mark this task as running
                        self._one_task_state[0] = self.RUNNING
                        self.write_task_state(task, self.RUNNING)
                        self._win.Put(
                            [self._one_task_state, 1, self._mpitype],
                            self._root,
                            target=[task, 1, self._mpitype],
                        )
                        break

                # Flush
                self._win.Flush(self._root)

                # Release the window lock
                if self._debug:
                    msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                    msg += "^^^^^^  Releasing Lock  ^^^^^^"
                    print(msg, flush=True)
                self._win.Unlock(self._root)

            if self._worker_comm is not None:
                result = self._worker_comm.bcast(result, root=0)

        if self._debug and self._worker_rank == 0:
            if result is not None:
                state_str = self.state_to_string(oldstate)
                msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                msg += f"set {state_str} task {task} to RUNNING"
                print(msg, flush=True)

        return result

    def set_task_state(self, task, state):
        """Set the state of a given task.

        Access to the global list of task states is serialized.

        Args:
            task (int):  The task index to modify.
            state (int):  The state to set.

        Returns:
            None

        """
        state_str = self.state_to_string(state)
        if self._debug:
            msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
            msg += f"set task {task} to {state_str}"
            print(msg, flush=True)

        if self._comm is None:
            self.write_task_state(task, state)
            self._all_task_state[task] = state
        else:
            from mpi4py import MPI

            # Update the task state
            self._one_task_state[0] = state

            # Lock the window
            self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)
            if self._debug:
                msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                msg += "vvvvvv  Acquired Lock   vvvvvv"
                print(msg, flush=True)

            if self._debug:
                msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                msg += f"Set task state to {self.state_to_string(state)}"
                print(msg, flush=True)

            self.write_task_state(task, state)
            self._win.Put(
                [self._one_task_state, 1, self._mpitype],
                self._root,
                target=[task, 1, self._mpitype],
            )

            # Flush
            self._win.Flush(self._root)

            # Release the window lock
            if self._debug:
                msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                msg += "^^^^^^  Releasing Lock  ^^^^^^"
                print(msg, flush=True)
            self._win.Unlock(self._root)

    def get_task_state(self, task):
        """Get the state of a given task.

        Access to the global list of task states is serialized.

        Args:
            task (int):  The task index to modify.

        Returns:
            (int):  The state.

        """
        if self._debug:
            msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
            msg += f"get state of task {task}"
            print(msg, flush=True)

        if self._comm is None:
            if self._task_fs_root is not None:
                # The filesystem is the source of truth, since another job might
                # have modified state.
                state = self.read_task_state(task)
                self._all_task_state[task] = state
            else:
                state = self._all_task_state[task]
        else:
            from mpi4py import MPI

            # Lock the window
            self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)
            if self._debug:
                msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                msg += "vvvvvv  Acquired Lock   vvvvvv"
                print(msg, flush=True)

            # Get the current state in memory
            self._win.Get(
                [self._one_task_state, 1, self._mpitype],
                self._root,
                target=[task, 1, self._mpitype],
            )

            # Get the state from the filesystem
            if self._task_fs_root is not None:
                # The filesystem is the source of truth, since another job might
                # have modified state.
                state = self.read_task_state(task)
            else:
                state = self._one_task_state[0]

            # If the two are different, update the memory copy
            if state != self._one_task_state[0]:
                if self._debug:
                    msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                    msg += f"task {task} has filesystem state {state} and mem state"
                    msg += f" {self._one_task_state[0]}.  Updating memory."
                    print(msg, flush=True)
                self._one_task_state[0] = state
                self._win.Put(
                    [self._one_task_state, 1, self._mpitype],
                    self._root,
                    target=[task, 1, self._mpitype],
                )

            if self._debug:
                msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                msg += f"Got task state {state}"
                print(msg, flush=True)

            # Flush
            self._win.Flush(self._root)

            # Release the window lock
            if self._debug:
                msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
                msg += "^^^^^^  Releasing Lock  ^^^^^^"
                print(msg, flush=True)
            self._win.Unlock(self._root)

        if self._debug:
            state_str = self.state_to_string(state)
            msg = f"MPIBatch[{self._tag}]:  Worker {self._worker_str()} "
            msg += f"task {task} has state {state_str}"
            print(msg, flush=True)
        return state
