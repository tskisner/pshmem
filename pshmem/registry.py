##
# Copyright (c) 2017-2025, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import sys
import atexit
import signal


class MPISharedRegistry:
    """Registry of shared memory buffers.

    This tracks all MPIShared memory buffers on the current process and
    ensures they are cleaned up when the process is terminated.

    """
    def __init__(self):
        self.reg = dict()

    def register(self, name, buffer, noderank):
        self.reg[name] = (buffer, noderank)

    def unregister(self, name):
        del self.reg[name]

    def cleanup(self):
        for name, (buf, noderank) in self.reg.items():
            buf.close()
            if noderank == 0:
                buf.unlink()
        self.reg.clear()


"""Single instance of the registry"""
registry = MPISharedRegistry()


def _signal_handler(sig, frame):
    global registry
    registry.cleanup()
    sys.exit(0)


@atexit.register
def _atexit_handler():
    global registry
    registry.cleanup()


def _register_signals():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGQUIT, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)


# Register signal handlers on import
_register_signals()
