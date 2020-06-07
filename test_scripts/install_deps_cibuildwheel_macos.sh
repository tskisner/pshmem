#!/bin/bash
#
# This installs mpich using homebrew and then installs mpi4py with pip.
#

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(pwd)
popd >/dev/null 2>&1

# Install mpich
brew install mpich
