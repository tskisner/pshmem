#!/bin/bash
#
# This installs mpich using homebrew and then installs mpi4py with pip.
#

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(pwd)
popd >/dev/null 2>&1

echo "Before brew mpich:  python = $(which python3), pip = $(which pip3)"

# Install mpich
brew install mpich

echo "After brew mpich:  python = $(which python3), pip = $(which pip3)"

# Install mpi4py
pip3 install setuptools
pip3 install wheel
pip3 install numpy
pip3 install mpi4py
