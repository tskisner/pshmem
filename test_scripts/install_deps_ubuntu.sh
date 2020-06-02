#!/bin/bash
#
# This installs mpich using apt and then installs mpi4py with pip.
#

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(pwd)
popd >/dev/null 2>&1

# Install mpich
sudo apt-get -y update
sudo apt-get install -y build-essential libmpich-dev

# Install mpi4py
pip3 install setuptools
pip3 install numpy
pip3 install mpi4py
