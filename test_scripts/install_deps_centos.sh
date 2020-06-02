#!/bin/bash
#
# This installs mpich using yum and then installs mpi4py with pip.
#

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(pwd)
popd >/dev/null 2>&1

# Install mpich
yum -y update
yum -y install gcc mpich-3.2-devel.x86_64 mpich-3.2-autoload.x86_64

# Reload environment to get MPI compilers
exec "${BASH}"

# Install mpi4py
pip3 install numpy
pip3 install mpi4py
