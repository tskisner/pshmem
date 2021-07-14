#!/bin/bash


docker run \
-it \
-v $(pwd):/home/pshmem \
quay.io/pypa/manylinux2014_x86_64:latest \
/bin/bash

# export PATH=/opt/python/cp38-cp38/bin:${PATH}
# python3 -m pip install --upgrade pip
# yum -y update
# yum -y install mpich-3.2-devel.x86_64 mpich-3.2-autoload.x86_64
# source /etc/profile.d/modules.sh
# source /etc/profile.d/mpich-3.2-x86_64.sh
