import os
import sys

from setuptools import find_packages, setup

import versioneer


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="pshmem",
    provides="pshmem",
    version=versioneer.get_version(),
    description="Parallel shared memory and locking with MPI",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Theodore Kisner",
    author_email="work@theodorekisner.com",
    url="https://github.com/tskisner/pshmem",
    packages=["pshmem"],
    scripts=None,
    license="BSD",
    python_requires=">=3.6.0",
    install_requires=["numpy", "posix_ipc"],
    extras_require={"mpi": ["mpi4py>=3.0"]},
    cmdclass=versioneer.get_cmdclass(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Utilities",
    ],
)
