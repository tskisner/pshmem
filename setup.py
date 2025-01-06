import setuptools
import versioneer

setuptools.setup(
    packages=["pshmem"],
    version=versioneer.get_version(),
)
