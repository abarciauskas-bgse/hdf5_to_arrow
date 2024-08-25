from setuptools import setup, find_packages

setup(
    name="hdf5_to_arrow",
    version="0.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
