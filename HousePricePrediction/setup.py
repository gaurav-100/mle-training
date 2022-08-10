# Always prefer setuptools over distutils
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="housingPrediction",  # Required
    version="1.0.0",  # Required
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.7, <4",
)
