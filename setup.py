from setuptools import setup, find_packages
from ecoc import __name__, __version__, __author__, __email__

# with open('requirements.txt') as f:
# requirements = [l for l in f.read().splitlines() if l]

setup(
    name=__name__,
    version=__version__,
    author=__author__,
    author_email=__email__,

    # install_requires=requirements,
    python_requires=">=3.6",
    packages=find_packages()
)
