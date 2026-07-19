import os
import re
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

# README
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# VERSION
with open(os.path.join(HERE, "magi", "__init__.py"), encoding="utf-8") as f:
    VERSION = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setup(
    name="magi",
    version=VERSION,
    author="Francesco Monastra",
    author_email="francesco.monastra@inaf.it",
    description="MAGI: a CVAE-based generative toolkit for Geant4 particle-source modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fchewie/MAGI",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "optuna",
        "astropy",
        "h5py",
        "seaborn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
        ],
        "docs": [
            "sphinx",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)