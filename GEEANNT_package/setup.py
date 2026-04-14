import os
import re
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

# README
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# VERSION
with open(os.path.join(HERE, "GEEANNT", "__init__.py"), encoding="utf-8") as f:
    VERSION = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setup(
    name="GEEANNT",
    version=VERSION,
    author="Francesco Monastra",
    author_email="francesco.monastra@inaf.it",
    description="Geant4 Efficiency Enhancing Artificial Neural Network Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/GEEANNT",  # TODO: update with real repository
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