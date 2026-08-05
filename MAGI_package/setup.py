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
    description=(
        "MAGI (Multivariate Autoencoder for particle Generative Inference): "
        "a CVAE-based generative particle source for Monte Carlo transport"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/francescomonastra/MAGI",
    project_urls={
        "Source": "https://github.com/francescomonastra/MAGI",
        "Issues": "https://github.com/francescomonastra/MAGI/issues",
        "User manual": (
            "https://github.com/francescomonastra/MAGI/blob/main/"
            "docs/manual/magi_manual.pdf"
        ),
    },
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-learn",
        # Not imported by magi/ itself, but scripts/generate_geant_source.py -
        # the entry point the companion Geant4 project shells out to - loads
        # the saved *_quantile_transformers.joblib with it. Declared directly
        # rather than leaning on scikit-learn pulling it in transitively.
        "joblib",
        # TensorFlow 2.16 moved to Keras 3, which TFP's legacy tf.keras usage
        # does not work against. TFP only pulls tf-keras under its optional
        # [tf] extra, so a plain "tensorflow_probability" requirement installs
        # a TFP that imports but fails the moment core/flows.py builds a
        # spline bijector. The extra is what makes a fresh install actually
        # work; do not drop it back to the bare name.
        "tensorflow>=2.16",
        "tensorflow_probability[tf]>=0.24",
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)