# MAGI

A CVAE-based generative toolkit for Geant4 particle-source modeling: learns the
phase-space distribution of particles crossing a detector surface in a Geant4 Monte Carlo
simulation, and generates new physically-consistent events at a fraction of the cost of
running full Geant4 transport.

**Full usage guide, model overview, and — importantly — current limitations (e.g. the
sphere-only geometry assumption) are in [`docs/USAGE.md`](docs/USAGE.md).** A runnable,
fully-commented walkthrough on synthetic data is at
[`../Example_Usage.ipynb`](../Example_Usage.ipynb).

## Installation

```bash
git clone https://github.com/Fchewie/MAGI.git
cd MAGI
pip install -e MAGI_package/
```

```python
import magi
magi.initialize_environment(seed=42)
```

## Package structure

```
MAGI_package/
├── setup.py
├── README.md
├── docs/
│   └── USAGE.md
└── magi/
    ├── __init__.py       # public API surface
    ├── magi.py            # simplified high-level API (setup, build_model, train_model)
    ├── config.py          # environment/seed initialization
    ├── core/               # CVAE model classes, losses, geometry transforms
    │   ├── model.py
    │   ├── losses.py
    │   └── geometry.py
    ├── data/                # load/preprocess/dataset-build raw detector tables
    │   ├── io.py
    │   ├── preprocessing.py
    │   └── dataset.py
    ├── training/            # compile/fit wrappers, adaptive loss scheduling, checkpointing
    │   ├── train.py
    │   ├── adaptive_callbacks.py
    │   └── checkpointing.py
    ├── generation/          # sample from a trained model, reconstruct physics, export files
    │   ├── sampling.py
    │   ├── reconstruction.py
    │   └── export.py
    ├── validation/          # Wasserstein / histogram-residual comparisons
    │   ├── metrics.py
    │   └── compare.py
    └── utils/                # plotting, model introspection
        ├── plotting.py
        └── model_inspection.py
```

## Author

Francesco Monastra (INAF) — `francesco.monastra@inaf.it`
