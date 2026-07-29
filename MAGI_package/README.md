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

## Tests

A minimal regression suite (v0.8.1 Phase 4) covers the pieces that fail silently
rather than loudly: flow/prior round-trip identity, line-integral recovery on
synthetic spectra, the coupling prior actually fitting an injected coupling, the
checkpoint config-match guard, and the Geant4 export script end-to-end. Runs in
under 30s, CPU only.

```bash
pip install pytest
pytest MAGI_package/tests/
```

## Package structure

```
MAGI_package/
├── setup.py
├── README.md
├── tests/              # minimal regression suite, see above
├── docs/
│   └── USAGE.md
└── magi/
    ├── __init__.py        # public API surface
    ├── magi.py            # high-level convenience API (setup, train_model, plot_training)
    ├── config.py          # environment/seed initialization
    ├── core/              # CVAE model classes, losses, geometry transforms
    │   ├── model.py       #   one class per version in the lineage
    │   ├── losses.py
    │   ├── geometry.py
    │   ├── flows.py       #   conditional spline flow (v0.8 continuum)
    │   └── priors.py      #   conditional coupling prior p(z|cond) (v0.8)
    ├── data/              # load/preprocess/dataset-build raw detector tables
    │   ├── io.py
    │   ├── preprocessing.py
    │   └── dataset.py
    ├── training/          # compile/fit wrappers, adaptive loss scheduling, checkpointing
    │   ├── train.py
    │   ├── adaptive_callbacks.py
    │   └── checkpointing.py
    ├── generation/        # sample from a trained model, reconstruct physics, export files
    │   ├── sampling.py
    │   ├── reconstruction.py
    │   └── export.py
    ├── validation/        # Wasserstein / histogram-residual / line-recovery metrics
    │   ├── metrics.py
    │   └── compare.py
    └── utils/             # plotting, model introspection
        ├── plotting.py
        └── model_inspection.py
```

`magi.print_model_structure(model)` and `magi.print_model_tree_with_params(model)` print
a description of a built model — its generative structure and the formulas behind it, the
configured mixture/prior/flow settings, and the parameter counts per block.

## Author

Francesco Monastra (INAF) — `francesco.monastra@inaf.it`
