# MAGI

A CVAE-based generative toolkit for Geant4 particle-source modeling: learns the
phase-space distribution of particles crossing a detector surface in a Geant4 Monte Carlo
simulation, and generates new physically-consistent events at a fraction of the cost of
running full Geant4 transport.

**Full usage guide, model overview, and — importantly — current limitations (e.g. the
sphere-only geometry assumption) are in [`docs/USAGE.md`](docs/USAGE.md).** A runnable,
fully-commented walkthrough on synthetic data is at
[`../Example_Usage.ipynb`](../Example_Usage.ipynb). The long-form reference is the
user manual, [`../docs/manual/magi_manual.pdf`](../docs/manual/magi_manual.pdf).

> **v0.8.2 is a beta.** It reproduces the joint phase-space distribution
> (energy–position–direction correlations) and the energy continuum to within its
> acceptance bars, on three seeds and two reference sources. It does **not** yet
> reproduce spectral-line *intensities*: measured per-line recovery runs 0.7×–4.9× and
> is unstable across seeds. Line *positions* are reliable to ≤0.5 eV. See
> [Accuracy you can rely on](docs/USAGE.md#accuracy-you-can-rely-on) before quoting any
> number derived from a generated source.

## Installation

```bash
git clone https://github.com/francescomonastra/MAGI.git
cd MAGI
pip install -e MAGI_package/
```

```python
import magi
magi.initialize_environment(seed=42)
```

## Tests

135 tests covering the pieces that fail silently rather than loudly: flow/prior
round-trip identity, line-integral recovery on synthetic spectra, the coupling
prior actually fitting an injected coupling, prior zone-conditioning, gate-target
construction, the checkpoint config-match guard, the Geant4 export script
end-to-end, the two visualization tools' HTML/JS generation, and public-API
docstring coverage. CPU only, ~3 minutes.

```bash
pip install pytest
pytest MAGI_package/tests/
```

They catch mechanical regressions. They do **not** tell you a trained model
reproduced your source — that is `tools/acceptance_v0_8.py`'s job.

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
    └── utils/             # plotting, model introspection, interactive HTML diagnostics
        ├── plotting.py
        ├── model_inspection.py
        ├── circuit_viz.py     #   save_routing_circuit: gate routing per type
        └── full_circuit.py    #   save_full_circuit: per-unit network-usage attribution
```

`magi.print_model_structure(model)` and `magi.print_model_tree_with_params(model)` print
a description of a built model — its generative structure and the formulas behind it, the
configured mixture/prior/flow settings, and the parameter counts per block.

`magi.save_routing_circuit(...)` and `magi.save_full_circuit(...)` write a self-contained,
interactive HTML file for inspecting a trained run in a browser: the gate's per-type
routing, and how heavily every unit in the network is used per particle type (Expected
Conductance), with a per-layer absolute-magnitude panel for asking whether a layer is
wider than it needs to be. See
[Inspecting a trained model](docs/USAGE.md#inspecting-a-trained-model) in the usage guide,
which also covers what the usage metric does and does not license you to conclude.

## Author

Francesco Monastra (INAF) — `francesco.monastra@inaf.it`
