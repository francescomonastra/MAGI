# MAGI

A CVAE (Conditional Variational Autoencoder) that learns the phase-space distribution of
particles crossing a detector surface in a Geant4 Monte Carlo simulation, and generates
new physically-consistent events at a fraction of the cost of running full Geant4
transport — aimed at improving simulation efficiency, especially in low-statistics cases.
Built on `keras`/`tensorflow`. Author: Francesco Monastra (INAF).

## Repo layout

- **[`MAGI_package/`](MAGI_package/)** — the installable Python library (`import magi`).
  See [`MAGI_package/README.md`](MAGI_package/README.md) for installation and
  [`MAGI_package/docs/USAGE.md`](MAGI_package/docs/USAGE.md) for the full usage guide,
  including current limitations (e.g. the sphere-only geometry assumption) to be aware of
  before training or adapting the model.
- **[`Example_Usage.ipynb`](Example_Usage.ipynb)** — a runnable, fully-commented
  walkthrough of the whole pipeline on synthetic data. Start here if you're new to MAGI.
- **`MAGI_v0_*.ipynb`** at the repo root — the actual experiment notebooks (highest
  version number = current pipeline). Each is a full run: load data → preprocess → build
  dataset → train → generate → validate → plot. `OldNotebooks/` holds deprecated
  versions kept for reference.
- **`TrainingData/`, `trained_models/`, `checkpoints/`** — raw/cleaned crossing data and
  trained-run artifacts (gitignored — large binaries stay local).
- **`scripts/generate_geant_source.py`** — generates a Geant4-ready particle source file
  from a trained model outside a notebook; this is what the companion Geant4 project's
  macros invoke directly (`/generator/mlScript ...`) for on-demand generation.

## Quick start

```bash
git clone https://github.com/Fchewie/MAGI.git
cd MAGI
pip install -e MAGI_package/
```

```python
import magi
magi.initialize_environment(seed=42)
```

Then see [`Example_Usage.ipynb`](Example_Usage.ipynb) or
[`MAGI_package/docs/USAGE.md`](MAGI_package/docs/USAGE.md) for the full workflow.
