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
  trained-run artifacts (large binaries are gitignored and stay local; each trained run's
  JSON config/metadata is versioned, and must be kept with its weights to reload).
- **`scripts/generate_geant_source.py`** — generates a Geant4-ready particle source file
  from a trained model outside a notebook; this is what the companion Geant4 project's
  macros invoke directly (`/generator/mlScript ...`) for on-demand generation.
- **[`tools/`](tools/)** — scripts around a run rather than inside it: full CR/Small
  training runs with live logging (`run_v0_8_real.py`), the pass/fail acceptance harness
  (`acceptance_v0_8.py`), the spectral-line centroid audit (`line_centroid_audit.py`),
  the candidate-line table builder from a Geant4 GDML mass model
  (`build_candidate_lines_from_geant4.py`), and the synthetic stress tests used to gate a
  change before spending a real run on it.
- **`CandidateLines/`** — candidate spectral-line tables built from a mass model, in the
  JSON form `magi.load_candidate_energy_lines` reads. Used to pin the v0.8 mixture head's
  line components at measured energies.
- **[`docs/`](docs/)** — development notes and plans for the v0.8 line: what was tried,
  what the measurements showed, and what is still open.

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
