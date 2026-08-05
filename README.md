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
- **[`MAGI_Colab_GPU_Benchmark.ipynb`](MAGI_Colab_GPU_Benchmark.ipynb)** — runs the
  v0.8.2 pipeline on a Google Colab GPU runtime and times it against the CPU baseline;
  see [GPU training](#gpu-training-optional) below.
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
  what the measurements showed, and what is still open. **[`docs/manual/`](docs/manual/)**
  holds the LaTeX user manual (`magi_manual.pdf`, built with `./build.sh`) — the
  long-form reference for structure, API, tools, a worked run, a worked validation,
  the theoretical foundation, and the accuracy envelope.

## Quick start

```bash
git clone https://github.com/francescomonastra/MAGI.git
cd MAGI
pip install -e MAGI_package/
```

```python
import magi
magi.initialize_environment(seed=42)
```

## GPU training (optional)

Training defaults to CPU (`cpu_only=True`) — on Apple Silicon, Metal/GPU was measured
*slower* than CPU for this op mix (see `MAGI_v0_8_2.ipynb`). On CUDA hardware it's a
different story: [`MAGI_Colab_GPU_Benchmark.ipynb`](MAGI_Colab_GPU_Benchmark.ipynb) runs
the same v0.8.2 pipeline (`magi.initialize_environment(cpu_only=False)`) on a free
Google Colab T4 GPU and measured **~2.2x faster** training on CryoSphere-CR — 22 min vs.
the ~48 min CPU reference, same 40-epoch config — with generated-sample accuracy
unchanged (Wasserstein log₁₀E 0.028, within the ≤0.05 bar below). Single run, one seed,
one GPU type — treat "~2x" as a ballpark, not a guarantee across sources or hardware.

## Status: v0.8.2 beta

Measured over three seeds (42/7/13) on two reference sources with
`tools/acceptance_v0_8.py`:

| Quantity | CryoSphere-CR | CryoSphere-Small | Bar |
|---|---|---|---|
| Coupling, max\|Δcorr\| | 0.0285 ± 0.0101 | 0.039 ± 0.011 | ≤0.05 ✅ |
| Wasserstein, log₁₀E | 0.0241 ± 0.0046 | 0.0065 ± 0.0017 | ≤0.05 ✅ |
| Per-line intensity | 0.96×–2.05×, unstable | 1.11×–3.25×, unstable | ❌ |

The joint distribution and the energy continuum are validated with error bars.
Spectral-line **intensities** are not — do not use generated output to estimate a
fluorescence or annihilation line flux. Line **positions** are reliable (≤0.5 eV,
audited). Details and the full per-line table:
[`MAGI_package/docs/USAGE.md`](MAGI_package/docs/USAGE.md#accuracy-you-can-rely-on).

Then see [`Example_Usage.ipynb`](Example_Usage.ipynb) or
[`MAGI_package/docs/USAGE.md`](MAGI_package/docs/USAGE.md) for the full workflow.
