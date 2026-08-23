# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MAGI is a research project, not a production application. A Conditional Variational
Autoencoder (CVAE), implemented in Keras/TensorFlow, learns the phase-space distribution
of particles (energy, position, direction) crossing a detector surface in Geant4 Monte
Carlo simulations, so that new physically-consistent events can be sampled cheaply
instead of running full Geant4 transport. Author/maintainer: Francesco Monastra (INAF).

The repo has two parts:
- `MAGI_package/` — the installable Python library (`import magi`).
- The repo root — Jupyter notebooks, raw/training data, trained model artifacts, and
  Optuna hyperparameter-search results that *use* the library. This is where actual
  experiments happen.

## Environment / install

```bash
pip install -r requirements.txt          # installs magi in editable mode from GitHub
# or, to develop the package itself:
pip install -e MAGI_package/
```

TensorFlow/Keras (`tensorflow`, `tensorflow-macos`, `tensorflow-metal` on Apple
Silicon), `optuna`, `astropy`, `joblib`, `scikit-learn` are core runtime deps. There is
no test suite, linter, or CI config in this repo — validation happens empirically via
notebooks and the `magi.validation` module (Wasserstein distances, histogram
comparisons between real and generated distributions).

## Typical workflow

See also `MAGI_package/docs/USAGE.md` for the user-facing version of this workflow, and
`Example_Usage.ipynb` for a runnable, fully-commented walkthrough on synthetic data.

1. Open the newest `MAGI_v0_*.ipynb` notebook at the repo root (highest version
   number = current pipeline; older ones are kept for reference, `OldNotebooks/` holds
   deprecated versions). Each notebook is a full run: load data → preprocess → build
   dataset → train → generate → validate → plot.
2. `magi.setup(...)` initializes the environment (seed, CPU-only TF, quiet logging) —
   call this before anything else in a notebook.
3. Data prep: `magi.data` (`load_detector_table` → `build_physical_features` →
   `build_feature_dataframe` / `fit_quantile_geometry_transforms` →
   `filter_particle_types_continuous_geometry` → `split_feature_data` →
   `scale_continuous_features` → `build_conditioning_and_weights` →
   `build_tf_datasets`). Most steps have a matching `report_*` function used purely for
   printed sanity checks in the notebook.
4. Model: pick a class from `magi.core` (see Architecture below), then
   `magi.training.compile_model` / `train_single_run` / `fit_model`.
5. Checkpointing: `magi.training.checkpointing` saves weights + JSON metadata +
   fitted transformers (`*_quantile_transformers.joblib`) under `trained_models/<run_name>/`
   or `checkpoints/<run_name>/`. `load_task_adaptive_model_for_generation` reloads a
   trained run for inference.
6. Generation: `magi.generation` samples from the trained model and reconstructs
   physical quantities (`reconstruct_generated_physics`), then
   `generated_physics_to_detector_dataframe` / `generate_detector_table_to_file` write
   Geant4-ready particle source files (text or binary).
7. Validation/plots: `magi.validation` (Wasserstein scores, real-vs-generated
   comparisons) and `magi.utils.plotting` (dist/pairgrid/correlation plots), saved
   into `Plots/`.
8. To generate a Geant4 input file from a trained run outside a notebook, use
   `scripts/generate_geant_source.py --save-dir ... --model-name ... --metadata-file ...
   --output-file ... --n-events ...` (reads the saved metadata JSON and, for quantile
   geometry models, the matching `*_quantile_transformers.joblib`).

## Architecture (`MAGI_package/magi/`)

- `core/model.py` — the CVAE architectures, each a `keras.Model` subclass. They form a
  version lineage; check which one a notebook actually instantiates before assuming
  behavior:
  - `CVAE_CatEnergy_CatUV` — original model: categorical energy head + discretized `u_v`.
  - `CVAE_CatEnergy_CatUV_TaskAdaptive` — adds task-adaptive loss weighting (see
    `training/adaptive_callbacks.py`).
  - `CVAE_CatEnergy_ContGeom_TaskAdaptive` — continuous geometry targets
    `[u_r, u_v, cphi_r, sphi_r, cphi_v, sphi_v]` (v0.7 quantile-transform line).
  - `CVAE_CatEnergy_ContPhi_TaskAdaptive` — continuous geometry incl. continuous
    `phi_r`/`phi_v` targets `[u_r, u_v, phi_r, phi_v]` (v0.7.2, the stable head).
  - `CVAE_MixEnergy_ContPhi_TaskAdaptive` — v0.7.2 geometry, but the categorical energy
    head becomes a gated mixture of a continuum density (a single Gaussian, or the
    conditional spline flow in `core/flows.py`) and fixed-position line components
    pinned at measured line energies; optionally with the learned conditional prior in
    `core/priors.py` (v0.8, newest head).
  Each model uses per-variable heads (categorical or mixture for logE, Gaussian for
  radial/angular variables, unit-circle-regularized 2D heads for angles) rather than one
  flat output — see `MAGI_package/docs/USAGE.md` for why, and
  `magi.print_model_structure(model)` for a printed description of a built model
  (generative structure, formulas, configured line table, parameter counts).
  `core/losses.py` and `core/geometry.py` hold the shared NLL/angular losses and the
  physical coordinate transforms (`xy_from_ur_phi`, `vxyz_from_uv_phi`, etc.) used both
  inside the model and during generation/reconstruction — keep these two in sync if you
  change a coordinate convention.
- `data/` — `io.py` (load/save the raw detector table; also
  `save_normalization_summary`/`load_normalization_summary` for exporting a measured
  primary-fraction correction factor as standalone JSON, e.g. for the Geant4-side
  analysis notebook to read), `preprocessing.py` (physical feature engineering, energy
  binning, quantile-transform fitting, `compute_primary_fraction` for flux
  normalization), `dataset.py` (splitting, scaling, one-hot conditioning, building
  `tf.data.Dataset`s). Functions are versioned by geometry convention
  (`filter_particle_types_and_discretize_uv` = legacy v0.6 discretized `u_v`;
  `filter_particle_types_continuous_geometry` = v0.7+ continuous). `geometry_transform`
  string values (`"arctanh_uv_discrete"`, `"quantile_u_r_u_v"`,
  `"quantile_u_r_u_v_phi_r_phi_v"`) select which transform a given run/model expects —
  this string is persisted in saved metadata and must match between training and
  generation. Geometry is currently hard-limited to a **sphere** (`center`/`radius`
  params) — see `MAGI_package/docs/USAGE.md` for what adapting this to other detector
  geometries would require.
- `training/` — `train.py` (compile/fit wrappers), `adaptive_callbacks.py`
  (`TaskAdaptiveLossScheduler` rebalances per-task loss weights during training,
  `TaskAdaptiveTrainingMonitor`/`ValidationEnergyDistributionMonitor` log diagnostics),
  `checkpointing.py` (saving/loading weights + metadata + transformers as a unit).
- `generation/` — `sampling.py` (draw latent codes/conditioning), `reconstruction.py`
  (invert model outputs back to physical `x,y,z,vx,vy,vz,E`), `export.py` (write
  Geant4-format particle source files, text or binary, chunked for large `n_events`).
- `validation/` — `metrics.py`/`compare.py`: Wasserstein distance and histogram-residual
  comparisons between real and generated distributions, the main way "did training
  work" is assessed here.
- `utils/` — `plotting.py` (all the `plot_*` helpers used in notebooks, with a
  dark/light theme switch via `set_plot_theme`), `model_inspection.py` (introspect
  layer/parameter structure of a built model), `circuit_viz.py`
  (`save_routing_circuit`: interactive HTML of the v0.8 gate's per-type routing,
  from a checkpoint's own `zone_probs`) and `full_circuit.py`
  (`save_full_circuit`: interactive HTML showing how heavily every unit in the
  network is used, per particle type plus a pooled "All types" view, measured
  with Expected Conductance — Integrated Gradients generalized to internal
  units, with baselines drawn from real held-out events rather than a zero
  vector, since inputs are quantile-transformed and "0" is a real quantile.
  Only the continuous feature block is interpolated along the attribution path;
  the one-hot type conditioning is held fixed. Covers every stage — encoder, z,
  decoder stem, trunk, and all three heads — over the real energy spectrum;
  re-derives the held-out split from the raw detector table, so unlike the other
  two it needs real data, not just the checkpoint, and takes minutes rather than
  seconds. **Node colors are percentile ranks, not magnitudes** — by
  construction some units always look dark, so the colors cannot answer "is this
  layer oversized"; the per-layer absolute-usage panel underneath carries the
  raw magnitudes (`share`, `load` = share-of-work over share-of-width,
  concentration, faint-unit fraction) for that question).
- `magi/magi.py` is a simplified high-level API (`setup`, `build_model`, `train_model`,
  `plot_training`) layered on top of the modules above for quick notebook use;
  `__init__.py` re-exports the full public surface (this is the canonical list of what's
  considered "public API").

## Data & artifact conventions

- Raw Geant4 output lands in `RawData/`, gets concatenated/cleaned (see
  `TrainingData/CommandsToDataCleaning.txt` for the `find | xargs cat` + `awk` cleaning
  recipe) into whitespace-delimited `.dat` files in `TrainingData/` with columns
  `EventId, ParticleName, Energy, X, Y, Z, Vx, Vy, Vz`. `RawData/` is treated as scratch
  and gets wiped after concatenation.
- `TrainingData/`, `checkpoints/generated_particles/`, and most raw data formats
  (`*.dat`, `*.root`, `*.npy`, `*.h5`, ...) are gitignored — large binary artifacts stay
  local, only code/notebooks/small JSON metadata are versioned.
- Trained runs live under `trained_models/<run_name>/` or `checkpoints/<run_name>/`, each
  containing `*.weights.h5`, `*_config.json`, `*_history.json`, `*_metadata.json`,
  `*_task_weights.json`, `*_summary.txt`, and (for quantile-geometry models)
  `*_quantile_transformers.joblib`. Treat this set of files as one unit — copy/load them
  together, since generation needs both the weights and the matching preprocessing
  metadata/transformers to reconstruct physical quantities correctly.
- `Versions Guideline.md` at the repo root tracks the model-variant naming history
  (e.g. `s_v`/`t_v` reparametrizations, mixture-of-Gaussians heads) for versions beyond
  what's in the current `core/model.py` — consult it when a notebook references a
  version-specific parameter not obviously present in the current code.
- `NormalFlow_Fioretti/` is an external/experimental script (increasing SPO via
  Normalizing Flows), not part of the MAGI package proper.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
