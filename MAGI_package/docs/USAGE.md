# MAGI usage guide

This is the user-facing guide to MAGI: what it does, how to use it, and — just as
importantly — what to watch for when adapting it to a new detector or source. For a
short overview see [`../README.md`](../README.md); for a runnable, fully-commented
walkthrough on synthetic data see [`../../Example_Usage.ipynb`](../../Example_Usage.ipynb).

## What MAGI does

MAGI (the package is imported as `magi`) is a Conditional Variational Autoencoder (CVAE),
implemented in Keras/TensorFlow, that learns the phase-space distribution of particles
crossing a detector surface in a Geant4 Monte Carlo simulation — energy, position,
direction, and particle type — and can then sample new, physically-consistent events at a
fraction of the cost of running full Geant4 transport.

The intended use is as a **drop-in replacement (or augmentation) for a Geant4 particle
source**: run the expensive full-physics simulation once to get a reference sample of
particles crossing some surface, train MAGI on that sample, then use MAGI's generated
output as the particle source for downstream simulations that would otherwise need many
repeats of the expensive upstream physics (e.g. re-running a full cosmic-ray shower or a
radioactive-decay chain through shielding for every downstream detector configuration you
want to test). This is most valuable when:

- large numbers of particles are needed downstream,
- the events reaching your surface of interest are rare relative to the number of
  primaries thrown (low statistics),
- you need to run the same downstream simulation many times with different geometry/
  configuration choices, and re-deriving the upstream flux every time is wasteful.

## Typical workflow

1. **Load** a raw Geant4 crossing-data file with `magi.data.load_detector_table`. Three
   column schemas are auto-detected: the legacy 9-column format
   (`EventId ParticleName Energy X Y Z Vx Vy Vz`), a 10-column format that adds a
   `PrimBool` primary/secondary flag, and a 13-column "lineage" format that additionally
   carries `ParticleId`, `ParentParticleId`, `CreatorProcessName` (needed for
   radioactive-source flux normalization, see below).
2. **Preprocess**: `build_physical_features` (raw → physical features: radial/angular
   position and direction relative to a sphere, energy) → `build_feature_dataframe`
   (bins energy, applies the geometry transform for the model version you're targeting) →
   `filter_particle_types_continuous_geometry` (drops rare particle types, builds
   per-type conditioning) → `split_feature_data` → `scale_continuous_features` →
   `build_conditioning_and_weights` → `build_tf_datasets`. Every step has a matching
   `report_*` function for printed sanity checks — use them, they catch most data issues
   before you waste time training.
3. **Train**: pick a model class from `magi.core` (see below), `magi.training.compile_model`
   then `train_single_run` or `fit_model`.
4. **Checkpoint**: `magi.training.checkpointing.save_final_trained_model` /
   `save_training_checkpoint` save weights + JSON metadata + fitted transformers as one
   unit under `trained_models/<run_name>/`. Always treat this set of files as a unit —
   generation needs the weights *and* the matching preprocessing metadata to reconstruct
   physical quantities correctly. `load_task_adaptive_model_for_generation` reloads a
   trained run for inference.
5. **Generate**: sample from the trained model and reconstruct physical quantities
   (`magi.generation.reconstruct_generated_physics`), then write a Geant4-ready particle
   source file — text or chunked binary — with `generated_physics_to_detector_dataframe`
   / `generate_detector_table_to_file`. `scripts/generate_geant_source.py` wraps this for
   direct invocation from a Geant4 macro (`/generator/mlScript ...`).
6. **Validate**: `magi.validation` (Wasserstein distances, histogram residuals between
   real and generated distributions) and `magi.utils.plotting` — this is the main way
   "did training work" gets assessed; there is no automated test suite (see below).

## Model variants (`magi.core`)

The models form a version lineage — check which one a given trained run actually is
(`model_config["model_class"]` in its metadata) before assuming behavior:

- `CVAE_CatEnergy_CatUV` — original model: categorical energy head, discretized `u_v`.
- `CVAE_CatEnergy_CatUV_TaskAdaptive` — adds task-adaptive loss weighting.
- `CVAE_CatEnergy_ContGeom_TaskAdaptive` — continuous geometry targets
  `[u_r, u_v, cphi_r, sphi_r, cphi_v, sphi_v]` (v0.7).
- `CVAE_CatEnergy_ContPhi_TaskAdaptive` — continuous geometry including continuous
  `phi_r`/`phi_v` targets `[u_r, u_v, phi_r, phi_v]` (v0.7.2, current default).

Each uses per-variable heads (categorical for logE, Gaussian for radial/angular
variables, unit-circle-regularized 2D heads for angles) rather than one flat output,
because narrow spectral energy lines and boundary-heavy angular variables (`u_v`
concentrates near ±1) don't fit a naive Gaussian/flat output well.

## Limitations & things to watch during training and adaptation

**Geometry is currently hard-limited to a sphere.** `build_physical_features`'s
`center`/`radius` parameters and the coordinate transforms in `core/geometry.py`
(`xy_from_ur_phi`, `vxyz_from_uv_phi`, etc., all parametrized by `sphere_R`) assume the
crossing surface is a sphere centered at a fixed point. Adapting MAGI to a planar,
cylindrical, or box-shaped detector surface is **not** a matter of passing different
parameters — it requires reworking the geometry-transform layer (both the physical
feature construction in `preprocessing.py` and the reconstruction in
`generation/reconstruction.py`, which must stay in sync).

**`geometry_transform` is a stringly-typed contract.** The string value
(`"arctanh_uv_discrete"`, `"quantile_u_r_u_v"`, `"quantile_u_r_u_v_phi_r_phi_v"`) and the
resulting `X_cont_cols` ordering are shared by preprocessing, the model's input layout,
and reconstruction. They're persisted in saved metadata and **must** match exactly
between the run that trained a model and the run that generates from it — there's no
validation that catches a mismatch early, it just produces physically wrong output.

**Particle-type filtering silently drops rare species.** `prob_threshold` in
`filter_particle_types_continuous_geometry` removes particle types below a given
population fraction. If a rare-but-relevant particle type disappears from your generated
output, check this first.

**Energy-binning mode affects spectral-line resolution.** `build_energy_bins` supports
`"fixed_width"`, `"min_counts"`, and `"log_fixed_count"` modes — the choice trades off
against the categorical energy head's ability to resolve narrow spectral lines (e.g.
characteristic decay-line gammas) versus giving every bin enough statistics to train on.

**Quantile transforms need enough samples.** The `QuantileTransformer`-based geometry
modes (`u_r_q`, `u_v_q`, `phi_r_q`, `phi_v_q`) are fit per-feature on your training data;
small datasets risk unreliable tail behavior in the fitted transform, which shows up as
implausible extreme values in generated output.

**Flux normalization depends on getting `source_type` right.** `build_physical_features`
accepts `source_type="cosmic_ray"` (default: a crossing is "primary" iff `PrimBool==1`,
i.e. the raw GPS-thrown particle reached the surface unmodified) or
`source_type="radioactive"` (a crossing is "primary" iff
`CreatorProcessName=="RadioactiveDecay"` — needed because for a decaying source embedded
in bulk material, the literal primary track is the decaying nucleus itself, which never
propagates anywhere, so `PrimBool` is *always* 0 and cannot be used). Radioactive sources
need the 13-column lineage schema (`CreatorProcessName` must be present in the raw file).
Picking the wrong `source_type` for your source doesn't error — it silently produces a
wrong flux-normalization factor. See `compute_primary_fraction` /
`save_normalization_summary` / `load_normalization_summary` for measuring and persisting
this factor.

**No automated test suite.** Validation is empirical, via `magi.validation` and notebook
plots — there's no CI, and a geometry round-trip identity test (physics → features →
transforms → reconstruction → physics, asserting the output matches the input within
tolerance) is a known, not-yet-built gap. If you change anything in the geometry-transform
layer, re-validate by hand against real data before trusting generated output.

**CPU-only by default.** `magi.initialize_environment(cpu_only=True)` is the default —
GPU/`tensorflow-metal` acceleration needs explicit opt-in.

## Data & artifact conventions

- Raw Geant4 output → concatenated/cleaned into whitespace-delimited `.dat` files (see
  `TrainingData/CommandsToDataCleaning.txt`). Large raw data and generated artifacts are
  gitignored; only code, notebooks, and small JSON metadata are versioned.
- Trained runs under `trained_models/<run_name>/` or `checkpoints/<run_name>/` each
  contain `*.weights.h5`, `*_config.json`, `*_history.json`, `*_metadata.json`,
  `*_task_weights.json`, `*_summary.txt`, and (for quantile-geometry models)
  `*_quantile_transformers.joblib`. Copy/load this set together.
