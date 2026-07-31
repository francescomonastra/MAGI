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
  `phi_r`/`phi_v` targets `[u_r, u_v, phi_r, phi_v]` (v0.7.2; the stable default, and
  the fallback if a v0.8 run does not meet its acceptance thresholds).
- `CVAE_MixEnergy_ContPhi_TaskAdaptive` — same geometry as v0.7.2, but the categorical
  energy head is replaced by a **continuous mixture** (v0.8): a gate routes each event
  between a continuum density and `n_lines` fixed-position Gaussian line components
  pinned at measured line energies. The continuum is either a single Gaussian or a
  conditional rational-quadratic-spline normalizing flow (`continuum_mode="flow"`), and
  the latent prior is either `N(0, I)` or a learned conditional coupling flow
  (`prior="coupling"`). See `magi.print_model_structure(model)` for a printed
  description of a built model, including the formulas and the configured line table.

  Two v0.8.2 additions are worth knowing about:

  - **`prior_zone_conditioning=True`** widens what the learned prior is conditioned on
    from the particle-type one-hot alone to `[type, zone]`, where zone is the
    continuum/line routing target. The gate is trained on `z ~ q(z|x)` but generates
    from `z ~ p(z|cond)`; without the zone in `cond` the prior cannot express which
    component an event belongs to, and rare lines are routed by whatever the type
    marginal happens to imply. Requires `zone_probs`, the per-type zone distribution
    measured from the training data. This fixed CR's Al Kα1 (0.702 ± 0.290 → 0.959 ±
    0.025) while leaving coupling intact.
  - **`build_gate_targets(..., bandwidth_mode="exact")`** decides which real events
    belong to a line by exact energy match rather than by a Gaussian kernel of the
    detector's resolution width. In raw Geant4 crossing data **no detector response has
    been applied**, so simulated fluorescence lines are exactly monoenergetic — every Cu
    Kα1 event shares one identical float64 energy, verified directly against the data.

    **This is a research option, not a recommendation. `"resolution"` remains the
    default.** Measured on CR, the exact label did *not* fix the Cu Kα1 overshoot it was
    built for (2.011 vs 2.052, inside the noise), regressed the previously-passing Al
    Kα1 from 0.959 ± 0.025 to 0.578, and pushed the coupling residual outside its bar.
    The tight tolerance also drove Cu Kα2's zone probability to zero, deleting it as a
    modelled component. The soft resolution kernel appears to regularize the gate in a
    way a hard 0/1 target does not. Full write-up:
    [`../../docs/v0.8.1_line_truth.md`](../../docs/v0.8.1_line_truth.md) §14.2.

  Checkpoints written by v0.8.2 carry `config_version: 2`. Older `config_version: 1`
  checkpoints load unchanged — `prior_zone_conditioning` defaults to `False`, which
  reconstructs the exact pre-existing architecture.

The first four use per-variable heads (categorical for logE, Gaussian for radial/angular
variables, unit-circle-regularized 2D heads for angles) rather than one flat output,
because narrow spectral energy lines and boundary-heavy angular variables (`u_v`
concentrates near ±1) don't fit a naive Gaussian/flat output well. v0.8 keeps that
principle for geometry and takes the energy head further: a categorical head can never
place a line more precisely than one bin, so the line positions become fixed physical
inputs instead of something the model has to learn.

## Inspecting a trained model

Two functions in `magi.utils` write a self-contained, interactive HTML file for
looking inside a trained v0.8 mixture run — no plotting library, no running
kernel, just open the file in a browser:

- **`magi.save_routing_circuit(save_dir, model_name)`** — the gate's conditional
  routing per type, colored by the checkpoint's own `zone_probs`. Reads only the
  checkpoint's saved config/metadata, so it runs in under a second. Requires
  `prior_zone_conditioning=True` at training time; raises `KeyError` otherwise
  (no `zone_probs` to plot).
- **`magi.save_full_circuit(save_dir, model_name, training_data_filepath, candidate_lines_file, center, radius)`**
  — a per-event trace: for one real held-out event per type, gradient×activation
  attribution through every stage (encoder, latent `z`, decoder stem, deep
  trunk, and all three heads), ending in the real marginal energy spectrum with
  the selected type's contribution shaded on top. Unlike the routing circuit,
  this needs a real held-out sample to trace, so it re-derives the
  training-time preprocessing (line matching, gate targets, quantile/log10
  transforms, the train/test split) from the raw detector table you point it
  at — budget roughly one preprocessing pass over your source (minutes, not
  seconds; longer for a bigger source). Both an adjustable "highlight top N%"
  slider and a light/dark toggle are built into the page (the toggle is manual
  rather than `prefers-color-scheme`-driven, since that wasn't reliably honored
  by every viewer this was tested in).

```python
magi.save_routing_circuit(save_dir="trained_models/v0_8_2_priorzone_CR", model_name="mix_CR")

magi.save_full_circuit(
    save_dir="trained_models/v0_8_2_priorzone_CR", model_name="mix_CR",
    training_data_filepath="TrainingData/alloutputDSCryoSphereCR.dat",
    candidate_lines_file="CandidateLines/CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json",
    center=(0.0, 0.0, -507.66), radius=100.0)
```

Both derive their layout from the loaded checkpoint's own architecture (number
of types, gate zones, encoder/branch depth) rather than assuming one source's
shape, so the same call works unchanged on CR, Small, or any other source.

## Accuracy you can rely on

**Read this before quoting any number a MAGI-generated source produces.** v0.8.2 is a
**beta**. The table below is what has actually been measured, as mean ± std over three
seeds (42/7/13) on the two reference sources, scored with `tools/acceptance_v0_8.py`.
Reproduce it yourself with:

```bash
python tools/acceptance_v0_8.py --sources CR Small --seeds 42 7 13
```

### Validated

| Quantity | CryoSphere-CR | CryoSphere-Small | Bar |
|---|---|---|---|
| Coupling, max\|Δcorr\| over (logE, u_r, u_v, φ_r, φ_v) | 0.0285 ± 0.0101 | 0.0357 ± 0.0102 | ≤ 0.05 |
| Energy marginal, Wasserstein on log₁₀E | 0.0241 ± 0.0046 | 0.0065 ± 0.0006 | ≤ 0.05 |

The **joint** distribution is what v0.8 exists for and it holds up with error bars: every
real cross-correlation between energy and the four geometry variables is reproduced
inside the bar, on both sources, across seeds. If your downstream use depends on
energy–direction or energy–position correlations, this is the result that matters, and
it is one the v0.7.2 categorical head does not deliver (its coupling residual is 0.226,
8× worse).

### Not validated — known limitations

Per-line **intensities** are wrong by factors of roughly 0.7× to 4.9×, and unstable
across seeds:

| Line | Recovery (mean ± std) | Status |
|---|---|---|
| CR Al Kα1 | 0.959 ± 0.025 | in band |
| CR Cu Kβ | 1.034 ± 0.050 | in band |
| CR e⁺e⁻ 511 keV | 1.825 ± 0.324 | **over-generated, unstable** |
| CR Cu Kα1 | 2.052 ± 0.034 (routing) | **over-generated** |
| Small Al Kα2 | 3.412 ± 0.376 | **far over** |
| Small Cu Kα1 | 4.903 ± 0.356 (routing) | **far over** |

Practical consequences:

- Line **positions** are reliable — they are pinned physical inputs, audited to ≤0.5 eV
  against the measured spectrum on all sources (`tools/line_centroid_audit.py`). It is
  the number of events in each line that is not.
- Do **not** use generated output to estimate a fluorescence or annihilation line flux,
  or any quantity dominated by one.
- The continuum between lines, and the total energy marginal, are within the bar.
- Rare lines are worse than strong ones, and the failure is source-dependent — measure
  on *your* source with the acceptance harness rather than assuming these numbers carry
  over.
- v0.7.2 (`CVAE_CatEnergy_ContPhi_TaskAdaptive`) remains the conservative fallback for
  the energy marginal alone (CR Wasserstein 0.0141 vs 0.0241), but it produces **zero**
  events in the 511 keV window against 52,225 real, and 8× worse coupling. Neither head
  is correct for lines today.

The open diagnostic work behind these numbers is written up in
[`../../docs/v0.8.1_line_truth.md`](../../docs/v0.8.1_line_truth.md); the remaining plan
is in [`../../docs/v0.8.2_RoadmapForAdoption.md`](../../docs/v0.8.2_RoadmapForAdoption.md).

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
With the v0.8 mixture head the binning no longer limits where a line can be *placed*
(positions are pinned physical inputs), but it still limits which lines
`detect_energy_lines` can *find*: on a log grid spanning many decades a bin can be
hundreds of eV wide, so close doublets fall in one bin and only one member is matched.
Pass `refine_bin_width_mev` to refine each detected peak to sub-bin precision, and check
the result with `tools/line_centroid_audit.py`.

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

**Line intensities are not yet validated — see [Accuracy you can rely on](#accuracy-you-can-rely-on)
below before quoting a generated line flux.**

**Unit tests cover the machinery, not the physics.** `MAGI_package/tests/` holds 135
tests (`python -m pytest MAGI_package/tests/`) covering flow round-trips, checkpoint
save/load config matching, gate-target construction, prior zone-conditioning, the two
visualization tools, and public API docstring coverage. They catch mechanical
regressions. They do **not** tell you a
trained model reproduced your source: that is still empirical, via `magi.validation`,
`tools/acceptance_v0_8.py` and notebook plots. If you change anything in the
geometry-transform layer, re-validate by hand against real data before trusting
generated output.

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
