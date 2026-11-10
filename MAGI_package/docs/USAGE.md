# MAGI user manual (v1.0.0)

This manual describes how to train a MAGI model on a Geant4 crossing sample, how to generate particle sources from it, how to judge the result, and which measures improve a model that does not yet meet the requirements of the application. It is written for a user who has a reference simulation of their own and wishes to adapt MAGI to it; every limit is stated with the number that was measured, and a cause is given where one has been established and marked as open where it has not. A short overview is given in [`../README.md`](../README.md), and a runnable, fully commented walkthrough on synthetic data is provided by [`../../Example_Usage.ipynb`](../../Example_Usage.ipynb). The changes with respect to earlier versions are listed in [`../../CHANGELOG.md`](../../CHANGELOG.md).

MAGI v1.0.0 is released under the GPL-3.0 licence ([`../../LICENSE`](../../LICENSE)). Work that uses it is cited through [`../../CITATION.cff`](../../CITATION.cff). The trained checkpoints and the Geant4 training data of the reference sources are large, are ignored by git and are not distributed with the repository.

**Which section to read.**

| You want to | Read |
|---|---|
| Train and generate for one source | 1, 2, 3, 4 |
| Decide whether the result can be trusted | 8, 9 |
| Improve a model that misses a requirement | 7 |
| Condition on the primary (advanced) | 5, 6 |
| Adapt to a new detector | 11 |
| Look inside a trained model | 10 |

## 1. Purpose and scope

MAGI (the package is imported as `magi`) is a Conditional Variational Autoencoder (CVAE), implemented in Keras/TensorFlow, that learns the phase-space distribution of the particles crossing a closed surface in a Geant4 Monte Carlo simulation, namely their energy, position, direction and type, and samples new crossings at a small fraction of the cost of the full transport.

The intended use is as a replacement for, or an augmentation of, a Geant4 particle source. The expensive upstream simulation is run once to collect a reference sample of the particles crossing the surface, MAGI is trained on that sample, and the generated crossings then drive the downstream simulations that would otherwise require the upstream physics to be repeated for every configuration. The approach is most valuable when large numbers of particles are needed downstream, when the events reaching the surface are rare with respect to the primaries thrown, and when the same downstream simulation has to be repeated for many geometry or configuration choices.

Two limits define the scope of version 1.0.0. The crossing surface must be a sphere (Section 11), and the crossings of one primary are drawn independently given its type and energy (Section 9). The follow-up architecture, an autoregressive flow that removes the second limit, is developed in the `MAGIv2/` folder of the repository and is not covered here.

**Quickstart.** The notebook [`../../Example_Usage.ipynb`](../../Example_Usage.ipynb) runs the whole chain on a synthetic source in a few minutes of CPU: load, preprocess, build the mixture model, train eight epochs, save, reload, generate a source file and check it against the real sample. The calls it makes, in order, are the following.

```python
import magi
magi.setup(seed=42)
prep = magi.build_physical_features(df, center=center, radius=R, source_type="cosmic_ray")
pack = magi.build_feature_dataframe(prep, geometry_transform="quantile_u_r_u_v_phi_r_phi_v")
model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(n_types=n_types, latent_dim=4, hidden=(64, 64))
history = magi.fit_model(model, train_ds, val_ds, epochs=8, callbacks=magi.build_default_callbacks())
magi.generate_detector_table_to_file(model=model, filepath="source.dat", n_events=2000, ...)
```

The lines elide the arguments the notebook builds between them (gate targets, datasets, type probabilities), and Section 3 explains each stage for a real source.

## 2. Installation

```bash
pip install -r requirements.txt        # installs magi in editable mode
pip install -e MAGI_package/           # to develop the package itself
pip install -e "MAGI_package[dev]"     # adds pytest
```

TensorFlow/Keras (`tensorflow-macos` and `tensorflow-metal` on Apple Silicon), `optuna`, `astropy`, `joblib` and `scikit-learn` are the core runtime dependencies. `magi.initialize_environment(cpu_only=True)` is the default, and GPU acceleration requires an explicit opt-in. Call `magi.setup(...)` before anything else in a notebook, since it fixes the seed and silences the logging.

## 3. Workflow for a single source

1. **Load** the raw crossing table with `magi.data.load_detector_table`. Four column schemas are detected automatically: the legacy 9-column format (`EventId ParticleName Energy X Y Z Vx Vy Vz`), a 10-column format that adds the `PrimBool` primary flag, a 13-column lineage format that adds `ParticleId`, `ParentParticleId` and `CreatorProcessName` (needed for the flux normalisation of radioactive sources), and the 16- and 17-column formats of Sections 5 and 6.
2. **Preprocess** with `build_physical_features` (radial and angular position and direction relative to the sphere, and energy), `build_feature_dataframe` (energy binning and the geometry transform of the targeted model version), `filter_particle_types_continuous_geometry` (removal of rare types and construction of the per-type conditioning), `split_feature_data`, `scale_continuous_features`, `build_conditioning_and_weights` and `build_tf_datasets`. Each step has a matching `report_*` function that prints a sanity check, and these catch most data issues before any training time is spent.
3. **Train** a model of Section 4 with `magi.training.compile_model` followed by `train_single_run` or `fit_model`. The mixture head is compiled with `clipnorm=5.0`, since occasional large gradients of the flow and of the line components otherwise produce loss spikes. The CDF warp is fitted with `exclude_y=line_positions_y` and the model is built with `energy_line_routing="supervised"`; both settings keep the pinned lines out of the continuum flow and are explained in Section 7.1, and the toy example of the quickstart can omit them. A long run is made resumable by passing `checkpoint_dir` (Section 12).
4. **Refit the prior** on the frozen encoder, which is the last stage of the training recipe (Section 7.4). The quickstart omits it, and the reference trainers apply it with `--refit-prior`.
5. **Checkpoint** with `magi.training.checkpointing.save_final_trained_model` or `save_training_checkpoint`, which write the weights, the JSON metadata and the fitted transformers as one unit under `trained_models/<run_name>/`. Generation requires the weights and the matching preprocessing metadata together, and a run directory has to be copied or loaded as a whole. `load_task_adaptive_model_for_generation` reloads a run for inference.
6. **Generate** with `magi.generation.reconstruct_generated_physics`, and write a Geant4-ready source file, as text or as chunked binary, with `generated_physics_to_detector_dataframe` or `generate_detector_table_to_file`. `scripts/generate_geant_source.py --save-dir ... --model-name ... --metadata-file ... --output-file ... --n-events ...` wraps this step for direct use from a Geant4 macro (`/generator/mlScript`).
7. **Validate** with the protocol of Section 8 before any number obtained from the generated source is quoted.

The reference trainers are `tools/train_cr_v1.py` (CryoSphere-CR, with `--refit-prior` for the whole recipe), `tools/train_iaps.py` (the IAPS Triple sources, with `--extra-lines` to pin the Cu K fluorescence energies of 7.985, 8.006, 8.863 and 8.865 keV when each holds at least 100 crossings), `tools/train_dm12_primtruth.py` and `tools/train_athena_primtruth.py` (primary-conditioned models).

## 4. Models (`magi.core`)

The models form a version lineage, and the class of a trained run is stored in `model_config["model_class"]` of its metadata. A run has to be generated with the class it was trained with.

| Class | Energy head | Geometry targets | Status in v1.0.0 |
|---|---|---|---|
| `CVAE_MixEnergy_ContPhi_TaskAdaptive` | gated mixture of a continuum density and pinned lines | `[u_r, u_v, phi_r, phi_v]` or the local frame | the model of v1.0.0 |
| `CVAE_CatEnergy_ContPhi_TaskAdaptive` | categorical | `[u_r, u_v, phi_r, phi_v]` | conservative fallback for the energy marginal alone |
| `CVAE_CatEnergy_ContGeom_TaskAdaptive` | categorical | `[u_r, u_v, cphi_r, sphi_r, cphi_v, sphi_v]` | loads, kept for old runs |
| `CVAE_CatEnergy_CatUV_TaskAdaptive`, `CVAE_CatEnergy_CatUV` | categorical | discretised `u_v` | loads, kept for old runs |

All models use one head per variable, with a categorical or mixture head for log E, Gaussian heads for the radial and angular variables and unit-circle-regularised two-dimensional heads for the angles. This choice is due to the shape of the data, since narrow spectral lines and boundary-heavy variables (`u_v` concentrates near ±1) do not fit a flat Gaussian output. The mixture model takes the energy head further: a categorical head can never place a line more precisely than one bin, and the line positions are therefore fixed physical inputs, with the network learning only how often each line occurs.

In the mixture model a gate routes every event between a continuum density and `n_lines` Gaussian components pinned at the measured line energies. The continuum is either a single Gaussian or a conditional rational-quadratic-spline flow (`continuum_mode="flow"`), and the latent prior is either `N(0, I)` or a learned conditional coupling flow (`prior="coupling"`). `magi.print_model_structure(model)` prints the structure of a built model, with the formulas, the configured line table and the parameter counts.

`prior_zone_conditioning=True` widens the conditioning of the learned prior from the type one-hot to `[type, zone]`, where the zone is the continuum or line routing target. The gate is trained on `z ~ q(z|x)` and generates from `z ~ p(z|cond)`; without the zone in `cond` the prior cannot express which component an event belongs to, and rare lines are routed according to the type marginal. This option requires `zone_probs`, the per-type zone distribution measured on the training data, and it raised the Al Kα1 recovery of CR from 0.702 ± 0.290 to 0.959 ± 0.025.

`build_gate_targets(..., bandwidth_mode="exact")` assigns the events to a line by exact energy match, since Geant4 crossing data carry no detector response and every Cu Kα1 event shares one float64 energy. This is a research option and `"resolution"` remains the default. On CR the exact label did not remove the Cu Kα1 overshoot it was built for (2.011 against 2.052), it lowered the Al Kα1 recovery from 0.959 to 0.578, and it deleted Cu Kα2 as a modelled component (see [`../../docs/v0.8.1_line_truth.md`](../../docs/v0.8.1_line_truth.md), Section 14.2). Checkpoints written since v0.8.2 carry `config_version: 2`, and `config_version: 1` checkpoints load unchanged.

## 5. Primary-conditioned models and grouped generation

This section and Section 6 are advanced. They are needed only when the downstream simulation requires every crossing to carry its primary, as in the HENABICS chain, and a single-source user can skip both.

HENABICS needs every generated crossing to carry the identity of the primary that produced it. A model trained on a table with primary truth learns p(crossing | crossing type, primary type, primary energy) and is sampled per primary.

**Data.** The 16-column table is the 13-column lineage layout plus `PrimaryTrackId`, `PrimaryPDG` and `PrimaryEnergy`, and `load_detector_table` detects it automatically. The species are simulated and merged separately, so that `EventId` can repeat across them and the group key is `(PrimaryPDG, EventId)`. The number of primaries thrown per species and energy bin, which includes the primaries that left no crossing, is read from the run summaries with `load_run_summaries`.

**Training.** `build_multiplicity_table` measures P(k | primary type, E_prim bin) from the run summaries, with k = 0 included, together with P(crossing type | primary type, E_prim bin). `build_primary_conditioning` appends `[onehot(primary type), qt(log10 E_prim)]` to the crossing-type one-hot, so that `cond_dim = n_types + n_primary_types + 1`, and builds a group id that makes `split_feature_data` split by primary; no shower then has crossings in both the training and the test set. Models trained without primary truth are unaffected, since `cond_dim` defaults to `n_types`.

**Generation.** `generate_primary_groups` draws the primaries from the GPS macro of the full simulation (`parse_gps_macro` reads its `/gps/hist` spectra), draws k and the crossing types from the multiplicity table, generates the k crossings conditioned on the primary and writes them contiguously under one group key. Primaries with k = 0 write nothing and are counted. The command-line entry point is `scripts/generate_geant_source.py --n-primaries N [--species mu|gamma|e-|<PDG>] [--primary-spectra macro] [--key-offset K]`, while `--n-events` keeps the per-crossing mode of the other models. Crossings that point outward are redrawn with the same conditioning by default (`inward_only=True`), since they cannot enter the detector from the source surface, and the number of outgoing first draws is recorded in the sidecar.

**Output.** The binary format is `GNTBIN` version 2, a fixed-size record per crossing that starts with the int64 group key, the primary track id, the primary PDG and the primary energy, followed by the fields of version 1; `read_detector_binary` reads both versions. A JSON sidecar records `N_primaries_sampled` per species and the histogram of the sampled primary energies, which is the denominator that normalises the MAGI arm of a comparison with the full simulation, in the same way as the number of primaries thrown normalises the full arm.

**Closure of the key.** `tools/henabics_closure_dm12.py` verifies that the key, the primary type and the primary energy survive the second Geant4 run (every post-MAGI row joins to exactly one primary, and the carried energy matches to 10⁻⁵ relative) and compares the per-primary deposit rate and the joint distribution of deposited and primary energy with the full simulation, per species.

**The v1.0.0 recipe.** Four options of `CVAE_MixEnergy_ContPhi_TaskAdaptive` and one generation option repair the defects exposed by the DM1.2 closure. The run used for the v1.0.0 results is `tools/train_dm12_primtruth.py --uncollided-atom --zone-by-primary --energy-loss-flow --loss-variable logit --first-entry --radius 105 --refit-prior`, followed by a weighted refit of the prior.

- *Uncollided atom* (`uncollided_atom=True`). A primary that reaches the surface unscattered crosses it at E = E_prim exactly, which is 53% of the γ crossings on DM1.2. The gate receives one more component pinned at log10 E_prim, allowed only for a crossing of the primary's own species (`add_uncollided_gate_target`), and the quantile transform of E_prim is inverted exactly in float64 (`primary_energy_knots`), since an interpolated inversion misses an atom 10⁻⁴ wide.
- *Zone table by primary* (`zone_primary_energy_edges`, `build_zone_probs_table`, `--zone-by-primary`). The zone is drawn from a table indexed by crossing type, primary type and one of eight E_prim bins. The per-type table averaged the γ primaries (55% uncollided) with the μ⁻ primaries (none) and generated 36% uncollided γ, while the table by primary gives 53.2–53.7% on three seeds against 53.2% real.
- *Energy loss flow* (`energy_loss_flow=True`, `loss_flow_variable="loglog"` or `"logit"`; requires the uncollided atom). The continuum crossings of the own species are modelled by a second flow on the energy lost, so that a sampled energy lies below E_prim by construction. Without it 6.3% of the μ⁻ and 2.0% of the γ own-species crossings exceeded E_prim, and with it none do; the condition-matched AUC of the μ⁻ primaries goes from 0.672 ± 0.024 to 0.628 ± 0.007 (three seeds).
- *Weighted prior refit* (Section 7.5).
- *Latent bank* (Section 7.6).
- *Multiplicity table* (`build_multiplicity_table(..., n_energy_quantiles=65)`). E_prim of the primaries that cross is redrawn from quantiles stored per coarse bin and k class, and the crossing types are tabulated on a 0.1 dex grid. The half-decade table placed the crossings evenly inside a bin where P(k > 0) changes steeply, with four times too many muons crossing below 50 MeV, and this alone accounted for most of the excess albedo of the first reinjection arm (1.50 → 1.087 of the full simulation). `--write-multiplicity-to` writes the table of a run.

## 6. Tracking surfaces that cut other volumes: first-entry crossings

The method of this section was developed and validated on the DM1.2 cryostat only, and no other mass model has been tested, the Athena one included.

MAGI learns the particles crossing a closed sphere around the region of interest, and the Geant4 run that uses a MAGI sample injects them at that sphere and simulates only the inside. The original scheme requires a sphere that lies entirely in empty space, since a sphere placed as a Geant4 volume overlaps whatever it cuts. On a real mass model such a position may not exist. Moreover, the particles collected on a sphere in material are not independent of what lies inside it, because the albedo returning from the interior is recorded as crossings and is produced again when the interior is simulated, so that a model trained on it counts it twice. The first-entry method removes both problems. It was developed and validated on the DM1.2 geometry (`S1GDML_DM1.2_MultiVerse`, branch `trackG-multiverse` of the Geant4 repository, documented in its `MULTIVERSE.md`; the design is in `docs/TrackG_design_review.md`).

**Simulation side.** The sphere is not a Geant4 volume. `SteppingAction` tests the straight segment of every step against the sphere and records the crossing from the outside to the inside, and the transport of the mass world is the standard one (a Geant4 parallel world cannot be used, since in Geant4 10.4.3 it switches the whole run to `G4CoupledTransportation`, which changed the secondaries by about a factor of ten, see `MULTIVERSE.md` Section 6). Every track carries a flag that states whether the track or one of its ancestors has been inside the sphere, set at birth for primaries, updated after every step and handed to the secondaries at creation. The crossing file has a 17th column, `AncestorInside`, which is 0 for a first entry (the whole history of the track lies outside the sphere, so it is independent of the interior) and 1 for a returning particle. The radius and the centre are macro commands, `/MultiVerse/radius` and `/MultiVerse/centre`, and one run writes both samples.

**MAGI side.**

```python
df = magi.load_detector_table(filepath="outputDSCryoSphere.dat", sep=r"\s+")   # 17 columns
df = magi.select_first_entry(df)     # keeps AncestorInside == 0 and prints the fraction removed
```

`--first-entry --radius <mm>` of the primary-conditioned trainers does this, and the metadata records `crossing_rule` (`first_entry` or `all_crossings`) next to the radius and the centre. A model trained on first entries has to be injected into a run that simulates the whole interior with the same outer geometry and the same radius, and `generate_geant_source.py` writes the positions on the sphere of the saved radius (the check is `|r − R| < 10⁻⁴ mm`).

**Choosing the sphere.** `g0_surface_scan` in the MultiVerse folder samples the surface of a candidate sphere and reports, from the geometry alone, which volumes and materials it cuts and which fraction of the area lies in material; with a list of crossing positions (`points` mode) it gives the fraction of the flux. The sphere may cut passive volumes, it must not cut a sensitive one, and it must enclose the sensitive detector. On DM1.2 the radius of 95 mm is the only one between 30 and 300 mm that is entirely free, while 105 mm cuts a copper plate (2.4% of the area, 6.9% of the crossings).

**Validation** (DM1.2, 72 × 10⁶ muons and 90 × 10⁶ photons, radii 95 and 105 mm). The decisive test uses no model: the first entries of a full run are injected into the same geometry, and the returning entries regenerated by the transport of the injected particles are compared with those of the full run.

| Muon primaries | R = 95 mm | R = 105 mm |
|---|---|---|
| Returning entries, reinjection / full | 1.010 ± 0.020 | 0.962 ± 0.015 |
| Energy deposited inside the sphere, reinjection / full | 1.004 ± 0.007 | 0.996 ± 0.007 |
| Triggering primaries of the detector, reinjection / full | 645 / 615 | 596 / 600 |

The first models produced 16–26% too many returning entries for the muon primaries, an excess that was traced to muons entering below 30 MeV, stopping inside the sphere and decaying, and that is due in part to the multiplicity table and in part to a bias of the learned prior at low E_prim. With the v1.0.0 recipe (R = 105 mm):

| Muon primaries, R = 105 mm, MAGI / full | α = 5 prior | latent bank | reinjection |
|---|---|---|---|
| Returning entries | 1.087 ± 0.016 | 0.975 ± 0.015 | 0.962 ± 0.015 |
| Energy deposited inside (keV per primary, full 12.106 ± 0.058) | 12.069 | 12.165 ± 0.058 | |
| Triggering primaries (per 10⁶, full 8.33 ± 0.34) | 8.86 | 9.14 ± 0.36 | |

For the γ primaries the α = 5 prior gives an interior deposit of 0.4286 keV per primary against 0.4193 ± 0.0053 (1.2σ), and the bank gives 0.4451 (3.4σ); the bank is therefore used for the muons only. For the muons the bank is chosen for the returning entries (0.975 ± 0.015 against 1.087 ± 0.016) and for the slow-muon fraction, and not for the trigger rate: its 9.14 triggering primaries per 10⁶ lie 2.4σ above the full 8.33 ± 0.34 (the α = 5 prior gives 8.86, 1.5σ), counting the error of the full run alone, and neither arm is demonstrated to agree at the detector level.

**Limits of the method.**
- The crossing state is taken at the pre-step point, which is the convention of the original recording and is exact for the position. Where the sphere cuts thick dense material, the energy lost between the pre-step point and the surface is neglected, and this has not been measured.
- A source lying inside the sphere, such as the activity of a passive layer that is partly inside, is flagged at birth and is not part of the first-entry sample. That part has to be simulated directly in the application run, and this case has not been tested.
- The returning entries are 1–3% of the crossings in the cases tested, and the detector-level closure (a few hundred triggering primaries, ±5%) cannot resolve an effect of that size. The test of the rule is thus the count of returning entries and the energy deposited inside.
- The method was validated on the DM1.2 cryostat only, and the Athena mass model, the case that motivated it, has not been tested.

## 7. Strategies to improve a trained model

The architecture of v1.0.0 is fixed, and the measures below act on the data transforms, on the energy warp, on the prior and on the generation. They are ordered by how early they enter the recipe. Each one states what it does, when it helps, and what it costs; none of them is applied silently, and each is reported in the metadata of the run.

| Strategy | Acts on | Use it when | Measured effect | Cost |
|---|---|---|---|---|
| 7.1 Exclude lines from the CDF warp | energy flow | a pinned line is over-generated | CR 511 keV: 4.84% → 3.40% of the γ within 50 eV (real 3.45%) | none |
| 7.2 Boost ranges of the warp | energy flow | the energy marginal misses a decade of the spectrum | Th warp2: energy per-bin RMS 5.20 → 4.83 | a tail above the real maximum must be capped |
| 7.3 Quantile transform fitted on all rows | geometry heads | geometry marginals show ripples at fixed positions | Th232 u_r 3.5 → 1.6, μ φ_r 3.7 → 1.3 (RMS per bin) | none at fit time; a refit of a shipped run changes its transformers |
| 7.4 Prior refit on the frozen encoder | prior | always, as the last stage | CR AUC 0.649 → 0.624 | about 35 minutes of laptop CPU |
| 7.5 Weighted prior refit | prior | the prior misses a region of the conditioning | DM1.2 μ⁻ slow-muon bias reduced by about two thirds | one more refit |
| 7.6 Latent bank | generation | the refit is not enough and the bias is declared | DM1.2 slow muons 0.098 → 0.058 (real 0.060) | replays training posteriors, 32 MB |
| 7.7 Line sampling | generation | a line has to be exactly monoenergetic | no change of the CR score | none |

### 7.1 Exclude the pinned lines from the CDF warp

Geant4 writes a line photon at the exact catalogued energy, so that every pinned line is an atom in the empirical energy CDF. `fit_cdf_warp_knots(energy_y)` then stacks several quantile knots on the atom, and the warp alone maps the probability mass of the line into a sliver a few eV wide: the continuum flow emits a spike at the line before it has learned anything, and an untrained flow already does so. On CryoSphere-CR this made the 511 keV line 1.4 times too strong (4.84% of the generated γ within 50 eV against 3.45% real), with the excess carrying continuum-like geometry, and it was the dominant coverage hole of the classifier. Neither the routed loss nor a warm fine-tune removes the spike, and a refit of the warp does.

Always pass the line positions with `fit_cdf_warp_knots(energy_y, ..., exclude_y=line_positions_y)`; the function warns when it finds an atom. New runs also set `energy_line_routing="supervised"`, so that each event trains only its own component (`argmax(gate_target)`) and the flow never sees a line photon. An existing checkpoint keeps its stored knots, and `tools/refit_energy_flow_posthoc.py` repairs one post hoc (refit warp, flow warm-started, everything upstream frozen), which brought the CR 511 fraction to 3.40% against 3.45% real. When a line looks over-produced, the first check is whether the stored knots repeat a value of y. The reference trainers `tools/train_cr_v1.py` and `tools/train_iaps.py` pass `exclude_y`, so that a run they produce carries the fix, while a run trained before the fix keeps the knots it was saved with until it is repaired; the run directory does not state which case it is, and this check is therefore the way to know.

### 7.2 Boost ranges and the line criterion

The warp of the energy flow is fitted on quantiles of the training energies, and a spectrum that spans several decades can leave a range under-resolved. `tools/train_iaps.py --warp-eps` sets the smallest probability step of the knots, and `--warp-boost LO_MEV,HI_MEV,N` (repeatable) adds N knots inside a range. The SRON Th and Ra models of v1.0.0 use an epsilon of 10⁻⁶, boosts of 0.001–0.07 and 0.07–0.12 MeV with 64 knots each, and a generation cap at the real maximum of the energy (3.96112 MeV for Th, 3.18362 MeV for Ra). Two choices of the first attempt proved harmful: a boost between 1.9 and 3.3 MeV placed crossings at 2.7 MeV (2483 per 10⁶ for Th, against 21 real), and an epsilon of 10⁻⁴ hid the 1–40 keV tail. The line criterion is the fraction of the crossings (0.1%) and not the prominence of the peak, since the latter lost lines (Ra from 9 to 4, Th from 9 to 5); all 26 Ra lines are kept with the fraction criterion.

The cap has to be chosen per source. The energy cap measured on SRON Th (3.96 MeV) applied to every IAPS source produced an apparent per-bin RMS of 17.6 for μ and 8.6 for γ, and the same models without it give 4.3 and 4.9, so the generation cap and the number of events (400 000 per subset) have to be stated whenever an RMS is quoted.

### 7.3 Fit the quantile transform on all rows

`_fit_quantile_column` previously left the `subsample` argument of the scikit-learn `QuantileTransformer` at its default of 10 000, so that the 10 000 quantiles of every model were computed from 10 000 rows, about one row per quantile. The transformed real data then lay 3 to 4 RMS away from N(0, 1), while a fit on all rows gives 0.9 to 1.0, and the geometry marginals showed ripples at fixed positions of u_r that had been attributed to the prior. The fit now uses `subsample=len(values)`, and v1.0.0 models trained from the release onward carry the correction.

The models trained before the correction can be repaired without retraining by swapping the transformers (`tools/refit_geometry_qt.py <run>` copies a run to `<run>_qtfull` and refits the geometry transforms on all training rows). The swap lowers the per-bin RMS of Th232 u_r from 3.5 to 1.6 and of μ φ_r from 3.7 to 1.3, and it has no effect on Ra226, K-40, γ and CR (differences below 0.1). The swap is not neutral for every variable: the u_r deviation of μ rises from 4.7 to 5.4, and the 5x detector closures of the swapped Th232 and μ models have not been repeated, so a shipped run is never replaced without that check. The energy is not affected, since it is transformed with log₁₀ and not with a quantile map.

### 7.4 Refit the prior on the frozen encoder

The learned prior of the v0.8 models is trained jointly with the encoder and the decoder, and it does not reach the aggregated posterior it is meant to match. This mismatch is the dominant part of the classifier AUC excess of the generated samples: on CryoSphere-CR, decoding the latents of held-out real events gives an AUC of 0.528 while decoding latents drawn from the prior gives 0.594 with the wide conditioner, and a classifier in latent space can barely separate the two (0.515), so that the decoder amplifies a small latent error into a visible error of the physical variables. A refit of the prior on the frozen encoder lowers the AUC of a retrained model from 0.649 to 0.624 (mean of three training seeds, standard deviation 0.004 between seeds), and it is therefore the last stage of the recipe, after `fit_model` and `save_final_trained_model`:

```python
result = magi.refit_prior(
    model, X_cont_train, cond_train, X_cont_val, cond_val,
    checkpoint_dir="trained_models/my_run/refit",   # per-epoch checkpoint, resumable
)
magi.save_refit_model(
    model, result.prior, src_dir="trained_models/my_run", out_dir="trained_models/my_run_final",
    model_name="mix_CR", n_types=n_types, radius=R, hidden=result.hidden,
    n_layers=result.n_layers, val_kl_before=result.val_kl_before,
    val_kl_after=result.val_kl_after, history=result.history)
```

`refit_prior_run` performs both steps for a saved run. The defaults are the measured recipe, a conditioner of 128 × 128 and six coupling layers, a fresh initialisation and 60 epochs at 10⁻³ with cosine decay, and the weights of the epoch with the lowest validation KL are kept. The written run directory is identical to the source except for the prior, and its metadata records the refit with the KL before and after. `prior_posterior_auc` gives the diagnostic (0.5 means that the prior reproduces the aggregated posterior), which was 0.51–0.52 after the refit on three seeds. A refit takes about 35 minutes on a laptop CPU for CR and continues after an interruption when the same call is repeated. `tools/train_cr_v1.py --refit-prior` runs the whole recipe for CryoSphere-CR, and `tools/train_dm12_primtruth.py --refit-prior` does so for the primary-conditioned models, where the refit brings the μ⁻ AUC of the uncollided-atom model from 0.734 back to 0.695.

### 7.5 Weighted refit

`sample_weight_train` and `sample_weight_val` weight the events of the refit, to pull the prior toward a region of the conditioning where it misses the aggregated posterior. On DM1.2 the learned prior over-produces slow muons for the μ⁻ primaries below 100 MeV, which then stop and decay inside the sphere, and weighting the events of those primaries by α = 5 removes about two thirds of the bias (`tools/dm12_weighted_refit.py <run> <out> <alpha> [E_cut]`). The value α = 5 is the reference because it keeps the detector level, and α = 20 is a sensitivity test.

### 7.6 Latent bank

`magi.build_latent_bank` (or `tools/build_latent_bank.py <run> <split npz> [K]`) stores the posterior means and variances of the training split with the model, and `generate_geant_source.py --latent-bank <npz>` then draws z from the posterior q(z|x′) of a training event x′ with the same crossing type, the same primary type and one of the K = 50 nearest values of qt(E_prim), instead of from the prior. The heads still draw their own noise, so no training crossing is reproduced. The bank is, however, a replay of training posteriors: it gives up the interpolation of p(z | cond) between training conditionings, it occupies 32 MB on DM1.2 (522 642 events), and rows whose conditioning group is absent fall back to the prior. It is available in group mode only. On DM1.2 it brings the fraction of slow muons (E < 10 MeV, E_prim < 100 MeV) from 0.098 to 0.058 against 0.060 real, while for the γ primaries it hardens the crossings (interior deposit +6%, 3.4σ) and the α = 5 prior is used instead. The bank masks a failing prior and does not repair it, and a result obtained with it has to be declared as such.

### 7.7 Line sampling and the inward condition

`line_sampling="delta"` (`--line-sampling delta` in `scripts/generate_geant_source.py`) emits the line events at exactly the line energy. It did not improve the classifier score on CR, and the default remains `"gaussian"`.

The direction head models the global variables `u_v` and `phi_v`, and whether a crossing points inward therefore depends on its position on the sphere, which the head does not enforce. After the prior refit 1.4% of the generated CR crossings point outward, against 13–15% on DM1.2 before the redraw. `inward_only=True` in `generate_primary_groups` redraws them with the conditioning fixed. The alternative is `geometry_transform="quantile_u_r_mu_phi_r_psi"` (`--local-direction`), which models the direction in the local frame of the crossing point with mu = −r̂·v, the cosine with the inward normal, and psi, the azimuth around it, so that every generated crossing is inward by construction. On DM1.2 this option costs realism outside the seed spread (AUC μ⁻ 0.656 against 0.629–0.634, γ 0.688 against 0.678–0.683), because the coupling of the energy to the zenith is lost (A3 in `docs/CVAE_vs_AutoRegFlow.md`), and v1.0.0 keeps the global frame with the redraw.

## 8. Validating a model

The unit tests check the machinery only, and whether a trained model reproduces the source is settled by the protocol of this section.

### 8.1 Classifier two-sample test

The figure of merit used to compare models is the AUC of a gradient-boosting classifier that separates real from generated crossings (`tools/score_prior_refit.py`; features log₁₀E, the geometry variables and the particle type). An AUC of 0.5 means that the classifier cannot tell the two samples apart, and the value obtained for two disjoint random subsets of the real data is the floor given by sampling noise. The Wasserstein distances and correlation residuals of the acceptance harness remain useful to locate a defect, whereas the AUC is the single number that responds to any difference of the joint distribution. Three points of protocol decide whether a number can be quoted.

- **Random real subsets and an interval.** `tools/auc_intervals.py` draws, for each replication, an independent generation seed, an independent random subset of the real crossings and an independent classifier split, and reports the mean and standard deviation, the paired difference between models and a real-versus-real null. The first N rows of a raw file are not a random sample of it: on CryoSphere-CR they contain a single shower of 1364 crossings at ten times its share of the file. With ten replications on 400 000 crossings the null is 0.5004 ± 0.0012, and two single-run AUCs are considered different only when they differ by more than about 0.008 (3σ of a difference).
- **Folds grouped by shower** whenever the data have showers. The crossings of one primary are correlated, and with row-level folds the classifier recognises individual showers: on DM1.2 two real samples give 0.54 for μ⁻ against 0.50 with folds grouped by primary (`tools/dm12_condmatched_auc.py`).
- **Matched conditioning** for the primary-conditioned task, generating one crossing for every real test crossing with the same crossing type, primary type and primary energy.

The intervals contain the sampling and classifier variance only. The variance between independent trainings is measured by retraining at different seeds (`tools/train_cr_v1.py --seed`), and an 80-epoch retraining on three seeds lowers the standard deviation of the AUC between seeds from 0.0042 to 0.0020 without changing the per-bin RMS.

| CryoSphere-CR model | AUC (10 replications) | inward crossings only |
|---|---|---|
| v0.8.2, original | 0.6613 ± 0.0018 | 0.6376 ± 0.0043 |
| + wide prior refit and line-free warp (v0.9.4) | 0.6191 ± 0.0022 | 0.6156 ± 0.0025 |
| retrained from scratch with the fixes built in (seed 42) | 0.6490 ± 0.0015 | 0.6279 ± 0.0032 |
| the same with the prior refit on the frozen encoder, seed 42 | 0.6270 ± 0.0025 | 0.6180 ± 0.0025 |
| the same, seed 7 | 0.6196 ± 0.0026 | 0.6145 ± 0.0025 |
| the same, seed 13 | 0.6267 ± 0.0031 | 0.6159 ± 0.0025 |
| real versus real | 0.5004 ± 0.0012 | 0.5010 ± 0.0007 |

### 8.2 Per-bin deviation of the marginals

`magi.marginal_bin_rms(real, gen)` gives the Poisson-weighted RMS deviation of a generated marginal from the real one over 50 equal-count bins of the real sample; a value of 1 means agreement within counting noise, and the real-versus-real null is 1.00 ± 0.03. Exact atoms have to be left out of the continuum marginal through `exclude_windows`, because a bin edge on a line splits its slightly smeared generated copy and produces a spurious deviation of tens of sigma, and the lines are scored by the line recovery instead. `tools/marginal_rms_intervals.py` evaluates it with the replications of the AUC protocol. On CryoSphere-CR (±0.01 in log₁₀E removed around each pinned line):

| CryoSphere-CR model | log₁₀E | u_r | u_v | φ_r, φ_v (cos, sin) | mean of the seven |
|---|---|---|---|---|---|
| v0.8.2, original | 4.75 ± 0.08 | 4.93 | 4.87 | 4.25–5.40 | 4.90 ± 0.06 |
| + wide prior refit and line-free warp | 3.43 ± 0.10 | 4.33 | 4.67 | 4.08–4.66 | 4.30 ± 0.06 |
| retrained with the fixes and the prior refit (3 seeds) | 2.33–3.09 | 4.46–4.48 | 4.48–4.90 | 4.13–4.77 | 4.23–4.28 |
| real versus real | 1.01 | 1.05 | 0.97 | 0.97–1.01 | 1.00 ± 0.03 |

The energy marginal improves from 4.75 to 2.3–3.1 with the fixes, while the geometry marginals stay at 4.1–4.9 on every model, and the deviation that remains therefore lies in the geometry and is the same for the original, the repaired and the retrained models. On the IAPS Triple sources, generated without the SRON cap, the mean per-bin RMS is 4.2 for K-40, 4.3 for μ, 4.9 for γ, 5.3 for Ra and 5.5 for Th, at the level of CR and SRON, with a null of 0.9 to 1.1. For μ the residual comes from the prior (decoding the posterior gives 1 to 1.5, the prior 2.9 to 3.8) and, to a smaller extent, from the quantile transform of Section 7.3.

### 8.3 Detector-level closure

The generated crossings are injected into the downstream Geant4 run, and the rate of events per detector is compared with the run driven by the real crossings. Two caveats have to be quoted with every number of this subsection. The real arm of the SRON CR closure re-uses crossings that belong to the training sample, so that the test is not independent of the training data, and the v1.0.0 fixes are bundled in one retraining, so that the closure does not attribute the change to any single one of them.

On SRON CR (84 chunks, 6.0 × 10⁶ MAGI against 6.0 × 10⁶ real crossings) the ratio is R = 1.006 ± 0.028 over all energies, 1.073 ± 0.082 for 0.1–30 keV, 0.979 ± 0.082 for 1–20 keV and 1.039 ± 0.117 for 1–7 MIP; the 8–15% deficit that the beta had measured (R = 0.846 ± 0.088) is not reproduced. The closure spectrum between 0.1 keV and 1 MeV agrees with the full simulation (χ² = 28.8 for 23 bins, Kolmogorov–Smirnov D = 0.039, p = 0.22).

On IAPS Triple, injecting five times the statistics of each source gives R = 0.97 ± 0.02 pooled over the five sources, computed as the equal-weight sum of the per-primary rates, since a sum of the counts over the primaries gives a wrong 0.18. The Th and Ra closures at the two laboratories are the weakest numbers of the validation and disagree in sign:

| Source | Detector set-up | Model | R | Reading |
|---|---|---|---|---|
| Th | IAPS Triple, 5x | IAPS model, SRON cap not applied | 0.88 | 2.4σ low |
| Ra | IAPS Triple, 5x | IAPS model, SRON cap not applied | 0.86 | 2.8σ low |
| K-40, γ, μ | IAPS Triple, 5x | IAPS models | 0.97 each | consistent with 1 |
| Th | SRON, warp2 model | boosted CDF warp, capped at the real maximum | 1.40 ± 0.22 | high |
| Ra | SRON, warp model | boosted CDF warp, capped at the real maximum | 1.31 ± 0.19 | high |

The two pairs differ in the detector, in the mass model and in the energy model (the SRON warp models carry the tail defects of Section 9), so that they are not repeated measurements of one quantity, and no common cause of the opposite signs has been established. Until it is, a detector-level rate obtained from a Th or Ra source is to be quoted with the deviation of the table, which is 12–14% low at IAPS and 31–40% high at SRON.

## 9. Accuracy and known limits

The numbers of this section are the measured state of v1.0.0, and the acceptance harness of the v0.8.2 beta that preceded the prior refit and the warp fix has been superseded by Section 8. Each limit is stated with its measured size. A cause is given where it has been established, and the limits whose cause is open are marked as such. The open items listed at the end move to the 1.0.1 patch.

- **Line positions are reliable, line intensities are source-dependent.** The positions are pinned physical inputs audited to ≤ 0.5 eV against the measured spectrum on all sources (`tools/line_centroid_audit.py`). The intensity of a line is learned, and it is the quantity that is least stable: the beta measured recoveries between 0.9 and 3.3 times the real value for the weak lines. The 511 keV line of CR, after the warp fix, is 3.40% of the generated γ within 50 eV against 3.45% real, and the 26 lines of Ra are all modelled. A generated line flux is not to be used as an estimate of a fluorescence or annihilation line without measuring it on the source with `tools/acceptance_v0_8.py`, and a line below 5σ significance is correctly left unmodelled, so that no recovery is quoted for it.
- **The prior does not reproduce the aggregated posterior closely enough.** Approximately 80% of the excess AUC of the original model came from this source, and the refit of Section 7.4 removes most of it. The remaining AUC of 0.62–0.63 on CR is due to the prior together with the per-variable heads.
- **Exact discrete atoms need a dedicated component.** An atom that is not listed is spread by the continuum density over the neighbouring energies. The unscattered primary at E = E_prim on DM1.2 (53% of the γ crossings) came out at 2.4% before it received its own component, and with the atom and the zone table by primary it is 53.2–53.7%.
- **Tail defects of the SRON Th and Ra models.** The Th warp2 model generates 722 crossings per 10⁶ at 2.7 MeV against 21 real (0.07%), and the Ra tail between 2.6 and 4.0 MeV is 3.7 times too high, while the ratio between 20 and 40 keV is 0.43 for Th. The closures of the warp models give R = 1.40 ± 0.22 for Th and 1.31 ± 0.19 for Ra at SRON, while the IAPS models give 0.88 and 0.86 (Section 8.3), and both are declared.
- **Grazing entries.** About 5% of the u_r of Th and Ra points outward, and the grazing entries on DM1.2 (cosine with the inward normal below 0.1) are about three times too many. This is not an effect of the width of the head (it was scanned) and it is declared. The Th232 φ_v deviation of 5–6 in every arm has an open cause, either the head or the transform, and no experiment has separated them.
- **Crossings are independent given the conditioning.** The correlations inside one shower are not represented: a single shower of 1364 crossings, of which 492 pass through a spot of about 1 mm with a common direction, appears in the CR file and MAGI does not reproduce it, and on DM1.2 the energies of the k crossings of a primary can sum to more than E_prim. The energy loss flow keeps the primary's own crossing below E_prim by construction, but nothing in the model enforces the budget of the secondaries. The autoregressive follow-up removes this limit.
- **The energy head misplaces the muons between 10 and 100 MeV** for low E_prim on DM1.2 (too few at 10–30 MeV, too many at 60–100 MeV), in the posterior arm as well, so that neither the refit nor the bank removes it; the cause is open and lies in the head or in the transform of the energy. The condition-matched AUC is 0.68 for the γ primaries and 0.63 for the μ⁻ primaries, against a null of 0.50.
- **Open items of the 1.0.1 patch.** The Th232 and μ `_qtfull` 5x closures, the quantile swap of the SRON models (which needs the split dumps), the K-40 delta and the seeds 8 and 9 of the validation. The sources in passive material and the Athena mass model (Section 6) are not covered.

## 10. Inspecting a trained model

Two functions in `magi.utils` write a self-contained, interactive HTML file for looking inside a trained mixture run, and no plotting library or running kernel is needed to open it.

- **`magi.save_routing_circuit(save_dir, model_name)`** draws the conditional routing of the gate per type, coloured by the `zone_probs` of the checkpoint. It reads the saved config and metadata only and runs in under a second, and it requires `prior_zone_conditioning=True` at training time (it raises `KeyError` otherwise).
- **`magi.save_full_circuit(save_dir, model_name, training_data_filepath, candidate_lines_file, center, radius)`** shows how heavily every unit of the network is used, per particle type, across many real held-out events. Each unit of every stage (encoder, latent `z`, decoder stem, deep trunk and the three heads) is coloured by its usage, and wires join the 10 most used units of each stage to the next, taking the colour of their weaker end. One button selects a type, and an "All types" button pools the events of every type. This function needs real held-out events, so it re-derives the preprocessing of the training (line matching, gate targets, quantile and log₁₀ transforms, train/test split) from the raw detector table and then runs the attribution, which takes several minutes per source (4–8 minutes for the 3.4 × 10⁶ events of CR).

```python
magi.save_routing_circuit(save_dir="trained_models/v0_8_2_priorzone_CR", model_name="mix_CR")

magi.save_full_circuit(
    save_dir="trained_models/v0_8_2_priorzone_CR", model_name="mix_CR",
    training_data_filepath="TrainingData/alloutputDSCryoSphereCR.dat",
    candidate_lines_file="CandidateLines/CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json",
    center=(0.0, 0.0, -507.66), radius=100.0)
```

Both functions derive the layout from the architecture of the loaded checkpoint (number of types, gate zones, encoder and branch depth), so the same call works unchanged on any source. The light and dark themes are switched by a button in the page.

### 10.1 What "usage" means

`save_full_circuit` measures the Expected Conductance, the generalisation of Integrated Gradients from the input features to the internal units ([Dhamdhere et al. 2019](https://arxiv.org/abs/1805.12233)), averaged over baselines drawn from the real data instead of a single zero vector ([Erion et al. 2021](https://arxiv.org/abs/1906.10670)). A zero baseline would be meaningless, since the inputs are quantile-transformed and "0" is a specific quantile and not the absence of a signal. Only the continuous feature block is interpolated along the attribution path, while the one-hot type conditioning is held fixed, which makes "usage when the network processes type *t*" a well-defined quantity. The parameters are `n_events_per_type` (default 128), `n_baselines` (16) and `n_steps` (32).

**The node colours are percentile ranks and not magnitudes.** Usage is heavy-tailed, so that a linear or logarithmic scale leaves most units in the dark lower third of the colormap. Ranking spreads the units evenly over the colour range, pooled over all types so that the buttons stay comparable, and the consequence is that some units always look dark whatever the true distribution is: a dark unit is low-ranked and not necessarily unused, and the colours are not evidence of spare capacity. That question is answered by the per-layer absolute usage panel below the diagram, which reports the unnormalised magnitudes: the `share` of the total conductance of the network held by the layer, its `load` (the share divided by its share of the units, where 1.0× means that the layer carries its weight and 0.2× means that it holds five times more units than its contribution justifies), the `top-20%` concentration, and the `faint` fraction of units below 1% of the busiest unit of the layer. A row is flagged when the load is below 0.5× and a real population of faint units exists.

A flag marks the place where an experiment is needed and is not a verdict. The attribution states that the gradient of the loss does not flow much through a unit at the optimum, which is not the same as stating that the network computes the same function without it, and it is blind to redundancy in both directions: two duplicated units each look moderately used, whereas a unit dormant on average may carry the rare line events. Only shrinking the layer, retraining and re-scoring with `tools/acceptance_v0_8.py` settles the question.

The follow-up architecture has its own version of this figure (`MAGIv2/scripts/make_circuit.py`), which draws the network usage of the eight stages of the autoregressive factorisation and the variables that link them.

## 11. Adapting MAGI to a new detector or source

**The geometry is limited to a sphere.** The `center` and `radius` parameters of `build_physical_features` and the coordinate transforms of `core/geometry.py` (`xy_from_ur_phi`, `vxyz_from_uv_phi` and others, all parametrised by `sphere_R`) assume a sphere centred at a fixed point. A planar, cylindrical or box-shaped surface is not obtained by passing different parameters: it requires reworking the geometry-transform layer, both the physical features of `preprocessing.py` and the reconstruction of `generation/reconstruction.py`, which have to stay in sync.

**`geometry_transform` is a contract between training and generation.** The string (`"arctanh_uv_discrete"`, `"quantile_u_r_u_v"`, `"quantile_u_r_u_v_phi_r_phi_v"`, `"quantile_u_r_mu_phi_r_psi"`) and the resulting ordering of `X_cont_cols` are shared by the preprocessing, the input layout of the model and the reconstruction, and they are stored in the metadata. They have to match exactly between the run that trained a model and the run that generates from it, since no check catches a mismatch early and the result is a physically wrong output.

**Rare particle types are dropped silently.** `prob_threshold` of `filter_particle_types_continuous_geometry` removes the types below a given population fraction, and this is the first place to look when a rare but relevant type is missing from the output.

**The energy-binning mode affects the resolution of the lines.** `build_energy_bins` supports `"fixed_width"`, `"min_counts"` and `"log_fixed_count"`, and the choice trades the ability of a categorical head to resolve narrow lines against the statistics of every bin. With the mixture head the binning no longer limits where a line can be placed, but it still limits which lines `detect_energy_lines` can find: on a logarithmic grid spanning many decades a bin can be hundreds of eV wide, close doublets fall in one bin, and only one member is matched. Pass `refine_bin_width_mev` to refine each detected peak to sub-bin precision, and check the result with `tools/line_centroid_audit.py`.

**Quantile transforms need enough samples.** The `QuantileTransformer` geometry modes (`u_r_q`, `u_v_q`, `phi_r_q`, `phi_v_q`) are fitted per feature on the training data, and a small dataset gives unreliable tails that show up as implausible extreme values in the generated output (see also Section 7.3).

**The flux normalisation depends on `source_type`.** `build_physical_features` accepts `source_type="cosmic_ray"` (default), for which a crossing is primary when `PrimBool == 1`, that is when the raw particle thrown by the GPS reached the surface unmodified, and `source_type="radioactive"`, for which a crossing is primary when `CreatorProcessName == "RadioactiveDecay"`. The second mode is needed because the literal primary of a decaying source embedded in bulk material is the decaying nucleus, which never propagates, so that `PrimBool` is always 0. Radioactive sources require the 13-column lineage schema. A wrong choice does not raise an error and silently produces a wrong normalisation factor; `compute_primary_fraction`, `save_normalization_summary` and `load_normalization_summary` measure and store it.

**Unit tests.** `MAGI_package/tests/` holds 238 tests (`python -m pytest MAGI_package/tests/`; `pytest` is in the `dev` extra and in `requirements.txt`) that cover the flow round-trips, the matching of the configuration at checkpoint save and load, the construction of the gate targets, the zone conditioning of the prior, the line-atom warp and routing, the loading of primary truth and the multiplicity tables, the Geant4 binary export, the resumable training, the two visualisation tools and the coverage of the public docstrings. They catch mechanical regressions and do not tell whether a trained model reproduces the source. Any change of the geometry-transform layer has to be validated by hand against real data before the generated output is trusted.

## 12. Resuming an interrupted or early-stopped run

A fit can end before it is finished, because the process is killed, a Colab session expires, or `EarlyStopping` fires a few epochs too soon. Passing `checkpoint_dir` to `fit_model` (or `train_single_run`) writes a resumable checkpoint after every epoch, and `resume=True` continues from it:

```python
history = magi.fit_model(
    model, train_ds, val_ds, epochs=40,
    callbacks=magi.build_default_callbacks(),
    checkpoint_dir="trained_models/my_run/resume",
    resume=True,
)
```

Running the same call again after an interruption picks up at the last completed epoch, and if `checkpoint_dir` holds nothing yet, training starts from scratch, so that one command serves for the first run and for every restart. `epochs` is the total, as in Keras, and the returned history covers all epochs, including those before the interruption. The model has to be compiled with the same optimizer and `clipnorm` as the interrupted run, and a mismatch raises an error instead of loading the wrong optimizer state.

- **Restored state:** the weights, the Adam moments and step count, the learning rate, the state of `EarlyStopping` (best value, patience counter, best weights), `ReduceLROnPlateau` and `TaskAdaptiveLossScheduler`, the decayed `model.task_weights` and the metric history. With an unshuffled dataset a resumed run reproduces an uninterrupted one weight for weight (`tests/test_resume_training.py`), while with shuffled data it is equivalent statistically, since the order of the batches is not restored.
- **Early stopping ended too soon:** raise `epochs`, optionally raise `patience` of the `EarlyStopping` instance, and pass `reset_early_stopping=True`. The patience counter is cleared and the best value found so far is kept, so the run continues from the best weights.
- **Kill during the write:** the checkpoint is written to a temporary directory and swapped in, so that an interruption leaves the checkpoint of the previous epoch intact. `magi.load_resume_state(dir)` reads how far a run got without loading a model.

The resume checkpoint is a training aid and not a model artifact: generation still needs the files of `save_final_trained_model`, which are written when the fit ends. Jobs longer than the limit of a tool or session are run detached (`python tools/run_detached.py <log> <cmd...>`) together with the per-epoch checkpoint.

## 13. Data and artifact conventions

- The raw Geant4 output is concatenated and cleaned into whitespace-delimited `.dat` files (a `find | xargs cat` concatenation followed by an `awk` cleaning pass; the recipe file lives in the ignored `TrainingData/` folder and is not part of the repository). Large raw data and generated artifacts are ignored by git, and only code, notebooks and small JSON metadata are versioned.
- A trained run lives under `trained_models/<run_name>/` or `checkpoints/<run_name>/` and contains `*.weights.h5`, `*_config.json`, `*_history.json`, `*_metadata.json`, `*_task_weights.json`, `*_summary.txt` and, for quantile-geometry models, `*_quantile_transformers.joblib`. These files are copied and loaded together.
- Analysis and validation that produce plots are written as notebooks, one per topic, while training, parallel jobs, monitors and the package itself stay in Python files.
