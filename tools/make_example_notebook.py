"""Generate Example_Usage.ipynb — the v0.8.2 walkthrough.

Kept as a script so the notebook can be regenerated rather than hand-edited
cell by cell. Run from the repo root, then execute the notebook to verify.
"""
import json, pathlib

C = []


def md(text):
    C.append({"cell_type": "markdown", "metadata": {},
              "source": text.strip("\n").split("\n")})


def code(text):
    C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": text.strip("\n").split("\n")})


# Cell sources are written with a trailing newline per line, matching nbformat.
def _fix():
    for c in C:
        c["source"] = [l + "\n" for l in c["source"][:-1]] + [c["source"][-1]]


md(r"""
# MAGI — Example Usage (v0.8.2)

A runnable, fully-commented walkthrough of the whole MAGI pipeline on **synthetic
data**: load → preprocess → detect spectral lines → build dataset → train →
checkpoint → reload → generate → validate → export a Geant4 source file.

It uses the **v0.8 mixture-energy head**
(`CVAE_MixEnergy_ContPhi_TaskAdaptive`), which is what the v0.8.2 beta is. The
synthetic source deliberately contains two narrow spectral lines on top of a
power-law continuum, so the line machinery is actually exercised rather than
just mentioned.

Everything here is small and fast — a few thousand events and a handful of
epochs — so it runs end to end in a couple of minutes on a CPU. The numbers it
produces are therefore **not** meaningful physics; the point is that every step
runs and connects to the next.

> **This release is a beta.** MAGI v0.8.2 reproduces the joint phase-space
> distribution (energy–position–direction correlations) and the energy continuum
> to within its acceptance bars. It does **not** yet reproduce spectral-line
> *intensities* — measured recovery runs 0.7×–4.9× and is unstable across seeds.
> Line *positions* are reliable. See `docs/manual/magi_manual.pdf` §7 before
> quoting any number from a real run.
""")

code(r"""
# CPU-only must be set BEFORE magi is imported: importing magi pulls in
# tensorflow_probability, which initializes the device, after which
# set_visible_devices no longer takes effect.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import magi

magi.initialize_environment(seed=42, cpu_only=True, quiet=True)
magi.print_tf_info()
""")

md(r"""
## 1. A synthetic source

MAGI reads whitespace-delimited Geant4 crossing files with
`magi.load_detector_table`. Rather than ship a data file, we build an
equivalent DataFrame in memory.

The source below is a sphere of radius 20 mm. Each event gets:

- a **particle type** (`gamma`, `e-`, `mu-`),
- an **energy** drawn from a power-law continuum **plus two narrow lines**,
- a **position** on the sphere surface,
- a **direction**, deliberately **correlated with energy** — high-energy events
  travel preferentially in −z. That correlation is the thing the conditional
  prior exists to reproduce, and we measure it at the end.

MAGI parametrizes both position and direction by a **global-frame** polar
cosine plus an azimuth: `u_r`/`phi_r` for the crossing point, `u_v`/`phi_v` for
the direction, where `u_v` is simply the z-component of the unit direction
vector. Building the source directly in those variables (rather than in a local
surface frame) keeps the injected correlation intact through preprocessing.

The two lines are placed at 8.0 keV and 511 keV and are **exactly
monoenergetic**, matching how Geant4 emits fluorescence and annihilation lines
in raw crossing data: no detector response has been applied at this stage.
""")

code(r"""
rng = np.random.default_rng(42)

N_CONT = 6000            # continuum events
N_LINE_A, N_LINE_B = 700, 500
E_LINE_A = 0.00800571    # MeV — a Cu K-alpha1-like fluorescence line
E_LINE_B = 0.510999      # MeV — e+e- annihilation

center = (0.0, 0.0, -50.0)   # mm
R = 20.0                     # mm

# --- energies: power-law continuum spanning ~4 decades, plus two exact lines
u = rng.uniform(0.0, 1.0, N_CONT)
E_cont = 1e-3 * (1.0 / (1.0 - 0.999 * u)) ** 0.6          # MeV, 1e-3 .. ~1e1
E_cont = np.clip(E_cont, 1e-3, 20.0)

E = np.concatenate([
    E_cont,
    np.full(N_LINE_A, E_LINE_A),      # exactly monoenergetic - see the note above
    np.full(N_LINE_B, E_LINE_B),
])
N = E.size
order = rng.permutation(N)
E = E[order]

# --- particle type: energy-dependent, so the type conditioning has real work
p_gamma = np.clip(0.7 - 0.15 * np.log10(E / 1e-3), 0.05, 0.9)
which = rng.uniform(size=N)
ParticleName = np.where(which < p_gamma, "gamma",
                np.where(which < p_gamma + 0.25, "e-", "mu-"))

# --- position: uniform on the sphere surface.
# MAGI parametrizes position by u_r = z-component of the unit radius vector
# (i.e. cos of the polar angle) and phi_r = its azimuth. Both are GLOBAL-frame,
# not a local surface frame - see build_physical_features.
u_r = rng.uniform(-1.0, 1.0, N)
phi_r = rng.uniform(0.0, 2 * np.pi, N)
sin_r = np.sqrt(1.0 - u_r ** 2)
X = center[0] + R * sin_r * np.cos(phi_r)
Y = center[1] + R * sin_r * np.sin(phi_r)
Z = center[2] + R * u_r

# --- direction: CORRELATED WITH ENERGY, which is the point of this example.
# Direction is parametrized the same way: u_v = z-component of the unit
# direction vector, phi_v its azimuth. We make u_v depend on energy, so
# high-energy events travel preferentially in -z.
forward = np.clip(0.15 + 0.30 * np.log10(E / 1e-3), 0.0, 1.0)
u_v = np.clip(rng.normal(loc=-forward, scale=0.35, size=N), -0.999, 0.999)
phi_v = rng.uniform(0.0, 2 * np.pi, N)
sin_v = np.sqrt(1.0 - u_v ** 2)

Vx = sin_v * np.cos(phi_v)
Vy = sin_v * np.sin(phi_v)
Vz = u_v                      # by construction, this is MAGI's u_v

print("injected corr(log10 E, u_v) =",
      round(float(np.corrcoef(np.log10(E), u_v)[0, 1]), 3))

df = pd.DataFrame({
    "ParticleName": ParticleName,
    "Energy": E,
    "X": X, "Y": Y, "Z": Z,
    "Vx": Vx, "Vy": Vy, "Vz": Vz,
    "PrimBool": rng.integers(0, 2, N),
})

magi.report_basic_table_checks(df)
""")

md(r"""
## 2. Physical features

`build_physical_features` converts raw Cartesian `X,Y,Z,Vx,Vy,Vz` into the
sphere-surface parametrization the model works in: radial position
(`u_r`, `phi_r`) and direction (`u_v`, `phi_v`). It also measures the primary
fraction used for flux normalization.

`source_type` decides what "primary" means: `"cosmic_ray"` uses `PrimBool == 1`;
`"radioactive"` uses `CreatorProcessName == "RadioactiveDecay"`, because for a
source embedded in bulk material the literal Geant4 primary is the decaying
nucleus, which never propagates. **Choosing wrongly does not raise** — it
silently produces a wrong normalization factor.
""")

code(r"""
prep = magi.build_physical_features(
    df, center=center, radius=R, source_type="cosmic_ray")
magi.print_physical_summary(prep)
""")

md(r"""
## 3. Spectral lines — detect *before* building the feature frame

This is the part of the pipeline most specific to v0.8, and the ordering
matters. Lines are detected on the **raw energies**, on their own fine binning,
*before* `build_feature_dataframe` runs. The feature frame's binning is a
separate, coarser thing: with the mixture head it no longer constrains where a
line can be *placed*, only how the continuum is described.

A candidate table would normally come from
`magi.load_candidate_energy_lines("CandidateLines/....json")`, built from a GDML
mass model by `tools/build_candidate_lines_from_geant4.py`. Here we write the
two candidates by hand.

`refine_bin_width_mev` refines each detected peak to sub-bin precision. On a log
grid spanning many decades a detection bin can be hundreds of eV wide — far
broader than a real line — so the bin centre is a poor position estimate.
""")

code(r"""
candidate_lines = [
    {"label": "synthetic Cu K-alpha1", "energy_mev": E_LINE_A, "origin": "instrumental"},
    {"label": "synthetic e+e- 511",    "energy_mev": E_LINE_B, "origin": "instrumental"},
]

RESOLUTION_EV = 4.0   # X-IFU-like; the GENERATIVE line width, see below

E_raw = prep["features"]["Energy"].to_numpy()

res = magi.detect_energy_lines(
    E_raw,
    binning_mode="log_fixed_count", n_bins=256,
    prominence_factor=3.0, window=5,
    candidate_lines=candidate_lines,
    refine_bin_width_mev=RESOLUTION_EV * 1e-6,
)
magi.print_detected_energy_lines(res)

# Keep only lines with enough events to model. On real data this is the 5-sigma
# Poisson floor discussion in the manual; here a plain count filter.
matched = [m for m in res["matched_lines"] if m["count"] >= 100]

# Doublets closer together than one detection bin never become separate peaks,
# so re-measure any unmatched candidate at eV resolution and confirm the real
# ones. (Nothing new to find in this toy source; on real data it recovers
# e.g. Cu K-alpha2, 21 eV from K-alpha1.)
matched += magi.confirm_unresolved_candidate_lines(
    E_raw, candidate_lines, matched, resolution_ev=RESOLUTION_EV)

print("\nmodelled lines:", [m["label"] for m in matched])
""")

md(r"""
## 4. Feature dataframe

`build_feature_dataframe` bins energy and fits the quantile geometry
transforms. Two settings matter here:

- `geometry_transform="quantile_u_r_u_v_phi_r_phi_v"` is the v0.7.2+ layout.
  This string is a **contract**: it is persisted in the checkpoint metadata and
  must match between training and generation. A mismatch does not raise, it
  produces physically wrong output.
- `energy_transform="log10"` gives the mixture head the target
  `y = log10(E/MeV)` it models in.
""")

code(r"""
feature_pack = magi.build_feature_dataframe(
    prep,
    energy_binning_mode="log_fixed_count",
    n_bins=64,                 # small, because this toy source is small
    min_counts=5,
    geometry_transform="quantile_u_r_u_v_phi_r_phi_v",
    n_quantiles=1000,
    random_state=42,
    energy_transform="log10",
)
magi.report_feature_dataframe(feature_pack)
""")

md(r"""
## 5. Gate targets and pinned line widths

Two distinct uses of the detector resolution, which are easy to conflate:

1. **`build_gate_targets`** builds the *training label* saying which real events
   belong to which mixture component. `bandwidth_mode="resolution"` uses a
   Gaussian kernel of the detector FWHM for that decision.
2. **`line_logsigma_from_resolution`** sets the model's pinned *generative*
   line width — how wide the lines it produces are.

The lines in this synthetic source are exactly monoenergetic, which suggests an
exact-match label would be better, and `bandwidth_mode="exact"` exists for that.
**On real data it measured worse**: it did not fix the line it was built for, it
broke a previously passing line, and it pushed the coupling residual out of
band. `"resolution"` is the default and is what every confirmed result used.
See `docs/v0.8.1_line_truth.md` §14.2.

The gate targets then ride into the dataset as extra feature columns.
""")

code(r"""
E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()

gate_targets = magi.build_gate_targets(
    E_full, feature_pack["energy_bins"], matched,
    bandwidth_mode="resolution",
    bandwidth_fwhm_mev=RESOLUTION_EV * 1e-6,
)
print("gate_targets shape:", gate_targets.shape,
      "= [continuum, line_1..line_L] per event")
print("mean routing fraction per component:", gate_targets.mean(axis=0).round(5))

line_positions_y = np.log10(
    [m["candidate_energy_mev"] for m in matched]).astype(np.float32)
line_logsigma = magi.line_logsigma_from_resolution(
    10.0 ** line_positions_y, RESOLUTION_EV, fwhm=True)

print("line positions (y = log10 E/MeV):", line_positions_y)
print("pinned line log-sigmas          :", np.asarray(line_logsigma).round(6))
""")

md(r"""
## 6. Dataset build

Filter rare particle types, split (stratified on type), scale, build the type
one-hot conditioning and per-type loss weights, then the `tf.data` pipelines.

`scale_cols=()` is deliberate: the quantile-transformed geometry columns are
already standardized by their own transform, and scaling them twice is wrong.
""")

code(r"""
feat = feature_pack["feat"].copy()
for j in range(gate_targets.shape[1]):
    feat[f"gate_target_{j}"] = gate_targets[:, j]

cont_cols = ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q", "energy_y") + tuple(
    f"gate_target_{j}" for j in range(gate_targets.shape[1]))

dataset_pack = magi.filter_particle_types_continuous_geometry(
    feat=feat, prob_threshold=1e-5, cont_cols=cont_cols,
    normalization=prep.get("normalization"))

split_pack = magi.split_feature_data(dataset_pack, random_state=42)
magi.report_split_summary(split_pack, dataset_pack["n_types"])

scaled_pack = magi.scale_continuous_features(split_pack, scale_cols=())
condition_pack = magi.build_conditioning_and_weights(
    scaled_pack, dataset_pack["n_types"],
    idx_to_type=dataset_pack["idx_to_type"], alpha=0.5)
magi.report_conditioning(condition_pack, dataset_pack["n_types"])

ds_pack = magi.build_tf_datasets(condition_pack, batch_size=256)
magi.report_tf_datasets(ds_pack)
""")

md(r"""
## 7. Per-type zone probabilities

v0.8.2's `prior_zone_conditioning` widens what the learned prior is conditioned
on, from the particle-type one-hot alone to `[type, zone]`, where zone is the
continuum/line routing target.

Why it is needed: the gate is trained on `z ~ q(z|x)` but generation draws
`z ~ p(z|cond)`. Without the zone in `cond`, the prior has no way to express
which mixture component an event belongs to, so at generation time rare lines
get routed by whatever the type marginal happens to imply. On the real
cosmic-ray source this change moved Al K-alpha1 from 0.702 ± 0.290 to
0.959 ± 0.025 while leaving coupling intact.

At generation time the zone is sampled from each type's empirical frequency,
which is what `zone_probs` records.
""")

code(r"""
n_zones = gate_targets.shape[1]
zone_cols = dataset_pack["X_cont_raw"][:, -n_zones:]
y_type = dataset_pack["y_type"]

zone_probs = np.zeros((dataset_pack["n_types"], n_zones), dtype=np.float64)
for t in range(dataset_pack["n_types"]):
    mask = (y_type == t)
    row = zone_cols[mask].mean(axis=0) if mask.any() else np.zeros(n_zones)
    s = row.sum()
    zone_probs[t] = (row / s) if s > 0 else np.eye(n_zones)[0]

print("zone_probs [type x (continuum, line_1..line_L)]:")
for t in range(dataset_pack["n_types"]):
    print(f"  {dataset_pack['idx_to_type'][t]:>8s}",
          np.array2string(zone_probs[t], precision=5))
""")

md(r"""
## 8. Build and train the v0.8 model

`CVAE_MixEnergy_ContPhi_TaskAdaptive` keeps v0.7.2's geometry heads and replaces
the categorical energy head with a gated mixture of a continuum density and
fixed-position line components.

The settings that matter:

| Setting | Why |
|---|---|
| `continuum_mode="flow"` | A single Gaussian cannot represent a multi-decade continuum. |
| `continuum_flow_warp="cdf"` | Pre-warps the flow input by the empirical CDF, so spline knots go where the events are. |
| `energy_flow_condition="z_cond"` | Conditions the continuum flow on both latent and conditioning. |
| `prior="coupling"` | The learned conditional prior. This is what makes the correlations come out right. |
| `prior_zone_conditioning=True` | Section 7 above. |
| `gate_focal_gamma=1.0` | Focal weighting so rare lines get routed. **Use 1** — at γ ≥ 2 the gate digs the continuum out from between the lines. |

A real run is 40 epochs and takes ~90 minutes on CPU. Here: 8 epochs on a tiny
sample, purely to show the loop runs and the loss moves.
""")

code(r"""
# CDF pre-warp knots for the continuum flow, fitted on the energy target.
ey = feat["energy_y"].to_numpy()
warp_yk, warp_zk = magi.fit_cdf_warp_knots(ey, n_knots=64, eps=1e-4)

model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
    n_types=dataset_pack["n_types"],
    latent_dim=4,
    hidden=(64, 64),
    line_positions_y=line_positions_y,
    line_logsigma_init=line_logsigma,
    line_logsigma_trainable=False,   # widths are pinned, not fitted
    continuum_mode="flow",
    continuum_flow_warp="cdf",
    continuum_flow_warp_y_knots=warp_yk,
    continuum_flow_warp_z_knots=warp_zk,
    continuum_flow_bins=16,
    continuum_flow_transforms=3,
    energy_flow_condition="z_cond",
    prior="coupling",
    prior_zone_conditioning=True,
    zone_probs=zone_probs,
    gate_focal_gamma=1.0,
    w_gate_aux=2.0,
    type_weights=condition_pack["type_weights"],
)

magi.compile_model(model, learning_rate=2e-3, optimizer="adam")
""")

code(r"""
history = magi.fit_model(
    model, ds_pack["train_ds"], ds_pack["val_ds"],
    epochs=8, callbacks=magi.build_default_callbacks(early_patience=20),
    verbose=2,
)
""")

md(r"""
`print_model_structure` prints the generative structure with the formulas the
code actually implements, the configured line table, and the parameter counts —
useful for confirming the model is the one you meant to build.
""")

code(r"""
magi.print_model_structure(model, summaries=False)
""")

md(r"""
## 9. Save the run

`save_final_trained_model` writes weights, config, metadata, history, task
weights and a human-readable summary as **one unit**. Generation needs the
config and metadata as much as the weights, so copy or move them together.

Pass `model.to_generation_config()` rather than a hand-written dict: for v0.8
heads the loader *requires* every key it emits and raises on a missing one.
That guard exists because the old `.get(key, default)` behaviour silently
rebuilt a different architecture — the defaults select a code path, not just a
layer shape, so generation was wrong instead of failing.
""")

code(r"""
import tempfile, joblib, os

SAVE_DIR = os.path.join(tempfile.gettempdir(), "magi_example_run")
MODEL_NAME = "example_mix"

preprocessing_metadata = {
    "geometry_transform": "quantile_u_r_u_v_phi_r_phi_v",
    "energy_transform": "log10",
    "energy_bins": [float(b) for b in feature_pack["energy_bins"]],
    "center": list(center),
    "radius": R,
    "resolution_ev": RESOLUTION_EV,
    "matched_lines": [
        {"label": m["label"], "candidate_energy_mev": m["candidate_energy_mev"]}
        for m in matched
    ],
    "cont_cols": list(cont_cols),
}

paths = magi.save_final_trained_model(
    model=model,
    save_dir=SAVE_DIR,
    model_name=MODEL_NAME,
    history=history,
    model_config=model.to_generation_config(),
    preprocessing_metadata=preprocessing_metadata,
    normalization_metadata=prep.get("normalization"),
    notes="Example_Usage.ipynb synthetic walkthrough",
)

# The fitted transformers are part of the same unit.
qt = feature_pack["quantile_transformers"]
joblib.dump(qt, os.path.join(SAVE_DIR, f"{MODEL_NAME}_quantile_transformers.joblib"))

print("\nrun directory:", SAVE_DIR)
for f in sorted(os.listdir(SAVE_DIR)):
    print("  ", f)
""")

md(r"""
## 10. Reload the checkpoint

Reloading exercises the config-match guard. If `to_generation_config()` and the
loader ever drift apart, this raises here rather than silently generating from
the wrong architecture.
""")

code(r"""
model_config = magi.load_json(os.path.join(SAVE_DIR, f"{MODEL_NAME}_config.json"))
print("config_version:", model_config.get("config_version"),
      "| model_class:", model_config.get("model_class"))

reloaded = magi.load_task_adaptive_model_for_generation(
    save_dir=SAVE_DIR,
    model_name=MODEL_NAME,
    model_config=model_config,
    n_types=dataset_pack["n_types"],
    type_weights=condition_pack["type_weights"],
    radius=R,
    verbose=1,
)
""")

md(r"""
## 11. Generate and reconstruct

Sample latent codes conditioned on particle type, decode, then invert the
transforms to get back physical quantities.

**Generate as many events as you have real ones** when validating. The
line-recovery metrics compare raw counts, so a smaller generated sample deflates
every ratio by exactly the shortfall.
""")

code(r"""
# The real reference. `filtered_prep` holds the physical features AFTER the
# energy cuts the feature frame applied, which is the population the model was
# actually trained on - so this is the honest comparison set. (This is exactly
# what tools/acceptance_v0_8.py does; reconstruct_real_test_physics is the
# alternative when you want the held-out split specifically.)
f_real = feature_pack["filtered_prep"]["features"]
real = {
    "E":     f_real["Energy"].to_numpy(),
    "logE":  np.log10(f_real["Energy"].to_numpy()),
    "u_r":   f_real["u_r"].to_numpy(),
    "u_v":   f_real["u_v"].to_numpy(),
    "phi_r": f_real["phi_r"].to_numpy(),
    "phi_v": f_real["phi_v"].to_numpy(),
}

N_GEN = real["E"].size    # match the real count - see the note above

gen_pack = magi.generate_latent_outputs(
    reloaded, n_samples=N_GEN,
    type_probs=dataset_pack["type_probs"],
    n_types=dataset_pack["n_types"],
    idx_to_type=dataset_pack["idx_to_type"],
    rng=np.random.default_rng(0),
)

qt = feature_pack["quantile_transformers"]   # the fitted geometry transforms

reco = magi.reconstruct_generated_features(
    gen_pack,
    energy_head_mode="mixture",
    energy_transform="log10",
    geometry_mode="quantile_u_r_u_v_phi_r_phi_v",
    qt_u_r=qt["qt_u_r"], qt_u_v=qt["qt_u_v"],
    qt_phi_r=qt["qt_phi_r"], qt_phi_v=qt["qt_phi_v"],
)

# reconstruct_generated_features returns "<name>_gen" keys; repack them under
# the same names as `real` so the validation helpers can compare the two.
gen = {
    "E":     reco["E_gen"],
    "logE":  np.log10(reco["E_gen"]),
    "u_r":   reco["u_r_gen"],
    "u_v":   reco["u_v_gen"],
    "phi_r": reco["phi_r_gen"],
    "phi_v": reco["phi_v_gen"],
    # which mixture component each event came from - the line-recovery metric
    # uses it to separate routing from the raw in-window count.
    "comp":  gen_pack["energy_component_idx_gen"],
}

# For the Cartesian x,y,z / vx,vy,vz an exported source file needs:
gen_phys = magi.reconstruct_generated_physics(reco, center=center, radius=R)

print(f"real: {real['E'].size:,} events   generated: {gen['E'].size:,} events")
print("generated physics keys:", sorted(gen_phys.keys()))
""")

md(r"""
## 12. Validate

Four checks, in increasing order of what they tell you.
""")

code(r"""
# (a) Constraints that must hold BY CONSTRUCTION, not by fit. Failures here
#     mean a reconstruction/transform mismatch, not an under-trained model.
magi.report_generated_constraints(gen_phys, radius=R)
""")

code(r"""
# (b) Marginals: 1-D Wasserstein distance per variable.
#
# We call scipy directly because `real`/`gen` here carry plain physical arrays
# keyed "E", "logE", "u_r", ... - the convention tools/acceptance_v0_8.py uses.
# magi.compute_wasserstein_scores() does the same job but is tied to the
# "<name>_real"/"<name>_gen" keys the reconstruct_*_physics helpers emit, and
# raises rather than skipping if one is missing.
from scipy.stats import wasserstein_distance

for k in ["logE", "u_r", "u_v", "phi_r", "phi_v"]:
    print(f"  {k:8s} {wasserstein_distance(real[k], gen[k]):.4f}")
print("\n  (bar on real data: global logE Wasserstein <= 0.05)")
""")

code(r"""
# (c) Per-line recovery. Note `line_significance`: below ~5 the line is not
#     significantly detected in the REAL data either, so its ratio is noise
#     and should be ignored rather than reported as a failure.
recovery = magi.compute_line_integral_recovery(
    real["E"], gen["E"], matched, feature_pack["energy_bins"],
    energy_component_idx_gen=gen["comp"],
    resolution_ev=RESOLUTION_EV,
    neighbour_lines=candidate_lines,
)

print(f"{'line':28s} {'n_real':>8s} {'n_gen':>8s} {'recovery':>9s} {'sig':>8s}")
for r in recovery:
    rr = r["recovery_ratio"]
    print(f"  {r['label']:26s} {r['n_real_line']:8.0f} {r['n_gen_line']:8.0f} "
          f"{(f'{rr:9.3f}' if rr is not None else '      n/a')} "
          f"{r['line_significance']:8.1f}")
""")

md(r"""
### (d) The joint distribution — the one that matters

This is what v0.8 exists for. We injected an energy↔direction correlation into
the synthetic source in section 1; the check is whether the generated sample
reproduces it. The **coupling residual** is the largest absolute difference
between the real and generated correlation matrices.

On real sources, over three seeds, this lands at 0.029 (CR) and 0.036 (Small)
against a bar of 0.05. With this toy's 7k events and 8 epochs, expect it to be
much worse — the point is the measurement, not the value.
""")

code(r"""
cols = ["logE", "u_r", "u_v", "phi_r", "phi_v"]

corr_real = pd.DataFrame({c: real[c] for c in cols}).corr()
corr_gen = pd.DataFrame({c: gen[c] for c in cols}).corr()
residual = (corr_gen - corr_real).abs().to_numpy().max()

print("real correlations:\n", corr_real.round(3), "\n")
print("generated correlations:\n", corr_gen.round(3), "\n")
print(f"coupling residual  max|d corr| = {residual:.3f}   (bar on real data: 0.05)")
""")

code(r"""
magi.compare_hist_with_residuals(
    np.log10(real["E"]), np.log10(gen["E"]),
    name="log10 E [MeV]", bins=40, ratio_clip=2.0,
)
""")

md(r"""
## 13. Export a Geant4-ready source file

`generate_detector_table_to_file` samples straight to disk in chunks, so an
arbitrarily large source file stays memory-safe. Neutrinos are filtered out by
default; with `generate_until_n_written=True` generation continues until the
requested number of *transportable* particles has been written.

For a one-call version that also loads the checkpoint, use
`magi.generate_detector_input_file` — that is what
`scripts/generate_geant_source.py` wraps, and what the companion Geant4
project's `/generator/mlScript` macro drives.
""")

code(r"""
OUT = os.path.join(SAVE_DIR, "example_source.dat")

summary = magi.generate_detector_table_to_file(
    model=reloaded,
    filepath=OUT,
    n_events=2000,
    type_probs=dataset_pack["type_probs"],
    n_types=dataset_pack["n_types"],
    idx_to_type=dataset_pack["idx_to_type"],
    geometry_metadata={
        "geometry_transform": "quantile_u_r_u_v_phi_r_phi_v",
        "qt_u_r": qt["qt_u_r"], "qt_u_v": qt["qt_u_v"],
        "qt_phi_r": qt["qt_phi_r"], "qt_phi_v": qt["qt_phi_v"],
    },
    energy_head_mode="mixture",
    energy_transform="log10",
    center=center, radius=R,
    chunk_size=1000, seed=42,
    output_format="text",
    verbose=1,
)

print("\nfirst 3 lines of the source file:")
with open(OUT) as f:
    for _ in range(3):
        print("  ", f.readline().rstrip())
""")

md(r"""
## Next steps

- **`docs/manual/magi_manual.pdf`** — the full user manual: code structure, the
  API and its settings, the tools, a worked full run, a worked validation, the
  theoretical foundation, and the validity envelope.
- **`MAGI_package/docs/USAGE.md`** — the short-form version of the same.
- **`tools/run_v0_8_real.py`** — a real run on real data, with live logging.
- **`tools/acceptance_v0_8.py`** — the pass/fail harness. Always use
  `--seeds 42 7 13`: on the cosmic-ray source the per-band Wasserstein has a
  seed-to-seed σ/μ of 40–50%, and two changes adopted on single-seed evidence
  during development both evaporated under a three-seed grid.
- **`docs/v0.8.1_line_truth.md`** — the measurement record: what was tried, what
  it showed, and what has already been ruled out.

### Before you trust a number

This walkthrough is a mechanical demonstration. On real data:

- Coupling and the energy marginal are validated, with error bars.
- Per-line **intensities** are not — 0.7× to 4.9×, unstable across seeds. Do not
  use generated output to estimate a line flux.
- Line **positions** are reliable, audited to ≤0.5 eV.
- Geometry is hard-limited to a **sphere**.
- `geometry_transform` mismatches between training and generation do not raise;
  they produce physically wrong output.
""")

_fix()

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path("Example_Usage.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} with {len(C)} cells")
