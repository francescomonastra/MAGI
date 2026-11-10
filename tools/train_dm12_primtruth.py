#!/usr/bin/env python
"""Train the HENABICS primary-conditioned MAGI on the DM1.2 primary-truth set.

Cloned from tools/train_dm12_iso_seed.py (same v0.8.2 head, same DM1.2
sphere and line table) with three differences:

  * the conditioning vector is [onehot(crossing type), onehot(primary type),
    qt(log10 E_prim)] - cond_dim = n_types + n_primary_types + 1 - so the
    model generates the crossings of a given primary
    (docs/MAGI_HENABICS_integration_requirements.md);
  * the train/val/test split is grouped by primary (PrimaryPDG, EventId), so
    one shower never sits on both sides of it;
  * clipnorm = 5.0, the v0.9.3 fix for the loss spikes.

It also measures and persists what grouped generation needs next to the
weights: the multiplicity table P(k | primary, E) / P(crossing type | ...),
the GPS macros the primaries were thrown from, and qt_primary_energy (in the
quantile_transformers joblib).

The mu- and gamma- runs stay in separate files up to here so each species can
be flux-normalized on its own (S1GDML_DM1.2/henabics_run/run_henabics_dm12.sh);
they are merged only for training. EventId restarts per species, so the group
key is (PrimaryPDG, EventId).

    python tools/train_dm12_primtruth.py [--seed 42] [--max-crossings N]
"""
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import magi

# --------------------------------------------------------------------------
SOURCE = "DM1_2_primtruth"
G4_DIR = "/Volumes/T7/Geant4-projects/Geant4-10.4.3/S1GDML_DM1.2"
RUN_DIR = f"{G4_DIR}/henabics_run"
SPECIES = ("mu", "gamma")
GPS_MACROS = {13: f"{G4_DIR}/gpsMu_iso.mac", 22: f"{G4_DIR}/gpsGamma.mac"}

CENTER = (0.0, 0.0, 5.0)      # GDMLDetectorConstruction.cc:167
R = 95.0                      # post-overlap-fix radius

X_IFU_RESOLUTION_EV = 4.0
CANDIDATE_LINES_FILE = ("/Volumes/T7/MAGI/CandidateLines/"
                        "CANDIDATE_ENERGY_LINES_"
                        "CriostatoDM1_2_Richiesta11_06_2026-worldVOL_EADL_escape.json")

LEARNING_RATE = 2e-4
CLIPNORM = 5.0
MODEL_NAME = "mix_DM1_2_primtruth"
# --------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--crossings", nargs="+", default=[
    f"{RUN_DIR}/allDSCryoSphere_DM1_2_primtruth_{sp}.dat" for sp in SPECIES])
parser.add_argument("--summaries", nargs="+", default=[
    f"{RUN_DIR}/run_summaries_{sp}" for sp in SPECIES])
parser.add_argument("--max-crossings", type=int, default=None,
                    help="train on a random subset of whole showers of about this "
                         "many crossings; the multiplicity table always uses all")
parser.add_argument("--save-dir", default=None)
parser.add_argument("--energy-loss-flow", action="store_true",
                    help="v1.0.0 (needs --uncollided-atom): model the energy of the continuum crossings of the "
                         "primary's own species as v = log10(log10(E_prim/E)), so that E < E_prim by construction")
parser.add_argument("--loss-variable", choices=["loglog", "logit"], default="loglog",
                    help="variable of the energy loss flow (see CVAE_MixEnergy_ContPhi_TaskAdaptive)")
parser.add_argument("--uncollided-atom", action="store_true",
                    help="v1.0.0: add the uncollided-primary atom (E = E_prim exactly) to "
                         "the energy head, with supervised line routing")
parser.add_argument("--radius", type=float, default=R,
                    help="radius of the tracking sphere of the crossing files in mm (default 95, the "
                         "DM1.2 shell; Track G runs use other radii of the virtual ball)")
parser.add_argument("--first-entry", action="store_true",
                    help="Track G: the crossing files have 17 columns (AncestorInside); train on the "
                         "first entries only (magi.select_first_entry) and record crossing_rule=first_entry")
parser.add_argument("--zone-by-primary", action="store_true",
                    help="v1.0.0: draw the zone at generation from a table per (crossing type, "
                         "primary type, primary-energy bin) instead of per crossing type")
parser.add_argument("--local-direction", action="store_true",
                    help="A3: direction in the local frame of the crossing point (mu, psi), "
                         "geometry_transform quantile_u_r_mu_phi_r_psi; non-inward crossings dropped")
parser.add_argument("--energy-condition-geometry", action="store_true",
                    help="condition the energy head (gate, continuum and loss flow) on the geometry targets, "
                         "teacher-forced in training and sampled first at generation (v0.8.3 flag); with "
                         "--local-direction the energy-zenith coupling no longer passes through z alone")
parser.add_argument("--init-seed", type=int, default=None,
                    help="seed of the weight initialisation and of the sampling; --seed keeps "
                         "fixing the split and the transforms, so that independent trainings "
                         "share the test showers of the condition-matched benchmark")
parser.add_argument("--refit-prior", action="store_true",
                    help="last stage of the v1.0 recipe: refit the prior on the frozen encoder "
                         "(magi.refit_prior); the jointly trained run is saved as <save dir>_joint")
parser.add_argument("--refit-epochs", type=int, default=60)
parser.add_argument("--resume", action="store_true",
                    help="continue from <save dir>/resume if it exists (epochs is the total)")
parser.add_argument("--write-multiplicity-to", default=None,
                    help="rebuild the multiplicity table (with the E_prim quantiles of the crossing primaries, "
                         "v1.0.0) from these crossings and write it into the metadata of this run directory, "
                         "keeping the old metadata as *_metadata.pre_equantiles.json; exits without training")
parser.add_argument("--dump-split", default=None,
                    help="write the validation split (model inputs and physical truth) to this .npz after the "
                         "preprocessing and exit without training; used by tools/dm12_posterior_energy.py")
args = parser.parse_args()

SEED = args.seed
FINAL_DIR = args.save_dir or f"/Volumes/T7/MAGI-dev/trained_models/v0_9_{SOURCE}_seed{SEED}"
SAVE_DIR = f"{FINAL_DIR}_joint" if args.refit_prior else FINAL_DIR

magi.initialize_environment(seed=SEED if args.init_seed is None else args.init_seed, cpu_only=True)
print(f"magi {magi.__version__} from {magi.__file__}", flush=True)

# ---- data ----------------------------------------------------------------
df = pd.concat(
    [magi.load_detector_table(filepath=p, sep=r"\s+") for p in args.crossings],
    ignore_index=True,
)
if args.first_entry:
    df = magi.select_first_entry(df)
magi.report_basic_table_checks(df)
run_summary = magi.load_run_summaries(args.summaries)
for pdg, thrown in run_summary["thrown"].items():
    print(f"PDG {pdg}: {int(thrown.sum()):,} primaries thrown, "
          f"{int((df['PrimaryPDG'] == pdg).sum()):,} crossings", flush=True)

R = float(args.radius)
prep = magi.build_physical_features(df, center=CENTER, radius=R)
magi.print_physical_summary(prep)

feat_base = prep["features"]
E_all = feat_base["Energy"].to_numpy()
print(f"\n{SOURCE}: {len(df):,} rows, {E_all.size:,} valid energies", flush=True)

# ---- lines ---------------------------------------------------------------
candidate_payload = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)
candidate_lines = candidate_payload["lines"]
print(f"Loaded {candidate_payload['n_lines']} candidate lines for "
      f"{candidate_payload['mass_model']}", flush=True)

line_result = magi.detect_energy_lines(
    E_all,
    binning_mode="log_fixed_count",
    n_bins=1024,
    prominence_factor=3.0,
    window=5,
    candidate_lines=candidate_lines,
    refine_bin_width_mev=X_IFU_RESOLUTION_EV * 1e-6,
)
magi.print_detected_energy_lines(line_result)

matched_all = line_result["matched_lines"]
matched = [m for m in matched_all if m["count"] >= 100]
print(f"\n{len(matched_all)} matched lines; {len(matched)} feed the mixture head: "
      f"{[m['label'] for m in matched]}", flush=True)

FWHM_MEV = X_IFU_RESOLUTION_EV * 1e-6

# ---- features ------------------------------------------------------------
GEOMETRY_TRANSFORM = ("quantile_u_r_mu_phi_r_psi" if args.local_direction
                      else "quantile_u_r_u_v_phi_r_phi_v")
feature_pack = magi.build_feature_dataframe(
    prep,
    energy_binning_mode="log_fixed_count",
    n_bins=512,
    geometry_transform=GEOMETRY_TRANSFORM,
    n_quantiles=10000,
    random_state=SEED,
    energy_transform="log10",
)
magi.report_feature_dataframe(feature_pack)

energy_bins = feature_pack["energy_bins"]
quantile_transformers = feature_pack["quantile_transformers"]

line_positions_mev = np.array([m["candidate_energy_mev"] for m in matched],
                              dtype=np.float64)
line_positions_y = np.log10(line_positions_mev).astype(np.float32)

E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
gate_targets = magi.build_gate_targets(
    E_full, feature_pack["energy_bins"], matched,
    bandwidth_mode="resolution", bandwidth_fwhm_mev=FWHM_MEV,
)
if args.uncollided_atom:
    # An uncollided crossing is the primary itself: same species, E = E_prim.
    PDG_NAME = {13: "mu-", 22: "gamma"}
    f0 = feature_pack["feat"]
    pdgs = sorted(int(p) for p in f0["PrimaryPDG"].unique())
    names = sorted(set(f0["ParticleName"]) | {PDG_NAME[p] for p in pdgs})
    cross_code = f0["ParticleName"].map({n: i for i, n in enumerate(names)}).to_numpy()
    prim_code = f0["PrimaryPDG"].map({p: i for i, p in enumerate(pdgs)}).to_numpy()
    gate_targets, is_unc = magi.add_uncollided_gate_target(
        gate_targets, E_full, f0["PrimaryEnergy"].to_numpy(), cross_code, prim_code,
        [names.index(PDG_NAME[p]) for p in pdgs],
    )
    for pdg in pdgs:
        m = (f0["PrimaryPDG"].to_numpy() == pdg) & (f0["ParticleName"].to_numpy() == PDG_NAME[pdg])
        print(f"uncollided PDG {pdg}: {is_unc[m].mean():.4f} of its own-species crossings "
              f"({int(is_unc.sum()):,} atom events in all)", flush=True)

feat = feature_pack["feat"].copy()
for j in range(gate_targets.shape[1]):
    feat[f"gate_target_{j}"] = gate_targets[:, j]

cont_cols = tuple(feature_pack["geometry_metadata"]["X_cont_cols"]) + ("energy_y",) + tuple(
    f"gate_target_{j}" for j in range(gate_targets.shape[1])
)

dataset_pack = magi.filter_particle_types_continuous_geometry(
    feat=feat, prob_threshold=1e-5, cont_cols=cont_cols,
)
magi.report_continuous_geometry_features(dataset_pack)

# ---- multiplicity (all crossings of the kept types, before subsampling) ---
multiplicity_table = magi.build_multiplicity_table(
    dataset_pack["feat"], run_summary, dataset_pack["type_to_idx"],
)
if args.write_multiplicity_to:
    import glob as _glob, shutil as _shutil
    mfile = _glob.glob(f"{args.write_multiplicity_to}/*_metadata.json")[0]
    bak = mfile.replace("_metadata.json", "_metadata.pre_equantiles.json")
    if not os.path.exists(bak):
        _shutil.copy2(mfile, bak)
    meta = json.load(open(mfile))
    meta["preprocessing_metadata"]["multiplicity_table"] = multiplicity_table
    json.dump(meta, open(mfile, "w"), indent=2)
    print(f"multiplicity table with E_prim quantiles written to {mfile} (backup {bak})", flush=True)
    raise SystemExit(0)
for pdg, sp in multiplicity_table["species"].items():
    k_counts = np.asarray(sp["k_counts"])
    k = np.arange(k_counts.shape[1])
    n_prim = k_counts.sum()
    print(f"PDG {pdg}: P(k=0) = {k_counts[:, 0].sum() / n_prim:.4f}, "
          f"<k | k>0> = {(k_counts * k).sum() / k_counts[:, 1:].sum():.3f}, "
          f"k_max = {k[-1]}", flush=True)

dataset_pack = magi.build_primary_conditioning(dataset_pack, random_state=SEED)
quantile_transformers["qt_primary_energy"] = dataset_pack["qt_primary_energy"]

if args.max_crossings is not None and args.max_crossings < len(dataset_pack["group_id"]):
    # Whole showers, so the subset keeps the within-group structure.
    rng = np.random.default_rng(SEED)
    gid = dataset_pack["group_id"]
    n_groups = int(gid.max()) + 1
    keep_groups = rng.random(n_groups) < args.max_crossings / len(gid)
    keep = keep_groups[gid]
    for key, val in list(dataset_pack.items()):
        if isinstance(val, pd.DataFrame) and len(val) == len(gid):
            dataset_pack[key] = val[keep].reset_index(drop=True)
        elif isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == len(gid):
            dataset_pack[key] = val[keep]
    print(f"\nsubsampled {keep.sum():,} / {len(gid):,} crossings "
          f"({keep_groups.sum():,} showers)", flush=True)

split_pack = magi.split_feature_data(
    dataset_pack, test_size_total=0.30, val_size_from_temp=0.50, random_state=SEED,
)
magi.report_split_summary(split_pack, n_types=dataset_pack["n_types"])

scaled_pack = magi.scale_continuous_features(split_pack, scale_cols=())
condition_pack = magi.build_conditioning_and_weights(
    scaled_pack, n_types=dataset_pack["n_types"],
    idx_to_type=dataset_pack["idx_to_type"], alpha=0.5,
)
COND_DIM = condition_pack["cond_dim"]
print(f"cond_dim = {COND_DIM} ({dataset_pack['n_types']} crossing types + "
      f"{dataset_pack['n_primary_types']} primary types + qt(log10 E_prim))",
      flush=True)
tf_pack = magi.build_tf_datasets(condition_pack, batch_size=4096,
                                 shuffle_buffer_cap=200_000)
magi.report_tf_datasets(tf_pack)

# ---- v0.8.2 per-type zone probabilities ----------------------------------
n_zones = gate_targets.shape[1]
zone_cols = dataset_pack["X_cont_raw"][:, -n_zones:]
y_type_for_zones = dataset_pack["y_type"]
zone_probs = np.zeros((dataset_pack["n_types"], n_zones), dtype=np.float64)
for t in range(dataset_pack["n_types"]):
    mask = (y_type_for_zones == t)
    row = zone_cols[mask].mean(axis=0) if mask.any() else np.zeros(n_zones)
    s = row.sum()
    zone_probs[t] = (row / s) if s > 0 else np.eye(n_zones)[0]

zone_labels = ["continuum"] + [m["label"] for m in matched] + (
    ["uncollided"] if args.uncollided_atom else [])
print(f"\nzone_probs ({zone_labels}):")
for t, tname in dataset_pack["idx_to_type"].items():
    print(f"    {tname:12s} {np.round(zone_probs[t], 4)}")

# ---- CDF pre-warp knots ---------------------------------------------------
energy_y_all = dataset_pack["feat"]["energy_y"].to_numpy()
if args.uncollided_atom:
    # The flow never sees an atom event (supervised routing), so neither does its warp.
    energy_y_all = energy_y_all[dataset_pack["feat"][f"gate_target_{gate_targets.shape[1] - 1}"].to_numpy() < 0.5]
loss_kwargs = {}
if args.energy_loss_flow:
    if not args.uncollided_atom:
        raise SystemExit("--energy-loss-flow needs --uncollided-atom")
    fd = dataset_pack["feat"]
    own_name = fd["PrimaryPDG"].map(PDG_NAME).to_numpy()
    own = (fd["ParticleName"].to_numpy() == own_name)
    cont = fd["gate_target_0"].to_numpy() > 0.5
    keep = (fd[f"gate_target_{gate_targets.shape[1] - 1}"].to_numpy() < 0.5)   # not the atom
    w = np.log10(fd["PrimaryEnergy"].to_numpy(dtype=np.float64)) - fd["energy_y"].to_numpy(dtype=np.float64)
    sel = own & cont
    if np.any(w[sel] <= 0):
        raise SystemExit(f"{int(np.sum(w[sel] <= 0))} own-species continuum crossings have E >= E_prim")
    if args.loss_variable == "loglog":
        v_own = np.log10(w[sel])
    else:
        f = 10.0 ** (-w[sel]); v_own = np.log(f) - np.log1p(-f)
    lk_y, lk_z = magi.fit_cdf_warp_knots(v_own, n_knots=256, eps=1e-4)
    loss_kwargs = dict(energy_loss_flow=True, loss_flow_variable=args.loss_variable,
                       loss_flow_warp_y_knots=lk_y, loss_flow_warp_z_knots=lk_z)
    print(f"loss flow ({args.loss_variable}): {int(sel.sum()):,} own-species continuum crossings, v quantiles 1/50/99 % = "
          f"{np.round(np.quantile(v_own, [.01, .5, .99]), 3)}", flush=True)
    # the continuum flow never sees them, so neither does its warp
    energy_y_all = fd["energy_y"].to_numpy()[keep & ~sel]
warp_y_knots, warp_z_knots = magi.fit_cdf_warp_knots(energy_y_all, n_knots=256, eps=1e-4,
                                                     exclude_y=line_positions_y)

n_types = dataset_pack["n_types"]
Y_CONT_DIM = tf_pack["X_cont_train_s"].shape[1]

# ---- model ----------------------------------------------------------------
line_logsigma_init = magi.line_logsigma_from_resolution(
    line_positions_mev, X_IFU_RESOLUTION_EV, fwhm=True,
)

preprocessing_metadata = {
    "source": SOURCE,
    "source_files": list(args.crossings),
    "run_summaries": list(args.summaries),
    "center": list(CENTER),
    "radius": R,
    "sphere_center": list(CENTER),
    "sphere_radius": R,
    "geometry_transform": GEOMETRY_TRANSFORM,
    "n_outward_dropped": feature_pack["geometry_metadata"].get("n_outward_dropped", 0),
    "energy_transform": "log10",
    "cont_cols": list(cont_cols),
    "y_cont_dim": Y_CONT_DIM,
    "energy_bins": energy_bins,
    "type_probs": dataset_pack["type_probs"],
    "idx_to_type": dataset_pack["idx_to_type"],
    "n_types": n_types,
    "energy_binning_mode": feature_pack["energy_config"]["mode"],
    "energy_config": feature_pack["energy_config"],
    "candidate_lines_file": CANDIDATE_LINES_FILE,
    "gate_target_bandwidth_mode": "resolution",
    "x_ifu_resolution_ev": X_IFU_RESOLUTION_EV,
    "matched_lines": [m["label"] for m in matched],
    # HENABICS primary truth: what generate_primary_groups needs.
    "cond_dim": int(COND_DIM),
    "n_primary_types": int(dataset_pack["n_primary_types"]),
    "idx_to_primary_type": {int(i): int(p) for i, p in
                            dataset_pack["idx_to_primary_type"].items()},
    "primary_join_key_rule": magi.data.PRIMARY_JOIN_KEY_RULE,
    "gps_macros": {str(p): m for p, m in GPS_MACROS.items()},
    "multiplicity_table": multiplicity_table,
    # Track G: which crossings the model was trained on. A first-entry model must be injected
    # into a run whose interior is simulated (see docs/TrackG_design_review.md).
    "crossing_rule": "first_entry" if args.first_entry else "all_crossings",
}

if args.dump_split:
    vi = split_pack["idx_val"] if "idx_val" in split_pack else None
    out = {"X_cont_val_s": np.asarray(tf_pack["X_cont_val_s"], dtype=np.float32),
           "cond_val": np.asarray(condition_pack["cond_val"], dtype=np.float32),
           "y_val": np.asarray(split_pack["y_val"]),
           "X_cont_train_s": np.asarray(tf_pack["X_cont_train_s"], dtype=np.float32),
           "cond_train": np.asarray(condition_pack["cond_train"], dtype=np.float32)}
    for part in ("val", "train"):
        ix = split_pack.get(f"idx_{part}")
        if ix is None:
            continue
        f = dataset_pack["feat"].iloc[ix]
        sfx = "" if part == "val" else "_train"
        out.update({f"E{sfx}": 10 ** f["energy_y"].to_numpy(), f"E_prim{sfx}": f["PrimaryEnergy"].to_numpy(),
                    f"primary_pdg{sfx}": f["PrimaryPDG"].to_numpy(),
                    f"particle{sfx}": f["ParticleName"].to_numpy().astype(str)})
    np.savez(args.dump_split, **out)
    print("dumped the validation split ->", args.dump_split, {k: v.shape for k, v in out.items()}, flush=True)
    raise SystemExit(0)

atom_kwargs = {}
if args.uncollided_atom:
    idx_to_primary = {int(i): int(p) for i, p in dataset_pack["idx_to_primary_type"].items()}
    atom_kwargs = dict(
        uncollided_atom=True,
        energy_line_routing="supervised",
        primary_energy_knots=magi.primary_energy_knots_from_transformer(
            quantile_transformers["qt_primary_energy"]),
        primary_to_crossing_type=[
            dataset_pack["type_to_idx"][PDG_NAME[idx_to_primary[i]]]
            for i in range(len(idx_to_primary))],
    )
    preprocessing_metadata["uncollided_atom"] = True
    atom_kwargs.update(loss_kwargs)

magi.initialize_environment(seed=SEED if args.init_seed is None else args.init_seed,
                            cpu_only=True, quiet=True)
if args.zone_by_primary:
    zone_probs, zone_edges = magi.build_zone_probs_table(
        dataset_pack["X_cont_raw"][:, -n_zones:], dataset_pack["y_type"],
        dataset_pack["y_primary_type"], dataset_pack["primary_cond"][:, -1],
        n_types=dataset_pack["n_types"], n_primary_types=dataset_pack["n_primary_types"])
    atom_kwargs["zone_primary_energy_edges"] = zone_edges
    print(f"zone table by primary: {zone_probs.shape}", flush=True)
model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
    n_types=n_types,
    line_positions_y=line_positions_y,
    latent_dim=8,
    hidden=(128, 128, 64),
    beta=0.2,
    continuum_mode="flow",
    continuum_flow_bins=24,
    continuum_flow_transforms=3,
    continuum_flow_warp="cdf",
    continuum_flow_warp_y_knots=warp_y_knots,
    continuum_flow_warp_z_knots=warp_z_knots,
    energy_flow_condition="z_cond",
    energy_condition_geometry=args.energy_condition_geometry,
    prior="coupling",
    w_gate_aux=2.0,
    gate_focal_gamma=1.0,
    gate_class_weights=None,
    line_logsigma_init=line_logsigma_init,
    line_logsigma_trainable=False,
    prior_zone_conditioning=True,
    zone_probs=zone_probs,
    cond_dim=COND_DIM,
    **atom_kwargs,
)
magi.compile_model(model, learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)
callbacks = magi.build_default_callbacks(
    monitor="val_loss", early_patience=8, lr_patience=6,
    factor=0.5, min_lr=1e-5, verbose=1,
)

t0 = time.time()
history = magi.fit_model(
    model=model, train_ds=tf_pack["train_ds"], val_ds=tf_pack["val_ds"],
    epochs=args.epochs, callbacks=callbacks, verbose=2,
    checkpoint_dir=f"{SAVE_DIR}/resume", resume=args.resume,
)
print(f"\n{SOURCE}: trained {len(history.history['loss'])} epochs in "
      f"{time.time() - t0:.0f}s", flush=True)

# ---- save -----------------------------------------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
magi.save_final_trained_model(
    model=model,
    save_dir=SAVE_DIR,
    model_name=MODEL_NAME,
    history=history,
    model_config=model.to_generation_config(),
    preprocessing_metadata=preprocessing_metadata,
    training_metadata={
        "source": SOURCE,
        "epochs_requested": args.epochs,
        "learning_rate": LEARNING_RATE,
        "clipnorm": CLIPNORM,
        "seed": SEED,
        "max_crossings": args.max_crossings,
        "split": "grouped by (PrimaryPDG, EventId)",
        "geometry_mode": GEOMETRY_TRANSFORM,
        "energy_transform": "log10",
        "device": "cpu",
        "sphere_center": list(CENTER),
        "sphere_radius": R,
    },
    callbacks=callbacks,
    notes=(
        "HENABICS primary-conditioned v0.8.2 head on the DM1.2 primary-truth "
        "set (mu- and gamma primaries, separate Geant4 runs merged here). "
        "cond = [onehot(crossing type), onehot(primary type), qt(log10 E_prim)]; "
        "generate whole showers with magi.generate_primary_groups or "
        "scripts/generate_geant_source.py --n-primaries."
    ),
)
joblib.dump(quantile_transformers, f"{SAVE_DIR}/{MODEL_NAME}_quantile_transformers.joblib")
print("saved ->", SAVE_DIR, flush=True)

if args.refit_prior:
    result = magi.refit_prior(
        model,
        tf_pack["X_cont_train_s"], condition_pack["cond_train"].numpy()
        if hasattr(condition_pack["cond_train"], "numpy") else condition_pack["cond_train"],
        tf_pack["X_cont_val_s"], condition_pack["cond_val"].numpy()
        if hasattr(condition_pack["cond_val"], "numpy") else condition_pack["cond_val"],
        epochs=args.refit_epochs, checkpoint_dir=f"{SAVE_DIR}/refit")
    magi.save_refit_model(
        model, result.prior, src_dir=SAVE_DIR, out_dir=FINAL_DIR, model_name=MODEL_NAME,
        n_types=n_types, radius=R, hidden=result.hidden, n_layers=result.n_layers,
        val_kl_before=result.val_kl_before, val_kl_after=result.val_kl_after,
        history=result.history)
    print(f"prior refit val KL {result.val_kl_before:.4f} -> {result.val_kl_after:.4f}; "
          f"saved -> {FINAL_DIR}", flush=True)
