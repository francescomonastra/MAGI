#!/usr/bin/env python
"""Train MAGI v0.8.2 on the DM1.2 500k CryoSphere training set.

Mirrors MAGI_v0_8_1_DM1.2.ipynb cells 4-40 with the v0.8.2 head, run as a
script so it can go in the background.

Two things differ from the SRON models and MUST NOT be copied from them:

  * the DM1.2 CryoSphere is centre (0, 0, 5) mm, R = 105 mm, against SRON's
    (0, 0, -507.6) mm, R = 100 mm. `center`/`radius` feed
    build_physical_features and reconstruct_generated_physics, so a wrong
    sphere silently produces wrong u_r/phi_r and unusable generated positions.
  * the crossings file has 13 columns with position at 7-9 and direction at
    10-12, against SRON's 3-5 and 6-8.

Source data: S1GDML_DM1.2/allDSCryoSphere_DM1_2_iso.dat
  353,026 crossings from 500,000 muons, neutrino-free, 0.00% outgoing.
  mu- 78.09%, gamma 14.93%, e- 6.96%, e+ 0.03%.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time

import joblib
import numpy as np

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import magi

# --------------------------------------------------------------------------
SOURCE = "DM1_2_iso"
SOURCE_FILE = ("/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDML_DM1.2/"
               "allDSCryoSphere_DM1_2_iso.dat")

CENTER = (0.0, 0.0, 5.0)      # GDMLDetectorConstruction.cc:167
R = 95.0                      # POST-FIX radius - the overlapping R=105 sphere lost 39% of crossings

X_IFU_RESOLUTION_EV = 4.0
CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
                        "CANDIDATE_ENERGY_LINES_"
                        "CriostatoDM1_2_Richiesta11_06_2026-worldVOL_EADL_escape.json")

EPOCHS = 40
LEARNING_RATE = 2e-4
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SAVE_DIR = f"/Volumes/X10Pro/MAGI/trained_models/v0_8_2_{SOURCE}_seed{SEED}"
MODEL_NAME = "mix_DM1_2_iso"
# --------------------------------------------------------------------------

magi.initialize_environment(seed=SEED, cpu_only=True)
print(f"magi {magi.__version__} from {magi.__file__}", flush=True)

# ---- data ----------------------------------------------------------------
df = magi.load_detector_table(filepath=SOURCE_FILE, sep=r"\s+")
magi.report_basic_table_checks(df)

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
feature_pack = magi.build_feature_dataframe(
    prep,
    energy_binning_mode="log_fixed_count",
    n_bins=512,
    geometry_transform="quantile_u_r_u_v_phi_r_phi_v",
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

feat = feature_pack["feat"].copy()
for j in range(gate_targets.shape[1]):
    feat[f"gate_target_{j}"] = gate_targets[:, j]

cont_cols = ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q", "energy_y") + tuple(
    f"gate_target_{j}" for j in range(gate_targets.shape[1])
)

dataset_pack = magi.filter_particle_types_continuous_geometry(
    feat=feat, prob_threshold=1e-5, cont_cols=cont_cols,
)
magi.report_continuous_geometry_features(dataset_pack)

split_pack = magi.split_feature_data(
    dataset_pack, test_size_total=0.30, val_size_from_temp=0.50, random_state=SEED,
)
magi.report_split_summary(split_pack, n_types=dataset_pack["n_types"])

scaled_pack = magi.scale_continuous_features(split_pack, scale_cols=())
condition_pack = magi.build_conditioning_and_weights(
    scaled_pack, n_types=dataset_pack["n_types"],
    idx_to_type=dataset_pack["idx_to_type"], alpha=0.5,
)
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

zone_labels = ["continuum"] + [m["label"] for m in matched]
print(f"\nzone_probs ({zone_labels}):")
for t, tname in dataset_pack["idx_to_type"].items():
    print(f"    {tname:12s} {np.round(zone_probs[t], 4)}")

# ---- CDF pre-warp knots ---------------------------------------------------
energy_y_all = dataset_pack["feat"]["energy_y"].to_numpy()
warp_y_knots, warp_z_knots = magi.fit_cdf_warp_knots(energy_y_all, n_knots=256,
                                                     eps=1e-4)

n_types = dataset_pack["n_types"]
Y_CONT_DIM = tf_pack["X_cont_train_s"].shape[1]

# ---- model ----------------------------------------------------------------
line_logsigma_init = magi.line_logsigma_from_resolution(
    line_positions_mev, X_IFU_RESOLUTION_EV, fwhm=True,
)

preprocessing_metadata = {
    "source": SOURCE,
    "source_file": SOURCE_FILE,
    # `center`/`radius` are the names scripts/generate_geant_source.py looks up
    # to reconstruct physical (x,y,z) from generated (u_r, phi_r). Spelling them
    # `sphere_center`/`sphere_radius` only made those lookups miss and silently
    # fall back to the SRON sphere, emitting every DM1.2 particle 512 mm away.
    # Both spellings are written now; `center`/`radius` are the operative ones.
    "center": list(CENTER),
    "radius": R,
    "sphere_center": list(CENTER),
    "sphere_radius": R,
    "geometry_transform": "quantile_u_r_u_v_phi_r_phi_v",
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
}

magi.initialize_environment(seed=SEED, cpu_only=True, quiet=True)
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
    prior="coupling",
    w_gate_aux=2.0,
    gate_focal_gamma=1.0,
    gate_class_weights=None,
    line_logsigma_init=line_logsigma_init,
    line_logsigma_trainable=False,
    prior_zone_conditioning=True,
    zone_probs=zone_probs,
)
magi.compile_model(model, learning_rate=LEARNING_RATE)
callbacks = magi.build_default_callbacks(
    monitor="val_loss", early_patience=8, lr_patience=6,
    factor=0.5, min_lr=1e-5, verbose=1,
)

t0 = time.time()
history = magi.fit_model(
    model=model, train_ds=tf_pack["train_ds"], val_ds=tf_pack["val_ds"],
    epochs=EPOCHS, callbacks=callbacks, verbose=2,
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
        "epochs_requested": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "geometry_mode": "quantile_u_r_u_v_phi_r_phi_v",
        "energy_transform": "log10",
        "device": "cpu",
        "sphere_center": list(CENTER),
        "sphere_radius": R,
    },
    callbacks=callbacks,
    notes=(
        "v0.8.2 head trained on the DM1.2 500k CryoSphere set (353,026 crossings "
        "from 500,000 muons, neutrino-free, 0.00% outgoing). DM1.2 sphere is "
        "centre (0,0,5) mm R=105 mm - NOT the SRON (0,0,-507.6) R=100 sphere - "
        "and the crossings file is 13-column with position at 7-9 and direction "
        "at 10-12. Candidate lines come from the DM1.2 GDML with escape peaks."
    ),
)
joblib.dump(quantile_transformers, f"{SAVE_DIR}/{MODEL_NAME}_quantile_transformers.joblib")
print("saved ->", SAVE_DIR, flush=True)
