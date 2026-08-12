#!/usr/bin/env python
"""Train a v0.8.3 CR model at a given seed.  usage: train_v083_cr_seed.py <seed>

Why seeds matter here: three v0.8.2 CR checkpoints differing ONLY in training
seed give median-E-vs-b ratios at b<10mm of 1.222 (seed42), 1.057 (seed7) and
0.495 (seed13), while every energy decade stays within a few percent in all
three. The energy x geometry coupling is therefore a high-variance property of
a training run, not a fixed property of the architecture - so a single v0.8.3
seed cannot demonstrate a fix. This script exists to produce the replicates.
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

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SOURCE_FILE = "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR.dat"
CENTER = (0.0, 0.0, -507.66)
R = 100.0
X_IFU_RESOLUTION_EV = 4.0
FWHM_MEV = X_IFU_RESOLUTION_EV * 1e-6
CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
                        "CANDIDATE_ENERGY_LINES_"
                        "SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json")
EPOCHS = 40
LEARNING_RATE = 2e-4
LATENT_DIM = 12
SAVE_DIR = f"/Volumes/X10Pro/MAGI/trained_models/v0_8_3_geomcond_CR_seed{SEED}"
MODEL_NAME = "mix_CR"

magi.initialize_environment(seed=SEED, cpu_only=True)
print(f"=== v0.8.3 CR, seed {SEED} ===", flush=True)

df = magi.load_detector_table(filepath=SOURCE_FILE, sep=r"\s+")
prep = magi.build_physical_features(df, center=CENTER, radius=R)
E_all = prep["features"]["Energy"].to_numpy()

candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]
line_result = magi.detect_energy_lines(
    E_all, binning_mode="log_fixed_count", n_bins=1024, prominence_factor=3.0,
    window=5, candidate_lines=candidate_lines, refine_bin_width_mev=FWHM_MEV)
matched = [m for m in line_result["matched_lines"] if m["count"] >= 100]
print("lines:", [m["label"] for m in matched], flush=True)

feature_pack = magi.build_feature_dataframe(
    prep, energy_binning_mode="log_fixed_count", n_bins=512,
    geometry_transform="quantile_u_r_u_v_phi_r_phi_v",
    n_quantiles=10000, random_state=SEED, energy_transform="log10")

line_positions_mev = np.array([m["candidate_energy_mev"] for m in matched], float)
line_positions_y = np.log10(line_positions_mev).astype(np.float32)
E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
gate_targets = magi.build_gate_targets(
    E_full, feature_pack["energy_bins"], matched,
    bandwidth_mode="resolution", bandwidth_fwhm_mev=FWHM_MEV)

feat = feature_pack["feat"].copy()
for j in range(gate_targets.shape[1]):
    feat[f"gate_target_{j}"] = gate_targets[:, j]
cont_cols = ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q", "energy_y") + tuple(
    f"gate_target_{j}" for j in range(gate_targets.shape[1]))
assert cont_cols[:4] == ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q")

dataset_pack = magi.filter_particle_types_continuous_geometry(
    feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
split_pack = magi.split_feature_data(
    dataset_pack, test_size_total=0.30, val_size_from_temp=0.50, random_state=SEED)
scaled_pack = magi.scale_continuous_features(split_pack, scale_cols=())
condition_pack = magi.build_conditioning_and_weights(
    scaled_pack, n_types=dataset_pack["n_types"],
    idx_to_type=dataset_pack["idx_to_type"], alpha=0.5)
tf_pack = magi.build_tf_datasets(condition_pack, batch_size=4096,
                                 shuffle_buffer_cap=200_000)

n_zones = gate_targets.shape[1]
zone_cols = dataset_pack["X_cont_raw"][:, -n_zones:]
zone_probs = np.zeros((dataset_pack["n_types"], n_zones))
for t in range(dataset_pack["n_types"]):
    m = (dataset_pack["y_type"] == t)
    row = zone_cols[m].mean(axis=0) if m.any() else np.zeros(n_zones)
    zone_probs[t] = row / row.sum() if row.sum() > 0 else np.eye(n_zones)[0]

warp_y_knots, warp_z_knots = magi.fit_cdf_warp_knots(
    dataset_pack["feat"]["energy_y"].to_numpy(), n_knots=256, eps=1e-4)

magi.initialize_environment(seed=SEED, cpu_only=True, quiet=True)
model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
    n_types=dataset_pack["n_types"], line_positions_y=line_positions_y,
    latent_dim=LATENT_DIM, hidden=(128, 128, 64), beta=0.2,
    continuum_mode="flow", continuum_flow_bins=24, continuum_flow_transforms=3,
    continuum_flow_warp="cdf", continuum_flow_warp_y_knots=warp_y_knots,
    continuum_flow_warp_z_knots=warp_z_knots,
    energy_flow_condition="z_cond",
    energy_condition_geometry=True,
    prior="coupling", w_gate_aux=2.0, gate_focal_gamma=1.0,
    gate_class_weights=None,
    line_logsigma_init=magi.line_logsigma_from_resolution(
        line_positions_mev, X_IFU_RESOLUTION_EV, fwhm=True),
    line_logsigma_trainable=False,
    prior_zone_conditioning=True, zone_probs=zone_probs)
magi.compile_model(model, learning_rate=LEARNING_RATE)
callbacks = magi.build_default_callbacks(
    monitor="val_loss", early_patience=8, lr_patience=6, factor=0.5,
    min_lr=1e-5, verbose=1)

t0 = time.time()
history = magi.fit_model(model=model, train_ds=tf_pack["train_ds"],
                         val_ds=tf_pack["val_ds"], epochs=EPOCHS,
                         callbacks=callbacks, verbose=2)
print(f"seed {SEED}: {len(history.history['loss'])} epochs in {time.time()-t0:.0f}s",
      flush=True)

os.makedirs(SAVE_DIR, exist_ok=True)
magi.save_final_trained_model(
    model=model, save_dir=SAVE_DIR, model_name=MODEL_NAME, history=history,
    model_config=model.to_generation_config(),
    preprocessing_metadata={
        "source": "CR", "source_file": SOURCE_FILE,
        "center": list(CENTER), "radius": R,
        "geometry_transform": "quantile_u_r_u_v_phi_r_phi_v",
        "energy_transform": "log10", "cont_cols": list(cont_cols),
        "y_cont_dim": tf_pack["X_cont_train_s"].shape[1],
        "energy_bins": feature_pack["energy_bins"],
        "type_probs": dataset_pack["type_probs"],
        "idx_to_type": dataset_pack["idx_to_type"],
        "n_types": dataset_pack["n_types"],
        "energy_binning_mode": feature_pack["energy_config"]["mode"],
        "energy_config": feature_pack["energy_config"],
        "candidate_lines_file": CANDIDATE_LINES_FILE,
        "gate_target_bandwidth_mode": "resolution",
        "x_ifu_resolution_ev": X_IFU_RESOLUTION_EV,
        "matched_lines": [m["label"] for m in matched]},
    training_metadata={"source": "CR", "epochs_requested": EPOCHS,
                       "learning_rate": LEARNING_RATE, "seed": SEED,
                       "latent_dim": LATENT_DIM,
                       "energy_condition_geometry": True, "device": "cpu"},
    callbacks=callbacks,
    notes=(f"v0.8.3 CR replicate, seed {SEED}. energy_condition_geometry=True, "
           f"latent_dim={LATENT_DIM}. Exists because the energy x geometry "
           "coupling is seed-variable in v0.8.2 (b<10mm ratio 0.495/1.057/1.222 "
           "over three seeds), so a single seed cannot demonstrate a fix."))
joblib.dump(feature_pack["quantile_transformers"],
            f"{SAVE_DIR}/{MODEL_NAME}_quantile_transformers.joblib")
print("saved ->", SAVE_DIR, flush=True)
