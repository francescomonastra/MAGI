"""Diagnostic: real per-epoch training time for the v0.8 flow+coupling model on
CR, using the exact notebook pipeline. Streams per-epoch timing so we can pick a
realistic EPOCHS (the synthetic probe badly underestimated it)."""
import time, numpy as np, tensorflow as tf, magi

magi.initialize_environment(seed=42)
print("magi:", magi.__file__)

TRAINING_DATA_DIR = "/Volumes/X10Pro/MAGI/TrainingData"
path = f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereCR.dat"
center = (0.0, 0.0, -507.66); R = 100.0

t0 = time.time()
df = magi.load_detector_table(filepath=path, sep=r"\s+")
prep = magi.build_physical_features(df, center=center, radius=R)
E = prep["features"]["Energy"].to_numpy()
print(f"[{time.time()-t0:.0f}s] loaded CR: {E.size:,} events", flush=True)

CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed.json")
candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]
res = magi.detect_energy_lines(E, binning_mode="log_fixed_count", n_bins=1024,
                              prominence_factor=3.0, window=5, candidate_lines=candidate_lines)
matched = [m for m in res["matched_lines"] if m["count"] >= 100]
print(f"[{time.time()-t0:.0f}s] matched lines: {[m['label'] for m in matched]}", flush=True)

feature_pack = magi.build_feature_dataframe(
    prep, energy_binning_mode="log_fixed_count", n_bins=512,
    geometry_transform="quantile_u_r_u_v_phi_r_phi_v", n_quantiles=10000,
    random_state=42, energy_transform="log10")
print(f"[{time.time()-t0:.0f}s] built feature dataframe", flush=True)

line_positions_y = np.log10([m["candidate_energy_mev"] for m in matched]).astype(np.float32)
E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
gate_targets = magi.build_gate_targets(E_full, feature_pack["energy_bins"], matched, resolution_mev=None)
feat = feature_pack["feat"].copy()
for j in range(gate_targets.shape[1]):
    feat[f"gate_target_{j}"] = gate_targets[:, j]
cont_cols = ("u_r_q","u_v_q","phi_r_q","phi_v_q","energy_y") + tuple(f"gate_target_{j}" for j in range(gate_targets.shape[1]))
dataset_pack = magi.filter_particle_types_continuous_geometry(feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
split_pack = magi.split_feature_data(dataset_pack, test_size_total=0.30, val_size_from_temp=0.50, random_state=42)
scaled_pack = magi.scale_continuous_features(split_pack, scale_cols=())
condition_pack = magi.build_conditioning_and_weights(scaled_pack, n_types=dataset_pack["n_types"], idx_to_type=dataset_pack["idx_to_type"], alpha=0.5)
tf_pack = magi.build_tf_datasets(condition_pack, batch_size=4096, shuffle_buffer_cap=200_000)
n_train = dataset_pack["E_idx"].size
print(f"[{time.time()-t0:.0f}s] tf datasets ready; total events {n_train:,}; n_types {dataset_pack['n_types']}", flush=True)

ey = feat["energy_y"].to_numpy()
lls = magi.line_logsigma_from_resolution(10.0**line_positions_y, 4.0, fwhm=True)
model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
    gate_focal_gamma=0.0,  # pinned: was the default before v0.8.2 flipped it
    n_types=dataset_pack["n_types"], line_positions_y=line_positions_y, latent_dim=8,
    hidden=(128,128,64), beta=0.2, continuum_mode="flow",
    continuum_flow_y_mean=float(ey.mean()), continuum_flow_y_scale=float(ey.std()),
    energy_flow_condition="z_cond", prior="coupling", w_gate_aux=2.0,
    line_logsigma_init=lls, line_logsigma_trainable=False)
magi.compile_model(model, learning_rate=2e-4)
print(f"[{time.time()-t0:.0f}s] model built; starting timed epochs", flush=True)

for e in range(3):
    te = time.time()
    model.fit(tf_pack["train_ds"], validation_data=tf_pack["val_ds"], epochs=1, verbose=0)
    print(f"  EPOCH {e+1}: {time.time()-te:.1f}s", flush=True)
print("DONE probe", flush=True)
