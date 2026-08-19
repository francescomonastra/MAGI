#!/usr/bin/env python
"""Does coupling the energy head to sampled geometry close the joint gap?

THE DEFECT (19/08). A classifier separates real from MAGI crossings at AUC 0.68,
with importance spread evenly over all six continuous variables and ~zero on
species: a joint/correlation mismatch, not a broken marginal. The sharpest
instance is a coverage hole at 511 keV, horizontal, near the sphere equator --
the model has the line and has the angular distribution but not their joint,
because the heads are conditionally independent given an 8-dim latent.

THE TEST. energy_condition_geometry (the v0.8.3 feature, implemented, tested and
NOT adopted) factorises p(geometry|z) * p(energy|geometry,z) with teacher forcing
in training and sampled geometry at generation. It is exactly the mechanism this
defect calls for, and it has never been scored against a metric that can see the
defect. Train both variants on IDENTICAL data, split, transformers and seed --
only the flag differs -- and score by AUC.

WHY THIS IS WORTH AN AFTERNOON. Every architecture decision so far was judged on
marginals, or on a downstream R carrying +/-0.09. AUC on 4e5 samples per class
has negligible statistical error and is directly sensitive to the coupling.
"""
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/Volumes/X10Pro/MAGI/MAGI_package")

import numpy as np
import pandas as pd
import magi

MAGI_DIR = "/Volumes/X10Pro/MAGI"
TRAIN = f"{MAGI_DIR}/TrainingData/alloutputDSCryoSphereCR_ingoingfix.dat"
REF = json.load(open(f"{MAGI_DIR}/trained_models/v0_8_2_CR_ingoingfix/mix_CR_config.json"))
OUT = f"{MAGI_DIR}/trained_models"
CENTER = (0.0, 0.0, -507.66)
SEED, EPOCHS = 42, 40
X_IFU_EV = 4.0

LINE_Y = np.asarray(REF["line_positions_y"], dtype=np.float32)
LINE_MEV = (10.0 ** LINE_Y).astype(np.float64)
MATCHED = [{"label": f"line_{i}", "candidate_energy_mev": float(e)}
           for i, e in enumerate(LINE_MEV)]

print("loading CR training set ...", flush=True)
rows = []
with open(TRAIN) as f:
    for line in f:
        a = line.split()
        if len(a) >= 13:
            rows.append((a[1], float(a[2]), float(a[7]), float(a[8]), float(a[9]),
                         float(a[10]), float(a[11]), float(a[12])))
df = pd.DataFrame(rows, columns=["ParticleName", "Energy", "X", "Y", "Z", "Vx", "Vy", "Vz"])
R = float(np.median(np.linalg.norm(
    df[["X", "Y", "Z"]].to_numpy(float) - np.asarray(CENTER), axis=1)))
print(f"{len(df):,} crossings, measured R = {R:.4f} mm", flush=True)

# ---- prepared ONCE, so both variants see byte-identical inputs --------------
prep = magi.build_physical_features(df, center=CENTER, radius=R)
fp = magi.build_feature_dataframe(
    prep, energy_binning_mode="log_fixed_count", n_bins=512,
    geometry_transform="quantile_u_r_u_v_phi_r_phi_v",
    n_quantiles=10000, random_state=SEED, energy_transform="log10")
E_full = fp["filtered_prep"]["features"]["Energy"].to_numpy()
gt = magi.build_gate_targets(E_full, fp["energy_bins"], MATCHED,
                             bandwidth_mode="resolution",
                             bandwidth_fwhm_mev=X_IFU_EV * 1e-6)
feat = fp["feat"].copy()
for j in range(gt.shape[1]):
    feat[f"gate_target_{j}"] = gt[:, j]
cont_cols = ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q", "energy_y") + tuple(
    f"gate_target_{j}" for j in range(gt.shape[1]))
dp = magi.filter_particle_types_continuous_geometry(
    feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
n_types, idx_to_type = dp["n_types"], dp["idx_to_type"]
nz = gt.shape[1]
zc, yt = dp["X_cont_raw"][:, -nz:], dp["y_type"]
zone_probs = np.zeros((n_types, nz))
for t in range(n_types):
    m = (yt == t)
    row = zc[m].mean(axis=0) if m.any() else np.zeros(nz)
    zone_probs[t] = row / row.sum() if row.sum() > 0 else np.eye(nz)[0]
sp = magi.split_feature_data(dp, test_size_total=0.30, val_size_from_temp=0.50,
                             random_state=SEED)
sc = magi.scale_continuous_features(sp, scale_cols=())
cp = magi.build_conditioning_and_weights(sc, n_types=n_types,
                                         idx_to_type=idx_to_type, alpha=0.5)
tfp = magi.build_tf_datasets(cp, batch_size=4096, shuffle_buffer_cap=200_000)
print(f"prepared once: n_types={n_types}, zones={nz}, "
      f"n_train={len(sp['idx_train']):,}", flush=True)


def train(coupled):
    tag = "coupled" if coupled else "baseline"
    t0 = time.time()
    magi.initialize_environment(seed=SEED, cpu_only=True, quiet=True)
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        n_types=n_types, line_positions_y=LINE_Y,
        latent_dim=REF["latent_dim"], hidden=tuple(REF["hidden"]), beta=REF["beta"],
        continuum_mode=REF["continuum_mode"],
        continuum_flow_bins=REF["continuum_flow_bins"],
        continuum_flow_transforms=REF["continuum_flow_transforms"],
        continuum_flow_warp=REF["continuum_flow_warp"],
        continuum_flow_warp_y_knots=np.asarray(REF["continuum_flow_warp_y_knots"]),
        continuum_flow_warp_z_knots=np.asarray(REF["continuum_flow_warp_z_knots"]),
        energy_flow_condition=REF["energy_flow_condition"], prior=REF["prior"],
        w_gate_aux=2.0, gate_focal_gamma=REF["gate_focal_gamma"],
        gate_class_weights=REF.get("gate_class_weights"),
        line_logsigma_init=magi.line_logsigma_from_resolution(LINE_MEV, X_IFU_EV, fwhm=True),
        line_logsigma_trainable=False,
        prior_zone_conditioning=REF["prior_zone_conditioning"],
        zone_probs=zone_probs,
        energy_condition_geometry=coupled)          # <-- the only difference
    magi.compile_model(model, learning_rate=2e-4)
    cbs = magi.build_default_callbacks()
    h = magi.fit_model(model=model, train_ds=tfp["train_ds"], val_ds=tfp["val_ds"],
                       epochs=EPOCHS, callbacks=cbs, verbose=2)
    sd = f"{OUT}/coupling_{tag}"
    os.makedirs(sd, exist_ok=True)
    magi.save_final_trained_model(
        model=model, save_dir=sd, model_name="mix_CR", history=h,
        model_config=model.to_generation_config(),
        preprocessing_metadata={
            "source": "CR", "center": list(CENTER), "radius": R,
            "geometry_transform": "quantile_u_r_u_v_phi_r_phi_v",
            "energy_transform": "log10", "n_types": n_types,
            "idx_to_type": idx_to_type, "type_probs": list(np.asarray(dp["type_probs"])),
            "cont_cols": list(cont_cols),
            "energy_bins": list(np.asarray(fp["energy_bins"]).ravel()),
            "geometry_metadata": fp.get("geometry_metadata")},
        training_metadata={"energy_condition_geometry": coupled, "seed": SEED,
                           "epochs_run": len(h.history["loss"]), "device": "m1-cpu",
                           "wall_s": round(time.time() - t0)},
        callbacks=cbs,
        notes=f"Head-coupling A/B ({tag}). Identical data/split/transformers/seed; "
              f"only energy_condition_geometry differs.")
    import joblib
    joblib.dump(fp["quantile_transformers"], f"{sd}/mix_CR_quantile_transformers.joblib")
    print(f"[{tag}] epochs={len(h.history['loss'])} "
          f"val_loss={h.history['val_loss'][-1]:+.4f} "
          f"val_kl={h.history['val_kl'][-1]:.3f} "
          f"{round(time.time()-t0)}s -> {sd}", flush=True)
    return {"tag": tag, "coupled": coupled, "save_dir": sd,
            "epochs": len(h.history["loss"]),
            "val_loss": float(h.history["val_loss"][-1]),
            "val_kl": float(h.history["val_kl"][-1]),
            "wall_s": round(time.time() - t0)}


res = [train(False), train(True)]
json.dump(res, open(f"{MAGI_DIR}/coupling_test/coupling_ab.json", "w"), indent=2)
print("\nTRAINING DONE")
for r in res:
    print(" ", r)
