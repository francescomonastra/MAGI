#!/usr/bin/env python
"""A2 of docs/CVAE_vs_AutoRegFlow.md: the CryoSphere-CR CVAE retrained from
scratch with the fixes built in, instead of patched onto a trained run.

Reference: v0_8_2_CR_ingoingfix (AUC 0.661) and its post-hoc repairs
(v0_8_2_CR_ingoingfix_priorrefit_wide_warpfix, AUC 0.619). The two differ in
exactly three config entries, and this script sets all three from the start:

    energy_line_routing  "supervised"   line events leave the continuum flow
    prior_hidden         (128, 128)     the wide prior conditioner
    CDF warp knots       exclude_y=line positions, fitted on the train split

plus clipnorm 5.0, the v0.9.3 fix for the loss spikes of the original run
(final val loss -13.5 against a best of -14.27). Everything else is the
reference run's setting (40 epochs, lr 2e-4, latent 8, beta 0.2,
w_gate_aux 2.0, focal gamma 1.0, early stopping patience 8, ReduceLROnPlateau).

The data, the line table and the 70/15/15 split are those of
tools/refit_prior_posthoc.build_dataset (split seed 42), so the seeds of
independent replicates change the initialisation and the sampling only, and
the spread of the resulting AUC is the training-seed variance. The fitted
quantile transformers and the preprocessing metadata are copied from the
reference run, whose split and transform seed are the same.

    python tools/train_cr_v1.py --seed 42
    python tools/train_cr_v1.py --seed 42 --resume     # after an interruption
    python tools/train_cr_v1.py --seed 42 --refit-prior --resume   # the full v1.0 recipe
    python tools/train_cr_v1.py --seed 7 --tag v1_0_CR_A2_seed7
A4 uses the same script with --latent-dim / --beta.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_prior_posthoc as rp  # noqa: E402  (CPU-only environment)

import magi  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", default=None, help="run directory under trained_models/")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--latent-dim", type=int, default=8)
ap.add_argument("--beta", type=float, default=0.2)
ap.add_argument("--prior-hidden", type=int, nargs="+", default=[128, 128])
ap.add_argument("--clipnorm", type=float, default=5.0)
ap.add_argument("--resume", action="store_true",
                help="continue from <run dir>/resume if it exists (epochs is the total)")
ap.add_argument("--reset-early-stopping", action="store_true",
                help="with --resume: clear the patience counter, e.g. to carry on a "
                     "run that early stopping ended too soon (raise --epochs too)")
ap.add_argument("--refit-prior", action="store_true",
                help="last stage of the recipe: refit the prior on the frozen encoder "
                     "(magi.refit_prior). The jointly trained model is saved as "
                     "<tag>_joint and the final run as <tag>")
ap.add_argument("--local-direction", action="store_true",
                help="A3: direction in the local frame (mu, psi), geometry_transform "
                     "quantile_u_r_mu_phi_r_psi, non-inward crossings dropped; the quantile "
                     "transformers are the ones fitted here, not the reference run's")
ap.add_argument("--energy-condition-geometry", action="store_true",
                help="condition the energy head on the geometry targets (v0.8.3 flag)")
ap.add_argument("--refit-epochs", type=int, default=60)
ap.add_argument("--trace-energy-dir", default=None,
                help="write the per-event validation energy NLL of every epoch there "
                     "(tools/energy_trace_callback.py)")
args = ap.parse_args()

TAG = args.tag or f"v1_0_CR_A2_seed{args.seed}"
FINAL_DIR = f"{rp.ROOT}/trained_models/{TAG}"
SAVE_DIR = f"{FINAL_DIR}_joint" if args.refit_prior else FINAL_DIR
NAME = rp.NAME
REF = rp.SAVE  # v0_8_2_CR_ingoingfix: same data, split and transformers

print(f"=== A2 CR retrain: {TAG}, seed {args.seed} ===", flush=True)
GEOMETRY_TRANSFORM = ("quantile_u_r_mu_phi_r_psi" if args.local_direction
                      else "quantile_u_r_u_v_phi_r_phi_v")
cfg_ref, R, dp, sp, fp = rp.build_dataset(GEOMETRY_TRANSFORM, return_feature_pack=True)
line_positions_y = np.asarray(cfg_ref["line_positions_y"], dtype=np.float32).reshape(-1)
line_positions_mev = 10.0 ** line_positions_y.astype(np.float64)
n_types, n_zones = dp["n_types"], len(line_positions_y) + 1

scaled_pack = magi.scale_continuous_features(sp, scale_cols=())
condition_pack = magi.build_conditioning_and_weights(
    scaled_pack, n_types=n_types, idx_to_type=dp["idx_to_type"], alpha=0.5)
tf_pack = magi.build_tf_datasets(condition_pack, batch_size=4096, shuffle_buffer_cap=200_000)

zone_cols = dp["X_cont_raw"][:, -n_zones:]
zone_probs = np.zeros((n_types, n_zones))
for t in range(n_types):
    m = dp["y_type"] == t
    row = zone_cols[m].mean(axis=0) if m.any() else np.zeros(n_zones)
    zone_probs[t] = row / row.sum() if row.sum() > 0 else np.eye(n_zones)[0]

# The warp is fitted on the training split with the line atoms left out: an
# atom in the empirical CDF stacks quantile knots on it (v0.9.4).
energy_y_train = dp["X_cont_raw"][sp["idx_train"], 4]
warp_y_knots, warp_z_knots = magi.fit_cdf_warp_knots(
    energy_y_train, n_knots=256, eps=1e-4, exclude_y=line_positions_y)

magi.initialize_environment(seed=args.seed, cpu_only=True, quiet=True)
model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
    n_types=n_types, line_positions_y=line_positions_y,
    latent_dim=args.latent_dim, hidden=(128, 128, 64), beta=args.beta,
    continuum_mode="flow", continuum_flow_bins=24, continuum_flow_transforms=3,
    continuum_flow_warp="cdf", continuum_flow_warp_y_knots=warp_y_knots,
    continuum_flow_warp_z_knots=warp_z_knots,
    energy_flow_condition="z_cond", energy_condition_geometry=args.energy_condition_geometry,
    energy_line_routing="supervised",
    prior="coupling", prior_hidden=tuple(args.prior_hidden),
    w_gate_aux=2.0, gate_focal_gamma=1.0, gate_class_weights=None,
    line_logsigma_init=magi.line_logsigma_from_resolution(
        line_positions_mev, rp.RES_EV, fwhm=True),
    line_logsigma_trainable=False,
    prior_zone_conditioning=True, zone_probs=zone_probs)
magi.compile_model(model, learning_rate=args.lr, clipnorm=args.clipnorm)
callbacks = magi.build_default_callbacks(
    monitor="val_loss", early_patience=8, lr_patience=6, factor=0.5,
    min_lr=1e-5, verbose=1)

if args.trace_energy_dir:
    from energy_trace_callback import EnergyTrace
    callbacks.append(EnergyTrace(tf_pack["val_ds"], args.trace_energy_dir))

t0 = time.time()
history = magi.fit_model(model=model, train_ds=tf_pack["train_ds"],
                         val_ds=tf_pack["val_ds"], epochs=args.epochs,
                         callbacks=callbacks, verbose=2,
                         checkpoint_dir=f"{SAVE_DIR}/resume", resume=args.resume,
                         reset_early_stopping=args.reset_early_stopping)
n_ep = len(history.history["loss"])
print(f"{TAG}: {n_ep} epochs in {time.time() - t0:.0f}s, "
      f"best val_loss {min(history.history['val_loss']):.4f}", flush=True)

ref_meta = json.load(open(f"{REF}/{NAME}_metadata.json"))
prep_meta = dict(ref_meta["preprocessing_metadata"])
prep_meta["radius"] = float(R)
if args.local_direction:
    prep_meta["geometry_transform"] = GEOMETRY_TRANSFORM
    prep_meta["cont_cols"] = list(fp["geometry_metadata"]["X_cont_cols"]) + list(
        prep_meta.get("cont_cols", [])[4:])
    prep_meta["n_outward_dropped"] = fp["geometry_metadata"]["n_outward_dropped"]
    prep_meta["geometry_metadata"] = dict(fp["geometry_metadata"])
    prep_meta["quantile_transformer_keys"] = sorted(fp["quantile_transformers"])
    prep_meta["type_probs"] = [float(p) for p in dp["type_probs"]]
    prep_meta["idx_to_type"] = {str(k): v for k, v in dp["idx_to_type"].items()}
os.makedirs(SAVE_DIR, exist_ok=True)
magi.save_final_trained_model(
    model=model, save_dir=SAVE_DIR, model_name=NAME, history=history,
    model_config=model.to_generation_config(),
    preprocessing_metadata=prep_meta,
    training_metadata={
        "source": "CR", "epochs_requested": args.epochs, "epochs_run": n_ep,
        "learning_rate": args.lr, "clipnorm": args.clipnorm, "seed": args.seed,
        "split_seed": 42, "latent_dim": args.latent_dim, "beta": args.beta,
        "prior_hidden": list(args.prior_hidden),
        "training_set": os.path.basename(rp.DATA), "device": "cpu",
        "fixes_built_in": ["energy_line_routing=supervised", "prior_hidden",
                           "warp exclude_y", "clipnorm"]},
    callbacks=callbacks,
    notes=(f"A2 retrain of docs/CVAE_vs_AutoRegFlow.md, seed {args.seed}: the "
           "post-hoc repairs of v0_8_2_CR_ingoingfix (supervised line routing, "
           "wide prior, line-free CDF warp) and clipnorm built in from the start."))
if args.local_direction:
    import joblib
    joblib.dump(fp["quantile_transformers"], f"{SAVE_DIR}/{NAME}_quantile_transformers.joblib")
else:
    shutil.copy(f"{REF}/{NAME}_quantile_transformers.joblib",
                f"{SAVE_DIR}/{NAME}_quantile_transformers.joblib")
print("saved ->", SAVE_DIR, flush=True)

if args.refit_prior:
    X_tr, X_va = sp["X_cont_train"], sp["X_cont_val"]
    c_tr = np.eye(n_types, dtype=np.float32)[sp["y_train"]]
    c_va = np.eye(n_types, dtype=np.float32)[sp["y_val"]]
    result = magi.refit_prior(
        model, X_tr, c_tr, X_va, c_va, epochs=args.refit_epochs,
        checkpoint_dir=f"{rp.ROOT}/checkpoints/refit_resume/{TAG}")
    magi.save_refit_model(
        model, result.prior, src_dir=SAVE_DIR, out_dir=FINAL_DIR, model_name=NAME,
        n_types=n_types, radius=float(R), hidden=result.hidden, n_layers=result.n_layers,
        val_kl_before=result.val_kl_before, val_kl_after=result.val_kl_after,
        history=result.history)
    print(f"{TAG}: prior refit val KL {result.val_kl_before:.4f} -> {result.val_kl_after:.4f}; "
          f"saved -> {FINAL_DIR}", flush=True)
