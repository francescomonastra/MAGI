#!/usr/bin/env python
"""3-detector IAPS model (S1GDML_DM1.2_Triple): one MAGI model per source, mu, gamma, K40, Th232 or Ra226.

Same recipe as tools/train_sron_chain.py. Only the data, the sphere and the names change.
Data: TrainingData/iaps_<SRC>_cluster.dat (tools/build_iaps_training.sh), 13 columns, first entries only,
and checkpoints/iaps_cluster_chi_<SRC>.json (thrown, crossings, chi).
Sphere (MultiVerse ball): centre (0, 0, 5) mm, R = 105 mm.

Lines: the decay-chain gammas that reach the sphere unscattered keep one exact energy, so a line is any energy
value (rounded to 1 eV) that holds at least --line-min-frac (default 0.1%) of the crossings. The mixture head gets
one pinned component per line, as in the DM1.2 and CR models.

Recipe of tools/train_cr_v1.py: supervised line routing, wide prior (128, 128), CDF warp fitted with the line
left out, clipnorm 5, 40 epochs, then the prior refit on the frozen encoder. 
    python tools/train_iaps.py --iso Th232 --seed 42 [--epochs 40] [--refit-prior] [--gpu] [--smoke N]
"""
import argparse
import json
import os
import sys
import time

T_START = time.time()
GPU = "--gpu" in sys.argv
if not GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gc

import joblib
import numpy as np

import magi

ROOT = os.environ.get("MAGI_ROOT", "/Volumes/T7/MAGI-dev")   # Colab: the Drive folder MAGI
ISO = sys.argv[sys.argv.index("--iso") + 1] if "--iso" in sys.argv else None
assert ISO in ("mu", "gamma", "K40", "Th232", "Ra226"), "--iso mu|gamma|K40|Th232|Ra226"
DATA = f"{ROOT}/TrainingData/iaps_{ISO}_cluster.dat"
CHI = json.load(open(f"{ROOT}/checkpoints/iaps_cluster_chi_{ISO}.json"))
LINES = f"{ROOT}/CandidateLines/CANDIDATE_ENERGY_LINES_CriostatoDM1_2_Richiesta11_06_2026-worldVOL_EADL_escape.json"
CENTER, RES_EV, NAME = (0.0, 0.0, 5.0), 4.0, f"mix_{ISO}"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--iso", required=True, choices=["mu", "gamma", "K40", "Th232", "Ra226"])
ap.add_argument("--line-criterion", choices=["frac", "prominence"], default="frac",
                help="frac: every exact energy holding --line-min-frac of the crossings (v1.0 Th/Ra first models). "
                     "prominence: exact energies of at least --line-min-count crossings whose energy bin passes "
                     "magi.detect_line_bins (count above --line-prominence x the local median), the DM1.2 and CR criterion")
ap.add_argument("--warp-eps", type=float, default=1e-4,
                help="probability of the outermost CDF warp knots; 1e-4 drops the lowest 0.01%% of the continuum "
                     "(the real Th/Ra crossings below 30 keV, 7e-5 of the continuum), 1e-6 reaches them")
ap.add_argument("--warp-boost", action="append", default=[], metavar="LO_MEV,HI_MEV,N",
                help="N extra warp knots uniform in probability between two energies (repeatable)")
ap.add_argument("--line-min-frac", type=float, default=0.001)
ap.add_argument("--line-min-count", type=int, default=100)
ap.add_argument("--extra-lines", default="0.007985,0.008006,0.008863,0.008865",
                help="exact energies [MeV] pinned whenever they hold at least --line-min-count crossings, below the frac threshold "
                     "(the Cu K fluorescence lines of the cryostat, 0.02 to 0.06%% of the K-40, Th-232 and Ra-226 crossings)")
ap.add_argument("--line-prominence", type=float, default=3.0)
ap.add_argument("--line-bins", type=int, default=1024)
ap.add_argument("--line-share", type=float, default=0.1,
                help="prominence criterion: an energy is dropped when a bigger energy in the same bin holds more than 1/share times its crossings "
                     "(a flagged bin would otherwise pull in the sub-0.1% neighbours of a strong line)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", default=None)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--latent-dim", type=int, default=8)
ap.add_argument("--beta", type=float, default=0.2)
ap.add_argument("--clipnorm", type=float, default=5.0)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--overwrite", action="store_true", help="allow replacing a finished model of the same tag")
ap.add_argument("--refit-prior", action="store_true")
ap.add_argument("--refit-epochs", type=int, default=60)
ap.add_argument("--gpu", action="store_true", help="use the GPU (Colab); every other trainer is CPU-only")
ap.add_argument("--dump-split", default=None, metavar="NPZ",
                help="after the data prep, save up to 300000 validation rows (X_cont_val, y_val) and 1000000 training rows to NPZ and exit without training")
ap.add_argument("--smoke", type=int, default=0, help="use the first N crossings and 1 epoch")
args = ap.parse_args()
TAG = args.tag or f"v1_0_IAPS_{ISO}_seed{args.seed}"
FINAL_DIR = f"{ROOT}/trained_models/{TAG}"
SAVE_DIR = f"{FINAL_DIR}_joint" if args.refit_prior else FINAL_DIR
if os.path.exists(f"{FINAL_DIR}/mix_{ISO}_metadata.json") and not args.overwrite and not args.smoke:
    sys.exit(f"{FINAL_DIR} holds a finished model; pick another --tag or pass --overwrite")
GEOM = "quantile_u_r_u_v_phi_r_phi_v"

magi.initialize_environment(seed=args.seed, cpu_only=not args.gpu, quiet=True)
df = magi.load_detector_table(filepath=DATA, sep=r"\s+")
if args.smoke:
    df = df.iloc[:args.smoke].reset_index(drop=True)
probe = df[["X", "Y", "Z"]].to_numpy()[:200_000]
R = float(np.median(np.linalg.norm(probe - np.asarray(CENTER), axis=1)))
print(f"{TAG}: {len(df):,} crossings, sphere radius from the data {R:.4f} mm", flush=True)
prep = magi.build_physical_features(df, center=CENTER, radius=R)
E = prep["features"]["Energy"].to_numpy().copy()
n_cross = len(df)



def select_lines(E, args):
    """Exact energies (rounded 1 eV) kept as pinned lines, with their counts."""
    vals, cnts = np.unique(np.round(E, 6), return_counts=True)
    extra = np.isin(np.round(vals, 6), np.round([float(x) for x in args.extra_lines.split(",") if x], 6)) & (cnts >= args.line_min_count)
    if args.line_criterion == "frac":
        keep = (cnts >= args.line_min_frac * len(E)) | extra
        return vals[keep], cnts[keep], f"holding >= {args.line_min_frac:.2%} of the crossings each"
    from magi.data.preprocessing import build_energy_bins, detect_line_bins
    bins = build_energy_bins(E, mode="log_fixed_count", n_bins=args.line_bins)
    edges = np.asarray(bins["edges"] if isinstance(bins, dict) else bins)
    h = np.histogram(E, edges)[0]
    flagged = np.zeros(len(h), dtype=bool)
    flagged[detect_line_bins(h, prominence_factor=args.line_prominence, window=5)[0]] = True
    b = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, len(h) - 1)
    keep = (cnts >= args.line_min_count) & flagged[b]
    top = np.zeros(len(h))
    np.maximum.at(top, b[keep], cnts[keep])
    keep &= cnts >= args.line_share * top[b]
    keep |= extra
    return vals[keep], cnts[keep], (f"of >= {args.line_min_count} crossings in a bin above {args.line_prominence:g} x the "
                                    f"local median ({args.line_bins} bins)")


lvals, lcnts, why = select_lines(E, args)
matched = [{"label": f"{ISO} chain line {v:.4f}", "candidate_energy_mev": float(v), "count": float(c)}
           for v, c in zip(lvals, lcnts)]
print(f"{len(matched)} lines {why}, {lcnts.sum() / len(E):.2%} in total:", flush=True)
for m in matched:
    print(f"   {m['candidate_energy_mev']:.6f} MeV  {m['count'] / len(E):.3%}", flush=True)
line_positions_mev = np.array([m["candidate_energy_mev"] for m in matched])
line_positions_y = np.log10(line_positions_mev).astype(np.float32)

fp = magi.build_feature_dataframe(prep, energy_binning_mode="log_fixed_count", n_bins=512,
                                  geometry_transform=GEOM, n_quantiles=10000,
                                  random_state=42, energy_transform="log10")
Ef = fp["filtered_prep"]["features"]["Energy"].to_numpy()
del df, prep
gc.collect()
gt = magi.build_gate_targets(Ef, fp["energy_bins"], matched, bandwidth_mode="resolution",
                             bandwidth_fwhm_mev=RES_EV * 1e-6)
feat = fp["feat"].copy()
for j in range(gt.shape[1]):
    feat[f"gate_target_{j}"] = gt[:, j]
cont_cols = tuple(fp["geometry_metadata"]["X_cont_cols"]) + ("energy_y",) + tuple(
    f"gate_target_{j}" for j in range(gt.shape[1]))
dp = magi.filter_particle_types_continuous_geometry(feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
fp.pop("filtered_prep", None)
fp.pop("feat", None)
del feat
gc.collect()
sp = magi.split_feature_data(dp, test_size_total=0.30, val_size_from_temp=0.50, random_state=42)
n_types, n_zones = dp["n_types"], len(line_positions_y) + 1
if args.dump_split:
    _i = np.random.default_rng(0).choice(len(sp["y_val"]), min(300_000, len(sp["y_val"])), replace=False)
    _j = np.random.default_rng(1).choice(len(sp["y_train"]), min(1_000_000, len(sp["y_train"])), replace=False)
    np.savez(args.dump_split, X_cont_val=sp["X_cont_val"][_i].astype(np.float32), y_val=sp["y_val"][_i], n_types=n_types,
             X_cont_train=sp["X_cont_train"][_j].astype(np.float32), y_train=sp["y_train"][_j])
    sys.exit(f"dumped {len(_i)} validation rows -> {args.dump_split}")
print(f"types {dp['idx_to_type']}, type_probs {np.round(dp['type_probs'], 5)}", flush=True)

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
print("zone_probs (continuum, then the lines) by type:", {dp["idx_to_type"][t]: np.round(zone_probs[t], 4).tolist()
                                                    for t in range(n_types)}, flush=True)

BOOST = [(float(np.log10(float(b.split(",")[0]))), float(np.log10(float(b.split(",")[1]))), int(b.split(",")[2]))
         for b in args.warp_boost] or None
print(f"warp: eps {args.warp_eps:g}, boost ranges {BOOST}", flush=True)
warp_y, warp_z = magi.fit_cdf_warp_knots(dp["X_cont_raw"][sp["idx_train"], 4], n_knots=256, eps=args.warp_eps,
                                         boost_ranges=BOOST, exclude_y=line_positions_y)
magi.initialize_environment(seed=args.seed, cpu_only=not args.gpu, quiet=True)
model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
    n_types=n_types, line_positions_y=line_positions_y, latent_dim=args.latent_dim,
    hidden=(128, 128, 64), beta=args.beta, continuum_mode="flow", continuum_flow_bins=24,
    continuum_flow_transforms=3, continuum_flow_warp="cdf", continuum_flow_warp_y_knots=warp_y,
    continuum_flow_warp_z_knots=warp_z, energy_flow_condition="z_cond",
    energy_line_routing="supervised", prior="coupling", prior_hidden=(128, 128),
    w_gate_aux=2.0, gate_focal_gamma=1.0, gate_class_weights=None,
    line_logsigma_init=magi.line_logsigma_from_resolution(line_positions_mev, RES_EV, fwhm=True),
    line_logsigma_trainable=False, prior_zone_conditioning=True, zone_probs=zone_probs)
magi.compile_model(model, learning_rate=args.lr, clipnorm=args.clipnorm)
callbacks = magi.build_default_callbacks(monitor="val_loss", early_patience=8, lr_patience=6,
                                         factor=0.5, min_lr=1e-5, verbose=1)
epochs = 1 if args.smoke else args.epochs
print(f"TIMING prep_s={time.time() - T_START:.1f}", flush=True)
t0 = time.time()
history = magi.fit_model(model=model, train_ds=tf_pack["train_ds"], val_ds=tf_pack["val_ds"],
                         epochs=epochs, callbacks=callbacks, verbose=2,
                         checkpoint_dir=f"{SAVE_DIR}/resume", resume=args.resume)
n_ep = len(history.history["loss"])
print(f"TIMING main_s={time.time() - t0:.1f} main_epochs={n_ep}", flush=True)
print(f"{TAG}: {n_ep} epochs in {time.time() - t0:.0f}s, best val_loss "
      f"{min(history.history['val_loss']):.4f}", flush=True)

prep_meta = {
    "source": f"IAPS_{ISO}_cluster", "source_file": DATA,
    "center": list(CENTER), "radius": R, "sphere_center": list(CENTER), "sphere_radius": R,
    "geometry_transform": GEOM, "energy_transform": "log10",
    "cont_cols": list(cont_cols), "y_cont_dim": tf_pack["X_cont_train_s"].shape[1],
    "energy_bins": fp["energy_bins"], "type_probs": dp["type_probs"], "idx_to_type": dp["idx_to_type"],
    "n_types": n_types, "energy_binning_mode": fp["energy_config"]["mode"],
    "energy_config": fp["energy_config"], "candidate_lines_file": LINES,
    "gate_target_bandwidth_mode": "resolution", "x_ifu_resolution_ev": RES_EV,
    "matched_lines": [m["label"] for m in matched]}
os.makedirs(SAVE_DIR, exist_ok=True)
magi.save_final_trained_model(
    model=model, save_dir=SAVE_DIR, model_name=NAME, history=history,
    model_config=model.to_generation_config(), preprocessing_metadata=prep_meta,
    training_metadata={"source": f"IAPS {ISO} cluster", "n_lines": len(matched), "epochs_requested": epochs, "epochs_run": n_ep,
                       "learning_rate": args.lr, "clipnorm": args.clipnorm, "seed": args.seed,
                       "split_seed": 42, "latent_dim": args.latent_dim, "beta": args.beta,
                       "n_crossings": int(n_cross), "n_thrown": int(CHI["thrown"]), "chi": CHI["chi"],
                       "device": "gpu" if args.gpu else "cpu"},
    callbacks=callbacks,
    notes=f"IAPS {ISO} model: v1.0 recipe on the 90-job Triple cluster run, lines from exact repeated energies.")
joblib.dump(fp["quantile_transformers"], f"{SAVE_DIR}/{NAME}_quantile_transformers.joblib")
print("saved ->", SAVE_DIR, flush=True)

if args.refit_prior:
    t_refit = time.time()
    c_tr = np.eye(n_types, dtype=np.float32)[sp["y_train"]]
    c_va = np.eye(n_types, dtype=np.float32)[sp["y_val"]]
    result = magi.refit_prior(model, sp["X_cont_train"], c_tr, sp["X_cont_val"], c_va,
                              epochs=args.refit_epochs,
                              checkpoint_dir=f"{ROOT}/checkpoints/refit_resume/{TAG}")
    magi.save_refit_model(model, result.prior, src_dir=SAVE_DIR, out_dir=FINAL_DIR, model_name=NAME,
                          n_types=n_types, radius=float(R), hidden=result.hidden, n_layers=result.n_layers,
                          val_kl_before=result.val_kl_before, val_kl_after=result.val_kl_after,
                          history=result.history)
    print(f"TIMING refit_s={time.time() - t_refit:.1f} refit_epochs={len(result.history)}", flush=True)
    print(f"{TAG}: prior refit val KL {result.val_kl_before:.4f} -> {result.val_kl_after:.4f}; "
          f"saved -> {FINAL_DIR}", flush=True)
