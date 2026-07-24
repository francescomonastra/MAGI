"""Real CR+Small v0.8 run (flow continuum + coupling prior + z_cond, per-line
widths pinned to X-IFU 4 eV, w_gate_aux=2.0). Mirrors MAGI_v0_8.ipynb's Phase
A/B pipeline but runs as a script with live per-epoch logging and chunked
generation, so a long run is observable and OOM-safe (the notebook's nbconvert
execution timed out blindly at 2 h). Saves checkpoints + spectra + a recovery
report per source.
"""
import os
# Force CPU: tensorflow-metal is slower than CPU for this op mix on this Mac
# (per the v0.7.2 notebook). Hide the GPU BEFORE importing magi -> which imports
# tensorflow_probability and would otherwise initialize the Metal device, after
# which set_visible_devices no longer takes effect.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, time, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
import magi

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--gen-chunk", type=int, default=1_000_000)
parser.add_argument("--sources", nargs="+", default=["CR", "Small"])
args = parser.parse_args()

magi.initialize_environment(seed=42, cpu_only=True)
print("magi:", magi.__file__, flush=True)
print("visible GPUs:", tf.config.get_visible_devices("GPU"), flush=True)

TRAINING_DATA_DIR = "/Volumes/X10Pro/MAGI/TrainingData"
SOURCE_FILES = {
    "CR": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereCR.dat",
    "Small": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereSmall.dat",
}
center = (0.0, 0.0, -507.66); R = 100.0
X_IFU_RESOLUTION_EV = 4.0
os.makedirs("Plots", exist_ok=True)

CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed.json")
candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def chunked_generate(model, n_gen, type_probs, n_types, idx_to_type, feature_pack, chunk):
    """Generate + reconstruct in chunks to bound peak memory; returns concatenated
    E_gen and energy_component_idx_gen."""
    qt = feature_pack["quantile_transformers"]
    E_parts, idx_parts = [], []
    done = 0
    while done < n_gen:
        m = min(chunk, n_gen - done)
        gp = magi.generate_latent_outputs(model, m, type_probs, n_types=n_types, idx_to_type=idx_to_type)
        rc = magi.reconstruct_generated_features(
            gp, energy_head_mode="mixture", energy_transform="log10",
            geometry_mode="quantile_u_r_u_v_phi_r_phi_v",
            qt_u_r=qt["qt_u_r"], qt_u_v=qt["qt_u_v"], qt_phi_r=qt["qt_phi_r"], qt_phi_v=qt["qt_phi_v"])
        E_parts.append(rc["E_gen"])
        idx_parts.append(gp["energy_component_idx_gen"])
        done += m
        log(f"    generated {done:,}/{n_gen:,}")
    return np.concatenate(E_parts), np.concatenate(idx_parts)


for name in args.sources:
    log("=" * 60)
    log(f"SOURCE: {name}")
    log("=" * 60)
    path = SOURCE_FILES[name]

    df = magi.load_detector_table(filepath=path, sep=r"\s+")
    prep = magi.build_physical_features(df, center=center, radius=R)
    E = prep["features"]["Energy"].to_numpy()
    log(f"loaded {E.size:,} events")

    res = magi.detect_energy_lines(E, binning_mode="log_fixed_count", n_bins=1024,
                                  prominence_factor=3.0, window=5, candidate_lines=candidate_lines)
    matched = [m for m in res["matched_lines"] if m["count"] >= 100]
    log(f"matched lines: {[m['label'] for m in matched]}")

    feature_pack = magi.build_feature_dataframe(
        prep, energy_binning_mode="log_fixed_count", n_bins=512,
        geometry_transform="quantile_u_r_u_v_phi_r_phi_v", n_quantiles=10000,
        random_state=42, energy_transform="log10")

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
    log(f"setup done; n_types={dataset_pack['n_types']} lines={line_positions_y}")

    ey = feat["energy_y"].to_numpy()
    lls = magi.line_logsigma_from_resolution(10.0**line_positions_y, X_IFU_RESOLUTION_EV, fwhm=True)
    # Cycle 1: CDF pre-warp for the continuum flow. A monotone empirical-CDF->N(0,1)
    # standardization makes the spline knots density-proportional, so CR's sharp
    # low-E Compton edge (buried under the huge muon tail) AND Small's far, rare
    # low-E tail both land inside [-B,B] with resolution where the events are.
    # Validated on the --sparse-tail synthetic (deep-tail window 0.00 affine ->
    # 0.96 cdf). See docs/v0.8_v072_comparison.md sec 6(b).
    warp_yk, warp_zk = magi.fit_cdf_warp_knots(ey, n_knots=256, eps=1e-4)
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        n_types=dataset_pack["n_types"], line_positions_y=line_positions_y, latent_dim=8,
        hidden=(128,128,64), beta=0.2, continuum_mode="flow",
        continuum_flow_bins=24, continuum_flow_transforms=3,
        continuum_flow_warp="cdf",
        continuum_flow_warp_y_knots=warp_yk, continuum_flow_warp_z_knots=warp_zk,
        energy_flow_condition="z_cond", prior="coupling", w_gate_aux=2.0,
        # Cycle 2: focal down-weighting of the easy ~99% continuum majority in the
        # gate CE, so rare fluorescence lines (CR Al/Ni) get routed. gamma=2 lifted
        # 0.2%-rare synthetic lines 0.79->1.13 (all 0.88-1.29) with only mild
        # continuum cost; self-limiting (no blow-up, unlike inverse-freq weights).
        gate_focal_gamma=2.0,
        line_logsigma_init=lls, line_logsigma_trainable=False)
    magi.compile_model(model, learning_rate=2e-4)
    log(f"training {args.epochs} epochs (warp=cdf {warp_yk.size} knots [{warp_yk[0]:.2f},{warp_yk[-1]:.2f}]; "
        f"pinned logsigma {np.round(model._line_logsigma_clipped().numpy(),2)})")

    # Full run: validation + v0.7.2 callbacks (early stop + LR anneal on val_loss).
    # verbose=2 prints one line per epoch (captured in the log for monitoring).
    callbacks = magi.build_default_callbacks(
        monitor="val_loss", early_patience=8, lr_patience=6,
        factor=0.5, min_lr=1e-5, verbose=1)
    t_tr = time.time()
    h = model.fit(tf_pack["train_ds"], validation_data=tf_pack["val_ds"],
                  epochs=args.epochs, callbacks=callbacks, verbose=2)
    ne = len(h.history["loss"])
    log(f"  trained {ne} epochs in {time.time()-t_tr:.0f}s  "
        f"val_nll={h.history['val_energy_mixture_nll'][-1]:.3f} "
        f"val_gate_aux={h.history['val_gate_aux_loss'][-1]:.3f} "
        f"val_kl={h.history['val_kl'][-1]:.2f}")

    # Checkpoint
    save_dir = f"trained_models/v0_8_{name}"
    magi.save_final_trained_model(model=model, save_dir=save_dir, model_name=f"mix_{name}",
                                  model_config=model.to_generation_config())
    log(f"checkpoint saved -> {save_dir}/")

    # Generate + validate (chunked). Cap at 1M generated events - recovery uses
    # fractions so it stays valid, and this keeps the end-of-run "check" fast.
    n_gen = min(dataset_pack["E_idx"].size, 1_000_000)
    E_gen, comp_idx = chunked_generate(model, n_gen, dataset_pack["type_probs"],
                                       dataset_pack["n_types"], dataset_pack["idx_to_type"],
                                       feature_pack, args.gen_chunk)
    E_real = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
    recovery = magi.compute_line_integral_recovery(
        E_real, E_gen, matched, feature_pack["energy_bins"], energy_component_idx_gen=comp_idx)
    log(f"{name} line-integral recovery:")
    for r in recovery:
        log(f"    {r['label']:18s} n_real={r['n_real']:7d} n_gen={r['n_gen']:7d} "
            f"recovery={r['recovery_ratio']:.3f} comp_frac={r['component_fraction']:.5f}")

    # Spectrum plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.asarray(feature_pack["energy_bins"])
    rc, _ = np.histogram(E_real, bins=bins); gc, _ = np.histogram(E_gen, bins=bins)
    gc = gc * (E_real.size / max(E_gen.size, 1))  # scale up capped generation to real N
    ctr = 0.5*(bins[:-1]+bins[1:])
    ax.step(ctr, rc, where="mid", color="#2b6cb0", label="real")
    ax.step(ctr, gc, where="mid", color="#dd6b20", alpha=0.8, label=f"generated x{E_real.size/max(E_gen.size,1):.1f} (flow+coupling)")
    for m in matched:
        ax.axvline(m["candidate_energy_mev"], color="#38a169", lw=0.7, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Energy [MeV]"); ax.set_ylabel("counts / bin")
    ax.set_title(f"{name}: real vs generated (v0.8 flow + coupling prior, {args.epochs} ep)")
    ax.legend(fontsize=8)
    out = f"Plots/v0_8_real_{name}_spectrum.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    log(f"spectrum saved -> {out}")

log("ALL DONE")
