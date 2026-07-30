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
import argparse, time, numpy as np, joblib
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
parser.add_argument("--run-tag", default="v0_8",
                    help="prefix for checkpoint dirs and plot filenames "
                         "(trained_models/<tag>_<source>/, Plots/<tag>_real_<source>_*). "
                         "Use a new tag per configuration - a run with the same tag "
                         "OVERWRITES the previous checkpoint's weights.")
parser.add_argument("--seed", type=int, default=42,
                    help="RNG seed; runs with seed != 42 checkpoint to a _seed<N> "
                         "directory so the reference run is never overwritten.")
parser.add_argument("--n-gen", type=int, default=0,
                    help="Events to generate for validation. 0 (default) = the real "
                         "event count. Generating fewer used to silently deflate the "
                         "line-recovery ratio; compute_line_integral_recovery now "
                         "normalizes for it, but matching the real count keeps the "
                         "spectra and the sparse-tail Poisson noise honest.")
parser.add_argument("--continuum-flow-bins", type=int, default=24,
                    help="RQS spline bins per transform for the continuum flow. "
                         "v0.8.1 Phase 3 attempt #1 found 32 degraded every CR "
                         "Wasserstein band incl. the untouched high-E/muon one when "
                         "combined with --warp-boost-cr in the same run; this flag "
                         "lets the two levers be tested independently. See "
                         "docs/v0.8.1_line_truth.md section 11.")
parser.add_argument("--warp-boost-cr", action="store_true",
                    help="Add extra CDF-warp knots across CR's 0.39-13.2 keV line "
                         "band (log10 MeV in [-3.41,-1.88], +64 knots). No-op for "
                         "other sources. See docs/v0.8.1_line_truth.md section 11.")
parser.add_argument("--gate-target-bandwidth-mode", default="exact",
                    choices=["bins", "resolution", "exact"],
                    help="How magi.build_gate_targets decides which real events "
                         "are 'line' events. 'exact' (new default) matches the "
                         "simulation's own ground truth: fluorescence lines are "
                         "exactly monoenergetic in the raw data, so membership is "
                         "a tight numerical match, not a detector-resolution "
                         "kernel. Pass 'resolution' to reproduce pre-v0.8.2 runs. "
                         "See docs/v0.8.1_line_truth.md section 14.2.")
parser.add_argument("--prior-zone-conditioning", action="store_true",
                    help="v0.8.2 Phase C candidate 1: widen the coupling prior's "
                         "conditioning with the per-event [continuum, line_1..line_L] "
                         "zone (real at train time, sampled from each type's "
                         "empirical zone frequency at generation) instead of "
                         "particle type alone. See docs/v0.8.2_RoadmapForAdoption.md "
                         "section 6 and docs/v0.8.1_line_truth.md section 13.2.")
args = parser.parse_args()

magi.initialize_environment(seed=args.seed, cpu_only=True)
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
    # v0.8.1: EADL energies (what Geant4 actually emitted). The Bearden table
    # this replaced put every fluorescence line 4-11 detector FWHM off.
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json")
candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]

# v0.8.1 remainder item 2: per-line gate_class_weights.
# CALIBRATION ATTEMPTED AND REVERTED - deliberately empty, see section 10 of
# docs/v0.8.1_line_truth.md for the full measurement. The weights below were
# tried on both sources (trained_models/v0_8_1_gatew_{CR,Small}) and did not
# work:
#
#   CR:    511  w=0.7631 -> routing 1.721 -> 1.424   (-17%)
#          Al Ka1 w=1.7613 -> routing 0.319 -> 0.381 (+19%)
#          Cu Kbeta w=1.0 (CONTROL, untouched) -> 0.893 -> 1.129 (+27%)
#   Small: 511  w=1.3424 -> routing 0.557 -> 2.180   (+292%, overshot past 1.0)
#
# The CR control line - weight never changed - moved MORE (+27%) than either
# deliberately-weighted line (-17%, +19%). So on CR the calibration's effect is
# not distinguishable from run-to-run variance. And the implied response
# exponent (routing ratio = w^beta) is 0.31, 0.50, 0.70, 0.90, 4.64 across the
# five weighted lines - a 15x spread, so there is no stable gain to iterate on.
# Coupling residuals also degraded on both sources (CR 0.042 -> 0.050, at the
# threshold; Small 0.023 -> 0.043) and no acceptance verdict changed anywhere.
#
# Root cause: the calibration assumed routing is set by the gate CE alone. It
# is not - the mixture NLL (the dominant reconstruction term), the gate CE
# (w_gate_aux=2.0) and the continuum flow all compete for the same density, so
# this knob is a soft nudge on one of several terms rather than a controller
# with predictable gain.
#
# PREREQUISITE before retrying: multi-seed, to establish the actual run-to-run
# noise floor per line (Phase 4 item 17, which this result promotes to a
# blocker rather than a follow-up - a +-27% measurement cannot calibrate a
# +-20% effect). The plumbing below is correct and tested; drop a calibrated
# table back in once the noise floor is known.
GATE_CLASS_WEIGHTS_BY_LABEL = {}

# v0.8.1 Phase 3 (continuum polish) attempt #1: CR's worst per-band Wasserstein
# score is [-3.41,-1.88) in log10(E/MeV) = 0.39-13.2 keV, ~5x the global score,
# and every modelled CR fluorescence line lives inside it. Tried a CR-only CDF-warp
# knot boost there (--run-tag v0_8_1_phase3) together with continuum_flow_bins
# 24->32 on both sources. NEGATIVE RESULT: every CR band got worse (target band
# +14%, global +23%, the high-E/muon guard band +45%, coupling 0.041->0.048), and
# the two levers were confounded in one run so neither can be attributed. Small
# (bins bump only) was mixed to mildly positive. See docs/v0.8.1_line_truth.md
# section 11 for the full table. Reverted to empty here; the plumbing in
# magi.fit_cdf_warp_knots(boost_ranges=...) is correct and tested - only the
# calibration is withdrawn. trained_models/v0_8_1_cuka2_CR and v0_8_1_Small
# remain the reference checkpoints, NOT the v0_8_1_phase3_* ones. Now
# controllable via --continuum-flow-bins / --warp-boost-cr to split the two
# confounded levers apart.
WARP_BOOST_RANGES_BY_SOURCE = {"CR": [(-3.41, -1.88, 64)]} if args.warp_boost_cr else {}

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
                                  prominence_factor=3.0, window=5, candidate_lines=candidate_lines,
                                  # v0.8.1: refine each peak to the pinned line
                                  # width before matching - a detection bin is
                                  # 21 keV wide at 511 keV on CR's log grid.
                                  refine_bin_width_mev=X_IFU_RESOLUTION_EV * 1e-6)
    matched = [m for m in res["matched_lines"] if m["count"] >= 100]
    # v0.8.1 Cu Kalpha2: detect_energy_lines's coarse bins (166 eV at 8 keV)
    # can never separate a 21 eV doublet, so Cu Kalpha2 never becomes a
    # detected peak regardless of matching logic - confirmed real (3,940 CR
    # events, 0.0 FWHM off) by the fine eV-resolution measurement instead.
    # See docs/v0.8.1_line_truth.md section 6.2.
    fine_matched = magi.confirm_unresolved_candidate_lines(
        E, candidate_lines, matched, resolution_ev=X_IFU_RESOLUTION_EV)
    if fine_matched:
        log(f"fine-detected lines (unresolved by coarse binning): "
            f"{[m['label'] for m in fine_matched]}")
        matched += fine_matched
    log(f"matched lines: {[m['label'] for m in matched]}")

    feature_pack = magi.build_feature_dataframe(
        prep, energy_binning_mode="log_fixed_count", n_bins=512,
        geometry_transform="quantile_u_r_u_v_phi_r_phi_v", n_quantiles=10000,
        random_state=42, energy_transform="log10")

    line_positions_y = np.log10([m["candidate_energy_mev"] for m in matched]).astype(np.float32)
    # v0.8.1 remainder item 2: [continuum, line_1..line_L] in the same order as
    # `matched` / line_positions_y; unlisted labels default to the neutral 1.0.
    gcw_by_label = GATE_CLASS_WEIGHTS_BY_LABEL.get(name, {})
    gate_class_weights = [1.0] + [gcw_by_label.get(m["label"], 1.0) for m in matched]
    log(f"gate_class_weights: {dict(zip(['continuum'] + [m['label'] for m in matched], gate_class_weights))}")
    E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
    # v0.8.1 Phase 2: gate-target bandwidth from the pinned line width, not the
    # detection bin width. The bin-width bandwidth was +/-43 keV at 511 keV and
    # +/-320 eV at 8.9 keV (31-10000 detector FWHM), labelling 3.5-4.6x too many
    # events as line events and training the gate to over-route by that factor.
    # Resolution mode labels 0.98x the true line fraction on both sources.
    #
    # v0.8.2 Phase C follow-up (docs/v0.8.1_line_truth.md section 14.2): even
    # "resolution" mode is wrong in a subtler way - it uses the DETECTOR's
    # resolution as a Gaussian kernel to decide which real events are line
    # events, but the raw simulation has no detector response applied. A
    # fluorescence line's real events are exactly monoenergetic (confirmed:
    # CR's Cu Kalpha1 has 7,542 events at the identical float64 energy,
    # Kalpha2 3,939 at another, everything else nearby a singleton), so any
    # event not at that exact value is genuine continuum, not a broadened
    # line photon. "exact" mode assigns membership by a tight numerical
    # match instead of a physical-width kernel - the 4 eV detector FWHM still
    # applies exactly where it always did (line_logsigma_init below), only
    # the training LABEL changes. This matters most for adjacent lines: Cu
    # Kalpha1/Kalpha2 are 21 eV apart, well inside "resolution" mode's ~5 eV
    # kernel width, which was blurring their identification into each other
    # and is the suspected cause of Cu Kalpha1's routing overshoot after the
    # zone-conditioning fix (section 14.1) - 0.688->2.052, correct direction,
    # wrong magnitude.
    gate_targets = magi.build_gate_targets(
        E_full, feature_pack["energy_bins"], matched,
        bandwidth_mode=args.gate_target_bandwidth_mode,
        bandwidth_fwhm_mev=X_IFU_RESOLUTION_EV * 1e-6)
    feat = feature_pack["feat"].copy()
    for j in range(gate_targets.shape[1]):
        feat[f"gate_target_{j}"] = gate_targets[:, j]
    cont_cols = ("u_r_q","u_v_q","phi_r_q","phi_v_q","energy_y") + tuple(f"gate_target_{j}" for j in range(gate_targets.shape[1]))
    dataset_pack = magi.filter_particle_types_continuous_geometry(feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)

    # v0.8.2 Phase C candidate 1: each type's empirical [continuum, line_1..
    # line_L] zone frequency, from the same real gate_target columns just
    # appended - used only if --prior-zone-conditioning is set, to sample the
    # prior's zone at generation time (there is no real event to read it from
    # then). Computed unconditionally (cheap) so it's available either way.
    n_zones = gate_targets.shape[1]
    zone_cols = dataset_pack["X_cont_raw"][:, -n_zones:]
    y_type_for_zones = dataset_pack["y_type"]
    zone_probs = np.zeros((dataset_pack["n_types"], n_zones), dtype=np.float64)
    for t in range(dataset_pack["n_types"]):
        mask = (y_type_for_zones == t)
        row = zone_cols[mask].mean(axis=0) if mask.any() else np.zeros(n_zones)
        row_sum = row.sum()
        zone_probs[t] = (row / row_sum) if row_sum > 0 else np.eye(n_zones)[0]
    if args.prior_zone_conditioning:
        log("zone_probs (per type, [continuum," + ",".join(m["label"] for m in matched) + "]):")
        for t, tname in dataset_pack["idx_to_type"].items():
            log(f"    {tname:12s} {np.round(zone_probs[t], 4)}")
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
    warp_yk, warp_zk = magi.fit_cdf_warp_knots(
        ey, n_knots=256, eps=1e-4,
        boost_ranges=WARP_BOOST_RANGES_BY_SOURCE.get(name))
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        n_types=dataset_pack["n_types"], line_positions_y=line_positions_y, latent_dim=8,
        hidden=(128,128,64), beta=0.2, continuum_mode="flow",
        continuum_flow_bins=args.continuum_flow_bins, continuum_flow_transforms=3,
        continuum_flow_warp="cdf",
        continuum_flow_warp_y_knots=warp_yk, continuum_flow_warp_z_knots=warp_zk,
        energy_flow_condition="z_cond", prior="coupling", w_gate_aux=2.0,
        # Focal down-weighting of the easy ~99% continuum majority in the gate CE.
        # v0.8.1 lowers gamma 2 -> 1: with the gate targets no longer over-labelling
        # (Phase 2 above), gamma=2 over-boosts and digs the continuum beside the
        # lines. On the narrow-line synthetic, gamma=1 gave lines 1.05-1.57 (vs
        # 1.08-2.57), between-line continuum 0.97/0.81/0.59 (vs 0.43/0.16/0.30) and
        # continuum core 0.807 (vs 0.655).
        gate_focal_gamma=1.0, gate_class_weights=gate_class_weights,
        line_logsigma_init=lls, line_logsigma_trainable=False,
        prior_zone_conditioning=args.prior_zone_conditioning,
        zone_probs=zone_probs if args.prior_zone_conditioning else None)
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

    # Checkpoint (seed-tagged unless this is the reference seed 42, so multi-seed
    # runs never clobber the reference checkpoint)
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    save_dir = f"trained_models/{args.run_tag}_{name}{suffix}"
    # scripts/generate_geant_source.py (what a Geant4 macro's /generator/mlScript
    # invokes) reads n_types/type_probs/idx_to_type/geometry_transform/
    # energy_transform from metadata["preprocessing_metadata"] and the fitted
    # quantile transformers from a sibling .joblib file - both omitted here
    # before v0.8.1 Phase 4, which left every checkpoint from this script
    # unusable for Geant4 export (KeyError on preprocessing_metadata["idx_to_type"]
    # against the empty dict save_final_trained_model defaults to). Mirrors what
    # MAGI_v0_8_1.ipynb's save cell has always done.
    preprocessing_metadata = {
        "source": name,
        "geometry_transform": "quantile_u_r_u_v_phi_r_phi_v",
        "energy_transform": "log10",
        "energy_bins": np.asarray(feature_pack["energy_bins"]).tolist(),
        "type_probs": np.asarray(dataset_pack["type_probs"]).tolist(),
        "idx_to_type": dataset_pack["idx_to_type"],
        "n_types": int(dataset_pack["n_types"]),
        "radius": R,
        "center": list(center),
    }
    magi.save_final_trained_model(model=model, save_dir=save_dir, model_name=f"mix_{name}",
                                  model_config=model.to_generation_config(),
                                  preprocessing_metadata=preprocessing_metadata)
    joblib.dump(feature_pack["quantile_transformers"],
                f"{save_dir}/mix_{name}_quantile_transformers.joblib")
    log(f"checkpoint saved -> {save_dir}/")

    # Generate + validate (chunked). Default n_gen = the real event count.
    n_gen = args.n_gen if args.n_gen > 0 else int(dataset_pack["E_idx"].size)
    E_gen, comp_idx = chunked_generate(model, n_gen, dataset_pack["type_probs"],
                                       dataset_pack["n_types"], dataset_pack["idx_to_type"],
                                       feature_pack, args.gen_chunk)
    E_real = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
    # Resolution-scaled window (+/-5 sigma at the pinned 4 eV FWHM) + local
    # continuum subtraction + size normalization: the legacy bin-width window was
    # ~190x the line width, so it measured continuum, not lines.
    recovery = magi.compute_line_integral_recovery(
        E_real, E_gen, matched, feature_pack["energy_bins"],
        energy_component_idx_gen=comp_idx,
        resolution_ev=X_IFU_RESOLUTION_EV,
        # See acceptance_v0_8.py: check contamination against all real lines.
        neighbour_lines=candidate_lines)
    log(f"{name} line-integral recovery (window +/-5sigma @ {X_IFU_RESOLUTION_EV} eV FWHM, "
        f"continuum-subtracted, N_real/N_gen={recovery[0]['gen_scale']:.3f}):")
    for r in recovery:
        rec = "  n/a" if r["recovery_ratio"] is None else f"{r['recovery_ratio']:.3f}"
        warn = f"  OVERLAPS {r['overlaps_lines']}" if r["overlaps_lines"] else ""
        log(f"    {r['label']:18s} n_real={r['n_real']:7d} (line {r['n_real_line']:9.1f}) "
            f"n_gen={r['n_gen']:7d} (line {r['n_gen_line']:9.1f}) "
            f"recovery={rec} comp_rec={r['component_recovery'] or float('nan'):.3f}{warn}")

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
    out = f"Plots/{args.run_tag}_real_{name}{suffix}_spectrum.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    log(f"spectrum saved -> {out}")

log("ALL DONE")
