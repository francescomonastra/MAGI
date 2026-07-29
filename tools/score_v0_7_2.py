"""Phase A1 (docs/v0.8.2_RoadmapForAdoption.md S5, S6): score v0.7.2
(CVAE_CatEnergy_ContPhi_TaskAdaptive) on the same axes as
tools/acceptance_v0_8.py, so the v0.8 go/no-go decision has an actual v0.7.2
number to compare against instead of the qualitative comparison in
docs/v0.8_v072_comparison.md S5.

v0.7.2 has no line components and no gate - it is a single categorical energy
head over log-spaced bins - so this script reuses
magi.compute_line_integral_recovery with energy_component_idx_gen=None (the
routing cross-check is simply absent) plus the same Wasserstein/coupling
machinery as the v0.8 harness.

Everything about how to rebuild the pipeline (energy cuts, bin count,
geometry transform) is read from the checkpoint's OWN saved
preprocessing_metadata rather than re-derived, because it turns out to
matter: the Small checkpoint here was trained with e_min_cut=0.05,
e_max_cut=0.8 MeV, which excludes every fluorescence line below 50 keV (Al
Kalpha, Cu Kalpha/Kbeta) from what the model ever saw. That is reported below
as "excluded by training energy window", not as a recovery failure.

Usage:
  python tools/score_v0_7_2.py --sources CR Small
  python tools/score_v0_7_2.py --sources CR Small --json docs/_data/v072_acceptance.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, json, time
import numpy as np
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
import magi

parser = argparse.ArgumentParser()
parser.add_argument("--sources", nargs="+", default=["CR", "Small"])
parser.add_argument("--checkpoint-dir", action="append", default=None,
                    help="save_dir override as SOURCE=path (repeatable). "
                         "Default: trained_models/task_adaptive_energy_contphi_cr_run_001 "
                         "(CR), trained_models/task_adaptive_energy_contphi_run_001 (Small).")
parser.add_argument("--resolution-ev", type=float, default=4.0,
                    help="detector FWHM used for the line-recovery window "
                         "(v0.7.2 has no pinned line width of its own - use "
                         "the same X-IFU value v0.8 is scored against).")
parser.add_argument("--n-gen", type=int, default=0,
                    help="events to generate; 0 (default) = the real event count")
parser.add_argument("--gen-chunk", type=int, default=1_000_000)
parser.add_argument("--json", default="")
parser.add_argument("--line-lo", type=float, default=0.8)
parser.add_argument("--line-hi", type=float, default=1.2)
parser.add_argument("--wasserstein-max", type=float, default=0.05)
parser.add_argument("--coupling-max", type=float, default=0.05)
args = parser.parse_args()

magi.initialize_environment(seed=42, cpu_only=True)

TRAINING_DATA_DIR = "/Volumes/X10Pro/MAGI/TrainingData"
SOURCE_FILES = {
    "CR": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereCR.dat",
    "Small": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereSmall.dat",
}
DEFAULT_CHECKPOINTS = {
    "CR": "trained_models/task_adaptive_energy_contphi_cr_run_001",
    "Small": "trained_models/task_adaptive_energy_contphi_run_001",
}
checkpoint_overrides = {}
for item in (args.checkpoint_dir or []):
    src, _, path = item.partition("=")
    checkpoint_overrides[src] = path

MODEL_NAME = "task_adaptive_cvae_energy_contphi"
CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json")
center = (0.0, 0.0, -507.66); R = 100.0

VARS = ["logE", "u_r", "u_v", "phi_r", "phi_v"]

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def band_wasserstein(logE_real, logE_gen, n_bands=6):
    from scipy.stats import wasserstein_distance
    out = {"global": float(wasserstein_distance(logE_real, logE_gen))}
    edges = np.linspace(logE_real.min(), logE_real.max(), n_bands + 1)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        a = logE_real[(logE_real >= lo) & (logE_real < hi)]
        b = logE_gen[(logE_gen >= lo) & (logE_gen < hi)]
        key = f"[{lo:.2f},{hi:.2f})"
        out[key] = (float(wasserstein_distance(a, b))
                    if a.size > 100 and b.size > 100 else None)
    return out


def coupling_residuals(real, gen):
    Xr = np.column_stack([real[v] for v in VARS])
    Xg = np.column_stack([gen[v] for v in VARS])
    Cr, Cg = np.corrcoef(Xr, rowvar=False), np.corrcoef(Xg, rowvar=False)
    D = Cg - Cr
    iu = np.triu_indices(len(VARS), k=1)
    worst = int(np.argmax(np.abs(D[iu])))
    return {
        "max_abs": float(np.max(np.abs(D[iu]))),
        "worst_pair": f"{VARS[iu[0][worst]]}<->{VARS[iu[1][worst]]}",
        "real": float(Cr[iu[0][worst], iu[1][worst]]),
        "gen": float(Cg[iu[0][worst], iu[1][worst]]),
        "matrix_real": Cr.tolist(), "matrix_gen": Cg.tolist(),
    }


def verdict(value, lo, hi):
    if value is None:
        return "n/a"
    return "PASS" if lo <= value <= hi else "FAIL"


def score_checkpoint(name, save_dir):
    meta_path = os.path.join(save_dir, f"{MODEL_NAME}_metadata.json")
    if not os.path.exists(meta_path):
        log(f"  SKIP: no checkpoint at {meta_path}")
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    model_config = meta["model_config"]
    pm = meta["preprocessing_metadata"]
    energy_bins = np.asarray(pm["energy_bins"], dtype=np.float64)
    n_types = int(pm["n_types"])
    idx_to_type = {int(k): v for k, v in pm["idx_to_type"].items()}
    type_probs = pm["type_probs"]
    econf = pm["energy_config"]
    geometry_transform = pm["geometry_transform"]
    gmeta = pm.get("geometry_metadata", {})

    log(f"  checkpoint energy window: e_min_cut={econf.get('e_min_cut')} "
        f"e_max_cut={econf.get('e_max_cut')} n_bins={econf.get('n_bins')} "
        f"geometry_transform={geometry_transform}")

    df = magi.load_detector_table(filepath=SOURCE_FILES[name], sep=r"\s+")
    prep = magi.build_physical_features(df, center=center, radius=R)
    feature_pack = magi.build_feature_dataframe(
        prep, energy_binning_mode=econf.get("mode", "log_fixed_count"),
        e_min_cut=econf.get("e_min_cut"), e_max_cut=econf.get("e_max_cut"),
        bin_width=econf.get("bin_width", 0.5), n_bins=econf["n_bins"],
        min_counts=econf.get("min_counts", 20),
        geometry_transform=geometry_transform,
        n_quantiles=gmeta.get("quantile_n_quantiles", 10000),
        random_state=gmeta.get("random_state", 42))

    rebuilt_bins = np.asarray(feature_pack["energy_bins"], dtype=np.float64)
    if rebuilt_bins.shape != energy_bins.shape or not np.allclose(
            rebuilt_bins, energy_bins, rtol=1e-5, atol=1e-9):
        log(f"  WARNING: rebuilt energy_bins ({rebuilt_bins.shape[0]}) differ "
            f"from the checkpoint's saved bins ({energy_bins.shape[0]}) - "
            f"using the checkpoint's own bins for scoring, since those are "
            f"what the categorical head was actually trained against.")

    f_real = feature_pack["filtered_prep"]["features"]
    real = {"E": f_real["Energy"].to_numpy(),
            "logE": np.log10(f_real["Energy"].to_numpy()),
            "u_r": f_real["u_r"].to_numpy(), "u_v": f_real["u_v"].to_numpy(),
            "phi_r": f_real["phi_r"].to_numpy(), "phi_v": f_real["phi_v"].to_numpy()}
    n_real = real["E"].size

    e_lo = econf.get("e_min_cut"); e_hi = econf.get("e_max_cut")
    excluded_lines = []
    if e_lo is not None or e_hi is not None:
        lo = e_lo if e_lo is not None else -np.inf
        hi = e_hi if e_hi is not None else np.inf
        excluded_lines = [c["label"] for c in candidate_lines
                          if not (lo <= float(c["energy_mev"]) <= hi)]
        if excluded_lines:
            log(f"  {len(excluded_lines)} candidate line(s) fall outside this "
                f"checkpoint's training energy window [{lo}, {hi}] MeV and "
                f"were never in its training distribution: {excluded_lines}")

    res = magi.detect_energy_lines(
        real["E"], binning_mode="log_fixed_count", n_bins=1024,
        prominence_factor=3.0, window=5, candidate_lines=candidate_lines,
        refine_bin_width_mev=args.resolution_ev * 1e-6)
    matched = [m for m in res["matched_lines"] if m["count"] >= 100]
    matched += magi.confirm_unresolved_candidate_lines(
        real["E"], candidate_lines, matched, resolution_ev=args.resolution_ev)
    log(f"  {len(matched)} line(s) matched in the real spectrum within the "
        f"trained window: {[m['label'] for m in matched]}")

    save_dir_resolved = checkpoint_overrides.get(name, save_dir)
    model = magi.load_task_adaptive_model_for_generation(
        save_dir=save_dir_resolved, model_name=MODEL_NAME, model_config=model_config,
        energy_bins=energy_bins, n_types=n_types, radius=R,
        compile_model_fn=magi.compile_model, verbose=0)

    qt = feature_pack["quantile_transformers"]
    n_gen = n_real if args.n_gen <= 0 else args.n_gen
    log(f"  model reloaded; n_types={n_types} n_real={n_real:,} n_gen={n_gen:,}")

    parts, done = [], 0
    while done < n_gen:
        m = min(args.gen_chunk, n_gen - done)
        gp = magi.generate_latent_outputs(
            model, m, type_probs, n_types=n_types, idx_to_type=idx_to_type)
        rc = magi.reconstruct_generated_features(
            gp, energy_bins=energy_bins, energy_head_mode="categorical",
            geometry_mode=geometry_transform,
            qt_u_r=qt["qt_u_r"], qt_u_v=qt["qt_u_v"],
            qt_phi_r=qt["qt_phi_r"], qt_phi_v=qt["qt_phi_v"])
        parts.append({
            "E": rc["E_gen"], "logE": np.log10(rc["E_gen"]),
            "u_r": rc["u_r_gen"], "u_v": rc["u_v_gen"],
            "phi_r": rc["phi_r_gen"], "phi_v": rc["phi_v_gen"],
        })
        done += m
        log(f"    generated {done:,}/{n_gen:,}")
    gen = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    recovery = magi.compute_line_integral_recovery(
        real["E"], gen["E"], matched, energy_bins,
        energy_component_idx_gen=None, resolution_ev=args.resolution_ev,
        neighbour_lines=candidate_lines)
    wass = band_wasserstein(real["logE"], gen["logE"])
    coup = coupling_residuals(real, gen)

    src_pass = True
    print()
    if matched:
        print(f"  {name}: LINE INTEGRAL RECOVERY "
              f"(+/-{recovery[0]['window_half_width_mev']*1e6:.1f} eV window, "
              f"continuum-subtracted, N_real/N_gen={recovery[0]['gen_scale']:.3f})")
        print(f"    {'line':22s} {'E [keV]':>9s} {'real_line':>10s} {'gen_line':>10s} "
              f"{'recovery':>9s}  verdict")
        for r in recovery:
            v = verdict(r["recovery_ratio"], args.line_lo, args.line_hi)
            src_pass &= (v != "FAIL")
            rec = "n/a" if r["recovery_ratio"] is None else f"{r['recovery_ratio']:9.3f}"
            note = f"  OVERLAPS {r['overlaps_lines']}" if r["overlaps_lines"] else ""
            print(f"    {r['label']:22s} {r['candidate_energy_mev']*1e3:9.4f} "
                  f"{r['n_real_line']:10.1f} {r['n_gen_line']:10.1f} {rec}  {v}{note}")
    else:
        print(f"  {name}: no lines fell inside this checkpoint's trained energy "
              f"window - line recovery is not applicable.")
    if excluded_lines:
        print(f"  {name}: excluded by training energy window (never modelled): "
              f"{excluded_lines}")

    print(f"\n  {name}: ENERGY MARGINAL - Wasserstein on log10(E)")
    for k, v_ in wass.items():
        if v_ is None:
            continue
        vv = "PASS" if v_ <= args.wasserstein_max else "FAIL"
        if k == "global":
            src_pass &= (vv != "FAIL")
        print(f"    {k:22s} {v_:.5f}  {vv if k == 'global' else ''}")

    vc = "PASS" if coup["max_abs"] <= args.coupling_max else "FAIL"
    src_pass &= (vc != "FAIL")
    print(f"\n  {name}: COUPLING - max |corr_gen - corr_real| = {coup['max_abs']:.3f} "
          f"({coup['worst_pair']}: real {coup['real']:+.3f} -> gen {coup['gen']:+.3f})  {vc}")

    print(f"\n  ==> {name}: {'PASS' if src_pass else 'FAIL'}\n")
    return {
        "n_real": int(n_real), "n_gen": int(gen["E"].size),
        "energy_window": {"e_min_cut": e_lo, "e_max_cut": e_hi},
        "excluded_lines": excluded_lines,
        "recovery": recovery, "wasserstein_logE": wass, "coupling": coup,
        "pass": bool(src_pass),
    }


candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]
report = {"model_class": "CVAE_CatEnergy_ContPhi_TaskAdaptive (v0.7.2)",
          "resolution_ev": args.resolution_ev,
          "thresholds": {"line": [args.line_lo, args.line_hi],
                         "wasserstein_max": args.wasserstein_max,
                         "coupling_max": args.coupling_max},
          "sources": {}}
all_pass = True

for name in args.sources:
    log("=" * 62)
    log(f"SOURCE: {name}")
    save_dir = DEFAULT_CHECKPOINTS[name]
    r = score_checkpoint(name, save_dir)
    if r is None:
        continue
    all_pass &= r["pass"]
    report["sources"][name] = r

report["pass"] = bool(all_pass)
print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")

if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"report -> {args.json}")
