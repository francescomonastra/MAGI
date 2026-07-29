"""One pass/fail acceptance report for a v0.8 checkpoint.

Grades a trained CVAE_MixEnergy_ContPhi_TaskAdaptive run against explicit
thresholds, so an ~80-minute real run is scored objectively instead of by eye:

  1. per-line integral recovery   - resolution-scaled window (+/-5 sigma at the
                                    pinned detector FWHM), local continuum
                                    subtracted, normalized for N_real/N_gen
  2. near-line continuum ratio    - did the gate dig a hole beside each line?
  3. energy-marginal Wasserstein  - global and per decade band, on log10(E)
  4. coupling residuals           - max |corr_gen - corr_real| over
                                    (logE, u_r, u_v, phi_r, phi_v)

The same numbers double as the paper's validation table; --json writes them out.

Usage:
  python tools/acceptance_v0_8.py --sources CR Small
  python tools/acceptance_v0_8.py --sources CR --save-dir-suffix _seed7 --seed 7
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
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-tag", default="v0_8_1",
                    help="checkpoint dir prefix: trained_models/<run-tag>_<source>/ - "
                         "must match the --run-tag the checkpoint was saved with "
                         "(run_v0_8_real.py's default is v0_8).")
parser.add_argument("--save-dir-suffix", default="",
                    help="e.g. _seed7, to score a multi-seed checkpoint. Ignored "
                         "when --seeds is given (the suffix is then derived per "
                         "seed, matching run_v0_8_real.py's own convention).")
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Score a checkpoint already trained for each of these "
                         "seeds (e.g. --seeds 42 7 13) and report mean+/-std per "
                         "metric across seeds, in addition to each seed's own "
                         "table. Checkpoint dirs are derived the same way "
                         "run_v0_8_real.py names them: trained_models/<run-tag>_"
                         "<source> for seed 42, trained_models/<run-tag>_<source>"
                         "_seed<N> otherwise. A single per-band Wasserstein score "
                         "swings by tens of percent between seeds of the "
                         "identical config (docs/v0.8.1_line_truth.md section "
                         "11.2) - do not draw conclusions from a single seed.")
parser.add_argument("--n-gen", type=int, default=0,
                    help="events to generate; 0 (default) = the real event count")
parser.add_argument("--gen-chunk", type=int, default=1_000_000)
parser.add_argument("--resolution-ev", type=float, default=4.0,
                    help="detector FWHM the line widths are pinned to")
parser.add_argument("--json", default="",
                    help="optional path to write the full report as JSON")
# thresholds
parser.add_argument("--line-lo", type=float, default=0.8)
parser.add_argument("--line-hi", type=float, default=1.2)
parser.add_argument("--continuum-lo", type=float, default=0.85)
parser.add_argument("--continuum-hi", type=float, default=1.15)
parser.add_argument("--wasserstein-max", type=float, default=0.05,
                    help="max Wasserstein distance on log10(E)")
parser.add_argument("--coupling-max", type=float, default=0.05,
                    help="max |corr_gen - corr_real| over the variable pairs")
args = parser.parse_args()

magi.initialize_environment(seed=args.seed, cpu_only=True)

TRAINING_DATA_DIR = "/Volumes/X10Pro/MAGI/TrainingData"
SOURCE_FILES = {
    "CR": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereCR.dat",
    "Small": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereSmall.dat",
}
CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
    # v0.8.1: EADL energies (what Geant4 actually emitted). The Bearden table
    # this replaced put every fluorescence line 4-11 detector FWHM off.
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json")
center = (0.0, 0.0, -507.66); R = 100.0

VARS = ["logE", "u_r", "u_v", "phi_r", "phi_v"]

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def rebuild_pipeline(name, candidate_lines, line_positions_y=None):
    """Deterministic rebuild of tools/run_v0_8_real.py's pipeline for one source.

    line_positions_y : array-like of log10(E/MeV) or None
        The checkpoint's OWN line positions (model_config["line_positions_y"]).
        When given, `matched` is reconstructed to have exactly this line set -
        each position matched back to its candidate_lines entry by nearest
        energy, with its count either read from the coarse detect_energy_lines
        match or, if the coarse detector never separated it (a close doublet,
        e.g. Cu Kalpha2), fine-measured the same way
        confirm_unresolved_candidate_lines does.

        This is the correct way to score an EXISTING checkpoint: its line
        count is whatever it was trained with, which need not be what
        detect_energy_lines + confirm_unresolved_candidate_lines would decide
        today (that decision can change - Cu Kalpha2 was added mid-project).
        Re-deriving `matched` independently of the checkpoint risks a length
        mismatch against the model's actual n_lines that would silently
        misalign gate targets rather than raise a clear error. Pass None only
        when there is no checkpoint yet to defer to (line-set exploration).
    """
    df = magi.load_detector_table(filepath=SOURCE_FILES[name], sep=r"\s+")
    prep = magi.build_physical_features(df, center=center, radius=R)
    E = prep["features"]["Energy"].to_numpy()

    res = magi.detect_energy_lines(E, binning_mode="log_fixed_count", n_bins=1024,
                                   prominence_factor=3.0, window=5,
                                   candidate_lines=candidate_lines,
                                   refine_bin_width_mev=args.resolution_ev * 1e-6)

    if line_positions_y is not None:
        coarse_by_label = {m["label"]: m for m in res["matched_lines"]}
        E_sorted = np.sort(E)
        candidate_energies = [(c["label"], float(c["energy_mev"])) for c in candidate_lines]
        matched = []
        for y in np.asarray(line_positions_y, dtype=np.float64).reshape(-1):
            E_c = float(10.0 ** y)
            cand = min(candidate_lines, key=lambda c: abs(float(c["energy_mev"]) - E_c))
            m = coarse_by_label.get(cand["label"])
            if m is None:
                r = magi.measure_line_centroid(
                    E_sorted, float(cand["energy_mev"]), candidate_energies,
                    args.resolution_ev)
                count = float(r["n_line"]) if r["verdict"] == "ok" else 0.0
                m = {"label": cand["label"], "origin": cand.get("origin", ""),
                     "candidate_energy_mev": float(cand["energy_mev"]), "count": count}
            matched.append(m)
        n_expected = len(np.asarray(line_positions_y).reshape(-1))
        assert len(matched) == n_expected, (
            f"rebuilt {len(matched)} lines but the checkpoint has {n_expected} "
            f"(two positions likely matched the same candidate - check "
            f"candidate_lines for duplicate/near-duplicate energies)")
    else:
        matched = [m for m in res["matched_lines"] if m["count"] >= 100]
        matched += magi.confirm_unresolved_candidate_lines(
            E, candidate_lines, matched, resolution_ev=args.resolution_ev)

    feature_pack = magi.build_feature_dataframe(
        prep, energy_binning_mode="log_fixed_count", n_bins=512,
        geometry_transform="quantile_u_r_u_v_phi_r_phi_v", n_quantiles=10000,
        random_state=42, energy_transform="log10")

    E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
    # Must match the training-time construction (v0.8.1: resolution bandwidth),
    # so the rebuilt y_cont has the same columns the checkpoint was trained on.
    gate_targets = magi.build_gate_targets(
        E_full, feature_pack["energy_bins"], matched,
        bandwidth_mode="resolution", bandwidth_fwhm_mev=args.resolution_ev * 1e-6)
    feat = feature_pack["feat"].copy()
    for j in range(gate_targets.shape[1]):
        feat[f"gate_target_{j}"] = gate_targets[:, j]
    cont_cols = ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q", "energy_y") + tuple(
        f"gate_target_{j}" for j in range(gate_targets.shape[1]))
    dataset_pack = magi.filter_particle_types_continuous_geometry(
        feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
    return feature_pack, dataset_pack, matched


def generate(model, n_gen, dataset_pack, feature_pack, chunk):
    """Chunked generation returning the full physical feature set."""
    qt = feature_pack["quantile_transformers"]
    parts, done = [], 0
    while done < n_gen:
        m = min(chunk, n_gen - done)
        gp = magi.generate_latent_outputs(
            model, m, dataset_pack["type_probs"],
            n_types=dataset_pack["n_types"], idx_to_type=dataset_pack["idx_to_type"])
        rc = magi.reconstruct_generated_features(
            gp, energy_head_mode="mixture", energy_transform="log10",
            geometry_mode="quantile_u_r_u_v_phi_r_phi_v",
            qt_u_r=qt["qt_u_r"], qt_u_v=qt["qt_u_v"],
            qt_phi_r=qt["qt_phi_r"], qt_phi_v=qt["qt_phi_v"])
        parts.append({
            "E": rc["E_gen"], "logE": np.log10(rc["E_gen"]),
            "u_r": rc["u_r_gen"], "u_v": rc["u_v_gen"],
            "phi_r": rc["phi_r_gen"], "phi_v": rc["phi_v_gen"],
            "comp": gp["energy_component_idx_gen"],
        })
        done += m
        log(f"    generated {done:,}/{n_gen:,}")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def near_line_continuum_ratios(E_real, E_gen, recovery, scale):
    """Ratio of generated to real continuum in the side-bands beside each line.

    This is the 'did the gate dig a hole next to the line' check: the line
    integral can look right while the continuum around it has been evacuated
    to feed it (seen in the gamma sweeps at gate_focal_gamma >= 2).
    """
    out = []
    for r in recovery:
        c, w = r["candidate_energy_mev"], r["window_half_width_mev"]
        lo_in, lo_out = c - 3.0 * w, c - 2.0 * w
        hi_in, hi_out = c + 2.0 * w, c + 3.0 * w
        nr = (np.sum((E_real >= lo_in) & (E_real <= lo_out))
              + np.sum((E_real >= hi_in) & (E_real <= hi_out)))
        ng = (np.sum((E_gen >= lo_in) & (E_gen <= lo_out))
              + np.sum((E_gen >= hi_in) & (E_gen <= hi_out)))
        out.append({
            "label": r["label"],
            "n_real": int(nr), "n_gen": int(ng),
            "ratio": (float(ng * scale / nr) if nr else None),
        })
    return out


def band_wasserstein(logE_real, logE_gen, n_bands=6):
    """Wasserstein on log10(E) globally and inside equal-width log bands, so a
    local defect (e.g. the CR Compton edge) is not averaged away."""
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


def aggregate_over_seeds(seed_reports):
    """Mean+/-std per metric across a list of per-seed report dicts (as stored
    in report["sources"][name]["seeds"][seed]). Skips None values (e.g. a line
    with no valid recovery_ratio in some seed) rather than propagating NaN, and
    reports how many seeds actually contributed to each number."""
    def mean_std(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()),
                "std": float(arr.std()) if arr.size > 1 else 0.0,
                "n": int(arr.size)}

    agg = {"n_seeds": len(seed_reports)}

    labels = [r["label"] for r in seed_reports[0]["recovery"]]
    agg["recovery"] = {}
    for lbl in labels:
        rows = [next((r for r in sr["recovery"] if r["label"] == lbl), None)
                for sr in seed_reports]
        agg["recovery"][lbl] = {
            "recovery_ratio": mean_std([r["recovery_ratio"] for r in rows if r]),
            "component_recovery": mean_std([r["component_recovery"] for r in rows if r]),
        }

    cont_labels = [c["label"] for c in seed_reports[0]["near_line_continuum"]]
    agg["near_line_continuum"] = {
        lbl: mean_std([
            next((c["ratio"] for c in sr["near_line_continuum"] if c["label"] == lbl), None)
            for sr in seed_reports])
        for lbl in cont_labels
    }

    wass_keys = list(seed_reports[0]["wasserstein_logE"].keys())
    agg["wasserstein_logE"] = {
        k: mean_std([sr["wasserstein_logE"].get(k) for sr in seed_reports])
        for k in wass_keys
    }

    agg["coupling_max_abs"] = mean_std([sr["coupling"]["max_abs"] for sr in seed_reports])
    agg["pass_fraction"] = sum(1 for sr in seed_reports if sr["pass"]) / len(seed_reports)
    return agg


def print_aggregate(name, agg):
    print(f"\n  {name}: MEAN +/- STD OVER {agg['n_seeds']} SEEDS")
    print(f"  {name}: line recovery")
    for lbl, d in agg["recovery"].items():
        rr, cr = d["recovery_ratio"], d["component_recovery"]
        rr_s = "n/a" if rr is None else f"{rr['mean']:.3f} +/- {rr['std']:.3f} (n={rr['n']})"
        cr_s = "n/a" if cr is None else f"{cr['mean']:.3f} +/- {cr['std']:.3f} (n={cr['n']})"
        print(f"    {lbl:22s} recovery={rr_s:28s} comp_rec={cr_s}")
    print(f"  {name}: near-line continuum")
    for lbl, d in agg["near_line_continuum"].items():
        s = "n/a" if d is None else f"{d['mean']:.3f} +/- {d['std']:.3f} (n={d['n']})"
        print(f"    {lbl:22s} ratio={s}")
    print(f"  {name}: Wasserstein on log10(E)")
    for k, d in agg["wasserstein_logE"].items():
        if d is None:
            continue
        print(f"    {k:22s} {d['mean']:.5f} +/- {d['std']:.5f} (n={d['n']})")
    d = agg["coupling_max_abs"]
    print(f"  {name}: coupling max|corr_gen-corr_real| = {d['mean']:.3f} +/- {d['std']:.3f} (n={d['n']})")
    print(f"  {name}: {agg['pass_fraction']*100:.0f}% of seeds PASS overall")


def score_checkpoint(name, save_dir, model_name):
    """Score one trained checkpoint; returns a report dict or None if missing."""
    cfg_path = os.path.join(save_dir, f"{model_name}_config.json")
    if not os.path.exists(cfg_path):
        log(f"  SKIP: no checkpoint at {cfg_path}")
        return None

    with open(cfg_path) as f:
        model_config = json.load(f)
    feature_pack, dataset_pack, matched = rebuild_pipeline(
        name, candidate_lines, line_positions_y=model_config.get("line_positions_y"))
    model = magi.load_task_adaptive_model_for_generation(
        save_dir=save_dir, model_name=model_name, model_config=model_config,
        energy_bins=feature_pack["energy_bins"], n_types=dataset_pack["n_types"],
        radius=R, compile_model_fn=magi.compile_model, verbose=0)

    f_real = feature_pack["filtered_prep"]["features"]
    real = {"E": f_real["Energy"].to_numpy(),
            "logE": np.log10(f_real["Energy"].to_numpy()),
            "u_r": f_real["u_r"].to_numpy(), "u_v": f_real["u_v"].to_numpy(),
            "phi_r": f_real["phi_r"].to_numpy(), "phi_v": f_real["phi_v"].to_numpy()}
    n_real = real["E"].size
    n_gen = n_real if args.n_gen <= 0 else args.n_gen
    log(f"  model reloaded; n_types={dataset_pack['n_types']} n_real={n_real:,} n_gen={n_gen:,}")

    gen = generate(model, n_gen, dataset_pack, feature_pack, args.gen_chunk)
    scale = n_real / gen["E"].size

    recovery = magi.compute_line_integral_recovery(
        real["E"], gen["E"], matched, feature_pack["energy_bins"],
        energy_component_idx_gen=gen["comp"], resolution_ev=args.resolution_ev,
        # Contamination is checked against every line real in the data, not
        # just the modelled ones - Small's Cu Kalpha2 (63 events, below the
        # modelling floor) still deflates Cu Kalpha1's subtracted count ~2x.
        neighbour_lines=candidate_lines)
    continuum = near_line_continuum_ratios(real["E"], gen["E"], recovery, scale)
    wass = band_wasserstein(real["logE"], gen["logE"])
    coup = coupling_residuals(real, gen)

    src_pass = True
    print()
    print(f"  {name}: LINE INTEGRAL RECOVERY "
          f"(+/-{recovery[0]['window_half_width_mev']*1e6:.1f} eV window, "
          f"continuum-subtracted, N_real/N_gen={recovery[0]['gen_scale']:.3f})")
    print(f"    {'line':22s} {'E [keV]':>9s} {'real_line':>10s} {'gen_line':>10s} "
          f"{'recovery':>9s} {'routing':>8s}  verdict")
    for r in recovery:
        v = verdict(r["recovery_ratio"], args.line_lo, args.line_hi)
        src_pass &= (v != "FAIL")
        rec = "n/a" if r["recovery_ratio"] is None else f"{r['recovery_ratio']:9.3f}"
        cr = r.get("component_recovery")
        crs = "n/a" if cr is None else f"{cr:8.3f}"
        note = f"  OVERLAPS {r['overlaps_lines']}" if r["overlaps_lines"] else ""
        print(f"    {r['label']:22s} {r['candidate_energy_mev']*1e3:9.4f} "
              f"{r['n_real_line']:10.1f} {r['n_gen_line']:10.1f} {rec} {crs}  {v}{note}")

    print(f"\n  {name}: NEAR-LINE CONTINUUM (gate must not dig holes beside the lines)")
    for c in continuum:
        v = verdict(c["ratio"], args.continuum_lo, args.continuum_hi)
        src_pass &= (v != "FAIL")
        rs = "n/a" if c["ratio"] is None else f"{c['ratio']:.3f}"
        print(f"    {c['label']:22s} real={c['n_real']:8d} gen={c['n_gen']:8d} "
              f"ratio={rs}  {v}")

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
        "recovery": recovery, "near_line_continuum": continuum,
        "wasserstein_logE": wass, "coupling": coup, "pass": bool(src_pass),
    }


candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]
report = {"resolution_ev": args.resolution_ev,
          "thresholds": {"line": [args.line_lo, args.line_hi],
                         "continuum": [args.continuum_lo, args.continuum_hi],
                         "wasserstein_max": args.wasserstein_max,
                         "coupling_max": args.coupling_max},
          "sources": {}}
all_pass = True

for name in args.sources:
    log("=" * 62)
    log(f"SOURCE: {name}")
    model_name = f"mix_{name}"

    if args.seeds:
        seed_reports = {}
        for seed in args.seeds:
            suffix = "" if seed == 42 else f"_seed{seed}"
            save_dir = f"trained_models/{args.run_tag}_{name}{suffix}"
            log(f"  -- seed {seed} ({save_dir}) --")
            magi.initialize_environment(seed=seed, cpu_only=True)
            r = score_checkpoint(name, save_dir, model_name)
            if r is not None:
                seed_reports[seed] = r
        if not seed_reports:
            continue
        reports_list = list(seed_reports.values())
        agg = aggregate_over_seeds(reports_list) if len(reports_list) > 1 else None
        if agg is not None:
            print_aggregate(name, agg)
            src_pass = agg["pass_fraction"] == 1.0
        else:
            log(f"  only 1/{ len(args.seeds) } seeds had a checkpoint - no aggregate")
            src_pass = reports_list[0]["pass"]
        all_pass &= src_pass
        report["sources"][name] = {"seeds": seed_reports, "aggregate": agg, "pass": bool(src_pass)}
    else:
        save_dir = f"trained_models/{args.run_tag}_{name}{args.save_dir_suffix}"
        r = score_checkpoint(name, save_dir, model_name)
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
