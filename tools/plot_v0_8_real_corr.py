"""Post-run multivariate validation for the v0.8 real CR+Small run.

Reloads each trained checkpoint (trained_models/v0_8_<name>/), regenerates events
with the FULL feature set retained (energy + geometry), and compares real vs
generated over ALL physical variables:

  variables: logE, u_r, u_v, phi_r, phi_v

Outputs per source:
  Plots/v0_8_real_<name>_pairgrid.png   corner scatter (lower tri) + marginals,
                                        real vs generated overlaid
  Plots/v0_8_real_<name>_cov.png        covariance matrices: real | gen | diff
  Plots/v0_8_real_<name>_corr.png       Pearson correlation: real | gen | diff

The pipeline (load -> features -> filter -> split -> condition) is rebuilt with
the same seed as tools/run_v0_8_real.py, so energy_bins / n_types / type_probs /
quantile transformers match the trained model exactly. The model architecture is
reconstructed from the saved <name>_config.json (model.to_generation_config()).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, json, time, numpy as np
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
parser.add_argument("--sources", nargs="+", default=["CR", "Small"])
parser.add_argument("--n-gen", type=int, default=400_000,
                    help="events to generate for the covariance (capped at n_real); "
                         "0 = match the real event count")
parser.add_argument("--n-scatter", type=int, default=8000,
                    help="points per class in the corner scatter")
parser.add_argument("--seed", type=int, default=42,
                    help="RNG seed; must match the seed the checkpoint was trained "
                         "with, so the rebuilt pipeline (bins, type_probs, quantile "
                         "transformers) is identical")
parser.add_argument("--save-dir-suffix", default="",
                    help="e.g. _seed7, to score a multi-seed checkpoint")
parser.add_argument("--run-tag", default="v0_8",
                    help="checkpoint dir prefix and plot filename prefix, e.g. v0_8_1 "
                         "for the v0.8.1 run - keeps each run's figures separate")
args = parser.parse_args()

magi.initialize_environment(seed=args.seed, cpu_only=True)

TRAINING_DATA_DIR = "/Volumes/X10Pro/MAGI/TrainingData"
SOURCE_FILES = {
    "CR": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereCR.dat",
    "Small": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereSmall.dat",
}
center = (0.0, 0.0, -507.66); R = 100.0
os.makedirs("Plots", exist_ok=True)

CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
    # v0.8.1: EADL energies (what Geant4 actually emitted). The Bearden table
    # this replaced put every fluorescence line 4-11 detector FWHM off.
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json")
candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]

VARS = ["logE", "u_r", "u_v", "phi_r", "phi_v"]
LABELS = [r"$\log_{10}E$", r"$u_r$", r"$u_v$", r"$\phi_r$", r"$\phi_v$"]

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def rebuild_pipeline(name):
    """Deterministic rebuild of the driver's Phase-A/B pipeline for one source.
    Returns everything needed to reload the model + real physical arrays."""
    path = SOURCE_FILES[name]
    df = magi.load_detector_table(filepath=path, sep=r"\s+")
    prep = magi.build_physical_features(df, center=center, radius=R)
    E = prep["features"]["Energy"].to_numpy()

    res = magi.detect_energy_lines(E, binning_mode="log_fixed_count", n_bins=1024,
                                   prominence_factor=3.0, window=5, candidate_lines=candidate_lines,
                                   refine_bin_width_mev=4.0e-6)
    matched = [m for m in res["matched_lines"] if m["count"] >= 100]

    feature_pack = magi.build_feature_dataframe(
        prep, energy_binning_mode="log_fixed_count", n_bins=512,
        geometry_transform="quantile_u_r_u_v_phi_r_phi_v", n_quantiles=10000,
        random_state=42, energy_transform="log10")

    E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
    gate_targets = magi.build_gate_targets(
        E_full, feature_pack["energy_bins"], matched,
        bandwidth_mode="resolution", bandwidth_fwhm_mev=4.0e-6)   # v0.8.1
    feat = feature_pack["feat"].copy()
    for j in range(gate_targets.shape[1]):
        feat[f"gate_target_{j}"] = gate_targets[:, j]
    cont_cols = ("u_r_q","u_v_q","phi_r_q","phi_v_q","energy_y") + tuple(f"gate_target_{j}" for j in range(gate_targets.shape[1]))
    dataset_pack = magi.filter_particle_types_continuous_geometry(feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
    return feature_pack, dataset_pack, matched


def real_dataframe(feature_pack):
    """Real physical variables of the filtered events."""
    f = feature_pack["filtered_prep"]["features"]
    return {
        "logE": np.log10(f["Energy"].to_numpy()),
        "u_r": f["u_r"].to_numpy(),
        "u_v": f["u_v"].to_numpy(),
        "phi_r": f["phi_r"].to_numpy(),
        "phi_v": f["phi_v"].to_numpy(),
    }


def gen_dataframe(model, n_gen, dataset_pack, feature_pack):
    qt = feature_pack["quantile_transformers"]
    gp = magi.generate_latent_outputs(model, n_gen, dataset_pack["type_probs"],
                                      n_types=dataset_pack["n_types"], idx_to_type=dataset_pack["idx_to_type"])
    rc = magi.reconstruct_generated_features(
        gp, energy_head_mode="mixture", energy_transform="log10",
        geometry_mode="quantile_u_r_u_v_phi_r_phi_v",
        qt_u_r=qt["qt_u_r"], qt_u_v=qt["qt_u_v"], qt_phi_r=qt["qt_phi_r"], qt_phi_v=qt["qt_phi_v"])
    return {
        "logE": np.log10(rc["E_gen"]),
        "u_r": rc["u_r_gen"],
        "u_v": rc["u_v_gen"],
        "phi_r": rc["phi_r_gen"],
        "phi_v": rc["phi_v_gen"],
    }


def _stack(d):
    return np.column_stack([d[v] for v in VARS])


def plot_pairgrid(real, gen, name, n_scatter):
    n = len(VARS)
    fig, axes = plt.subplots(n, n, figsize=(2.4*n, 2.4*n))
    ri = np.random.default_rng(0).choice(real["logE"].size, min(n_scatter, real["logE"].size), replace=False)
    gi = np.random.default_rng(1).choice(gen["logE"].size, min(n_scatter, gen["logE"].size), replace=False)
    # common per-variable ranges from the real data (robust percentiles)
    rng_ = {v: np.percentile(real[v], [0.5, 99.5]) for v in VARS}
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(real[VARS[i]], bins=80, range=tuple(rng_[VARS[i]]), density=True,
                        histtype="step", color="#2b6cb0", lw=1.3, label="real")
                ax.hist(gen[VARS[i]], bins=80, range=tuple(rng_[VARS[i]]), density=True,
                        histtype="step", color="#dd6b20", lw=1.3, label="gen")
                if i == 0:
                    ax.legend(fontsize=6, loc="upper right")
            elif i > j:  # lower triangle: scatter
                ax.scatter(real[VARS[j]][ri], real[VARS[i]][ri], s=3, alpha=0.15, color="#2b6cb0", edgecolors="none")
                ax.scatter(gen[VARS[j]][gi], gen[VARS[i]][gi], s=3, alpha=0.15, color="#dd6b20", edgecolors="none")
                ax.set_xlim(rng_[VARS[j]]); ax.set_ylim(rng_[VARS[i]])
            else:  # upper triangle: pearson r (real vs gen)
                rr = np.corrcoef(real[VARS[j]], real[VARS[i]])[0, 1]
                gg = np.corrcoef(gen[VARS[j]], gen[VARS[i]])[0, 1]
                ax.axis("off")
                ax.text(0.5, 0.6, f"real r={rr:+.2f}", ha="center", color="#2b6cb0", fontsize=8, transform=ax.transAxes)
                ax.text(0.5, 0.35, f"gen  r={gg:+.2f}", ha="center", color="#dd6b20", fontsize=8, transform=ax.transAxes)
            if i == n-1:
                ax.set_xlabel(LABELS[j], fontsize=9)
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS[i], fontsize=9)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{name}: real (blue) vs generated (orange) — v0.8 flow+coupling", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = f"Plots/{args.run_tag}_real_{name}_pairgrid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    log(f"  pairgrid -> {out}")


def plot_matrix_triptych(real, gen, name, kind):
    Xr, Xg = _stack(real), _stack(gen)
    if kind == "cov":
        Mr, Mg = np.cov(Xr, rowvar=False), np.cov(Xg, rowvar=False)
        cmap, title = "viridis", "Covariance"
        vlim = None
    else:
        Mr, Mg = np.corrcoef(Xr, rowvar=False), np.corrcoef(Xg, rowvar=False)
        cmap, title = "coolwarm", "Pearson correlation"
        vlim = (-1, 1)
    diff = Mg - Mr
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("real", Mr, cmap, vlim), ("generated", Mg, cmap, vlim),
              ("gen - real", diff, "coolwarm", (-np.abs(diff).max() or 1, np.abs(diff).max() or 1))]
    for ax, (ttl, M, cm, vl) in zip(axes, panels):
        kw = {} if vl is None else {"vmin": vl[0], "vmax": vl[1]}
        im = ax.imshow(M, cmap=cm, **kw)
        ax.set_xticks(range(len(VARS))); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(VARS))); ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_title(f"{ttl}", fontsize=10)
        for a in range(len(VARS)):
            for b in range(len(VARS)):
                ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm == "viridis" else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{name}: {title} — real vs generated (v0.8 flow+coupling)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"Plots/{args.run_tag}_real_{name}_{kind}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    log(f"  {kind} triptych -> {out}")


for name in args.sources:
    log("=" * 56)
    log(f"SOURCE: {name}")
    save_dir = f"trained_models/{args.run_tag}_{name}{args.save_dir_suffix}"; model_name = f"mix_{name}"
    cfg_path = os.path.join(save_dir, f"{model_name}_config.json")
    if not os.path.exists(cfg_path):
        log(f"  SKIP: no checkpoint at {cfg_path}")
        continue

    feature_pack, dataset_pack, matched = rebuild_pipeline(name)
    with open(cfg_path) as f:
        model_config = json.load(f)
    model = magi.load_task_adaptive_model_for_generation(
        save_dir=save_dir, model_name=model_name, model_config=model_config,
        energy_bins=feature_pack["energy_bins"], n_types=dataset_pack["n_types"],
        radius=R, compile_model_fn=magi.compile_model, verbose=0)
    log(f"  model reloaded; n_types={dataset_pack['n_types']}")

    real = real_dataframe(feature_pack)
    n_real = real["logE"].size
    n_gen = n_real if args.n_gen <= 0 else min(args.n_gen, n_real)
    gen = gen_dataframe(model, n_gen, dataset_pack, feature_pack)
    log(f"  n_real={n_real:,} n_gen={n_gen:,}; building plots")

    plot_pairgrid(real, gen, name, args.n_scatter)
    plot_matrix_triptych(real, gen, name, "cov")
    plot_matrix_triptych(real, gen, name, "corr")

log("ALL DONE")
