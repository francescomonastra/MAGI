"""Pairgrid + Pearson-correlation triptych for the v0.8.2 CR_ingoingfix checkpoint.

This is the checkpoint behind the report's post-fix SRON headline number
(R=0.963+-0.043), trained on alloutputDSCryoSphereCR_ingoingfix.dat with n_types=4
(no neutrinos) -- distinct from tools/plot_v0_8_real_corr.py's "CR" source, which
reloads the older, superseded 6-type checkpoint (docs/v0.8.1_line_truth.md line 382).

Save-dir does not follow the <run_tag>_<source> convention (see
docs/COLAB_CR_retrain_instructions.md Sect. 5): it is trained_models/v0_8_2_CR_ingoingfix/
with model_name mix_CR.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
import magi

magi.initialize_environment(seed=42, cpu_only=True)

TRAINING_DATA_DIR = "/Volumes/X10Pro/MAGI/TrainingData"
SOURCE_FILE = f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereCR_ingoingfix.dat"
center = (0.0, 0.0, -507.66); R = 100.0
SAVE_DIR = "trained_models/v0_8_2_CR_ingoingfix"
MODEL_NAME = "mix_CR"
N_GEN = 400_000
N_SCATTER = 8000
os.makedirs("Plots", exist_ok=True)

CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json")
candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]

VARS = ["u_r", "u_v", "phi_r", "phi_v"]
LABELS = [r"$u_r$", r"$u_v$", r"$\phi_r$", r"$\phi_v$"]

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def rebuild_pipeline(line_positions_y):
    df = magi.load_detector_table(filepath=SOURCE_FILE, sep=r"\s+")
    prep = magi.build_physical_features(df, center=center, radius=R)
    E = prep["features"]["Energy"].to_numpy()

    res = magi.detect_energy_lines(E, binning_mode="log_fixed_count", n_bins=1024,
                                   prominence_factor=3.0, window=5, candidate_lines=candidate_lines,
                                   refine_bin_width_mev=4.0e-6)

    coarse_by_label = {m["label"]: m for m in res["matched_lines"]}
    E_sorted = np.sort(E)
    candidate_energies = [(c["label"], float(c["energy_mev"])) for c in candidate_lines]
    matched = []
    for y in np.asarray(line_positions_y, dtype=np.float64).reshape(-1):
        E_c = float(10.0 ** y)
        cand = min(candidate_lines, key=lambda c: abs(float(c["energy_mev"]) - E_c))
        m = coarse_by_label.get(cand["label"])
        if m is None:
            r = magi.measure_line_centroid(E_sorted, float(cand["energy_mev"]),
                                           candidate_energies, 4.0)
            count = float(r["n_line"]) if r["verdict"] == "ok" else 0.0
            m = {"label": cand["label"], "origin": cand.get("origin", ""),
                 "candidate_energy_mev": float(cand["energy_mev"]), "count": count}
        matched.append(m)

    feature_pack = magi.build_feature_dataframe(
        prep, energy_binning_mode="log_fixed_count", n_bins=512,
        geometry_transform="quantile_u_r_u_v_phi_r_phi_v", n_quantiles=10000,
        random_state=42, energy_transform="log10")

    E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
    gate_targets = magi.build_gate_targets(
        E_full, feature_pack["energy_bins"], matched,
        bandwidth_mode="resolution", bandwidth_fwhm_mev=4.0e-6)
    feat = feature_pack["feat"].copy()
    for j in range(gate_targets.shape[1]):
        feat[f"gate_target_{j}"] = gate_targets[:, j]
    cont_cols = ("u_r_q","u_v_q","phi_r_q","phi_v_q","energy_y") + tuple(f"gate_target_{j}" for j in range(gate_targets.shape[1]))
    dataset_pack = magi.filter_particle_types_continuous_geometry(feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
    return feature_pack, dataset_pack


def real_dataframe(feature_pack):
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


def plot_pairgrid(real, gen, out):
    n = len(VARS)
    fig, axes = plt.subplots(n, n, figsize=(2.4*n, 2.4*n))
    ri = np.random.default_rng(0).choice(real["logE"].size, min(N_SCATTER, real["logE"].size), replace=False)
    gi = np.random.default_rng(1).choice(gen["logE"].size, min(N_SCATTER, gen["logE"].size), replace=False)
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
                    ax.legend(fontsize=8, loc="upper right")
            elif i > j:
                ax.scatter(real[VARS[j]][ri], real[VARS[i]][ri], s=4, alpha=0.18, color="#2b6cb0", edgecolors="none")
                ax.scatter(gen[VARS[j]][gi], gen[VARS[i]][gi], s=4, alpha=0.18, color="#dd6b20", edgecolors="none")
                ax.set_xlim(rng_[VARS[j]]); ax.set_ylim(rng_[VARS[i]])
            else:
                rr = np.corrcoef(real[VARS[j]], real[VARS[i]])[0, 1]
                gg = np.corrcoef(gen[VARS[j]], gen[VARS[i]])[0, 1]
                ax.axis("off")
                ax.text(0.5, 0.6, f"real r={rr:+.2f}", ha="center", color="#2b6cb0", fontsize=10, transform=ax.transAxes)
                ax.text(0.5, 0.35, f"gen  r={gg:+.2f}", ha="center", color="#dd6b20", fontsize=10, transform=ax.transAxes)
            if i == n-1:
                ax.set_xlabel(LABELS[j], fontsize=11)
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS[i], fontsize=11)
            ax.tick_params(labelsize=8)
    fig.suptitle("SRON CR (post-fix, v0.8.2): real (blue) vs generated (orange)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    log(f"  pairgrid -> {out}")


def plot_corr(real, gen, out):
    Xr, Xg = _stack(real), _stack(gen)
    Mr, Mg = np.corrcoef(Xr, rowvar=False), np.corrcoef(Xg, rowvar=False)
    diff = Mg - Mr
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("real", Mr, "coolwarm", (-1, 1)), ("generated", Mg, "coolwarm", (-1, 1)),
              ("gen - real", diff, "coolwarm", (-(np.abs(diff).max() or 1), np.abs(diff).max() or 1))]
    for ax, (ttl, M, cm, vl) in zip(axes, panels):
        im = ax.imshow(M, cmap=cm, vmin=vl[0], vmax=vl[1])
        ax.set_xticks(range(len(VARS))); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(VARS))); ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_title(ttl, fontsize=10)
        for a in range(len(VARS)):
            for b in range(len(VARS)):
                ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center", fontsize=6, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("SRON CR (post-fix, v0.8.2): Pearson correlation, real vs generated", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    log(f"  corr triptych -> {out}")


log("loading checkpoint config")
with open(os.path.join(SAVE_DIR, f"{MODEL_NAME}_config.json")) as f:
    model_config = json.load(f)

feature_pack, dataset_pack = rebuild_pipeline(model_config.get("line_positions_y"))
model = magi.load_task_adaptive_model_for_generation(
    save_dir=SAVE_DIR, model_name=MODEL_NAME, model_config=model_config,
    energy_bins=feature_pack["energy_bins"], n_types=dataset_pack["n_types"],
    radius=R, compile_model_fn=magi.compile_model, verbose=0)
log(f"model reloaded; n_types={dataset_pack['n_types']}")

real = real_dataframe(feature_pack)
n_real = real["logE"].size
n_gen = min(N_GEN, n_real)
gen = gen_dataframe(model, n_gen, dataset_pack, feature_pack)
log(f"n_real={n_real:,} n_gen={n_gen:,}; building plots")

plot_pairgrid(real, gen, "Plots/v0_8_2_ingoingfix_real_CR_pairgrid.png")
plot_corr(real, gen, "Plots/v0_8_2_ingoingfix_real_CR_corr.png")
log("ALL DONE")
