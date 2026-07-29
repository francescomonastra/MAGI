"""Phase A2 (docs/v0.8.2_RoadmapForAdoption.md S5.2a, S6): posterior-vs-prior
gate audit for a trained CVAE_MixEnergy_ContPhi_TaskAdaptive checkpoint.

Hypothesis under test: the gate is conditioned on z (energy_flow_condition=
"z_cond"). At training/reconstruction time it sees z ~ q(z|x_real) from the
encoder posterior; at generation time it sees z ~ p(z|cond) from the learned
ConditionalCouplingPrior, which conditions only on the particle-type one-hot.
val_kl ~= 21-28 at latent_dim=8 says these two z distributions differ
substantially - this script checks whether that difference actually moves the
gate's line/continuum fractions, which would explain the systematic per-line
intensity errors in docs/v0.8.1_line_truth.md S10 without needing another
gate-reweighting experiment (the thing already tried and shown not to work).

For each particle type and each gate slot (continuum component(s) + each
pinned line), reports:
  - mean gate probability under z ~ q(z|x_real)  (posterior, reconstruction-time)
  - mean gate probability under z ~ p(z|cond)    (prior, generation-time)
  - their ratio (prior/posterior) - far from 1 means the gate genuinely sees
    a different z distribution at generation time than it was fit against.

This is training-free: it loads an existing checkpoint and runs two forward
passes (encoder+decode, prior.sample+decode) over the real dataset the
checkpoint was trained on. No new model is fit.

Usage:
  python tools/gate_posterior_prior_audit.py --sources CR Small
  python tools/gate_posterior_prior_audit.py --sources CR --run-tag v0_8_1_cuka2 --max-events-per-type 200000
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
parser.add_argument("--run-tags", nargs="+", default=None,
                    help="one run-tag per --sources entry; default matches "
                         "docs/v0.8.2_RoadmapForAdoption.md S9: v0_8_1_cuka2 "
                         "for CR, v0_8_1 for Small.")
parser.add_argument("--max-events-per-type", type=int, default=300_000,
                    help="subsample cap per particle type, for tractable CPU runtime")
parser.add_argument("--chunk", type=int, default=200_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--json", default="")
args = parser.parse_args()

magi.initialize_environment(seed=args.seed, cpu_only=True)

TRAINING_DATA_DIR = "/Volumes/X10Pro/MAGI/TrainingData"
SOURCE_FILES = {
    "CR": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereCR.dat",
    "Small": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereSmall.dat",
}
DEFAULT_RUN_TAGS = {"CR": "v0_8_1_cuka2", "Small": "v0_8_1"}
run_tag_overrides = dict(zip(args.sources, args.run_tags)) if args.run_tags else {}

CANDIDATE_LINES_FILE = ("/Volumes/X10Pro/MAGI/CandidateLines/"
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json")
center = (0.0, 0.0, -507.66); R = 100.0
RESOLUTION_EV = 4.0

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def rebuild_dataset_pack(name, candidate_lines, line_positions_y):
    """Mirror tools/acceptance_v0_8.py's rebuild_pipeline, but only as far as
    the encoder input (X_cont_raw/y_type) - no generation needed here."""
    df = magi.load_detector_table(filepath=SOURCE_FILES[name], sep=r"\s+")
    prep = magi.build_physical_features(df, center=center, radius=R)
    E = prep["features"]["Energy"].to_numpy()

    res = magi.detect_energy_lines(E, binning_mode="log_fixed_count", n_bins=1024,
                                   prominence_factor=3.0, window=5,
                                   candidate_lines=candidate_lines,
                                   refine_bin_width_mev=RESOLUTION_EV * 1e-6)
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
                E_sorted, float(cand["energy_mev"]), candidate_energies, RESOLUTION_EV)
            count = float(r["n_line"]) if r["verdict"] == "ok" else 0.0
            m = {"label": cand["label"], "candidate_energy_mev": float(cand["energy_mev"]),
                 "count": count}
        matched.append(m)

    feature_pack = magi.build_feature_dataframe(
        prep, energy_binning_mode="log_fixed_count", n_bins=512,
        geometry_transform="quantile_u_r_u_v_phi_r_phi_v", n_quantiles=10000,
        random_state=42, energy_transform="log10")

    E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
    gate_targets = magi.build_gate_targets(
        E_full, feature_pack["energy_bins"], matched,
        bandwidth_mode="resolution", bandwidth_fwhm_mev=RESOLUTION_EV * 1e-6)
    feat = feature_pack["feat"].copy()
    for j in range(gate_targets.shape[1]):
        feat[f"gate_target_{j}"] = gate_targets[:, j]
    cont_cols = ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q", "energy_y") + tuple(
        f"gate_target_{j}" for j in range(gate_targets.shape[1]))
    dataset_pack = magi.filter_particle_types_continuous_geometry(
        feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
    return dataset_pack, matched


def audit_checkpoint(name, run_tag):
    save_dir = f"trained_models/{run_tag}_{name}"
    model_name = f"mix_{name}"
    cfg_path = os.path.join(save_dir, f"{model_name}_config.json")
    if not os.path.exists(cfg_path):
        log(f"  SKIP: no checkpoint at {cfg_path}")
        return None
    with open(cfg_path) as f:
        model_config = json.load(f)

    dataset_pack, matched = rebuild_dataset_pack(
        name, candidate_lines, model_config["line_positions_y"])
    n_types = dataset_pack["n_types"]
    idx_to_type = dataset_pack["idx_to_type"]
    n_lines = len(matched)
    n_cont = int(model_config.get("n_continuum_components", 1))
    slot_labels = [f"continuum_{k}" for k in range(n_cont)] + [m["label"] for m in matched]

    model = magi.load_task_adaptive_model_for_generation(
        save_dir=save_dir, model_name=model_name, model_config=model_config,
        n_types=n_types, radius=R, compile_model_fn=magi.compile_model, verbose=0)

    X_cont_raw = dataset_pack["X_cont_raw"]
    y_type = dataset_pack["y_type"]
    n_events = X_cont_raw.shape[0]
    log(f"  {name}: n_events={n_events:,} n_types={n_types} n_lines={n_lines} "
        f"n_continuum_components={n_cont}")

    rng = np.random.default_rng(args.seed)
    keep_idx = []
    for t in range(n_types):
        idx_t = np.flatnonzero(y_type == t)
        if idx_t.size > args.max_events_per_type:
            idx_t = rng.choice(idx_t, size=args.max_events_per_type, replace=False)
        keep_idx.append(idx_t)
    keep_idx = np.concatenate(keep_idx)
    X_cont_raw = X_cont_raw[keep_idx]
    y_type = y_type[keep_idx]
    n_sub = X_cont_raw.shape[0]
    log(f"  {name}: scoring on {n_sub:,} subsampled events "
        f"(cap {args.max_events_per_type:,}/type)")

    n_slots = n_cont + n_lines
    sum_post = np.zeros((n_types, n_slots), dtype=np.float64)
    sum_prior = np.zeros((n_types, n_slots), dtype=np.float64)
    count = np.zeros((n_types,), dtype=np.int64)

    done = 0
    while done < n_sub:
        m = min(args.chunk, n_sub - done)
        sl = slice(done, done + m)
        x_cont = tf.constant(X_cont_raw[sl], dtype=tf.float32)
        t_idx = y_type[sl]
        cond = tf.one_hot(t_idx, depth=n_types, dtype=tf.float32)
        x_in = tf.concat([x_cont, cond], axis=1)

        z_mean, z_logvar = model.encoder(x_in, training=False)
        z_post = model.sample_z(z_mean, z_logvar)
        params_post = model._decode_params(z_post, cond, training=False)
        gate_post = tf.nn.softmax(params_post["energy_gate_logits"], axis=-1).numpy()

        z_prior = model.prior.sample(cond)
        params_prior = model._decode_params(z_prior, cond, training=False)
        gate_prior = tf.nn.softmax(params_prior["energy_gate_logits"], axis=-1).numpy()

        for t in range(n_types):
            mask = (t_idx == t)
            if not np.any(mask):
                continue
            sum_post[t] += gate_post[mask].sum(axis=0)
            sum_prior[t] += gate_prior[mask].sum(axis=0)
            count[t] += int(mask.sum())

        done += m
        log(f"    {done:,}/{n_sub:,}")

    mean_post = sum_post / np.maximum(count[:, None], 1)
    mean_prior = sum_prior / np.maximum(count[:, None], 1)

    print(f"\n  {name}: POSTERIOR q(z|x_real) vs PRIOR p(z|cond) GATE FRACTIONS")
    print(f"    {'type':12s} {'slot':22s} {'post':>10s} {'prior':>10s} {'prior/post':>11s}")
    per_type = {}
    for t in range(n_types):
        tname = idx_to_type[t]
        per_type[tname] = {}
        for s, label in enumerate(slot_labels):
            p, q = mean_prior[t, s], mean_post[t, s]
            ratio = (p / q) if q > 1e-12 else (float("inf") if p > 1e-12 else float("nan"))
            per_type[tname][label] = {"posterior": float(q), "prior": float(p),
                                       "ratio_prior_over_post": float(ratio)}
            print(f"    {tname:12s} {label:22s} {q:10.4e} {p:10.4e} {ratio:11.3f}")

    # Population-level (marginalized over the real type mix) - this is the
    # number directly comparable to the recovery ratios in
    # docs/v0.8.1_line_truth.md S2.2, since real line fractions there are
    # also population-level, not per-type.
    type_probs = np.array([count[t] for t in range(n_types)], dtype=np.float64)
    type_probs /= type_probs.sum()
    pop_post = (mean_post * type_probs[:, None]).sum(axis=0)
    pop_prior = (mean_prior * type_probs[:, None]).sum(axis=0)
    print(f"\n  {name}: POPULATION-LEVEL (real type mix)")
    print(f"    {'slot':22s} {'post':>10s} {'prior':>10s} {'prior/post':>11s}")
    pop = {}
    for s, label in enumerate(slot_labels):
        p, q = pop_prior[s], pop_post[s]
        ratio = (p / q) if q > 1e-12 else (float("inf") if p > 1e-12 else float("nan"))
        pop[label] = {"posterior": float(q), "prior": float(p),
                      "ratio_prior_over_post": float(ratio)}
        print(f"    {label:22s} {q:10.4e} {p:10.4e} {ratio:11.3f}")

    return {"n_events_scored": int(n_sub), "slot_labels": slot_labels,
            "per_type": per_type, "population": pop}


candidate_lines = magi.load_candidate_energy_lines(CANDIDATE_LINES_FILE)["lines"]
report = {"sources": {}}
for name in args.sources:
    log("=" * 62)
    log(f"SOURCE: {name}")
    run_tag = run_tag_overrides.get(name, DEFAULT_RUN_TAGS[name])
    r = audit_checkpoint(name, run_tag)
    if r is not None:
        report["sources"][name] = r

if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"report -> {args.json}")
