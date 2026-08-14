#!/usr/bin/env python
"""v0.8.2 vs v0.8.3 on CR at MATCHED generated statistics.

The notebook run generated 380,980 events against the 1.6e6 used for the
original v0.8.2 measurement, so its small-b bins carry 4x fewer events and the
b<1mm comparison was confounded. Here both models generate the same N, and the
median ratios carry bootstrap CIs so "improved", "unchanged" and "too noisy to
say" can be told apart.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import joblib
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import magi
from magi.generation.export import NON_TRANSPORT_PARTICLES as NON_TRANSPORT

N_GEN = 2_000_000
CHUNK = 500_000
CENTER = (0.0, 0.0, -507.66)
R = 100.0
DET1 = np.array([-0.1, -0.25, -471.54])
CUTS = (np.inf, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0)

import sys
_ALL = {
    "v0.8.2":       ("trained_models/v0_8_2_priorzone_CR", "mix_CR"),
    "v082-seed7":   ("trained_models/v0_8_2_priorzone_CR_seed7", "mix_CR"),
    "v082-seed13":  ("trained_models/v0_8_2_priorzone_CR_seed13", "mix_CR"),
    "v082-exact":   ("trained_models/v0_8_2_exactzone_CR", "mix_CR"),
    "v0.8.3":       ("trained_models/v0_8_3_geomcond_CR", "mix_CR"),
    "v083-seed7":   ("trained_models/v0_8_3_geomcond_CR_seed7", "mix_CR"),
    "v083-seed13":  ("trained_models/v0_8_3_geomcond_CR_seed13", "mix_CR"),
}
_pick = sys.argv[1:] or ["v0.8.2", "v0.8.3"]
MODELS = {k: _ALL[k] for k in _pick}

TRAIN_FILE = "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR.dat"


def load_and_generate(save_dir, model_name, n_gen, seed):
    meta = json.load(open(f"{save_dir}/{model_name}_metadata.json"))
    cfg = json.load(open(f"{save_dir}/{model_name}_config.json"))
    pp = meta["preprocessing_metadata"]
    qts = joblib.load(f"{save_dir}/{model_name}_quantile_transformers.joblib")

    idx_to_type = {int(k): v for k, v in pp["idx_to_type"].items()} \
        if isinstance(pp["idx_to_type"], dict) else dict(enumerate(pp["idx_to_type"]))

    magi.initialize_environment(seed=seed, cpu_only=True, quiet=True)
    model = magi.load_task_adaptive_model_for_generation(
        save_dir=save_dir, model_name=model_name, model_config=cfg,
        energy_bins=pp["energy_bins"], u_v_bins=None,
        n_types=int(pp["n_types"]), type_weights=pp.get("type_weights"),
        radius=pp.get("radius", R), compile_model_fn=magi.compile_model, verbose=0,
    )
    print(f"  {save_dir}: latent_dim={model.latent_dim} "
          f"energy_condition_geometry={getattr(model,'energy_condition_geometry',False)}")

    parts, done = [], 0
    while done < n_gen:
        m = min(CHUNK, n_gen - done)
        raw = magi.generate_latent_outputs(
            model=model, n_samples=m, type_probs=pp["type_probs"],
            n_types=int(pp["n_types"]), idx_to_type=idx_to_type)
        gf = magi.reconstruct_generated_features(
            raw, energy_head_mode="mixture", energy_transform="log10",
            geometry_mode="quantile_u_r_u_v_phi_r_phi_v",
            qt_u_r=qts["qt_u_r"], qt_u_v=qts["qt_u_v"],
            qt_phi_r=qts["qt_phi_r"], qt_phi_v=qts["qt_phi_v"])
        parts.append(magi.reconstruct_generated_physics(
            gf, center=CENTER, radius=pp.get("radius", R)))
        done += m
    n0 = parts[0]["E_gen"].shape[0]
    out = {}
    for k in parts[0]:
        vals = [np.asarray(p[k]) for p in parts]
        if vals[0].ndim >= 1 and vals[0].shape[0] == n0:
            out[k] = np.concatenate(vals, axis=0)

    # CRITICAL: drop neutrinos, exactly as generation/export.py does when it
    # writes a Geant4 source (NON_TRANSPORT_PARTICLES). v0_8_2_priorzone_CR was
    # trained on a set that still contained them - its type_probs carry
    # anti_nu_e and nu_mu at 2.3% each - so sampling type_probs directly yields
    # ~4.6% neutrinos that the real pipeline never emits. Leaving them in
    # inflates the 10-100 MeV decade from 6.4% to 10.6% and corrupts every
    # median-energy comparison.
    names = np.asarray(out.get("ParticleName", []), dtype=object)
    if names.size == out["E_gen"].size:
        keep = ~np.isin(names, list(NON_TRANSPORT))
        out = {k: v[keep] for k, v in out.items() if v.shape[0] == keep.size}
        print(f"    dropped {int((~keep).sum()):,} neutrinos "
              f"({(~keep).mean():.2%}), {int(keep.sum()):,} kept")
    else:
        raise RuntimeError(
            "no ParticleName to filter neutrinos with - refusing to "
            "report energy statistics that the real pipeline would not produce"
        )
    return out


def impact(P, U, D):
    w = D - P
    t = np.einsum("ij,ij->i", w, U)
    b = np.linalg.norm(w - t[:, None] * U, axis=1)
    return b, t > 0


def med_ci(x, n_boot=400, rng=None):
    """Median with a bootstrap 68% CI - the honest error bar on a small bin."""
    rng = rng or np.random.default_rng(0)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.median(x))
    if x.size < 4:
        return m, np.nan, np.nan
    bs = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return m, float(np.percentile(bs, 16)), float(np.percentile(bs, 84))


# ---- real crossings --------------------------------------------------------
NU = {"nu_e", "anti_nu_e", "nu_mu", "anti_nu_mu", "nu_tau", "anti_nu_tau"}
P, U, E = [], [], []
for line in open(TRAIN_FILE):
    a = line.split()
    if len(a) < 9 or a[1] in NU:
        continue
    P.append((float(a[3]), float(a[4]), float(a[5])))
    U.append((float(a[6]), float(a[7]), float(a[8])))
    E.append(float(a[2]))
P = np.array(P); U = np.array(U); E = np.array(E)
b_r, in_r = impact(P, U, DET1)
print(f"real crossings: {len(E):,}")

gens = {}
for tag, (sd, mn) in MODELS.items():
    print(f"generating {N_GEN:,} from {tag}")
    g = load_and_generate(sd, mn, N_GEN, seed=1234)
    Pg = np.column_stack([g["x_gen"], g["y_gen"], g["z_gen"]])
    Ug = np.column_stack([g["vx_gen"], g["vy_gen"], g["vz_gen"]])
    bg, ing = impact(Pg, Ug, DET1)
    gens[tag] = dict(E=g["E_gen"], b=bg, ing=ing)

rng = np.random.default_rng(7)
print()
print("=" * 92)
print(f"median energy vs impact parameter to Detector 1, {N_GEN:,} generated per model")
print("=" * 92)
hdr = f"{'cut':>10s} {'n_real':>9s} |"
for tag in MODELS: hdr += f" {tag:>12s} {'n':>7s} |"
print(hdr)
for c in CUTS:
    mr = in_r & (b_r < c)
    if mr.sum() < 10:
        continue
    med_r = float(np.median(E[mr]))
    lab = "all" if not np.isfinite(c) else f"b<{c:g}mm"
    row = f"{lab:>10s} {mr.sum():9,} |"
    for tag in MODELS:
        d = gens[tag]
        mg = d["ing"] & (d["b"] < c)
        if mg.sum() < 4:
            row += f" {mg.sum():7,} {'--':>18s} |"
            continue
        m, lo, hi = med_ci(d["E"][mg], rng=rng)
        row += (f" {mg.sum():7,} {m/med_r:7.3f} "
                f"[{lo/med_r:.3f},{hi/med_r:.3f}] |")
    print(row)

print()
print("energy decades (marginal - must not regress)")
print(f"{'decade [MeV]':>20s} {'real':>9s}   ratios")
ed = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e6]
for lo, hi in zip(ed[:-1], ed[1:]):
    a = ((E >= lo) & (E < hi)).mean()
    if a < 1e-4:
        continue
    row = f"{lo:9.3g} - {hi:<8.3g} {a:9.4%}"
    for tag in MODELS:
        b = ((gens[tag]["E"] >= lo) & (gens[tag]["E"] < hi)).mean()
        row += f"  {tag}={b/a:.3f}"
    print(row)
