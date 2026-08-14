#!/usr/bin/env python
"""Is the median-E-vs-b systematic an ENERGY error or a SPECIES-MIX error?

Measured on DM1.2 (13/08): the aggregate metric reads 2.17 at b<10mm, which
looks catastrophic, while the SAME crossings give 1.021 for muons alone and the
global species fractions are correct to 0.3%. The aggregate number was reporting
the species mix, not the energy conditional: CR/muon sources are bimodal in
energy (a soft gamma/e- population and a ~350 MeV muon population) with the
median sitting on the boundary, so a 5-9% shift in mix swings the median by a
factor 2.

That matters directly for what goes in the paper. The 0.495-1.222 spread over
three v0.8.2 CR seeds (docs/v0.8.3_seed_variance.md) was measured with the
aggregate metric. If that spread is species-mix rather than energy, then:
  - the quoted systematic is describing the wrong channel,
  - and the ranked fix list in section 6.1 (which targets the energy conditional
    via importance weighting / conditional-moment penalties on log E) is aimed
    at the wrong term.

This script recomputes the metric PER SPECIES on the three v0.8.2 CR seeds.

Output to read: if the per-species ratios cluster near 1 while the aggregate
scatters, the systematic is a mix effect. If the per-species ratios scatter too,
it is a genuine energy-conditional error and section 6.1 stands.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json

import joblib
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import magi
from magi.generation.export import NON_TRANSPORT_PARTICLES as NON_TRANSPORT

N_GEN = 2_000_000
CHUNK = 500_000
CENTER = (0.0, 0.0, -507.66)
R = 100.0
DET1 = np.array([-0.1, -0.25, -471.54])
CUTS = (np.inf, 50.0, 20.0, 10.0)
SPECIES = ("mu-", "mu+", "gamma", "e-")
TRAIN_FILE = "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR.dat"
MODELS = {
    "seed42": "trained_models/v0_8_2_priorzone_CR",
    "seed7": "trained_models/v0_8_2_priorzone_CR_seed7",
    "seed13": "trained_models/v0_8_2_priorzone_CR_seed13",
}


def generate(save_dir, n_gen, seed):
    meta = json.load(open(f"{save_dir}/mix_CR_metadata.json"))
    cfg = json.load(open(f"{save_dir}/mix_CR_config.json"))
    pp = meta["preprocessing_metadata"]
    qts = joblib.load(f"{save_dir}/mix_CR_quantile_transformers.joblib")
    idx_to_type = ({int(k): v for k, v in pp["idx_to_type"].items()}
                   if isinstance(pp["idx_to_type"], dict)
                   else dict(enumerate(pp["idx_to_type"])))
    magi.initialize_environment(seed=seed, cpu_only=True, quiet=True)
    model = magi.load_task_adaptive_model_for_generation(
        save_dir=save_dir, model_name="mix_CR", model_config=cfg,
        energy_bins=pp["energy_bins"], u_v_bins=None,
        n_types=int(pp["n_types"]), type_weights=pp.get("type_weights"),
        radius=pp.get("radius", R), compile_model_fn=magi.compile_model, verbose=0)

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
    # neutrinos, exactly as generation/export.py drops them
    names = np.asarray(out["ParticleName"], dtype=object)
    keep = ~np.isin(names, list(NON_TRANSPORT))
    return {k: v[keep] for k, v in out.items() if v.shape[0] == keep.size}


def impact(P, U, D):
    w = D - P
    t = np.einsum("ij,ij->i", w, U)
    return np.linalg.norm(w - t[:, None] * U, axis=1), t > 0


# ---- real crossings --------------------------------------------------------
NU = {"nu_e", "anti_nu_e", "nu_mu", "anti_nu_mu", "nu_tau", "anti_nu_tau"}
P, U, E, NAME = [], [], [], []
for line in open(TRAIN_FILE):
    a = line.split()
    if len(a) < 9 or a[1] in NU:
        continue
    P.append((float(a[3]), float(a[4]), float(a[5])))
    U.append((float(a[6]), float(a[7]), float(a[8])))
    E.append(float(a[2]))
    NAME.append(a[1])
P, U, E = np.array(P), np.array(U), np.array(E)
NAME = np.array(NAME, dtype=object)
b_r, in_r = impact(P, U, DET1)
print(f"real crossings: {len(E):,}", flush=True)

gens = {}
for tag, sd in MODELS.items():
    print(f"generating {N_GEN:,} from {tag}", flush=True)
    g = generate(sd, N_GEN, seed=1234)
    Pg = np.column_stack([g["x_gen"], g["y_gen"], g["z_gen"]])
    Ug = np.column_stack([g["vx_gen"], g["vy_gen"], g["vz_gen"]])
    bg, ing = impact(Pg, Ug, DET1)
    gens[tag] = dict(E=g["E_gen"], b=bg, ing=ing,
                     name=np.asarray(g["ParticleName"], dtype=object))

print()
print("=" * 96)
print("SPECIES FRACTIONS among detector-aimed crossings")
print("=" * 96)
print(f"{'cut':>12s} {'species':>8s} {'real':>10s} " +
      " ".join(f"{t:>10s}" for t in MODELS))
for c in CUTS:
    mr = in_r & (b_r < c)
    lab = "all ingoing" if not np.isfinite(c) else f"b<{c:g}mm"
    for sp in SPECIES:
        a = (NAME[mr] == sp).mean()
        if a < 1e-5:
            continue
        row = f"{lab:>12s} {sp:>8s} {a:10.4%}"
        for tag in MODELS:
            d = gens[tag]
            mg = d["ing"] & (d["b"] < c)
            row += f" {(d['name'][mg] == sp).mean() / a:10.3f}"
        print(row)
    print()

print("=" * 96)
print("MEDIAN ENERGY vs b  --  aggregate (all species) then PER SPECIES")
print("=" * 96)
for sp in (None,) + SPECIES:
    lab_sp = "AGGREGATE" if sp is None else sp
    print(f"\n--- {lab_sp} ---")
    print(f"{'cut':>12s} {'n_real':>9s} " + " ".join(f"{t:>10s}" for t in MODELS))
    for c in CUTS:
        mr = in_r & (b_r < c)
        if sp is not None:
            mr = mr & (NAME == sp)
        if mr.sum() < 20:
            continue
        med_r = float(np.median(E[mr]))
        lab = "all ingoing" if not np.isfinite(c) else f"b<{c:g}mm"
        row = f"{lab:>12s} {mr.sum():9,}"
        for tag in MODELS:
            d = gens[tag]
            mg = d["ing"] & (d["b"] < c)
            if sp is not None:
                mg = mg & (d["name"] == sp)
            row += (f" {np.median(d['E'][mg]) / med_r:10.3f}"
                    if mg.sum() >= 20 else f" {'--':>10s}")
        print(row)
