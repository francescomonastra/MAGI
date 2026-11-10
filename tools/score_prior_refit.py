#!/usr/bin/env python
"""Score post-hoc prior refits (tools/refit_prior_posthoc.py) against the
v0_8_2_CR_ingoingfix baseline on the generated sample, with real generated
energy (the paper protocol), not the Phase-0 real-energy arms.

Per arm, against the same 400k real crossings:
    AUC            calibrated HGB classifier, reweight_magi_to_real features
    coverage       real:MAGI density ratio at classifier c > 0.7/0.8/0.9/0.95
                   (the v0.9.0/v0.9.1 protocol; AUC alone is not safe here,
                   docs/v1.0.0_release_plan.md sec 2.2)
    Pearson resid  mean |C_gen - C_real| over [log10E, u_r, u_v, phi_r, phi_v]
                   (tools/kdsource_vs_magi.py corr_resid)
    per species    species-internal AUC
    inward only    AUC restricted to r.v < 0 on both sides

Usage:
    python tools/score_prior_refit.py <run_dir> [<run_dir> ...]
"""
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reweight_magi_to_real as rw  # noqa: E402

ROOT = "/Volumes/T7/MAGI-dev"
PY = "/Users/francesco/mambaforge/envs/tf-metal/bin/python"
REAL = f"{ROOT}/TrainingData/alloutputDSCryoSphereCR_ingoingfix.dat"
GEN_DIR = f"{ROOT}/checkpoints/generated_particles"
N_GEN = N_REAL = 400_000
SEED = 20260819


def auc_calibrated(Xr, Xg, seed=0):
    X = np.vstack([Xr, Xg]); y = np.r_[np.ones(len(Xr)), np.zeros(len(Xg))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63, random_state=seed),
        method="isotonic", cv=3).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), clf


def auc_plain(Xr, Xg, seed=0):
    X = np.vstack([Xr, Xg]); y = np.r_[np.ones(len(Xr)), np.zeros(len(Xg))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                       random_state=seed).fit(Xtr, ytr)
    return roc_auc_score(yte, c.predict_proba(Xte)[:, 1])


def pearson_feats(X):
    return np.column_stack([X[:, 0], X[:, 1], X[:, 2],
                            np.arctan2(X[:, 4], X[:, 3]), np.arctan2(X[:, 6], X[:, 5])])


def corr_resid(ref, gen):
    Cr, Cg = np.corrcoef(ref, rowvar=False), np.corrcoef(gen, rowvar=False)
    d = np.abs(Cg - Cr)[np.triu_indices(ref.shape[1], 1)]
    return float(d.mean()), float(d.max())


def generate(sd):
    if sd.endswith(".txt"):  # an already-generated text file: score it as is
        return sd
    os.makedirs(GEN_DIR, exist_ok=True)
    out = f"{GEN_DIR}/priorrefit_{os.path.basename(sd)}_{N_GEN}.txt"
    if not os.path.exists(out):
        r = subprocess.run(
            [PY, f"{ROOT}/scripts/generate_geant_source.py", "--save-dir", sd,
             "--model-name", "mix_CR", "--metadata-file", f"{sd}/mix_CR_metadata.json",
             "--transformers-file", f"{sd}/mix_CR_quantile_transformers.joblib",
             "--output-file", out, "--n-events", str(N_GEN), "--seed", str(SEED),
             "--format", "text"], capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1",
                 "PYTHONPATH": f"{ROOT}/MAGI_package"})
        if r.returncode != 0:
            print(r.stdout[-700:], r.stderr[-1500:])
            raise SystemExit(f"generation failed for {sd}")
    return out


def main():
    Pr, Vr, Er, Nr = rw.load(REAL, "real", N_REAL)
    Xr = rw.feats(Pr, Vr, Er, Nr)
    in_r = np.einsum("ij,ij->i", Pr - rw.C, Vr) < 0
    print(f"real reference {len(Er):,}  inward fraction {in_r.mean():.4f}\n", flush=True)
    rows = []
    for sd in sys.argv[1:]:
        Pg, Vg, Eg, Ng = rw.load(generate(sd), "gen", N_GEN)
        Xg = rw.feats(Pg, Vg, Eg, Ng)
        n = min(len(Xr), len(Xg))
        auc, clf = auc_calibrated(Xr[:n], Xg[:n])
        cr, cg = clf.predict_proba(Xr[:n])[:, 1], clf.predict_proba(Xg[:n])[:, 1]
        ratios = {t: float((cr > t).mean()) / max(float((cg > t).mean()), 1e-9)
                  for t in (0.7, 0.8, 0.9, 0.95)}
        counts = {t: [int((cr > t).sum()), int((cg > t).sum())] for t in (0.7, 0.8, 0.9)}
        pm, px = corr_resid(pearson_feats(Xr[:n]), pearson_feats(Xg[:n]))
        per = {}
        for s in rw.SPEC:
            a, b = Xr[:n][Nr[:n] == s], Xg[:n][Ng[:n] == s]
            m = min(len(a), len(b))
            if m > 2000:
                per[s] = float(auc_plain(a[:m], b[:m]))
        in_g = np.einsum("ij,ij->i", Pg - rw.C, Vg)[:n] < 0
        a, b = Xr[:n][in_r[:n]], Xg[:n][in_g]
        m = min(len(a), len(b))
        row = dict(run=os.path.basename(sd), n=n, auc=float(auc), auc_plain=float(auc_plain(Xr[:n], Xg[:n])),
                   coverage=ratios, coverage_counts_real_gen=counts, pearson_mean=pm, pearson_max=px, auc_per_species=per,
                   inward_frac_gen=float(in_g.mean()), auc_inward=float(auc_plain(a[:m], b[:m])),
                   species_mix={s: float(np.mean(Ng[:n] == s)) for s in rw.SPEC})
        rows.append(row)
        print(f"{row['run']}\n  AUC {auc:.4f} (uncalibrated {row['auc_plain']:.4f})  "
              f"inward-only {row['auc_inward']:.4f}  gen inward {row['inward_frac_gen']:.4f}\n"
              f"  coverage c>0.7 {ratios[0.7]:.2f}x  c>0.8 {ratios[0.8]:.2f}x  "
              f"c>0.9 {ratios[0.9]:.2f}x  c>0.95 {ratios[0.95]:.2f}x\n"
              f"  counts real/gen c>0.7 {counts[0.7]}  c>0.8 {counts[0.8]}  c>0.9 {counts[0.9]}\n"
              f"  Pearson residual mean {pm:.4f}  max {px:.4f}\n"
              f"  per species {({k: round(v, 4) for k, v in per.items()})}\n"
              f"  species mix {({k: round(v, 4) for k, v in row['species_mix'].items()})}",
              flush=True)
    json.dump(rows, open(os.environ.get("SCORES_OUT", f"{ROOT}/checkpoints/prior_refit_scores.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
