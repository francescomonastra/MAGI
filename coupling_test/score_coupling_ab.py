#!/usr/bin/env python
"""Score the head-coupling A/B by the metric that can see the defect: classifier AUC.

Also re-scores the Colab reference checkpoint, so device/seed variation between
the reference (Colab GPU) and these two (M1 CPU) is visible rather than assumed.
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

MAGI = "/Volumes/X10Pro/MAGI"
PY = "/Users/francesco/mambaforge/envs/tf-metal/bin/python"
C = np.array([0.0, 0.0, -507.66])
SPEC = ["gamma", "mu-", "e-", "e+"]
N_GEN, N_REAL = 400_000, 400_000
ARMS = [("reference (Colab)", f"{MAGI}/trained_models/v0_8_2_CR_ingoingfix"),
        ("baseline (local)", f"{MAGI}/trained_models/coupling_baseline"),
        ("COUPLED  (local)", f"{MAGI}/trained_models/coupling_coupled")]


def load(path, kind, cap):
    P, V, E, N = [], [], [], []
    for l in open(path):
        if len(P) >= cap:
            break
        a = l.split()
        if kind == "real":
            if len(a) < 13:
                continue
            N.append(a[1]); E.append(float(a[2]))
            P.append((float(a[7]), float(a[8]), float(a[9])))
            V.append((float(a[10]), float(a[11]), float(a[12])))
        else:
            if len(a) < 8:
                continue
            N.append(a[0]); E.append(float(a[1]))
            P.append((float(a[2]), float(a[3]), float(a[4])))
            V.append((float(a[5]), float(a[6]), float(a[7])))
    P, V = np.array(P), np.array(V)
    V /= np.linalg.norm(V, axis=1)[:, None]
    return P, V, np.array(E), np.array(N, dtype=object)


def feats(P, V, E, N):
    r = P - C
    r = r / np.linalg.norm(r, axis=1)[:, None]
    pr = np.arctan2(r[:, 1], r[:, 0]); pv = np.arctan2(V[:, 1], V[:, 0])
    oh = np.zeros((len(E), 4))
    for i, s in enumerate(SPEC):
        oh[:, i] = (N == s)
    return np.column_stack([np.log10(np.clip(E, 1e-12, None)), r[:, 2], V[:, 2],
                            np.cos(pr), np.sin(pr), np.cos(pv), np.sin(pv), oh]), r


def auc_vs_real(Xg, Xr, seed=0):
    X = np.vstack([Xr, Xg]); y = np.r_[np.ones(len(Xr)), np.zeros(len(Xg))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63, random_state=seed),
        method="isotonic", cv=3).fit(Xtr, ytr)
    c = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, c), clf


Pr, Vr, Er, Nr = load(f"{MAGI}/TrainingData/alloutputDSCryoSphereCR_ingoingfix.dat",
                      "real", N_REAL)
Xr, rr = feats(Pr, Vr, Er, Nr)
print("=" * 82)
print("HEAD-COUPLING A/B -- scored by classifier AUC (0.5 = indistinguishable)")
print("=" * 82)
print(f"real reference: {len(Er):,} crossings\n")

rows = []
for lab, sd in ARMS:
    if not os.path.isdir(sd):
        print(f"{lab:20s} MISSING ({sd})")
        continue
    out = f"/Volumes/X10Pro/tmp/coup_{os.path.basename(sd)}.txt"
    if not os.path.exists(out):
        r = subprocess.run(
            [PY, f"{MAGI}/scripts/generate_geant_source.py", "--save-dir", sd,
             "--model-name", "mix_CR", "--metadata-file", f"{sd}/mix_CR_metadata.json",
             "--transformers-file", f"{sd}/mix_CR_quantile_transformers.joblib",
             "--output-file", out, "--n-events", str(N_GEN), "--seed", "20260819",
             "--format", "text"], capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
        if r.returncode != 0 or "WARNING" in r.stdout:
            print(f"{lab:20s} GENERATION PROBLEM"); print(r.stdout[-700:], r.stderr[-500:])
            continue
    Pg, Vg, Eg, Ng = load(out, "gen", N_GEN)
    Xg, rg = feats(Pg, Vg, Eg, Ng)
    auc, clf = auc_vs_real(Xg, Xr)
    cg = clf.predict_proba(Xg)[:, 1]
    cr = clf.predict_proba(Xr)[:, 1]
    hole = float((cr > 0.8).mean()) / max(float((cg > 0.8).mean()), 1e-9)
    hi = cr > 0.8
    row = dict(arm=lab, auc=float(auc), coverage_ratio=float(hole),
               frac_real_hi=float(hi.mean()),
               hole_gamma=float((Nr[hi] == "gamma").mean()) if hi.any() else float("nan"),
               hole_medE_keV=float(np.median(Er[hi]) * 1e3) if hi.any() else float("nan"))
    rows.append(row)
    print(f"{lab:20s} AUC = {auc:.4f}   c>0.8 density ratio {hole:6.1f}x   "
          f"hole: {row['hole_gamma']:.0%} gamma at {row['hole_medE_keV']:.0f} keV")

json.dump(rows, open(f"{MAGI}/coupling_test/coupling_auc.json", "w"), indent=2)
print("\n" + "-" * 82)
if len(rows) >= 2:
    b = next((r for r in rows if "baseline" in r["arm"]), None)
    c_ = next((r for r in rows if "COUPLED" in r["arm"]), None)
    if b and c_:
        d = c_["auc"] - b["auc"]
        print(f"coupling changes AUC by {d:+.4f}  ({b['auc']:.4f} -> {c_['auc']:.4f})")
        if d < -0.03:
            print("COUPLING HELPS. v0.8.3 was rejected on a metric that could not see this.")
        elif d > 0.01:
            print("COUPLING HURTS on this metric.")
        else:
            print("NO MEANINGFUL CHANGE. The conditional-independence edge between energy")
            print("and geometry is not what the classifier is reading -- look to the")
            print("geometry heads themselves (Gaussian/unit-circle, no flow) or the latent.")
