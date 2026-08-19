#!/usr/bin/env python
"""Reweight the MAGI population to the real one, to localise the conversion deficit.

THE QUESTION
    Arm A gives R = 0.846 +/- 0.088: a MAGI crossing yields ~15% fewer detector
    events than a real one. Aiming, species mix and normalisation are already
    excluded by measurement. What is left is a distortion of the JOINT
    distribution over the variables MAGI models.

THE TEST
    Force the MAGI sample to have the real joint distribution, then inject it.
      - deficit disappears  -> the defect is a density-modelling error in the
        variables we can see, and better density estimation fixes it.
      - deficit survives    -> the injected populations are statistically
        identical in every modelled variable yet still behave differently, so
        the defect lives in structure finer than the reweighting resolves, or in
        a correlation the feature set does not express.

METHOD
    Likelihood-ratio reweighting by a classifier, not histograms: 6 continuous
    variables plus species is too many for dense binning, and a classifier
    estimates p_real/p_magi in the full joint without a binning choice.
        w(x) = c(x) / (1 - c(x)),  c = P(real | x)
    Then draw the injection sample from the MAGI pool with probability
    proportional to w. Azimuths enter as (cos, sin) so periodicity is respected.

    The reweighting is VALIDATED before use: a second classifier, trained on the
    reweighted sample against real, must fall to AUC ~ 0.5. If it does not, the
    reweighting failed and the injection would prove nothing.
"""
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REAL = "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR_ingoingfix.dat"
POOL = "/Volumes/X10Pro/tmp/cr_magi_pool.txt"
OUT = "/Volumes/X10Pro/tmp/cr_rw_s%d.txt"
C = np.array([0.0, 0.0, -507.66])
# 12M pool -> 2M injected. The first attempt used 2:1 and could not reshape a
# diffuse 7-D mismatch (AUC 0.672 -> 0.608, gate 0.55). A 6:1 pool gives the
# weights room; 2M still yields ~800 all-deposit events, ~4.5% on R, which is a
# valid test at slightly worse precision rather than a precise invalid one.
N_SLICE, N_JOBS = 333_333, 6
SPEC = ["gamma", "mu-", "e-", "e+"]
RNG = np.random.default_rng(20260819)


def load(path, kind, cap):
    P, V, E, N = [], [], [], []
    for line in open(path):
        if len(P) >= cap:
            break
        a = line.split()
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
    onehot = np.zeros((len(E), len(SPEC)))
    for i, s in enumerate(SPEC):
        onehot[:, i] = (N == s)
    return np.column_stack([np.log10(np.clip(E, 1e-12, None)), r[:, 2], V[:, 2],
                            np.cos(pr), np.sin(pr), np.cos(pv), np.sin(pv), onehot])


def fit_ratio(Xa, Xb, seed=0, wb=None):
    """AUC and CALIBRATED classifier for a-vs-b (a=real=1).

    Calibration matters: w = c/(1-c) is only a density ratio if c is a
    calibrated probability, and gradient boosting is not calibrated by default.
    The uncalibrated first attempt reached only AUC 0.608 after reweighting.
    """
    X = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    sw = np.r_[np.ones(len(Xa)), (wb if wb is not None else np.ones(len(Xb)))]
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(
        X, y, sw, test_size=0.25, random_state=seed, stratify=y)
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                          max_leaf_nodes=63, random_state=seed)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr, ytr, sample_weight=str_)
    return clf, roc_auc_score(yte, clf.predict_proba(Xte)[:, 1], sample_weight=ste)


def main():
    n_need = N_SLICE * N_JOBS
    Pr, Vr, Er, Nr = load(REAL, "real", 1_500_000)
    Pp, Vp, Ep, Np = load(POOL, "gen", 12_000_000)
    print(f"real {len(Er):,} | MAGI pool {len(Ep):,} | need {n_need:,}")
    Xr, Xp = feats(Pr, Vr, Er, Nr), feats(Pp, Vp, Ep, Np)

    # ITERATED reweighting: each round refits the ratio on the CURRENT weights,
    # so residual mismatch is corrected instead of being estimated once.
    w = np.ones(len(Xp))
    auc0 = None
    for it in range(4):
        clf, auc = fit_ratio(Xr, Xp[:len(Xr)], seed=it, wb=w[:len(Xr)])
        if auc0 is None:
            auc0 = auc
            print(f"\nclassifier real-vs-MAGI AUC = {auc0:.4f}   (0.5 = indistinguishable)")
        c = np.clip(clf.predict_proba(Xp)[:, 1], 1e-6, 1 - 1e-6)
        w = w * (c / (1.0 - c))
        w = np.clip(w, 0, np.quantile(w, 0.9999))
        w = w / w.mean()
        print(f"  round {it}: AUC {auc:.4f} -> weights median {np.median(w):.3f} "
              f"max {w.max():.2f}")
        if auc < 0.515:
            break
    p = w / w.sum()
    ess = 1.0 / np.sum(p ** 2)
    print(f"weights: median {np.median(w):.3f}  max {w.max():.2f}  "
          f"effective sample size {ess:,.0f} of {len(w):,}")

    idx = RNG.choice(len(w), size=n_need, replace=True, p=p)
    dup = n_need / len(np.unique(idx))
    print(f"drew {n_need:,} with mean duplication {dup:.2f}x")

    _, auc1 = fit_ratio(Xr, Xp[idx[:len(Xr)]], seed=1)
    print(f"\nAFTER reweighting: AUC = {auc1:.4f}  (was {auc0:.4f})")
    if auc1 > 0.55:
        print("*** REWEIGHTING FAILED -- the samples are still separable.")
        print("*** Injecting would prove nothing. Stop here.")
        sys.exit(1)
    print("reweighting validated: the injected population now matches real")

    for j in range(N_JOBS):
        sl = idx[j * N_SLICE:(j + 1) * N_SLICE]
        with open(OUT % j, "w") as f:
            for i in sl:
                f.write(f"{Np[i]} {Ep[i]:.6e} {Pp[i,0]:.6e} {Pp[i,1]:.6e} {Pp[i,2]:.6e} "
                        f"{Vp[i,0]:.6e} {Vp[i,1]:.6e} {Vp[i,2]:.6e}\n")
        print(f"  wrote {OUT % j}")

    with open("/Volumes/X10Pro/tmp/cr_rw_meta.txt", "w") as f:
        f.write(f"auc_before {auc0:.6f}\nauc_after {auc1:.6f}\n"
                f"ess {ess:.0f}\ndup {dup:.4f}\npool {len(w)}\n")


if __name__ == "__main__":
    main()
