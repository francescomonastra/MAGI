#!/usr/bin/env python
"""KDSource vs MAGI on the same CR crossing population, same variables, same N.

WHY THIS DESIGN
---------------
KDE resampling is a smoothed bootstrap: evaluated against the sample it was
fitted on it cannot help but look near-perfect, so an in-sample comparison would
be meaningless and flattering to KDSource. MAGI is trained on a 70% split
(`split_feature_data(test_size_total=0.30)`), so a HELD-OUT protocol is both
fair and available:

    fit KDE on 70%   ->   compare BOTH generators against the held-out 30%

Caveat recorded honestly: MAGI's 70/30 draw used its own random_state, so the
two splits are statistically equivalent but not event-identical. The held-out
set here is therefore unseen by KDE by construction and unseen by MAGI only in
distribution. That favours MAGI slightly and is stated in the output.

WHY NOT KDSource's Geometry API
-------------------------------
KDSource ships no spherical-surface metric (Vol, SurfXY, SurfR, SurfR2,
SurfCircle, Guide, Isotrop, Polar, PolarMu) and our source is a spherical shell,
where a Vol metric would put the position density on a degenerate 2-D manifold
inside 3-D. Forcing that would handicap KDSource for a reason that has nothing
to do with its density model. Instead we apply KDSource's OWN recipe --
standardise by per-variable std, select bandwidth with kdsource.bw_silv /
bw_knn, resample by smoothed bootstrap exactly as KDS_sample does -- to the same
five variables MAGI uses. This is the most favourable fair setup for KDSource.

MULTISOURCE
-----------
MultiSource (C API only, not exported to Python) picks a source from a weight
CDF then samples it. Drawing N_s = w_s*N per species and concatenating is
mathematically identical, so no C binding is needed. We hand KDSource the TRUE
observed species fractions, which is the best case for it -- MultiSource weights
are user-supplied, not learned.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Volumes/X10Pro/Old-Simulations/Sim/KDSource/python")
from kdsource import bw_silv                      # noqa: E402
from kdsource.kde import bw_knn                   # noqa: E402

CENTER = np.array([0.0, 0.0, -507.66])
TRAIN = "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR.dat"
MAGI_GEN = "/Volumes/X10Pro/tmp/magi_crseed_s0.gntbin"
MAGI_CKPT = "/Volumes/X10Pro/MAGI/trained_models/v0_8_2_priorzone_CR"
NU = {"nu_e", "anti_nu_e", "nu_mu", "anti_nu_mu", "nu_tau", "anti_nu_tau"}
LABELS = ["log10E", "u_r", "u_v", "phi_r", "phi_v"]
BW_SUBSAMPLE = 150_000      # cap for the kNN bandwidth search
RNG = np.random.default_rng(20260814)


def features(P, U, E):
    r = P - CENTER
    r = r / np.linalg.norm(r, axis=1)[:, None]
    v = U / np.linalg.norm(U, axis=1)[:, None]
    return np.column_stack([np.log10(np.clip(E, 1e-12, None)),
                            r[:, 2], v[:, 2],
                            np.arctan2(r[:, 1], r[:, 0]),
                            np.arctan2(v[:, 1], v[:, 0])])


def load_real():
    P, U, E, N = [], [], [], []
    with open(TRAIN) as fh:
        for line in fh:
            a = line.split()
            if len(a) < 9 or a[1] in NU:
                continue
            P.append((float(a[3]), float(a[4]), float(a[5])))
            U.append((float(a[6]), float(a[7]), float(a[8])))
            E.append(float(a[2]))
            N.append(a[1])
    return features(np.array(P), np.array(U), np.array(E)), np.array(N, dtype=object)


def load_magi():
    REC = np.dtype([("pdg", "<i4"), ("v", "<f4", 7)])
    n = (os.path.getsize(MAGI_GEN) - 20) // 32
    g = np.fromfile(MAGI_GEN, dtype=np.uint8, count=20 + n * 32)[20:].view(REC)[:n]
    pdg2name = {22: "gamma", 11: "e-", -11: "e+", 13: "mu-", -13: "mu+", 2212: "proton"}
    names = np.array([pdg2name.get(int(p), str(p)) for p in g["pdg"]], dtype=object)
    return features(g["v"][:, 1:4].astype(float), g["v"][:, 4:7].astype(float),
                    g["v"][:, 0].astype(float)), names


def kde_resample(X_train, n_out, method, k_eff=100):
    """KDSource's model: standardise, pick bw with its own selector, smoothed
    bootstrap exactly as KDS_sample (draw a training particle, perturb by the
    kernel scaled by bw*scaling)."""
    scaling = X_train.std(axis=0)
    scaling[scaling == 0] = 1.0
    Z = X_train / scaling
    if method == "silv":
        bw = np.full(len(Z), bw_silv(Z.shape[1], len(Z)))
    else:
        # Per-point adaptive bandwidths on the FULL set. Do NOT subsample and
        # reassign: each bandwidth is tied to the local density of its own
        # point (the range spans ~30x), so shuffling them destroys the
        # adaptivity that is KDSource's headline feature and broadens the
        # sample badly -- measured at 0.65 RMS on Gaussian-vs-itself, against
        # 0.02 when done properly. bw_knn batches internally and costs ~4 s per
        # 400k points, so there is no reason to approximate.
        # weights must be explicit: kdsource.kde.bw_knn line 75 evaluates
        # `weights ** 2` without guarding the documented None default.
        bw = np.asarray(bw_knn(Z, weights=np.ones(len(Z)),
                              K_eff=k_eff)).ravel()
    idx = RNG.integers(0, len(Z), size=n_out)
    out = Z[idx] + RNG.standard_normal((n_out, Z.shape[1])) * bw[idx][:, None]
    return out * scaling


MIN_BIN = 50   # reference counts required before a bin enters the RMS


def marginal_rms(ref, gen):
    """RMS of the generated/reference density ratio, over well-populated bins.

    Bins are weighted by COUNTS, not density: with a threshold of MIN_BIN the
    sparse tails cannot dominate. Without it a Gaussian resampled from itself
    scores 2.5 purely from empty end-bins, which would wreck precisely the
    rare-species comparison this benchmark exists to make.
    """
    devs = []
    for k in range(ref.shape[1]):
        lo, hi = np.percentile(ref[:, k], [0.2, 99.8])
        b = np.linspace(lo, hi, 24)
        na, _ = np.histogram(ref[:, k], bins=b)
        nc, _ = np.histogram(gen[:, k], bins=b)
        ok = na >= MIN_BIN
        if ok.sum() < 5:
            devs.append(float("nan"))
            continue
        ratio = (nc[ok] / nc.sum()) / (na[ok] / na.sum())
        devs.append(float(np.std(ratio)))
    return devs


def corr_resid(ref, gen):
    Cr, Cg = np.corrcoef(ref, rowvar=False), np.corrcoef(gen, rowvar=False)
    iu = np.triu_indices(ref.shape[1], 1)
    d = np.abs(Cg - Cr)[iu]
    return float(d.mean()), float(d.max())


def main():
    t0 = time.time()
    X, names = load_real()
    print(f"real crossings (neutrino-free): {len(X):,}")
    species, counts = np.unique(names, return_counts=True)
    frac = counts / counts.sum()
    for s, c, f in zip(species, counts, frac):
        print(f"   {s:8s} {c:9,}  {f:7.4%}")

    # 70/30 split; KDE fits on train only, both generators judged on held-out
    perm = RNG.permutation(len(X))
    ntr = int(0.70 * len(X))
    tr, te = perm[:ntr], perm[ntr:]
    X_tr, n_tr = X[tr], names[tr]
    X_te, n_te = X[te], names[te]
    print(f"\nsplit: {len(X_tr):,} train / {len(X_te):,} held-out")

    Xg, ng = load_magi()
    N_OUT = len(Xg)
    print(f"MAGI generated: {N_OUT:,}\n")

    # KDSource's constructor default is bw='silv'. knn is its adaptive option;
    # at the default K_eff=100 it over-smooths badly (0.44-0.71 RMS on
    # Gaussian-vs-itself against 0.07 for silv), so we sweep K_eff and report
    # every setting rather than judging KDSource on one untuned choice.
    CONFIGS = [("silv", None), ("knn", 10), ("knn", 30), ("knn", 100)]
    results = {}
    for method, keff in CONFIGS:
        tag = method if keff is None else f"knn K={keff}"
        parts, pnames = [], []
        for s in species:
            m = n_tr == s
            n_s = int(round(N_OUT * frac[species.tolist().index(s)]))
            if m.sum() < 50 or n_s < 1:
                print(f"   [{tag}] {s}: only {m.sum()} train events, skipped")
                continue
            t = time.time()
            parts.append(kde_resample(X_tr[m], n_s, method, k_eff=keff or 100))
            pnames.append(np.full(n_s, s, dtype=object))
            print(f"   [{tag}] {s:8s} fit+resample {n_s:8,} in {time.time()-t:5.1f}s")
        results[tag] = (np.vstack(parts), np.concatenate(pnames))

    print()
    print("=" * 78)
    print("HELD-OUT COMPARISON  (all generators judged against the same 30%)")
    print("=" * 78)
    hdr = f"{'generator':<18s} {'mean|drho|':>11s} {'max':>7s}   " + \
          " ".join(f"{l:>7s}" for l in LABELS)
    print(hdr)
    rows = [("MAGI v0.8.2", Xg, ng)]
    rows += [(f"KDSource {m}", results[m][0], results[m][1]) for m in results]
    for tag, G, _ in rows:
        mr, mx = corr_resid(X_te, G)
        dv = marginal_rms(X_te, G)
        print(f"{tag:<18s} {mr:11.4f} {mx:7.4f}   " + " ".join(f"{d:7.3f}" for d in dv))

    print()
    print("PER-SPECIES marginal RMS (the rare-species prediction)")
    print(f"{'species':<8s} {'abundance':>10s} {'train N':>10s}  " +
          "  ".join(f"{t:>16s}" for t, _, _ in rows))
    for s in species:
        mt = n_te == s
        if mt.sum() < 200:
            continue
        line = (f"{s:<8s} {frac[species.tolist().index(s)]:10.4%} "
                f"{int((n_tr == s).sum()):10,}  ")
        for _, G, GN in rows:
            mg = GN == s
            line += (f"{np.mean(marginal_rms(X_te[mt], G[mg])):16.3f}  "
                     if mg.sum() > 200 else f"{'--':>16s}  ")
        print(line)

    ck = sum(os.path.getsize(os.path.join(MAGI_CKPT, f))
             for f in os.listdir(MAGI_CKPT) if not f.endswith(".html"))
    kde_bytes = X_tr.nbytes
    print()
    print("FOOTPRINT (what must be retained to generate)")
    print(f"  MAGI checkpoint         : {ck/1e6:8.2f} MB")
    print(f"  KDE training sample     : {kde_bytes/1e6:8.2f} MB "
          f"({len(X_tr):,} x 5 float64; KDE resampling cannot discard it)")
    print()
    print("CAVEAT: the held-out 30% is unseen by KDE by construction, and unseen")
    print("by MAGI only in distribution (its own split used a different draw).")
    print("That favours MAGI slightly.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
