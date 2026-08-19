#!/usr/bin/env python
"""Validate the corrected CR training set against the censored one it replaces.

Both campaigns threw 5.4e8 primaries into the same geometry, so everything here
is a PER-PRIMARY yield, not a fraction. Fractions are what made the bug hard to
see: the total crossing count changed by only 4.4 % while the composition moved
by tens of percent.

The load-bearing question is not "did the population change" -- the pilot
settled that -- but "did the AIMED population change", because that is what the
detector sees and what the 15-24 % CR deficit is about. A prediction made here,
before retraining, is worth more than the same number extracted afterwards.
"""
import numpy as np

OLD = "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR.dat"
NEW = "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR_ingoingfix.dat"
N_PRIM = 5.4e8                      # identical on both sides: 1e6*540 and 6*9e7
C = np.array([0.0, 0.0, -507.66])
DET1 = np.array([-0.1, -0.25, -471.54])
NU = {"nu_e", "anti_nu_e", "nu_mu", "anti_nu_mu", "nu_tau", "anti_nu_tau"}


def load(path):
    """RAW output is 13 columns (pos 7-9, dir 10-12); CLEANED is 9 (3-5, 6-8)."""
    P, V, E, N = [], [], [], []
    for line in open(path):
        a = line.split()
        if len(a) < 9 or a[1] in NU:
            continue
        pi, vi = (7, 10) if len(a) >= 13 else (3, 6)
        P.append((float(a[pi]), float(a[pi + 1]), float(a[pi + 2])))
        V.append((float(a[vi]), float(a[vi + 1]), float(a[vi + 2])))
        E.append(float(a[2]))
        N.append(a[1])
    P, V = np.array(P), np.array(V)
    V /= np.linalg.norm(V, axis=1)[:, None]
    return P, V, np.array(E), np.array(N, dtype=object)


def impact(P, U, D):
    w = D - P
    t = np.einsum("ij,ij->i", w, U)
    return np.linalg.norm(w - t[:, None] * U, axis=1), t > 0


def main():
    sets = {}
    for tag, path in (("old (censored)", OLD), ("new (fixed)", NEW)):
        P, V, E, N = load(path)
        r = P - C
        r /= np.linalg.norm(r, axis=1)[:, None]
        ur = (V * r).sum(1)
        b, ing = impact(P, V, DET1)
        sets[tag] = dict(P=P, V=V, E=E, N=N, ur=ur, rz=r[:, 2], b=b, ing=ing)
        print(f"{tag:15s}: {len(E):,} crossings   chi = {len(E)/N_PRIM:.4e}   "
              f"outgoing = {(ur > 0).mean():.3%}")
    o, n = sets["old (censored)"], sets["new (fixed)"]

    print("\n" + "=" * 74)
    print("PER-PRIMARY YIELD BY SPECIES  (what injection actually delivers)")
    print("=" * 74)
    print(f"{'species':<8s} {'old':>12s} {'new':>12s} {'new/old':>9s}")
    for s in ("gamma", "mu-", "e-", "e+", "proton"):
        yo = (o["N"] == s).sum() / N_PRIM
        yn = (n["N"] == s).sum() / N_PRIM
        if yo == 0 and yn == 0:
            continue
        print(f"{s:<8s} {yo:12.4e} {yn:12.4e} {yn/yo if yo else np.nan:9.3f}")

    print("\n" + "=" * 74)
    print("AIMED YIELD PER PRIMARY -- the quantity the detector responds to")
    print("=" * 74)
    print(f"{'cut':<14s} {'species':<8s} {'old':>12s} {'new':>12s} {'new/old':>9s}")
    for lab, bmax in (("all ingoing", None), ("aimed b<50", 50),
                      ("aimed b<20", 20), ("aimed b<10", 10)):
        mo = o["ing"] if bmax is None else (o["ing"] & (o["b"] < bmax))
        mn = n["ing"] if bmax is None else (n["ing"] & (n["b"] < bmax))
        yo, yn = mo.sum() / N_PRIM, mn.sum() / N_PRIM
        print(f"{lab:<14s} {'ALL':<8s} {yo:12.4e} {yn:12.4e} {yn/yo:9.3f}")
        for s in ("mu-", "gamma", "e-"):
            yos = (mo & (o["N"] == s)).sum() / N_PRIM
            yns = (mn & (n["N"] == s)).sum() / N_PRIM
            print(f"{'':<14s} {s:<8s} {yos:12.4e} {yns:12.4e} "
                  f"{yns/yos if yos else np.nan:9.3f}")

    print("\n" + "=" * 74)
    print("ANGULAR / ENERGY SHAPE")
    print("=" * 74)
    print(f"{'quantity':<26s} {'old':>10s} {'new':>10s} {'new/old':>9s}")
    for lab, ko, kn in (("mean u_r", o["ur"].mean(), n["ur"].mean()),
                        ("mean r_z", o["rz"].mean(), n["rz"].mean()),
                        ("mean v_z", o["V"][:, 2].mean(), n["V"][:, 2].mean()),
                        ("v_z>0 fraction", (o["V"][:, 2] > 0).mean(),
                         (n["V"][:, 2] > 0).mean())):
        print(f"{lab:<26s} {ko:10.4f} {kn:10.4f} {kn/ko:9.3f}")
    for s in ("mu-", "gamma"):
        eo, en = o["E"][o["N"] == s], n["E"][n["N"] == s]
        print(f"median E {s:<17s} {np.median(eo):10.4f} {np.median(en):10.4f} "
              f"{np.median(en)/np.median(eo):9.3f}  MeV")

    print("\nPREDICTION for the detector-level retest, stated before retraining:")
    mo = o["ing"] & (o["b"] < 20)
    mn = n["ing"] & (n["b"] < 20)
    r_all = (mn.sum() / mo.sum())
    r_mu = ((mn & (n["N"] == "mu-")).sum() / (mo & (o["N"] == "mu-")).sum())
    print(f"  aimed flux b<20      x {r_all:.3f}")
    print(f"  aimed muon flux b<20 x {r_mu:.3f}")
    print("  The 1-7 keV MIP region was 0.750 of the full simulation and is")
    print("  muon-dominated (84 % of reference events), so it should move with")
    print("  the aimed MUON factor. The soft band should move with the total.")


if __name__ == "__main__":
    main()
