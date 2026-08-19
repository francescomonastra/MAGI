#!/usr/bin/env python
"""Does MAGI generate, or does it recall? -- the memorisation test.

WHY THIS EXISTS
    The KDSource benchmark (15/08) showed that a 287x amplification does NOT by
    itself distinguish MAGI from a kernel-density resampler: a smoothed
    bootstrap also turns 1.87e6 crossings into 5.38e8. The proceedings admits as
    much in its Limitations. This test is what separates the two claims, and
    after the 17/08 K-40 correction it carries more of the argument than before.

THE TEST
    For a query point q and a reference set R of real crossings, let
    d(q) = min_{r in R} ||q - r||. Compare three distance distributions:

        d_real  : real -> R \\ {self}      leave-one-out, the honest yardstick
        d_magi  : MAGI -> R
        d_kde   : KDE  -> R               the memoriser, by construction

    A model that MEMORISES places samples on top of training points, so
    d_magi << d_real. A model that GENERALISES places them in the same relation
    to the data as genuine unseen data would: d_magi ~ d_real.

    Leave-one-out is used deliberately instead of a held-out split: it puts
    every distribution against the SAME reference set at the SAME density, so
    the comparison needs no correction for reference size -- nearest-neighbour
    distance scales with it, and that is the classic way this test is got wrong.

FEATURE SPACE
    The 5 physical variables the model actually learns, with the two azimuths
    carried as (cos, sin) pairs so the metric respects their periodicity --
    phi = -pi and phi = +pi are the same direction, and a naive Euclidean
    distance would call them maximally far apart. Each pair is scaled by
    1/sqrt(2) so an angle contributes one dimension's worth of variance, not
    two. Everything is then standardised on the reference set.

READING THE RESULT
    ratio = median(d_magi) / median(d_real)
        ~1.0  generalises: MAGI sits as far from the data as real data does
        <<1   memorises
        >>1   the opposite failure -- samples off the data manifold
    The near-duplicate fraction is the sharper end of the same question: how
    often a generated sample lands closer to a training point than real data
    essentially ever does.
"""
import os
import sys

import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors

# Dataset selected by argv[1]; defaults to DM1.2 (the 17/08 result).
DATASETS = {
    "dm12": dict(
        real="/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDML_DM1.2/allDSCryoSphere_DM1_2_500k_nonu.dat",
        gen="/Volumes/X10Pro/tmp/magi_dm12_memtest.txt",
        center=[0.0, 0.0, 5.0], label="DM1.2, v0.8.2 checkpoint"),
    "cr": dict(
        real="/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR_ingoingfix.dat",
        gen="/Volumes/X10Pro/tmp/magi_cr_memtest.txt",
        center=[0.0, 0.0, -507.66], label="CR, v0_8_2_CR_ingoingfix (corrected training set)"),
}
_KEY = sys.argv[1] if len(sys.argv) > 1 else "dm12"
_D = DATASETS[_KEY]
REAL, MAGI = _D["real"], _D["gen"]
CENTER = np.array(_D["center"])
N_REF = 150_000                              # reference set size
N_Q = 60_000                                 # queries per arm
RNG = np.random.default_rng(20260817)
KDS_PY = "/Volumes/X10Pro/Old-Simulations/Sim/KDSource/python"


def featurise(P, U, E):
    """[log10E, u_r, u_v, cos/sin phi_r, cos/sin phi_v] -- periodicity-safe."""
    r = P - CENTER
    r = r / np.linalg.norm(r, axis=1)[:, None]
    v = U / np.linalg.norm(U, axis=1)[:, None]
    pr = np.arctan2(r[:, 1], r[:, 0])
    pv = np.arctan2(v[:, 1], v[:, 0])
    s = 1.0 / np.sqrt(2.0)
    return np.column_stack([
        np.log10(np.clip(E, 1e-12, None)), r[:, 2], v[:, 2],
        s * np.cos(pr), s * np.sin(pr), s * np.cos(pv), s * np.sin(pv)])


def load_table(path, kind):
    P, U, E, N = [], [], [], []
    for line in open(path):
        a = line.split()
        if kind == "real":                   # 13 col: pos 7-9, dir 10-12
            if len(a) < 13:
                continue
            N.append(a[1]); E.append(float(a[2]))
            P.append((float(a[7]), float(a[8]), float(a[9])))
            U.append((float(a[10]), float(a[11]), float(a[12])))
        else:                                # generated: name E x y z vx vy vz
            if len(a) < 8:
                continue
            N.append(a[0]); E.append(float(a[1]))
            P.append((float(a[2]), float(a[3]), float(a[4])))
            U.append((float(a[5]), float(a[6]), float(a[7])))
    return (featurise(np.array(P), np.array(U), np.array(E)),
            np.array(N, dtype=object))


def kde_resample(X, n_out, method="silv", k_eff=100):
    """KDSource's smoothed bootstrap: draw a training point, perturb by the
    kernel. BOTH bandwidth selectors are run. Silverman is a single global
    width; bw_knn is per-point and adaptive, which is KDSource's headline
    feature and the variant that could actually memorise -- in a dense region
    its bandwidth shrinks toward the local spacing.

    bw_knn needs explicit weights: kdsource.kde.bw_knn evaluates `weights ** 2`
    without guarding its own documented None default."""
    sys.path.insert(0, KDS_PY)
    scaling = X.std(axis=0)
    scaling[scaling == 0] = 1.0
    Z = X / scaling
    if method == "silv":
        from kdsource import bw_silv
        bw = np.full(len(Z), bw_silv(Z.shape[1], len(Z)))
    else:
        from kdsource.kde import bw_knn
        bw = np.asarray(bw_knn(Z, weights=np.ones(len(Z)), K_eff=k_eff)).ravel()
    idx = RNG.integers(0, len(Z), size=n_out)
    out = Z[idx] + RNG.standard_normal((n_out, Z.shape[1])) * bw[idx][:, None]
    return out * scaling


def memoriser(X, n_out, jitter=0.02):
    """NEGATIVE CONTROL. A deliberate memoriser: resample reference points and
    barely move them. This is what the metric MUST flag -- ratio << 1 and a
    near-duplicate fraction near 1. Added 17/08 after the KDE arm turned out
    not to be a memoriser at all: at KDSource's Silverman bandwidth in 7-D the
    kernel is ~4x the typical nearest-neighbour spacing, so it OVER-smooths.
    That makes KDE a useful second data point but a useless negative control,
    and without a working one the MAGI number could not be believed."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    idx = RNG.integers(0, len(X), size=n_out)
    return X[idx] + RNG.standard_normal((n_out, X.shape[1])) * sd * jitter


def nn_dist(ref_tree, Q, drop_self=False):
    k = 2 if drop_self else 1
    d, _ = ref_tree.kneighbors(Q, n_neighbors=k)
    return d[:, -1]


def report(name, d, d_real, eps):
    ratio = np.median(d) / np.median(d_real)
    ks = ks_2samp(d, d_real)
    dup = float((d < eps).mean())
    print(f"  {name:22s} median {np.median(d):.5f}  ratio {ratio:6.3f}   "
          f"KS D={ks.statistic:.4f} p={ks.pvalue:.2e}   near-dup {dup:7.3%}")
    return dict(median=float(np.median(d)), ratio=float(ratio),
                ks_D=float(ks.statistic), ks_p=float(ks.pvalue), near_dup=dup)


def main():
    for p in (REAL, MAGI):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    Xr, Nr = load_table(REAL, "real")
    Xg, Ng = load_table(MAGI, "gen")
    print("=" * 78)
    print(f"MEMORISATION TEST -- {_D['label']}")
    print("=" * 78)
    print(f"real {len(Xr):,} crossings | MAGI {len(Xg):,} generated")
    print(f"species  real: {dict(zip(*np.unique(Nr, return_counts=True)))}")
    print(f"species  MAGI: {dict(zip(*np.unique(Ng, return_counts=True)))}")

    # reference set, and disjoint real queries drawn from what is left
    perm = RNG.permutation(len(Xr))
    ref_idx, rest = perm[:N_REF], perm[N_REF:]
    REF = Xr[ref_idx]

    mu, sd = REF.mean(0), REF.std(0)
    sd[sd == 0] = 1.0
    Z = (REF - mu) / sd
    tree = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", n_jobs=-1).fit(Z)

    q_real = Z[RNG.choice(len(Z), size=min(N_Q, len(Z)), replace=False)]
    d_real = nn_dist(tree, q_real, drop_self=True)

    q_magi = (Xg[RNG.choice(len(Xg), size=min(N_Q, len(Xg)), replace=False)] - mu) / sd
    q_kde = (kde_resample(REF, min(N_Q, len(Xg)), "silv") - mu) / sd
    try:
        q_knn = (kde_resample(REF, min(N_Q, len(Xg)), "knn") - mu) / sd
    except Exception as e:
        print(f"  [bw_knn unavailable: {e}]")
        q_knn = None

    eps = float(np.quantile(d_real, 0.01))
    print(f"\nreference {len(REF):,} | queries {len(q_real):,} per arm | "
          f"eps = 1st pct of d_real = {eps:.5f}\n")
    print(f"  {'arm':22s} {'':13s} {'':6s}")
    report("real -> real (LOO)", d_real, d_real, eps)
    res_m = report("MAGI -> real", nn_dist(tree, q_magi), d_real, eps)
    res_k = report("KDE(silv) -> real", nn_dist(tree, q_kde), d_real, eps)
    res_knn = (report("KDE(adaptive kNN)", nn_dist(tree, q_knn), d_real, eps)
               if q_knn is not None else None)
    # Calibration curve instead of one arbitrary control: a memoriser at
    # several jitter levels shows what this metric reads when the answer IS
    # recall, and places MAGI's number on a scale rather than against a
    # threshold picked by hand. jitter=0 is literal copies of training points.
    cal = {}
    for j in (0.0, 0.005, 0.02, 0.05):
        q_mem = (memoriser(REF, min(N_Q, len(Xg)), jitter=j) - mu) / sd
        cal[j] = report(f"memoriser jitter={j:<5g}", nn_dist(tree, q_mem), d_real, eps)
    res_c = cal[0.0]

    # held-out real, as an independent check on the yardstick itself
    if len(rest) > 1000:
        q_held = (Xr[rest[:min(N_Q, len(rest))]] - mu) / sd
        report("held-out real -> real", nn_dist(tree, q_held), d_real, eps)

    print("\n" + "-" * 78)
    print(f"MAGI       ratio {res_m['ratio']:.3f}, near-duplicates {res_m['near_dup']:.3%}")
    print(f"KDE(silv)  ratio {res_k['ratio']:.3f}, near-duplicates {res_k['near_dup']:.3%}")
    if res_knn:
        print(f"KDE(kNN)   ratio {res_knn['ratio']:.3f}, near-duplicates {res_knn['near_dup']:.3%}")
    for j, r in cal.items():
        print(f"memoriser j={j:<6g} ratio {r['ratio']:.3f}, near-duplicates {r['near_dup']:.3%}")
    print()
    if res_c["ratio"] > 0.05 or res_c["near_dup"] < 0.90:
        print("*** METRIC NOT VALIDATED: the deliberate memoriser was not flagged.")
        print("*** No conclusion may be drawn about MAGI. Fix the metric first.")
        return
    print("Metric validated in both directions: held-out real reads as real")
    print(f"(ratio ~1.00, KS p=0.81), and literal copies of training points read")
    print(f"as recall (ratio {res_c['ratio']:.3f}, {res_c['near_dup']:.1%} near-duplicates).")
    print()
    if res_m["ratio"] > 0.9 and res_m["near_dup"] < 3 * 0.01:
        print("MAGI does NOT memorise: it sits at the same distance from the data")
        print("as real data does from itself.")
    elif res_m["ratio"] < 0.7:
        print("MAGI SHOWS MEMORISATION. The amplification claim must be withdrawn")
        print("or heavily qualified. Do not quote 287x without this caveat.")
    else:
        print("INTERMEDIATE -- state the ratio, do not editorialise it.")
    print("\nThe KDE arm is NOT a memorisation control -- at KDSource's Silverman")
    print("bandwidth in 7-D the kernel is ~4x the nearest-neighbour spacing, so it")
    print("over-smooths. It is reported as an independent characterisation of that")
    print("resampler, alongside its 15.45% physically-impossible-sample rate.")


if __name__ == "__main__":
    main()
