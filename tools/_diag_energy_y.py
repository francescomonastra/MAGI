"""Diagnostic for Cycle 1: how the continuum-flow standardization maps the real
energy_y = log10(E) distribution into the RQS spline interval [-B, B].

For each source, prints percentiles of energy_y and, for the current std-based
scale vs a robust (clipped/IQR) scale, how much of [-5,5] the physically
interesting regions occupy — the CR Compton edge (~0.02-0.3 MeV) and the Small
low-E tail (~0.005-0.04 MeV). This tells us what y_scale / interval / n_bins to
use so knots land where the sharp structure is.

Fast: reads only the Energy column from the raw .dat (no feature pipeline).
"""
import numpy as np, pandas as pd

FILES = {
    "CR": "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR.dat",
    "Small": "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereSmall.dat",
}
# columns: EventId ParticleName Energy X Y Z Vx Vy Vz  -> Energy is index 2
EDGE_MEV = {"CR": (0.02, 0.3), "Small": (0.005, 0.04)}

for name, path in FILES.items():
    E = pd.read_csv(path, sep=r"\s+", header=None, usecols=[2]).to_numpy().ravel()
    E = E[E > 0]
    y = np.log10(E)
    pcts = [0.1, 0.5, 1, 5, 16, 50, 84, 95, 99, 99.5, 99.9]
    pv = np.percentile(y, pcts)
    print(f"\n===== {name}: {y.size:,} events =====")
    print("energy_y=log10(E) percentiles:")
    for p, v in zip(pcts, pv):
        print(f"  p{p:>5}: {v:+.3f}  (E={10**v:.4g} MeV)")

    mean, std = y.mean(), y.std()
    # robust: clip to [p1,p99] then std; and IQR/1.349
    ylo, yhi = np.percentile(y, [1, 99])
    yc = np.clip(y, ylo, yhi)
    rmean, rstd = yc.mean(), yc.std()
    iqr = np.percentile(y, 75) - np.percentile(y, 25)
    iqr_std = iqr / 1.349

    elo, ehi = EDGE_MEV[name]
    ylo_e, yhi_e = np.log10(elo), np.log10(ehi)
    print(f"raw:        mean={mean:+.3f} std={std:.3f}")
    print(f"clip[p1,p99]: mean={rmean:+.3f} std={rstd:.3f}")
    print(f"IQR/1.349:  {iqr_std:.3f}")
    for tag, m, s in [("raw std", mean, std), ("clip std", rmean, rstd), ("IQR/1.349", mean, iqr_std)]:
        # region of interest in y_std units under this standardization
        z_lo = (ylo_e - m) / s
        z_hi = (yhi_e - m) / s
        z_data_lo = (pv[0] - m) / s   # p0.1
        z_data_hi = (pv[-1] - m) / s  # p99.9
        frac = (z_hi - z_lo) / 10.0   # fraction of [-5,5] the ROI spans
        print(f"  [{tag:9}] ROI y_std=[{z_lo:+.2f},{z_hi:+.2f}] "
              f"-> {frac*100:4.1f}% of [-5,5];  data p0.1..p99.9 y_std=[{z_data_lo:+.2f},{z_data_hi:+.2f}]")
