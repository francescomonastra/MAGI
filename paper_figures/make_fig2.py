#!/usr/bin/env python
"""Figure 2 - CR crossing population, real vs MAGI, with the correlation matrix.

Lower triangle : 2D density contours, real (solid) over MAGI (dashed).
Diagonal       : 1D marginals.
Upper triangle : per-pair Pearson correlation, each cell split diagonally -
                 lower-left half = real, upper-right half = MAGI, on a common
                 colour scale, with the residual printed. A visually uniform
                 cell means the correlation is reproduced.

Source: SRON cosmic rays. Real = the training file (neutrino-free). Generated =
the seed-42 v0.8.2 checkpoint file already written for the cr_seed_run, so no
new generation is needed and the sample is exactly what Geant4 was fed.

The generated file carries no neutrinos (export.py drops NON_TRANSPORT_PARTICLES)
so the real side is filtered the same way - comparing an unfiltered real sample
against a filtered generated one is the mistake that corrupted the v0.8.3 study.
"""
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})

C_REF = "#1f4e79"
C_MAGI = "#c1440e"

CENTER = np.array([0.0, 0.0, -507.66])
R = 100.0
TRAIN = "/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR.dat"
GEN = "/Volumes/X10Pro/tmp/magi_crseed_s0.gntbin"
NU = {"nu_e", "anti_nu_e", "nu_mu", "anti_nu_mu", "nu_tau", "anti_nu_tau"}

LABELS = [r"$\log_{10}(E/\mathrm{MeV})$", r"$u_r$", r"$u_v$",
          r"$\phi_r$ [rad]", r"$\phi_v$ [rad]"]


def features(P, U, E):
    rhat = (P - CENTER)
    rhat = rhat / np.linalg.norm(rhat, axis=1)[:, None]
    vhat = U / np.linalg.norm(U, axis=1)[:, None]
    return np.column_stack([
        np.log10(np.clip(E, 1e-12, None)),
        rhat[:, 2], vhat[:, 2],
        np.arctan2(rhat[:, 1], rhat[:, 0]),
        np.arctan2(vhat[:, 1], vhat[:, 0]),
    ])


# ---- real -----------------------------------------------------------------
P, U, E = [], [], []
with open(TRAIN) as fh:
    for line in fh:
        a = line.split()
        if len(a) < 9 or a[1] in NU:
            continue
        P.append((float(a[3]), float(a[4]), float(a[5])))
        U.append((float(a[6]), float(a[7]), float(a[8])))
        E.append(float(a[2]))
X_real = features(np.array(P), np.array(U), np.array(E))
print(f"real crossings (neutrino-free): {X_real.shape[0]:,}")

# ---- generated ------------------------------------------------------------
REC = np.dtype([("pdg", "<i4"), ("v", "<f4", 7)])
n = (os.path.getsize(GEN) - 20) // 32
g = np.fromfile(GEN, dtype=np.uint8, count=20 + n * 32)[20:].view(REC)[:n]
X_gen = features(g["v"][:, 1:4].astype(float), g["v"][:, 4:7].astype(float),
                 g["v"][:, 0].astype(float))
print(f"MAGI crossings: {X_gen.shape[0]:,}")

D = len(LABELS)
RANGES = [(np.percentile(np.concatenate([X_real[:, k], X_gen[:, k]]), 0.2),
           np.percentile(np.concatenate([X_real[:, k], X_gen[:, k]]), 99.8))
          for k in range(D)]

C_real = np.corrcoef(X_real, rowvar=False)
C_gen = np.corrcoef(X_gen, rowvar=False)
resid = C_gen - C_real
iu = np.triu_indices(D, 1)
print(f"mean |rho_gen - rho_real| over the {len(iu[0])} pairs = "
      f"{np.abs(resid[iu]).mean():.4f}   max = {np.abs(resid[iu]).max():.4f}")

fig, axes = plt.subplots(D, D, figsize=(7.1, 7.1))
fig.subplots_adjust(hspace=0.07, wspace=0.07, left=0.085, right=0.9,
                    top=0.965, bottom=0.075)
norm = Normalize(-1, 1)
cmap = plt.get_cmap("RdBu_r")

for i in range(D):
    for j in range(D):
        ax = axes[i, j]
        if i == j:
            br = np.linspace(*RANGES[i], 70)
            ax.hist(X_real[:, i], bins=br, density=True, histtype="step",
                    color=C_REF, lw=1.0)
            ax.hist(X_gen[:, i], bins=br, density=True, histtype="step",
                    color=C_MAGI, lw=1.0, ls="--")
            ax.set_yticks([])
            ax.set_xlim(*RANGES[i])
        elif i > j:
            br = [np.linspace(*RANGES[j], 46), np.linspace(*RANGES[i], 46)]
            hr, xe, ye = np.histogram2d(X_real[:, j], X_real[:, i], bins=br)
            hg, _, _ = np.histogram2d(X_gen[:, j], X_gen[:, i], bins=br)
            # smooth before contouring: on the near-uniform azimuths the raw
            # counts are Poisson speckle and the contour levels are degenerate
            hr = gaussian_filter(hr, 1.3); hg = gaussian_filter(hg, 1.3)
            xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])

            def levels(h):
                v = np.sort(h[h > 0].ravel())[::-1]
                c = np.cumsum(v) / v.sum()
                return [v[np.searchsorted(c, f)] for f in (0.39, 0.68, 0.865)][::-1]

            ax.contourf(xc, yc, hr.T, levels=levels(hr) + [hr.max()],
                        colors=[C_REF], alpha=0.16)
            ax.contour(xc, yc, hr.T, levels=levels(hr), colors=C_REF,
                       linewidths=0.8)
            ax.contour(xc, yc, hg.T, levels=levels(hg), colors=C_MAGI,
                       linewidths=0.8, linestyles="--")
            ax.set_xlim(*RANGES[j]); ax.set_ylim(*RANGES[i])
        else:
            # upper triangle: split cell, real (lower-left) vs MAGI (upper-right)
            ax.add_patch(Polygon([[0, 0], [1, 0], [0, 1]], closed=True,
                                 facecolor=cmap(norm(C_real[i, j])),
                                 edgecolor="white", lw=0.5))
            ax.add_patch(Polygon([[1, 0], [1, 1], [0, 1]], closed=True,
                                 facecolor=cmap(norm(C_gen[i, j])),
                                 edgecolor="white", lw=0.5))
            ax.text(0.28, 0.24, f"{C_real[i, j]:+.2f}", fontsize=6.2,
                    ha="center", va="center", color="0.12")
            ax.text(0.72, 0.76, f"{C_gen[i, j]:+.2f}", fontsize=6.2,
                    ha="center", va="center", color="0.12")
            ax.text(0.5, 0.5, r"$\Delta$" + f"{resid[i, j]:+.3f}", fontsize=5.6,
                    ha="center", va="center", color="0.35", rotation=-45)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        if i < D - 1 or i == j == D - 1 and False:
            pass
        if i != D - 1:
            ax.set_xticklabels([])
        if j != 0 or i == 0:
            ax.set_yticklabels([])
        if i == D - 1:
            ax.set_xlabel(LABELS[j], fontsize=7.5)
        if j == 0 and i != 0:
            ax.set_ylabel(LABELS[i], fontsize=7.5)
        ax.tick_params(labelsize=6.5)

# legend + colourbar
axes[0, 0].plot([], [], color=C_REF, lw=1.0, label="real crossings")
axes[0, 0].plot([], [], color=C_MAGI, lw=1.0, ls="--", label="MAGI v0.8.2")
axes[0, 0].legend(loc="upper left", fontsize=6.8, bbox_to_anchor=(0.0, 1.02))

cax = fig.add_axes([0.915, 0.55, 0.016, 0.33])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"Pearson $\rho$   (lower-left: real, upper-right: MAGI)",
             fontsize=7)
cb.ax.tick_params(labelsize=6.5)

fig.text(0.9235, 0.505, r"mean $|\Delta\rho|$" + "\n" +
         f"= {np.abs(resid[iu]).mean():.3f}", fontsize=7.5, ha="center",
         va="top")

out = "/Volumes/X10Pro/MAGI/paper_figures/fig2_corner.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print("wrote", out)
