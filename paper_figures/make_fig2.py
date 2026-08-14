#!/usr/bin/env python
"""Figure 2 candidates - crossing population real vs MAGI, one per dataset.

Produces a corner plot for every source that has a loadable v0.8.2 checkpoint,
so the best one can be picked for the proceeding.

Lower triangle : 2D density contours, real (filled/solid) vs MAGI (dashed).
Diagonal       : 1D marginals.
Upper triangle : per-pair Pearson correlation, each cell split diagonally -
                 lower-left half = real, upper-right half = MAGI, on a common
                 colour scale, with the residual printed. A visually uniform
                 cell means the correlation is reproduced.

Neutrinos are dropped from the real side because export.py drops them when
writing a Geant4 source, so the generated file never contains any; comparing an
unfiltered real sample against a filtered generated one is the mistake that
corrupted the v0.8.3 study.

NOT INCLUDED: Torio and Radio. Their v0.8.2 checkpoints on this machine are
INCOMPLETE (weights still on Google Drive), so they cannot be generated from.
"""
import os
import sys

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
NU = {"nu_e", "anti_nu_e", "nu_mu", "anti_nu_mu", "nu_tau", "anti_nu_tau"}
LABELS = [r"$\log_{10}(E/\mathrm{MeV})$", r"$u_r$", r"$u_v$",
          r"$\phi_r$ [rad]", r"$\phi_v$ [rad]"]

DM12 = "/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDML_DM1.2"
TD = "/Volumes/X10Pro/MAGI/TrainingData"

DATASETS = {
    "CR": dict(
        real=f"{TD}/alloutputDSCryoSphereCR.dat",
        gen="/Volumes/X10Pro/tmp/magi_crseed_s0.gntbin",
        center=(0.0, 0.0, -507.66), radius=100.0,
        title=r"SRON XFDM $-$ cosmic rays $-$ MAGI v0.8.2"),
    "Small": dict(
        real=f"{TD}/alloutputDSCryoSphereSmall.dat",
        gen="/Volumes/X10Pro/tmp/magi_fig2_small.gntbin",
        center=(0.0, 0.0, -507.66), radius=100.0,
        title=r"SRON XFDM $-$ $^{40}$K (Small) $-$ MAGI v0.8.2"),
    "DM1_2_iso": dict(
        real=f"{DM12}/allDSCryoSphere_DM1_2_iso.dat",
        gen="/Volumes/X10Pro/tmp/magi_fig2_dm12.gntbin",
        center=(0.0, 0.0, 5.0), radius=95.0,
        title=r"DM1.2 $-$ muons (iso) $-$ MAGI v0.8.2"),
}


def features(P, U, E, center):
    rhat = P - np.asarray(center)
    rhat = rhat / np.linalg.norm(rhat, axis=1)[:, None]
    vhat = U / np.linalg.norm(U, axis=1)[:, None]
    return np.column_stack([
        np.log10(np.clip(E, 1e-12, None)),
        rhat[:, 2], vhat[:, 2],
        np.arctan2(rhat[:, 1], rhat[:, 0]),
        np.arctan2(vhat[:, 1], vhat[:, 0]),
    ])


def load_real(path, center):
    """Crossing files come in two layouts; pick the columns by field count.

    9 fields  : EventId ParticleName E x y z vx vy vz          (SRON training)
    13 fields : EventId ParticleName E c3 c4 c5 process x y z vx vy vz (DM1.2)
    """
    P, U, E = [], [], []
    with open(path) as fh:
        for line in fh:
            a = line.split()
            if len(a) >= 13:
                pi, vi = 7, 10
            elif len(a) >= 9:
                pi, vi = 3, 6
            else:
                continue
            if a[1] in NU:
                continue
            P.append((float(a[pi]), float(a[pi + 1]), float(a[pi + 2])))
            U.append((float(a[vi]), float(a[vi + 1]), float(a[vi + 2])))
            E.append(float(a[2]))
    return features(np.array(P), np.array(U), np.array(E), center)


REC = np.dtype([("pdg", "<i4"), ("v", "<f4", 7)])


def load_gen(path, center):
    n = (os.path.getsize(path) - 20) // 32
    g = np.fromfile(path, dtype=np.uint8, count=20 + n * 32)[20:].view(REC)[:n]
    return features(g["v"][:, 1:4].astype(float), g["v"][:, 4:7].astype(float),
                   g["v"][:, 0].astype(float), center)


def corner(tag, cfg):
    X_real = load_real(cfg["real"], cfg["center"])
    X_gen = load_gen(cfg["gen"], cfg["center"])
    print(f"[{tag}] real {X_real.shape[0]:,}   MAGI {X_gen.shape[0]:,}")

    D = len(LABELS)
    RANGES = [(np.percentile(np.concatenate([X_real[:, k], X_gen[:, k]]), 0.2),
               np.percentile(np.concatenate([X_real[:, k], X_gen[:, k]]), 99.8))
              for k in range(D)]
    C_real = np.corrcoef(X_real, rowvar=False)
    C_gen = np.corrcoef(X_gen, rowvar=False)
    resid = C_gen - C_real
    iu = np.triu_indices(D, 1)
    mean_res = np.abs(resid[iu]).mean()
    print(f"[{tag}] mean |drho| = {mean_res:.4f}   max = {np.abs(resid[iu]).max():.4f}")

    # 1D marginal agreement, the thing the eye cannot judge off a density plot
    devs = []
    for k in range(D):
        b = np.linspace(*RANGES[k], 24)
        ha, _ = np.histogram(X_real[:, k], bins=b, density=True)
        hb, _ = np.histogram(X_gen[:, k], bins=b, density=True)
        devs.append(np.nanstd(hb / np.where(ha > 0, ha, np.nan)))
    print(f"[{tag}] marginal RMS dev per bin: " +
          "  ".join(f"{l.split('$')[1] if '$' in l else l}={d:.3f}"
                    for l, d in zip(["logE", "u_r", "u_v", "phi_r", "phi_v"], devs)))

    fig, axes = plt.subplots(D, D, figsize=(7.1, 7.3))
    fig.subplots_adjust(hspace=0.07, wspace=0.07, left=0.085, right=0.875,
                        top=0.925, bottom=0.075)
    fig.suptitle(cfg["title"], fontsize=10, y=0.972)
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
                hr = gaussian_filter(hr, 1.3)
                hg = gaussian_filter(hg, 1.3)
                xc = 0.5 * (xe[:-1] + xe[1:])
                yc = 0.5 * (ye[:-1] + ye[1:])

                def levels(h):
                    v = np.sort(h[h > 0].ravel())[::-1]
                    c = np.cumsum(v) / v.sum()
                    lv = [v[min(np.searchsorted(c, f), v.size - 1)]
                          for f in (0.39, 0.68, 0.865)][::-1]
                    return sorted(set(lv))

                lr, lg = levels(hr), levels(hg)
                if lr:
                    ax.contourf(xc, yc, hr.T, levels=lr + [hr.max()],
                                colors=[C_REF], alpha=0.16)
                    ax.contour(xc, yc, hr.T, levels=lr, colors=C_REF, linewidths=0.8)
                if lg:
                    ax.contour(xc, yc, hg.T, levels=lg, colors=C_MAGI,
                               linewidths=0.8, linestyles="--")
                ax.set_xlim(*RANGES[j])
                ax.set_ylim(*RANGES[i])
            else:
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
                ax.text(0.5, 0.5, r"$\Delta$" + f"{resid[i, j]:+.3f}",
                        fontsize=5.6, ha="center", va="center", color="0.35",
                        rotation=-45)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

            if i != D - 1:
                ax.set_xticklabels([])
            if j != 0 or i == 0:
                ax.set_yticklabels([])
            if i == D - 1:
                ax.set_xlabel(LABELS[j], fontsize=7.5)
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS[i], fontsize=7.5)
            ax.tick_params(labelsize=6.5)

    axes[0, 0].plot([], [], color=C_REF, lw=1.0, label="real crossings")
    axes[0, 0].plot([], [], color=C_MAGI, lw=1.0, ls="--", label="MAGI v0.8.2")
    axes[0, 0].legend(loc="upper left", fontsize=6.8, bbox_to_anchor=(0.0, 1.02))

    cax = fig.add_axes([0.905, 0.55, 0.016, 0.30])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"Pearson $\rho$   (lower-left: real, upper-right: MAGI)",
                 fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.text(0.985, 0.505, r"mean $|\Delta\rho|$" + "\n" + f"= {mean_res:.3f}",
             fontsize=7.5, ha="right", va="top")

    out = f"/Volumes/X10Pro/MAGI/paper_figures/fig2_corner_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[{tag}] wrote {out}\n")
    return mean_res


picks = sys.argv[1:] or list(DATASETS)
res = {t: corner(t, DATASETS[t]) for t in picks}
print("=" * 60)
print("mean |drho| by dataset (lower is better):")
for t, v in sorted(res.items(), key=lambda kv: kv[1]):
    print(f"  {t:12s} {v:.4f}")
