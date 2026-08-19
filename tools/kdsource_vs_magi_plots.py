#!/usr/bin/env python
"""Plots for the KDSource-vs-MAGI held-out benchmark.

Reuses kdsource_vs_magi.py rather than reimplementing it, so the figures and the
printed table come from exactly the same split, bandwidths and resampling. The
module guards its own main(), so importing is side-effect free apart from the
seeded RNG -- which is the point: the split is reproducible.

Two figures:
  fig A  marginals, held-out reference vs MAGI vs KDSource, with pull panels
  fig B  correlation-residual heatmaps + per-species RMS + footprint

Honest framing baked into the captions: KDSource wins on the Pearson structure
because a smoothed bootstrap perturbs REAL training points, so the correlations
are inherited from the retained sample rather than learned. That is also why it
cannot discard the sample -- the footprint panel is the other side of that coin.
"""
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kdsource_vs_magi as K                                  # noqa: E402

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "figure.dpi": 200,
})

C_REF = "#1f4e79"     # held-out real
C_MAGI = "#c1440e"    # MAGI
C_KDS = "#2e8b57"     # KDSource (silv, its own default and its best setting)

PRETTY = [r"$\log_{10}E$", r"$u_r$", r"$u_v$", r"$\phi_r$", r"$\phi_v$"]
OUT = "/Volumes/X10Pro/MAGI/paper_figures"


def main():
    X, names = K.load_real()
    species, counts = np.unique(names, return_counts=True)
    frac = counts / counts.sum()

    perm = K.RNG.permutation(len(X))
    ntr = int(0.70 * len(X))
    tr, te = perm[:ntr], perm[ntr:]
    X_tr, n_tr = X[tr], names[tr]
    X_te, n_te = X[te], names[te]

    Xg, ng = K.load_magi()
    N_OUT = len(Xg)
    print(f"held-out {len(X_te):,}   MAGI {N_OUT:,}")

    # KDSource at its own default bandwidth, per species, true mixing weights
    parts, pnames = [], []
    for s in species:
        m = n_tr == s
        n_s = int(round(N_OUT * frac[species.tolist().index(s)]))
        if m.sum() < 50 or n_s < 1:
            continue
        parts.append(K.kde_resample(X_tr[m], n_s, "silv"))
        pnames.append(np.full(n_s, s, dtype=object))
        print(f"  KDE {s:8s} {n_s:,}")
    Xk = np.vstack(parts)
    nk = np.concatenate(pnames)

    # ---------------------------------------------------------------- fig A
    fig = plt.figure(figsize=(7.1, 3.4))
    gs = GridSpec(2, 5, height_ratios=[2.5, 1], hspace=0.08, wspace=0.30,
                  left=0.055, right=0.99, top=0.87, bottom=0.14)
    # u_r and u_v are direction cosines and phi are azimuths: all four are
    # HARD-BOUNDED. Plot past the bound so the KDE's leakage outside the
    # physical domain is visible rather than cropped away.
    BOUND = {1: 1.0, 2: 1.0, 3: np.pi, 4: np.pi}

    for k in range(5):
        ax = fig.add_subplot(gs[0, k])
        axr = fig.add_subplot(gs[1, k], sharex=ax)
        if k in BOUND:
            B = BOUND[k]
            lo, hi = -1.18 * B, 1.18 * B
        else:
            lo, hi = np.percentile(X_te[:, k], [0.2, 99.8])
        b = np.linspace(lo, hi, 46)
        cen = 0.5 * (b[1:] + b[:-1])
        h_r, _ = np.histogram(X_te[:, k], bins=b)
        h_m, _ = np.histogram(Xg[:, k], bins=b)
        h_k, _ = np.histogram(Xk[:, k], bins=b)
        d_r = h_r / h_r.sum()
        d_m = h_m / h_m.sum()
        d_k = h_k / h_k.sum()

        if k in BOUND:
            for s in (-1, 1):
                ax.axvspan(s * BOUND[k], s * 1.18 * BOUND[k],
                           color="#d62728", alpha=0.10, lw=0, zorder=0)
                axr.axvspan(s * BOUND[k], s * 1.18 * BOUND[k],
                            color="#d62728", alpha=0.10, lw=0, zorder=0)
                ax.axvline(s * BOUND[k], color="#d62728", lw=0.7, ls=":")

        ax.step(cen, d_r, where="mid", color=C_REF, lw=1.1)
        ax.step(cen, d_m, where="mid", color=C_MAGI, lw=0.9)
        ax.step(cen, d_k, where="mid", color=C_KDS, lw=0.9, ls="--")
        ax.set_title(PRETTY[k], fontsize=9)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_yticks([])
        if k in BOUND:
            out = np.abs(Xk[:, k]) > BOUND[k]
            ax.text(0.5, 0.955, f"KDE outside: {out.mean():.1%}",
                    transform=ax.transAxes, fontsize=6.0, ha="center",
                    va="top", color="#a01d1d",
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2))

        # ratio to the held-out reference, on well-populated bins only
        ok = h_r >= K.MIN_BIN
        axr.axhline(1.0, color="0.35", lw=0.7, ls="--")
        axr.axhspan(0.9, 1.1, color="0.90", zorder=0, lw=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.plot(cen[ok], (d_m / d_r)[ok], ".", ms=2.6, color=C_MAGI)
            axr.plot(cen[ok], (d_k / d_r)[ok], ".", ms=2.6, color=C_KDS)
        axr.set_ylim(0.55, 1.45)
        axr.set_yticks([0.7, 1.0, 1.3])
        if k:
            axr.tick_params(labelleft=False)
        else:
            axr.set_ylabel("gen / real", fontsize=7.5)
        axr.tick_params(labelsize=6.5)

    fig.legend(handles=[plt.Line2D([], [], color=C_REF, lw=1.3),
                        plt.Line2D([], [], color=C_MAGI, lw=1.3),
                        plt.Line2D([], [], color=C_KDS, lw=1.3, ls="--")],
               labels=["held-out real (30 %)", "MAGI v0.8.2",
                       "KDSource v0.2.2 (silv)"],
               loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.52, 1.045), handlelength=2.0, columnspacing=2.4)
    fig.text(0.52, 0.955,
             "shaded = physically impossible: direction cosines outside "
             r"$[-1,1]$, azimuths outside $[-\pi,\pi]$",
             fontsize=6.6, ha="center", color="#a01d1d")
    fig.savefig(f"{OUT}/fig_kds_marginals.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_kds_marginals.png", bbox_inches="tight", dpi=300)
    print("wrote fig_kds_marginals")

    # ---------------------------------------------------------------- fig B
    fig2 = plt.figure(figsize=(7.4, 2.9))
    # explicit narrow column for the colorbar: sharing one across the heatmaps
    # with fraction/pad steals width from the bar panel and the labels collide
    gs2 = GridSpec(1, 4, width_ratios=[1, 1, 0.055, 1.30], wspace=0.30,
                   left=0.055, right=0.985, top=0.84, bottom=0.21)
    Cr = np.corrcoef(X_te, rowvar=False)
    vmax = 0.05
    for j, (G, tag) in enumerate(((Xg, "MAGI v0.8.2"), (Xk, "KDSource (silv)"))):
        ax = fig2.add_subplot(gs2[0, j])
        D = np.abs(np.corrcoef(G, rowvar=False) - Cr)
        np.fill_diagonal(D, np.nan)
        im = ax.imshow(D, cmap="magma_r", vmin=0, vmax=vmax)
        iu = np.triu_indices(5, 1)
        ax.set_title(f"{tag}\nmean $|\\Delta\\rho|$ = {D[iu].mean():.4f}",
                     fontsize=8)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(PRETTY, fontsize=6.5)
        ax.set_yticklabels(PRETTY, fontsize=6.5)
        for a in range(5):
            for c in range(5):
                if a != c:
                    ax.text(c, a, f"{D[a, c]:.3f}", ha="center", va="center",
                            fontsize=5.2,
                            color="w" if D[a, c] > vmax * 0.55 else "0.15")
    cax = fig2.add_subplot(gs2[0, 2])
    cb = fig2.colorbar(im, cax=cax)
    cb.set_label(r"$|\Delta\rho|$ vs held-out real", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # per-species marginal RMS -- the rare-species prediction
    ax3 = fig2.add_subplot(gs2[0, 3])
    order = np.argsort(frac)
    labs, vm, vk = [], [], []
    for i in order:
        s = species[i]
        mt = n_te == s
        if mt.sum() < 200:
            continue
        mg, mk = ng == s, nk == s
        if mg.sum() < 200 or mk.sum() < 200:
            continue
        labs.append(f"{s} ({frac[i]:.2%})")
        vm.append(np.nanmean(K.marginal_rms(X_te[mt], Xg[mg])))
        vk.append(np.nanmean(K.marginal_rms(X_te[mt], Xk[mk])))
    y = np.arange(len(labs))
    ax3.barh(y - 0.19, vm, 0.36, color=C_MAGI, label="MAGI")
    ax3.barh(y + 0.19, vk, 0.36, color=C_KDS, label="KDSource")
    ax3.set_yticks(y); ax3.set_yticklabels(labs, fontsize=7)
    ax3.set_xlabel("mean marginal RMS  (lower = better)", fontsize=7.5)
    ax3.set_title("per species, by abundance", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.62))
    ax3.invert_yaxis()
    ax3.set_xlim(0, max(vm + vk) * 1.32)

    ck = sum(os.path.getsize(os.path.join(K.MAGI_CKPT, f))
             for f in os.listdir(K.MAGI_CKPT) if not f.endswith(".html"))
    ax3.text(0.985, 0.02,
             f"retained to generate:  MAGI {ck/1e6:.1f} MB  ·  KDE {X_tr.nbytes/1e6:.0f} MB",
             transform=ax3.transAxes, fontsize=6.2, ha="right", va="bottom",
             color="0.35")
    fig2.savefig(f"{OUT}/fig_kds_correlations.pdf", bbox_inches="tight")
    fig2.savefig(f"{OUT}/fig_kds_correlations.png", bbox_inches="tight", dpi=300)
    print("wrote fig_kds_correlations")

    # ------------------------------------------------- fig C: fluorescence lines
    # KDE picks ONE global bandwidth for a spectrum spanning six decades. On the
    # gamma component that is 0.126 dex, which smears 511 keV over 382-683 keV
    # and merges the Cu K-alpha1/alpha2/beta triplet -- 25 eV apart -- into one
    # featureless bump. This is structural, not a tuning choice.
    from kdsource import bw_silv
    mg = n_tr == "gamma"
    n_out = int((ng == "gamma").sum())
    Xk_g = K.kde_resample(X_tr[mg], n_out, "silv")
    bw_dex = bw_silv(5, int(mg.sum())) * X[names == "gamma"][:, 0].std()

    real_g = X_te[n_te == "gamma"][:, 0]
    magi_g = Xg[ng == "gamma"][:, 0]
    kde_g = Xk_g[:, 0]

    fig3 = plt.figure(figsize=(7.1, 2.6))
    gs3 = GridSpec(1, 2, wspace=0.22, left=0.075, right=0.985, top=0.86,
                   bottom=0.19)
    panels = [("Al K$\\alpha$  ·  Cu K$\\alpha_{1,2}$, K$\\beta$", 1.0, 12.0, 260,
               [1, 2, 3, 5, 8, 12]),
              (r"$e^+e^-$ annihilation, 511 keV", 470.0, 550.0, 200,
               [480, 500, 511, 520, 540])]
    LINES = [1.469, 7.984, 8.006, 8.866, 510.999]
    for j, (title, elo, ehi, nb, ticks) in enumerate(panels):
        ax = fig3.add_subplot(gs3[0, j])
        b = np.linspace(np.log10(elo * 1e-3), np.log10(ehi * 1e-3), nb)
        cen = 10 ** (0.5 * (b[1:] + b[:-1])) * 1e3
        for a, c, lw, ls, lab in ((real_g, C_REF, 1.0, "-", "held-out real"),
                                  (magi_g, C_MAGI, 0.9, "-", "MAGI v0.8.2"),
                                  (kde_g, C_KDS, 0.9, "--", "KDSource (silv)")):
            h, _ = np.histogram(a, bins=b)
            # normalise by the TOTAL gamma count, not by what lands in this
            # window: the three generators put very different fractions inside
            # it, so a per-panel normalisation would not be comparable.
            ax.step(cen, h / len(a), where="mid", color=c, lw=lw, ls=ls, label=lab)
        for L in LINES:
            if elo < L < ehi:
                ax.axvline(L, color="0.7", lw=0.5, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(elo, ehi)
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("energy [keV]", fontsize=8)
        if j == 0:
            ax.set_ylabel("fraction of all $\\gamma$ crossings", fontsize=8)
            ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)
    fig3.text(0.5, 0.965,
              f"KDE kernel = {bw_dex:.3f} dex on the $\\gamma$ component: one global "
              "bandwidth cannot resolve a 4 eV line and six decades of continuum",
              fontsize=6.8, ha="center", color="0.3")
    fig3.savefig(f"{OUT}/fig_kds_lines.pdf", bbox_inches="tight")
    fig3.savefig(f"{OUT}/fig_kds_lines.png", bbox_inches="tight", dpi=300)
    print("wrote fig_kds_lines")


if __name__ == "__main__":
    main()
