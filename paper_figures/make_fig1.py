#!/usr/bin/env python
"""Figure 1 - spectral comparison, MAGI vs full Geant4, on two independent setups.

Panel (a) DM1.2 laboratory cryostat, full deposited-energy band, normalized
          differential rate. Normalization follows the established lab-setup
          convention in S1GDML_DM1.2/Analysis/ProcessedDataAnalysis_pkg.ipynb
          (cell 13):
              Flux = PhiTotalMu * 4*pi*Rgen^2 * n / (Ngen * dE)   [cts/s/keV]
          with Rgen = 118 cm - the /gps/pos/radius of gpsMu_iso.mac - and
          PhiTotalMu the muon integral of Old-Simulations/Sim/OmniFluxes.tsv.
          NOTE this is the LAB OmniFluxes.tsv, not the SRON OmniFluxes_Run1.tsv
          used in panel (b): the two panels sit on different flux references.
          MAGI enters with Ngen_eff = N_injected / Chi_DM12, the same way the
          SRON side uses ngen/ChiCR.
          DM127 is an anti-coincidence detector (ACDSD), so no X-IFU band is
          drawn - the MIP peak is its band.

Panel (b) SRON XFDM, Detector 1 (8x8 TES array), flux against the RUN CRYOAC
          reference over 0-20 keV. Normalization copied from
          DataAnalysis_pkg.ipynb cell 63 rather than reinvented.

Ratio panels carry EXACT Poisson confidence intervals, not sqrt(N) propagation.
For a ratio of two Poisson counts the conditional distribution of n_MAGI given
the total is binomial, so a Clopper-Pearson interval on p = n_m/(n_m+n_r) maps
to an exact interval on the rate ratio via lambda = p/(1-p). This matters here:
several bins hold single-digit counts, where a Gaussian error is meaningless and
the true interval is strongly asymmetric.

Efficiency factors are quoted rounded to the precision of their own uncertainty.
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import beta, norm

try:                       # scipy >= 1.14 removed the trapz alias
    from scipy.integrate import trapz
except ImportError:
    from numpy import trapz

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "legend.frameon": False, "figure.dpi": 200,
})

C_REF = "#1f4e79"    # full Geant4
C_MAGI = "#c1440e"   # MAGI

DM12 = "/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDML_DM1.2"
SRON = "/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDMLSRON_CryoSphereEmission"
RUN3 = "/Volumes/X10Pro/Old-Simulations/Sim/S1GDMLSRON/Analysis/ProcessedData-run3"
OLD = "/Volumes/X10Pro/Old-Simulations/Sim/S1GDMLSRON"


def poisson_ratio_ci(n_m, n_r, expo_m, expo_r, sigma=1.0):
    """Exact CI on (n_m/expo_m)/(n_r/expo_r) for two Poisson counts.

    Conditional on T = n_m + n_r, n_m ~ Binomial(T, p) with
    p = mu_m/(mu_m+mu_r), so a Clopper-Pearson interval on p maps to the rate
    ratio through lambda = p/(1-p), scaled by expo_r/expo_m.
    """
    n_m = np.asarray(n_m, float)
    n_r = np.asarray(n_r, float)
    alpha = 2 * (1 - norm.cdf(sigma))
    T = n_m + n_r
    with np.errstate(divide="ignore", invalid="ignore"):
        p_lo = np.where(n_m > 0, beta.ppf(alpha / 2, n_m, T - n_m + 1), 0.0)
        p_hi = np.where(n_m < T, beta.ppf(1 - alpha / 2, n_m + 1, T - n_m), 1.0)
        lam_lo = p_lo / (1 - p_lo)
        lam_hi = np.where(p_hi < 1, p_hi / (1 - p_hi), np.inf)
        scale = expo_r / expo_m
        r = (n_m / expo_m) / (n_r / expo_r)
        lo = np.clip(r - lam_lo * scale, 0, None)
        hi = lam_hi * scale - r
    return r, lo, hi


def poisson_pull(n_m, n_r, expo_m, expo_r):
    """Signed-root likelihood-ratio pull for two Poisson counts, in sigma.

    Same conditional-binomial model as poisson_ratio_ci, so the two panels stay
    internally consistent: given T = n_m + n_r, the null "equal rates" makes
    n_m ~ Binomial(T, p0) with p0 = expo_m / (expo_m + expo_r). The deviance

        G2 = 2 [ n_m ln(n_m / T p0) + n_r ln(n_r / T(1-p0)) ]

    is chi2_1 under the null, so sign(n_m - T p0) sqrt(G2) is ~N(0,1).

    Preferred over (r-1)/sigma_r: at 3-10 counts per bin the ratio's error is
    strongly asymmetric and a Gaussian pull misreports the tension by a factor
    of a few. This form is calibrated down to single counts, with the usual
    0 ln 0 = 0 convention.
    """
    n_m = np.asarray(n_m, float)
    n_r = np.asarray(n_r, float)
    T = n_m + n_r
    p0 = expo_m / (expo_m + expo_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_m = np.where(n_m > 0, n_m * np.log(n_m / (T * p0)), 0.0)
        t_r = np.where(n_r > 0, n_r * np.log(n_r / (T * (1 - p0))), 0.0)
    g2 = 2 * (t_m + t_r)
    z = np.sign(n_m - T * p0) * np.sqrt(np.clip(g2, 0, None))
    return np.where(T > 0, z, np.nan)


# Residual panels: "sigma" (default) or "ratio". Set RESID=ratio to compare.
RESID = os.environ.get("RESID", "sigma")


def draw_resid(axr, cen, n_m, n_r, expo_m, expo_r, ok, color):
    """Bottom-panel residuals in whichever convention RESID selects."""
    if RESID == "ratio":
        rat, elo, ehi = poisson_ratio_ci(n_m, n_r, expo_m, expo_r)
        axr.axhline(1.0, color="0.35", lw=0.7, ls="--")
        axr.errorbar(cen[ok], rat[ok], yerr=[elo[ok], ehi[ok]], fmt="o", ms=2.2,
                     color=color, lw=0, elinewidth=0.7)
        axr.set_ylim(0.0, 2.0)
        axr.set_ylabel("MAGI / full", fontsize=7.5)
        return
    z = poisson_pull(n_m, n_r, expo_m, expo_r)
    axr.axhspan(-2, 2, color="0.90", zorder=0, lw=0)
    axr.axhspan(-1, 1, color="0.80", zorder=0, lw=0)
    axr.axhline(0.0, color="0.35", lw=0.7, ls="--")
    axr.plot(cen[ok], z[ok], "o", ms=2.4, color=color)
    axr.set_ylim(-4.2, 4.2)
    axr.set_yticks([-3, 0, 3])
    axr.set_ylabel(r"pull  [$\sigma$]", fontsize=7.5)


# ---------------------------------------------------------------- panel (a)
COLS_DM12 = ["EventId", "ParticleId", "particleInt", "EPreStep_keV", "Edep_keV",
             "x", "y", "z", "StartE_MeV", "GlobalTime"]
N_CROSS_REF_DM12 = 999_358
N_CROSS_MAGI_DM12 = 1_200_000
N_MU_REF_DM12 = 200_000_000                        # muons thrown by the full run
CHI_DM12 = N_CROSS_REF_DM12 / N_MU_REF_DM12
NGEN_EFF_MAGI_DM12 = N_CROSS_MAGI_DM12 / CHI_DM12  # equivalent muons
RGEN_DM12 = 118.0                                  # cm, gpsMu_iso.mac
AREA_DM12 = 2.0                                    # cm^2, DM1.2 detector area


def dm12_per_event(paths, offset=True):
    parts = []
    for k, f in enumerate(paths):
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        d = pd.read_table(f, names=COLS_DM12)
        if offset:
            d["EventId"] += k * 10 ** 9
        parts.append(d[["EventId", "Edep_keV"]])
    return pd.concat(parts).groupby("EventId")["Edep_keV"].sum().values


_ANG_LAB = pd.read_table("/Volumes/X10Pro/Old-Simulations/Sim/OmniFluxes.tsv",
                         names=["Energy", "OmniMuPlus", "OmniMuMinus",
                                "OmniElectron", "OmniPositron", "OmniGamma",
                                "OmniProton"], decimal=".").astype(float)
PHI_MU_LAB = trapz(_ANG_LAB.OmniMuPlus + _ANG_LAB.OmniMuMinus, _ANG_LAB.Energy)
FLUXNUM_DM12 = PHI_MU_LAB * 4 * np.pi * RGEN_DM12 ** 2
print(f"PhiTotalMu(lab) = {PHI_MU_LAB:.4e}   Chi_DM12 = {CHI_DM12:.4e}   "
      f"1/Chi = {1/CHI_DM12:.2f}")

e_ref_dm = dm12_per_event([f"{DM12}/allOutputGDML_DM1_2_iso.dat"], offset=False)
e_magi_dm = dm12_per_event(
    [f"{d}/outputGDML.dat" for d in sorted(glob.glob(f"{DM12}/magi_iso_run/job*"))])
print(f"DM1.2  ref {e_ref_dm.size:,}  MAGI {e_magi_dm.size:,}")

# ---------------------------------------------------------------- panel (b)
COLS_SRON = ["EventId", "StartE", "particleInt", "Edep", "pixelID", "PrimBool",
             "PrimEpre", "GlobalTime"]
ANG = pd.read_table(f"{OLD}/OmniFluxes_Run1.tsv",
                    names=["Energy", "OmniMuPlus", "OmniMuMinus", "OmniElectron",
                           "OmniPositron", "OmniGamma", "OmniProton"],
                    decimal=".").astype(float)
PhiTotalMu = trapz(ANG.OmniMuPlus + ANG.OmniMuMinus, ANG.Energy)
RBuilding = 4200.0
FluxScaleFactor = (2 - np.sqrt(2)) / 2
RenormNgen = 3.563e-3
FLUXNUM = PhiTotalMu * 4 * np.pi * RBuilding ** 2 * FluxScaleFactor ** 2 * RenormNgen
NgenCryoAC = 131_129_837_637
AREA_DET1 = 4e-2
BIN_W = 0.8
BINS_S = np.arange(0.0, 100.0 + BIN_W, BIN_W)
CEN_S = BINS_S[:-1] + BIN_W / 2

# MAGI series: ingoing-fixed ("new") checkpoint, pooled across cr_retest
# (the original 3 "new"-arm jobs) and cr_retest_morestat (the 12-job
# statistics-boosting follow-up, 20-21/08, launched because 114 Detector-1
# events left the improvement over the censored checkpoint at only 1.2sigma
# -- see docs/MAGI_state_reference.tex sec:retest). Same pooling as
# paper_figures/make_fig_cr_vs_full.py. chi_new = 6.9210e-3 is the crossing
# efficiency the ingoing-cut fix produced (pre-fix was 6.0840e-3, the value
# this panel used to use via the CryoAC RUN reference); both are recorded in
# make_fig_cr_vs_full.py's own docstring/CHI dict.
CR_RETEST = f"{SRON}/cr_retest"
CR_MORESTAT = f"{SRON}/cr_retest_morestat"
CHI_NEW = 6.9210e-3


def _job_ready(jd):
    det1 = glob.glob(f"{jd}/proc/Detector1_*.dat")
    return bool(det1) and any(os.path.getsize(f) > 0 for f in det1)


def per_event_edep(path):
    tot = {}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.array([])
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        tot[p[0]] = tot.get(p[0], 0.0) + float(p[3])
    return np.fromiter(tot.values(), dtype=float)


def last_event_id(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return -1
    with open(path, "rb") as fh:
        fh.seek(max(0, os.path.getsize(path) - 65536))
        for line in reversed(fh.read().split(b"\n")):
            if line.strip():
                return int(line.split(b"\t")[0])
    return -1


def sron_per_event(path):
    return pd.read_table(path, names=COLS_SRON).groupby("EventId")["Edep"].sum().values


_new_edep, _n_inj_new, _n_jobs_new = [], 0, 0
for jd in sorted(glob.glob(f"{CR_RETEST}/job[0-5]")):
    if open(f"{jd}/arm.txt").read().strip() != "new":
        continue
    if not _job_ready(jd):
        continue
    n = last_event_id(f"{jd}/outputGDML.dat") + 1
    if n <= 0:
        n = int(open(f"{jd}/n_injected.txt").read().strip())
    _n_inj_new += n
    _n_jobs_new += 1
    for f in glob.glob(f"{jd}/proc/Detector1_*.dat"):
        e = per_event_edep(f)
        if e.size:
            _new_edep.append(e)
for jd in sorted(glob.glob(f"{CR_MORESTAT}/job*")):
    if not _job_ready(jd):
        continue
    n = last_event_id(f"{jd}/outputGDML.dat") + 1
    if n <= 0:
        n = int(open(f"{jd}/n_injected.txt").read().strip())
    _n_inj_new += n
    _n_jobs_new += 1
    for f in glob.glob(f"{jd}/proc/Detector1_*.dat"):
        e = per_event_edep(f)
        if e.size:
            _new_edep.append(e)
e_magi_sr = np.concatenate(_new_edep) if _new_edep else np.array([])
NGEN_EFF_SRON = _n_inj_new / CHI_NEW

e_ref_sr = sron_per_event(f"{RUN3}/Detector1All.dat")
print(f"SRON Det1  ref {e_ref_sr.size:,}  MAGI {e_magi_sr.size:,}  "
      f"({_n_jobs_new} jobs, new-arm injected {_n_inj_new:,})")
print(f"ChiCR (new) = {CHI_NEW:.6e}   1/ChiCR = {1/CHI_NEW:.2f}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.1, 3.7))
gs = GridSpec(2, 2, height_ratios=[2.6, 1], hspace=0.06, wspace=0.26,
              left=0.075, right=0.985, top=0.855, bottom=0.13)

# --- (a) DM1.2 -------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
axr = fig.add_subplot(gs[1, 0], sharex=ax)
bins_a = np.logspace(np.log10(2.0), np.log10(3e3), 30)
cen_a = np.sqrt(bins_a[:-1] * bins_a[1:])
wa = np.diff(bins_a)
n_r, _ = np.histogram(e_ref_dm, bins=bins_a)
n_m, _ = np.histogram(e_magi_dm, bins=bins_a)
y_r = FLUXNUM_DM12 * n_r / (N_MU_REF_DM12 * wa * AREA_DM12)
y_m = FLUXNUM_DM12 * n_m / (NGEN_EFF_MAGI_DM12 * wa * AREA_DM12)
ey_m = FLUXNUM_DM12 * np.sqrt(np.maximum(n_m, 1)) / (NGEN_EFF_MAGI_DM12 * wa * AREA_DM12)

y_r_plot = np.where(n_r > 0, y_r, np.nan)   # empty bins -> gap, not a spike
h_ref, = ax.step(bins_a[:-1], y_r_plot, where="post", color=C_REF, lw=1.1,
                 label="full Geant4")
h_magi = ax.errorbar(cen_a, y_m, yerr=ey_m, fmt="o", ms=2.2, color=C_MAGI, lw=0,
                     elinewidth=0.7, label="MAGI v0.8.2")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"flux  [cts s$^{-1}$cm$^{-2}$keV$^{-1}$]")
ax.set_title(r"(a)  DM1.2  $\mu$", fontsize=8.5, loc="left")
ax.tick_params(labelbottom=False)
ax.set_xlim(2.0, 3e3)
ax.set_ylim(3e-8 / AREA_DM12, 3e-3 / AREA_DM12)

ok = (n_r >= 1) & (n_m >= 1)
draw_resid(axr, cen_a, n_m, n_r, NGEN_EFF_MAGI_DM12, N_MU_REF_DM12, ok, C_MAGI)
axr.set_xscale("log")
axr.set_xlim(2.0, 3e3)
axr.set_xlabel(r"deposited energy  $E_{\rm dep}$  [keV]")
axr.axvspan(100, 300, color="0.92", zorder=0)
ax.axvspan(100, 300, color="0.92", zorder=0)
axr.text(173, 3.35 if RESID == "sigma" else 1.72, "MIP",
         fontsize=6.5, ha="center", color="0.45")

ax.text(0.030, 0.965,
        r"$\varepsilon = 194 \pm 6$  (all)" + "\n"
        r"$\varepsilon = 200 \pm 7$  (MIP)" + "\n"
        r"ideal $1/\chi = 200$",
        transform=ax.transAxes, fontsize=6.6, ha="left", va="top",
        linespacing=1.5)

# --- (b) SRON XFDM ---------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2r = fig.add_subplot(gs[1, 1], sharex=ax2)
c_r, _ = np.histogram(e_ref_sr, bins=BINS_S)
c_m, _ = np.histogram(e_magi_sr, bins=BINS_S)
sf_r = FLUXNUM / (NgenCryoAC * AREA_DET1 * np.diff(BINS_S))
sf_m = FLUXNUM / (NGEN_EFF_SRON * AREA_DET1 * np.diff(BINS_S))
f_r, f_m = c_r * sf_r, c_m * sf_m
ef_m = np.sqrt(np.maximum(c_m, 1)) * sf_m
m = CEN_S <= 20

ax2.step(BINS_S[:-1][m], np.where(c_r[m] > 0, f_r[m], np.nan), where="post",
         color=C_REF, lw=1.1)
ax2.errorbar(CEN_S[m], f_m[m], yerr=ef_m[m], fmt="o", ms=2.2, color=C_MAGI,
             lw=0, elinewidth=0.7)
ax2.set_yscale("log")
ax2.set_ylabel(r"flux  [cts s$^{-1}$cm$^{-2}$keV$^{-1}$]")
ax2.set_title("(b)  SRON XFDM $-$ Detector 1, TES array", fontsize=8.5, loc="left")
ax2.tick_params(labelbottom=False)

ok2 = (c_r >= 1) & (c_m >= 1) & m
draw_resid(ax2r, CEN_S, c_m, c_r, NGEN_EFF_SRON, NgenCryoAC, ok2, C_MAGI)
ax2r.set_xlim(0, 20)
ax2r.set_xlabel(r"deposited energy  $E_{\rm dep}$  [keV]")

# epsilon = R/chi, same relation the DM1.2 panel's static annotation uses
# (194 = 0.970 x 200.13, 200 = 0.999 x 200.13). R measured over the plotted
# <=20 keV band from raw counts, not the "mean flux ratio" print at the
# bottom of this file (that's a per-bin mean, not count-weighted).
_cm_tot, _cr_tot = c_m[m].sum(), c_r[m].sum()
R_SRON = (_cm_tot / NGEN_EFF_SRON) / (_cr_tot / NgenCryoAC)
R_SRON_ERR = R_SRON * np.sqrt(1 / max(_cm_tot, 1) + 1 / max(_cr_tot, 1))
EPS_SRON, EPS_SRON_ERR = R_SRON / CHI_NEW, R_SRON_ERR / CHI_NEW
print(f"SRON <=20 keV epsilon = {EPS_SRON:.0f} +/- {EPS_SRON_ERR:.0f}  "
      f"(R = {R_SRON:.3f} +/- {R_SRON_ERR:.3f})")
ax2.text(0.972, 0.960,
         rf"$\varepsilon = {EPS_SRON:.0f} \pm {EPS_SRON_ERR:.0f}$" + "\n" +
         rf"ideal $1/\chi = {1/CHI_NEW:.0f}$",
         transform=ax2.transAxes, fontsize=6.6, ha="right", va="top",
         linespacing=1.5)

if RESID == "ratio":
    for a in (axr, ax2r):
        a.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])

# one legend for the whole figure, outside both panels at the top
fig.legend(handles=[h_ref, h_magi], labels=["full Geant4", "MAGI v0.8.2"],
           loc="upper center", ncol=2, fontsize=8.5,
           bbox_to_anchor=(0.53, 0.995), handlelength=1.8, columnspacing=2.2)

# Pull diagnostics. A good match scatters about zero with chi2/ndf ~ 1; a
# coherent offset shows up as <z> far from 0 with most bins one-signed, which
# is a far stronger statement than any single ratio.
for tag, zz, okk in (("DM1.2", poisson_pull(n_m, n_r, NGEN_EFF_MAGI_DM12, N_MU_REF_DM12), ok),
                     ("SRON ", poisson_pull(c_m, c_r, NGEN_EFF_SRON, NgenCryoAC), ok2)):
    z = zz[okk]
    print(f"{tag} pulls: n={len(z):3d}  <z>={z.mean():+.2f}  "
          f"chi2/ndf={np.sum(z**2)/len(z):5.2f}  "
          f"neg={np.mean(z < 0):.0%}  |z|>2: {np.sum(np.abs(z) > 2):2d}")

out = "/Volumes/X10Pro/MAGI/paper_figures/fig1_spectra.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print("wrote", out)

mip = (e_ref_dm >= 100) & (e_ref_dm <= 300)
mipm = (e_magi_dm >= 100) & (e_magi_dm <= 300)
print(f"DM1.2 MIP ratio  "
      f"{(mipm.sum()/N_CROSS_MAGI_DM12)/(mip.sum()/N_CROSS_REF_DM12):.3f}")
band = (CEN_S > 1) & (CEN_S < 20)
print(f"SRON 1-20 keV mean flux ratio "
      f"{np.nanmean(f_m[band])/np.nanmean(f_r[band]):.3f}")
