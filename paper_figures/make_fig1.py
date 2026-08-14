#!/usr/bin/env python
"""Figure 1 - spectral comparison, MAGI vs full Geant4, on two independent setups.

Panel (a) DM1.2 laboratory cryostat, full deposited-energy band, normalized
          differential rate. Normalization follows the established lab-setup
          convention in S1GDML_DM1.2/Analysis/ProcessedDataAnalysis_pkg.ipynb
          (cell 13):
              Flux = PhiTotalMu * 4*pi*Rgen^2 * n / (Ngen * dE)   [cts/s/keV]
          with Rgen = 118 cm - the /gps/pos/radius of gpsMu_iso.mac - and
          PhiTotalMu the muon integral of Old-Simulations/Sim/OmniFluxes.tsv.
          MAGI enters with Ngen_eff = N_injected / Chi_DM12, the same way the
          SRON side uses ngen/ChiCR.
          DM127 is an anti-coincidence detector (ACDSD), so no X-IFU band is
          drawn - the MIP peak is its band.

Panel (b) SRON XFDM, Detector 1 (8x8 TES array), flux against the RUN CRYOAC
          reference over 0-20 keV. Normalization copied from
          DataAnalysis_pkg.ipynb cell 63 rather than reinvented.

Both panels carry a ratio sub-panel, which is where the reader should look.
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

# ---------------------------------------------------------------- panel (a)
COLS_DM12 = ["EventId", "ParticleId", "particleInt", "EPreStep_keV", "Edep_keV",
             "x", "y", "z", "StartE_MeV", "GlobalTime"]
N_CROSS_REF_DM12 = 999_358
N_CROSS_MAGI_DM12 = 1_200_000
N_MU_REF_DM12 = 200_000_000                       # muons thrown by the full run
CHI_DM12 = N_CROSS_REF_DM12 / N_MU_REF_DM12
NGEN_EFF_MAGI_DM12 = N_CROSS_MAGI_DM12 / CHI_DM12  # equivalent muons
RGEN_DM12 = 118.0                                  # cm, /gps/pos/radius in gpsMu_iso.mac


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


import glob

# muon flux integral, the same OmniFluxes.tsv the lab-setup notebook uses
_ANG_LAB = pd.read_table("/Volumes/X10Pro/Old-Simulations/Sim/OmniFluxes.tsv",
                         names=["Energy", "OmniMuPlus", "OmniMuMinus", "OmniElectron",
                                "OmniPositron", "OmniGamma", "OmniProton"],
                         decimal=".").astype(float)
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
ChiCR = (3_443_885 - 158_538) / (1e6 * 540)
NGEN_MAGI_SRON = 1.85e7
AREA_DET1 = 4e-2
BIN_W = 0.8
BINS_S = np.arange(0.0, 100.0 + BIN_W, BIN_W)
CEN_S = BINS_S[:-1] + BIN_W / 2


def sron_per_event(path):
    return pd.read_table(path, names=COLS_SRON).groupby("EventId")["Edep"].sum().values


e_ref_sr = sron_per_event(f"{RUN3}/Detector1All.dat")
e_magi_sr = sron_per_event(
    f"{SRON}/Analysis/ProcessedDataFromMAGI/CR_v082/Detector1_082.dat")
print(f"SRON Det1  ref {e_ref_sr.size:,}  MAGI {e_magi_sr.size:,}")
print(f"ChiCR = {ChiCR:.6e}   1/ChiCR = {1/ChiCR:.2f}")


def flux(edep, ngen_eff, area):
    c, _ = np.histogram(edep, bins=BINS_S)
    sf = FLUXNUM / (ngen_eff * area * np.diff(BINS_S))
    return c, c * sf, np.sqrt(np.maximum(c, 1)) * sf


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.1, 3.5))
gs = GridSpec(2, 2, height_ratios=[2.6, 1], hspace=0.06, wspace=0.26,
              left=0.075, right=0.985, top=0.93, bottom=0.135)

# --- (a) DM1.2 -------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
axr = fig.add_subplot(gs[1, 0], sharex=ax)
bins_a = np.logspace(np.log10(2.0), np.log10(3e3), 30)
cen_a = np.sqrt(bins_a[:-1] * bins_a[1:])
wa = np.diff(bins_a)
n_r, _ = np.histogram(e_ref_dm, bins=bins_a)
n_m, _ = np.histogram(e_magi_dm, bins=bins_a)
y_r = FLUXNUM_DM12 * n_r / (N_MU_REF_DM12 * wa)
y_m = FLUXNUM_DM12 * n_m / (NGEN_EFF_MAGI_DM12 * wa)
ey_r = FLUXNUM_DM12 * np.sqrt(np.maximum(n_r, 1)) / (N_MU_REF_DM12 * wa)
ey_m = FLUXNUM_DM12 * np.sqrt(np.maximum(n_m, 1)) / (NGEN_EFF_MAGI_DM12 * wa)

y_r_plot = np.where(n_r > 0, y_r, np.nan)   # empty bins -> gap, not a spike
ax.step(bins_a[:-1], y_r_plot, where="post", color=C_REF, lw=1.1,
        label="full Geant4")
ax.errorbar(cen_a, y_m, yerr=ey_m, fmt="o", ms=2.2, color=C_MAGI, lw=0,
            elinewidth=0.7, label="MAGI v0.8.2")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylabel(r"rate  [cts s$^{-1}$keV$^{-1}$]")
ax.set_title(r"(a)  DM1.2  $\mu$", fontsize=8.5, loc="left")
ax.legend(loc="lower left", fontsize=7)
ax.tick_params(labelbottom=False)
ax.set_xlim(2.0, 3e3); ax.set_ylim(3e-8, 3e-3)

ok = (n_r >= 3) & (n_m >= 3)
rat = np.where(ok, y_m / np.where(y_r > 0, y_r, np.nan), np.nan)
erat = rat * np.sqrt(1 / np.maximum(n_r, 1) + 1 / np.maximum(n_m, 1))
axr.axhline(1.0, color="0.35", lw=0.7, ls="--")
axr.errorbar(cen_a[ok], rat[ok], yerr=erat[ok], fmt="o", ms=2.2,
             color=C_MAGI, lw=0, elinewidth=0.7)
axr.set_xscale("log"); axr.set_ylim(0.0, 2.0); axr.set_xlim(2.0, 3e3)
axr.set_xlabel(r"deposited energy  $E_{\rm dep}$  [keV]")
axr.set_ylabel("MAGI / full", fontsize=7.5)
# MIP band marker
axr.axvspan(100, 300, color="0.92", zorder=0)
ax.axvspan(100, 300, color="0.92", zorder=0)
axr.text(173, 1.72, "MIP", fontsize=6.5, ha="center", color="0.45")

# --- (b) SRON CR -----------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2r = fig.add_subplot(gs[1, 1], sharex=ax2)
c_r, f_r, ef_r = flux(e_ref_sr, NgenCryoAC, AREA_DET1)
c_m, f_m, ef_m = flux(e_magi_sr, NGEN_MAGI_SRON / ChiCR, AREA_DET1)
m = CEN_S <= 20

ax2.step(BINS_S[:-1][m], f_r[m], where="post", color=C_REF, lw=1.1,
         label="full Geant4 (RUN CRYOAC)")
ax2.errorbar(CEN_S[m], f_m[m], yerr=ef_m[m], fmt="o", ms=2.2, color=C_MAGI,
             lw=0, elinewidth=0.7, label="MAGI v0.8.2")
ax2.set_yscale("log")
ax2.set_ylabel(r"flux  [cts s$^{-1}$cm$^{-2}$keV$^{-1}$]")
ax2.set_title("(b)  SRON XFDM $-$ Detector 1, TES array", fontsize=8.5, loc="left")
ax2.legend(loc="upper right", fontsize=7)
ax2.tick_params(labelbottom=False)

ok2 = (c_r >= 3) & (c_m >= 3) & m
r2 = np.where(ok2, f_m / np.where(f_r > 0, f_r, np.nan), np.nan)
er2 = r2 * np.sqrt(1 / np.maximum(c_r, 1) + 1 / np.maximum(c_m, 1))
ax2r.axhline(1.0, color="0.35", lw=0.7, ls="--")
ax2r.errorbar(CEN_S[ok2], r2[ok2], yerr=er2[ok2], fmt="o", ms=2.2,
              color=C_MAGI, lw=0, elinewidth=0.7)
ax2r.set_ylim(0.0, 2.0)
ax2r.set_xlim(0, 20)
ax2r.set_xlabel(r"deposited energy  $E_{\rm dep}$  [keV]")
ax2r.set_ylabel("MAGI / full", fontsize=7.5)
band = (CEN_S > 1) & (CEN_S < 20)
ax2r.axvspan(1, 20, color="0.92", zorder=0)
ax2.axvspan(1, 20, color="0.92", zorder=0)

# efficiency factors, in place of the shot-particle counts
ax.text(0.030, 0.965,
        "$\\varepsilon = 194.2 \\pm 6.4$  (all)\n"
        "$\\varepsilon = 199.9 \\pm 7.4$  (MIP)\n"
        r"ideal $1/\chi = 200.1$",
        transform=ax.transAxes, fontsize=6.6, ha="left", va="top", linespacing=1.5)
ax2.text(0.030, 0.045,
         "$\\varepsilon = 124.8 \\pm 5.6$\n" + r"ideal $1/\chi = 164.4$",
         transform=ax2.transAxes, fontsize=6.6, ha="left", va="bottom", linespacing=1.5)

for a in (axr, ax2r):
    a.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])

out = "/Volumes/X10Pro/MAGI/paper_figures/fig1_spectra.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print("wrote", out)

# numbers for the caption / table
mip = (e_ref_dm >= 100) & (e_ref_dm <= 300)
mipm = (e_magi_dm >= 100) & (e_magi_dm <= 300)
print(f"DM1.2 MIP ratio  {(mipm.sum()/N_CROSS_MAGI_DM12)/(mip.sum()/N_CROSS_REF_DM12):.3f}")
print(f"SRON 1-20 keV mean flux ratio "
      f"{np.nanmean(f_m[band])/np.nanmean(f_r[band]):.3f}")
