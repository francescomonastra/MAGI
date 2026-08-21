#!/usr/bin/env python
"""CR at detector level: full Geant4 vs the old and new MAGI checkpoints.

Flux normalisation is identical to Fig. 1 panel (b), so this figure is directly
comparable with it. Each MAGI crossing stands for 1/chi primaries, and chi is
what the ingoing-cut fix changed (6.084e-3 -> 6.921e-3): the old checkpoint's
population was censored, the new one is not. Using per-injected-crossing rates
here would divide chi out and silently remove the very effect under test.

The "new" arm pools TWO campaigns: cr_retest/job{0,1,2} (the original 3 jobs,
18-19/08) and cr_retest_morestat/job0..11 (the statistics-boosting follow-up,
20-21/08, launched because the original 114-event Detector-1 sample left the
improvement over the censored checkpoint at only 1.2sigma -- see
docs/MAGI_state_reference.tex sec:retest). 15 jobs total, ~4.5x the original
statistics. job_ready() guards against counting a job's injected crossings
before its processed output exists (a real bug caught mid-campaign: doing so
silently deflates the rate by inflating the denominator with no matching
numerator). "old" (censored) arm is unchanged at its original 3 jobs -- this
campaign only ever needed to tighten the "new" number.
"""
import glob
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.integrate import trapz
except ImportError:
    from numpy import trapz

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "figure.dpi": 200,
})
C_REF, C_OLD, C_NEW = "#1f4e79", "#7a8b99", "#c1440e"
OLD = "/Volumes/X10Pro/Old-Simulations/Sim/S1GDMLSRON"
RUN3 = f"{OLD}/Analysis/ProcessedData-run3"
RT = "/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDMLSRON_CryoSphereEmission/cr_retest"
MORESTAT = "/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDMLSRON_CryoSphereEmission/cr_retest_morestat"
COLS = ["EventId", "StartE", "particleInt", "Edep", "pixelID", "PrimBool",
        "PrimEpre", "GlobalTime"]

ANG = pd.read_table(f"{OLD}/OmniFluxes_Run1.tsv",
                    names=["Energy", "OmniMuPlus", "OmniMuMinus", "OmniElectron",
                           "OmniPositron", "OmniGamma", "OmniProton"]).astype(float)
FLUXNUM = (trapz(ANG.OmniMuPlus + ANG.OmniMuMinus, ANG.Energy)
           * 4 * np.pi * 4200.0 ** 2 * ((2 - np.sqrt(2)) / 2) ** 2 * 3.563e-3)
NGEN_CRYOAC = 131_129_837_637
AREA = 4e-2
CHI = {"old": 6.0840e-3, "new": 6.9210e-3}
BW = 0.8
BINS = np.arange(0.0, 100.0 + BW, BW)
CEN = BINS[:-1] + BW / 2


def per_event(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.array([])
    return pd.read_table(path, names=COLS).groupby("EventId")["Edep"].sum().values


def last_event_id(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return -1
    with open(path, "rb") as fh:
        fh.seek(max(0, os.path.getsize(path) - 65536))
        for line in reversed(fh.read().split(b"\n")):
            if line.strip():
                return int(line.split(b"\t")[0])
    return -1


def job_ready(jd):
    det1 = glob.glob(f"{jd}/proc/Detector1_*.dat")
    return bool(det1) and any(os.path.getsize(f) > 0 for f in det1)


e_ref = per_event(f"{RUN3}/Detector1All.dat")
arms, inj, njobs = {"old": [], "new": []}, {"old": 0, "new": 0}, {"old": 0, "new": 0}

for jd in sorted(glob.glob(f"{RT}/job[0-5]")):
    if not job_ready(jd):
        continue
    a = open(f"{jd}/arm.txt").read().strip()
    n = last_event_id(f"{jd}/outputGDML.dat") + 1
    inj[a] += n if n > 0 else int(open(f"{jd}/n_injected.txt").read().strip())
    njobs[a] += 1
    for f in glob.glob(f"{jd}/proc/Detector1_*.dat"):
        v = per_event(f)
        if v.size:
            arms[a].append(v)

for jd in sorted(glob.glob(f"{MORESTAT}/job*")):
    if not job_ready(jd):
        continue
    n = last_event_id(f"{jd}/outputGDML.dat") + 1
    inj["new"] += n if n > 0 else int(open(f"{jd}/n_injected.txt").read().strip())
    njobs["new"] += 1
    for f in glob.glob(f"{jd}/proc/Detector1_*.dat"):
        v = per_event(f)
        if v.size:
            arms["new"].append(v)

arms = {k: (np.concatenate(v) if v else np.array([])) for k, v in arms.items()}
NGEN_EFF = {k: inj[k] / CHI[k] for k in arms}
print(f"jobs used: {njobs}")
print("Det1 events:", {k: len(v) for k, v in arms.items()},
      "| injected crossings:", inj,
      "| effective primaries:", {k: f"{v:.3e}" for k, v in NGEN_EFF.items()})

c_ref, _ = np.histogram(e_ref, bins=BINS)
sf_ref = FLUXNUM / (NGEN_CRYOAC * AREA * BW)
f_ref = c_ref * sf_ref

fig, (ax, axr) = plt.subplots(
    2, 1, figsize=(5.6, 4.4), sharex=True,
    gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.06})
m = CEN <= 20

ax.step(BINS[:-1][m], np.where(c_ref[m] > 0, f_ref[m], np.nan), where="post",
        color=C_REF, lw=1.2, label="full Geant4 (RUN CRYOAC)")
axr.axhline(1.0, color="0.35", lw=0.7, ls="--")
for k, c, lab in (("old", C_OLD, "MAGI, censored training set"),
                  ("new", C_NEW, f"MAGI, ingoing-fixed ({njobs['new']} jobs)")):
    cm, _ = np.histogram(arms[k], bins=BINS)
    sf = FLUXNUM / (NGEN_EFF[k] * AREA * BW)
    f, ef = cm * sf, np.sqrt(np.maximum(cm, 1)) * sf
    ax.errorbar(CEN[m], np.where(cm[m] > 0, f[m], np.nan), yerr=ef[m], fmt="o",
                ms=2.4, color=c, lw=0, elinewidth=0.7, label=lab)
    ok = (c_ref > 0) & (cm > 0) & m
    r = np.where(ok, f / np.where(c_ref > 0, f_ref, np.nan), np.nan)
    er = np.where(ok, r * np.sqrt(1 / np.maximum(cm, 1) + 1 / np.maximum(c_ref, 1)), np.nan)
    axr.errorbar(CEN[ok], r[ok], yerr=er[ok], fmt="o", ms=2.4, color=c,
                 lw=0, elinewidth=0.7)
    tot = (cm[m].sum() * sf) / (c_ref[m].sum() * sf_ref)
    mip = (CEN >= 2.5) & (CEN <= 4.0)
    tot_mip = (cm[mip].sum() * sf) / (c_ref[mip].sum() * sf_ref) if c_ref[mip].sum() > 0 else float("nan")
    print(f"{k}: 0-20 keV integrated MAGI/full = {tot:.3f}   "
          f"2.5-4 keV (MIP) integrated = {tot_mip:.3f}  (n={cm[mip].sum()})")

ax.set_yscale("log")
ax.set_ylabel(r"flux  [cts s$^{-1}$cm$^{-2}$keV$^{-1}$]")
ax.set_title("CR $-$ SRON XFDM, Detector 1 (TES array)", fontsize=8.5, loc="left")
ax.legend(loc="upper right", fontsize=6.8)
axr.set_ylim(0, 2.0)
axr.set_xlim(0, 20)
axr.set_ylabel("MAGI / full", fontsize=7.5)
axr.set_xlabel(r"deposited energy  $E_{\rm dep}$  [keV]")
for a_ in (ax, axr):
    a_.axvspan(2.5, 4.0, color="0.90", zorder=0, lw=0)
axr.text(3.25, 1.78, "MIP\n(2.5-4)", fontsize=6.0, ha="center", color="0.45")

fig.savefig("/Volumes/X10Pro/MAGI/paper_figures/fig_cr_vs_full.pdf", bbox_inches="tight")
fig.savefig("/Volumes/X10Pro/MAGI/paper_figures/fig_cr_vs_full.png", bbox_inches="tight", dpi=200)
print("wrote fig_cr_vs_full.{pdf,png}")
