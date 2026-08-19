#!/usr/bin/env python
"""Detector-level spectra, old (censored) vs new (ingoing-fixed) CR checkpoint.

Both arms: 3 jobs x 1.2e6 crossings, generated and injected in the same session
into SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed.gdml, matched N, both emitted
on the corrected sphere R = 99.0141 mm. The only difference is the training set.
"""
import glob
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 200,
})
C_OLD = "#1f4e79"     # old checkpoint, censored training set
C_NEW = "#c1440e"     # new checkpoint, corrected training set
R = "/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDMLSRON_CryoSphereEmission/cr_retest"


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


arms = {"old": [], "new": []}
inj = {"old": 0, "new": 0}
for jd in sorted(glob.glob(f"{R}/job[0-5]")):
    a = open(f"{jd}/arm.txt").read().strip()
    n = last_event_id(f"{jd}/outputGDML.dat") + 1
    if n <= 0:
        n = int(open(f"{jd}/n_injected.txt").read().strip())
    inj[a] += n
    for d in range(1, 7):
        for f in glob.glob(f"{jd}/proc/Detector{d}_*.dat"):
            e = per_event_edep(f)
            if e.size:
                arms[a].append(e)
arms = {k: (np.concatenate(v) if v else np.array([])) for k, v in arms.items()}
print({k: (len(v), inj[k]) for k, v in arms.items()})

bins = np.logspace(-1, 4, 26)                 # 0.1 keV - 10 MeV, 5 bins/decade
cen = np.sqrt(bins[:-1] * bins[1:])
w = np.diff(bins)

fig, (ax, axr) = plt.subplots(
    2, 1, figsize=(5.4, 4.3), sharex=True,
    gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.06})

hs = {}
for k, c, lab in (("old", C_OLD, "old checkpoint (censored set)"),
                  ("new", C_NEW, "new checkpoint (ingoing-fixed)")):
    h, _ = np.histogram(arms[k], bins=bins)
    hs[k] = h
    y = h / inj[k] / w
    ey = np.sqrt(h) / inj[k] / w
    ok = h > 0
    ax.errorbar(cen[ok], y[ok], yerr=ey[ok], fmt="o", ms=2.6, lw=0,
                elinewidth=0.8, color=c, label=f"{lab}  [{h.sum()} counts]")
    ax.step(bins[:-1], np.where(h > 0, y, np.nan), where="post", color=c, lw=0.9, alpha=0.55)

for a_ in (ax, axr):
    a_.axvspan(1.0, 7.0, color="0.92", zorder=0, lw=0)
ax.text(2.6, ax.get_ylim()[1] * 0.35 if ax.get_ylim()[1] > 0 else 1,
        "MIP\n1-7 keV", fontsize=6.5, ha="center", color="0.35")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylabel(r"counts per injected crossing per keV")
ax.set_title("CR detector-level retest -- SRON, Flower_fixed, matched $N$",
             fontsize=8.5, loc="left")
ax.legend(loc="lower left", fontsize=6.8)

ho, hn = hs["old"], hs["new"]
ok = (ho > 0) & (hn > 0)
ratio = np.where(ok, (hn / inj["new"]) / np.where(ho > 0, ho / inj["old"], np.nan), np.nan)
err = np.where(ok, ratio * np.sqrt(1 / np.maximum(ho, 1) + 1 / np.maximum(hn, 1)), np.nan)
axr.axhline(1.0, color="0.35", lw=0.7, ls="--")
axr.axhline(1.268, color="0.55", lw=0.8, ls=":")
axr.text(1.4e3, 1.30, "predicted 1.268", fontsize=6.2, color="0.45", va="bottom")
axr.errorbar(cen[ok], ratio[ok], yerr=err[ok], fmt="o", ms=2.6, lw=0,
             elinewidth=0.8, color="0.15")
axr.set_xscale("log")
axr.set_ylim(0, 2.6)
axr.set_xlabel("deposited energy per event  [keV]")
axr.set_ylabel("new / old", fontsize=7.5)
axr.text(0.115, 2.25, r"all deposits: $1.097\pm0.041$  (predicted 1.268)",
         fontsize=6.5, color="0.15")

fig.savefig("/Volumes/X10Pro/MAGI/paper_figures/fig_cr_retest.pdf", bbox_inches="tight")
fig.savefig("/Volumes/X10Pro/MAGI/paper_figures/fig_cr_retest.png", bbox_inches="tight", dpi=200)
print("wrote fig_cr_retest.{pdf,png}")
