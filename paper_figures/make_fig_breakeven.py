#!/usr/bin/env python
"""Break-even cost vs. number of inner design variants, DM1.2 and SRON.

Replaces the earlier hand-drawn SVG mock-up on the talk's "what it buys"
slide with an actual computed figure. All numbers are taken directly from
docs/MAGI_state_reference.tex (S:"Reuse across design variants" and
S:cost-sron) -- nothing here is fit or eyeballed.

Cost axis is normalized to "one full reference-equivalent simulation run"
for the geometry in question, because that is the only unit available for
both geometries on equal footing:

DM1.2 -- absolute core-hours are stated directly:
  full simulation:  N x 101.7 core-h            (2e8 primaries x 5.084e-7 core-h/primary)
  MAGI:             102.2 + N x 48.7 core-h      (102.2 = one full reference run to
                     produce the training crossings, 0.51 core-h actual training time;
                     48.7 = inner-only transport per variant, cross-checked two
                     independent ways to 0.03%)
  break-even N = 102.2 / (101.7-48.7) = 1.928  (state reference: 1.93)
  asymptotic speed-up = 101.7/48.7 = 2.088    (state reference: 2.09x)
  Normalized by 101.7 to plot alongside SRON.

SRON -- only the outer/inner cost SHARE is measured directly (70%/30%), not
an absolute total for one full reference run. The two headline numbers the
paper states -- break-even N=1.43 and asymptotic speed-up 3.33x -- are
exactly reproduced by the same model DM1.2 uses, in normalized units where
one full run = 1:
  full simulation:  N x 1
  MAGI:             1 + N x 0.30
  break-even N = 1/0.70 = 1.429               (state reference: 1.43)
  asymptotic speed-up = 1/0.30 = 3.333        (state reference: 3.33x)
This is the minimal model consistent with both stated numbers -- not an
independent fit -- so plotting it does not add information beyond what the
text already asserts, only a visual of the same arithmetic.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 200,
})

C_REF = "#1f4e79"    # full Geant4 -- matches Fig. 1 / Fig. 2
C_MAGI = "#c1440e"   # MAGI        -- matches Fig. 1 / Fig. 2

# ---- DM1.2, absolute core-hours, normalized to one full run (101.7 core-h)
DM12_FULL_PER_N = 101.7
DM12_MAGI_INTERCEPT = 102.2
DM12_MAGI_SLOPE = 48.7
dm12_full = lambda n: n * DM12_FULL_PER_N / DM12_FULL_PER_N
dm12_magi = lambda n: (DM12_MAGI_INTERCEPT + n * DM12_MAGI_SLOPE) / DM12_FULL_PER_N
dm12_breakeven = DM12_MAGI_INTERCEPT / (DM12_FULL_PER_N - DM12_MAGI_SLOPE)
dm12_asymptote = DM12_FULL_PER_N / DM12_MAGI_SLOPE

# ---- SRON, normalized units (only the 70/30 split and the two headline
# results are measured; see docstring)
SRON_INNER_SHARE = 0.30
sron_full = lambda n: n * 1.0
sron_magi = lambda n: 1.0 + n * SRON_INNER_SHARE
sron_breakeven = 1.0 / (1.0 - SRON_INNER_SHARE)
sron_asymptote = 1.0 / SRON_INNER_SHARE

print(f"DM1.2 break-even N={dm12_breakeven:.3f}  asymptote={dm12_asymptote:.3f}x")
print(f"SRON  break-even N={sron_breakeven:.3f}  asymptote={sron_asymptote:.3f}x")

N = np.linspace(0, 6, 200)

fig, ax = plt.subplots(figsize=(5.6, 3.55))

# Both "full simulation" curves are exactly y=N in these normalized units --
# that's a consequence of normalizing each geometry to its own one-full-run
# cost, not a coincidence to hide -- so one reference line represents both,
# rather than plotting two identical overlapping lines.
ax.plot(N, N, color=C_REF, lw=2.0, ls="-", label="full simulation (either geometry)")
ax.plot(N, dm12_magi(N), color=C_MAGI, lw=2.0, ls="-", label="DM1.2 — MAGI")
ax.plot(N, sron_magi(N), color=C_MAGI, lw=2.0, ls="--", label="SRON — MAGI")

for be in (dm12_breakeven, sron_breakeven):
    ax.plot([be], [be], "o", ms=6, mfc="#ffc44d", mec="0.15", mew=0.8, zorder=5)

# Both break-even labels share the same x column (anchored at the SRON
# point's x) so they read as a stacked pair rather than drifting apart.
label_x = sron_breakeven
ax.text(label_x, sron_breakeven - 0.14, f"N={sron_breakeven:.2f}",
        fontsize=8, fontweight="bold", color="0.15", ha="left", va="top", zorder=6)
ax.text(label_x, dm12_breakeven + 0.30, f"N={dm12_breakeven:.2f}",
        fontsize=8, fontweight="bold", color="0.15", ha="left", va="bottom", zorder=6)

# Label position = each curve's actual value at x=5.85 (NOT the asymptote
# number itself -- using the ratio as a cost-axis y-coordinate mislabeled
# the wrong line in an earlier version of this script).
ax.text(5.85, dm12_magi(5.85) + 0.10, f"{dm12_asymptote:.2f}×", fontsize=8.5,
        fontweight="bold", color=C_MAGI, ha="right", va="bottom")
ax.text(5.85, sron_magi(5.85) - 0.16, f"{sron_asymptote:.2f}×", fontsize=8.5,
        fontweight="bold", color=C_MAGI, ha="right", va="top")

ax.set_xlim(0, 6)
ax.set_ylim(0, 6.3)
ax.set_xlabel("inner design variants  N")
ax.set_ylabel("cumulative cost  [× one full simulation run]")
ax.legend(loc="upper left", fontsize=8, ncol=1, handlelength=2.4)
ax.set_title("Break-even, reuse across design variants", fontsize=9.5, loc="left")

fig.tight_layout()
out = "/Volumes/X10Pro/MAGI/paper_figures/fig_breakeven.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print("wrote", out)
