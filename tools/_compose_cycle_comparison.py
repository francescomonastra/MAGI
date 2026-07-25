"""Compose Cycle 0 / 1 / 2 progression figures for the v0.8 comparison doc.

Stacks the per-cycle real-vs-generated PNGs (already saved) as labelled rows,
one composite per (dataset, view). Cycle-0/1 checkpoints were overwritten, so
this composites the saved images rather than re-generating.

Sources:
  Cycle 0 : scratch/cycle0/{CR,Small}_{spectrum,corr}.png   (from git ec40d69)
  Cycle 1 : scratch/cycle1/v0_8_real_{CR,Small}_{spectrum,corr}.png
  Cycle 2 : Plots/v0_8_real_{CR,Small}_{spectrum,corr}.png   (current)
Outputs: Plots/v0_8_cycles_{CR,Small}_{spectrum,corr}.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

S = "/private/tmp/claude-501/-Volumes-X10Pro/7fcaacf6-6afd-46f1-86ec-5eef4cd746bc/scratchpad"
PLOTS = "/Volumes/X10Pro/MAGI/Plots"

LABELS = {
    "spectrum": {
        "CR":    ["Cycle 0 (affine flow, w_gate_aux=2)",
                  "Cycle 1 (+ CDF warp)",
                  "Cycle 2 (+ focal gate, gamma=2)"],
        "Small": ["Cycle 0 (affine flow, w_gate_aux=2)",
                  "Cycle 1 (+ CDF warp)",
                  "Cycle 2 (+ focal gate, gamma=2)"],
    },
}
LABELS["corr"] = LABELS["spectrum"]

def paths(view, ds):
    return [
        f"{S}/cycle0/{ds}_{view}.png",
        f"{S}/cycle1/v0_8_real_{ds}_{view}.png",
        f"{PLOTS}/v0_8_real_{ds}_{view}.png",
    ]

# per-view row height (inches) tuned to the source aspect ratio
ROW_H = {"spectrum": 3.2, "corr": 2.6}
FIG_W = {"spectrum": 11.0, "corr": 17.0}

for view in ("spectrum", "corr"):
    for ds in ("CR", "Small"):
        imgs = [mpimg.imread(p) for p in paths(view, ds)]
        labels = LABELS[view][ds]
        n = len(imgs)
        fig, axes = plt.subplots(n, 1, figsize=(FIG_W[view], ROW_H[view] * n))
        for ax, im, lab in zip(axes, imgs, labels):
            ax.imshow(im)
            ax.axis("off")
            ax.set_title(lab, fontsize=13, fontweight="bold", loc="left", pad=4)
        title = ("Energy spectrum" if view == "spectrum" else "Pearson correlation (real | generated | gen-real)")
        fig.suptitle(f"{ds}: {title} - Cycle 0 -> 1 -> 2", fontsize=15, y=0.997)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        out = f"{PLOTS}/v0_8_cycles_{ds}_{view}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)
print("DONE")
