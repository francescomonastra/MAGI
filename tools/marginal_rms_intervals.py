#!/usr/bin/env python
"""Per-bin RMS of the 1D marginals on CryoSphere-CR with the intervals of A1.

Same replications as tools/auc_intervals.py: replication r takes generation file
r of each arm and the random real subset of seed 1000 + r, and the real-vs-real
null compares two disjoint real subsets. The variables are the features of the
classifier: log10 E, u_r (z of the unit position vector), u_v (z of the
direction), and the cosines and sines of the two azimuths. The metric is
magi.marginal_bin_rms on 50 equal-count bins of the real subset, with +-0.01 in log10 E
removed around each pinned line; about 1 means agreement within Poisson noise.

    python tools/marginal_rms_intervals.py NAME=GLOB [NAME=GLOB ...] [--null K]
Output: checkpoints/marginal_rms.json (override with MARGINAL_RMS_OUT).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auc_intervals as ai  # noqa: E402

import magi  # noqa: E402

VARS = ["log10E", "u_r", "u_v", "cos_phi_r", "sin_phi_r", "cos_phi_v", "sin_phi_v"]


# Windows (in log10 E) removed around the pinned lines of the CR model: an atom is an
# exact value and is scored by the line recovery, not by the binned marginal.
CFG = json.load(open(f"{ai.ROOT}/trained_models/v1_0_CR_A2_seed42/mix_CR_config.json"))
HALF = float(os.environ.get("RMS_LINE_HALF_WIDTH", 0.01))
WINDOWS = [(y - HALF, y + HALF) for y in CFG["line_positions_y"]]


def rms_all(Xr, Xg):
    return {v: magi.marginal_bin_rms(Xr[:, i], Xg[:, i],
                                     exclude_windows=WINDOWS if v == "log10E" else None)["rms_z"]
            for i, v in enumerate(VARS)}


def summ(rows):
    out = {}
    for v in VARS + ["mean"]:
        x = np.array([r[v] for r in rows])
        out[v] = {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else None}
    return out


def main():
    args = sys.argv[1:]
    k_null = 0
    if "--null" in args:
        i = args.index("--null"); k_null = int(args[i + 1]); del args[i:i + 2]
    arms = {}
    for a in args:
        name, pat = a.split("=", 1)
        arms[name] = sorted(glob.glob(pat if os.path.isabs(pat) else f"{ai.ROOT}/{pat}"))
    Xall, _ = ai.real_pool()
    N = ai.N
    n_rep = max([len(f) for f in arms.values()] + [k_null])
    subsets = [np.random.default_rng(1000 + r).permutation(len(Xall))[:2 * N] for r in range(n_rep)]
    res = {}
    for name, files in arms.items():
        rows = []
        for r, f in enumerate(files):
            Xg, _ = ai.gen_feats(f)
            idx = subsets[r][:N]
            row = rms_all(Xall[idx[:len(Xg)]], Xg[:len(idx)])
            row["mean"] = float(np.mean(list(row.values())))
            rows.append(row)
            print(f"{name:>12s} rep {r}  mean {row['mean']:.2f}  " +
                  " ".join(f"{v} {row[v]:.2f}" for v in VARS), flush=True)
        res[name] = {"reps": rows, "summary": summ(rows)}
    if k_null:
        rows = []
        for r in range(k_null):
            a, b = subsets[r][:N], subsets[r][N:2 * N]
            row = rms_all(Xall[a], Xall[b])
            row["mean"] = float(np.mean(list(row.values())))
            rows.append(row)
        res["real_vs_real"] = {"reps": rows, "summary": summ(rows)}
    print()
    print(f"{'arm':>14s} " + " ".join(f"{v:>9s}" for v in VARS + ["mean"]))
    for name, d in res.items():
        print(f"{name:>14s} " + " ".join(
            f"{d['summary'][v]['mean']:6.2f}±{(d['summary'][v]['std'] or 0):.2f}" for v in VARS + ["mean"]))
    out = os.environ.get("MARGINAL_RMS_OUT", f"{ai.ROOT}/checkpoints/marginal_rms.json")
    json.dump(res, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
