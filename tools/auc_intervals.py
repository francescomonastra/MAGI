#!/usr/bin/env python
"""A1 of docs/CVAE_vs_AutoRegFlow.md: seed-to-seed intervals on the
classifier two-sample AUC of tools/score_prior_refit.py.

Replication r uses generation seed r of each arm (one file per seed), an
independent random subset of N real crossings drawn from the whole real file
(seed r, shared by all arms), and classifier split seed r. The spread over r
therefore contains the generation, the real-sample and the classifier
variance together. Since all arms share the real subset and the split of
replication r, arm differences are also given paired per replication, which
cancels the common part.

A real-vs-real null (two disjoint real subsets per replication, same
classifier) gives the AUC that sampling noise alone produces.

The headline protocol of score_prior_refit takes the *first* N real rows of
the file; the random subsets here are the reason the per-arm means can differ
slightly from those headline numbers.

Usage:
    python tools/auc_intervals.py NAME=GLOB [NAME=GLOB ...] [--null K]
e.g.
    python tools/auc_intervals.py \\
        warpfix='checkpoints/generated_particles/priorrefit_v0_8_2_CR_ingoingfix_priorrefit_wide_warpfix_400000*.txt' \\
        --null 5
Output: checkpoints/auc_intervals.json (override with AUC_INTERVALS_OUT).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reweight_magi_to_real as rw  # noqa: E402
import score_prior_refit as sc  # noqa: E402

ROOT = sc.ROOT
N = int(os.environ.get("AUC_N", sc.N_REAL))
CACHE = f"{ROOT}/checkpoints/real_cr_ingoingfix_feats.npz"


def real_pool():
    """All real crossings as (features, inward flag), cached."""
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        return c["X"], c["inward"]
    P, V, E, Nm = rw.load(sc.REAL, "real", 10 ** 9)
    X = rw.feats(P, V, E, Nm)
    inward = np.einsum("ij,ij->i", P - rw.C, V) < 0
    np.savez(CACHE, X=X, inward=inward)
    return X, inward


def gen_feats(path):
    P, V, E, Nm = rw.load(path, "gen", N)
    return rw.feats(P, V, E, Nm), np.einsum("ij,ij->i", P - rw.C, V) < 0


def score(Xr, in_r, Xg, in_g, seed):
    auc, _ = sc.auc_calibrated(Xr, Xg, seed=seed)
    a, b = Xr[in_r], Xg[in_g]
    m = min(len(a), len(b))
    return {"auc": float(auc), "auc_inward": float(sc.auc_plain(a[:m], b[:m], seed=seed))}


def summary(vals):
    v = np.asarray(vals)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def main():
    args = sys.argv[1:]
    k_null = 0
    if "--null" in args:
        i = args.index("--null"); k_null = int(args[i + 1]); del args[i:i + 2]
    arms = {}
    for a in args:
        name, pat = a.split("=", 1)
        files = sorted(glob.glob(pat if os.path.isabs(pat) else f"{ROOT}/{pat}"))
        if not files:
            raise SystemExit(f"no files for {name}: {pat}")
        arms[name] = files
    Xall, in_all = real_pool()
    print(f"real pool {len(Xall):,}  N per sample {N:,}", flush=True)
    n_rep = max([len(f) for f in arms.values()] + [k_null])
    subsets = [np.random.default_rng(1000 + r).permutation(len(Xall))[:2 * N] for r in range(n_rep)]

    res = {"N": N, "arms": {}, "paired_differences": {}}
    per = {}
    for name, files in arms.items():
        rows = []
        for r, f in enumerate(files):
            idx = subsets[r][:N]
            Xg, in_g = gen_feats(f)
            n = min(len(idx), len(Xg))
            s = score(Xall[idx[:n]], in_all[idx[:n]], Xg[:n], in_g[:n], seed=r)
            rows.append({"rep": r, "file": os.path.basename(f), **s})
            print(f"{name:>12s} rep {r}  AUC {s['auc']:.4f}  inward-only {s['auc_inward']:.4f}", flush=True)
        per[name] = rows
        res["arms"][name] = {"reps": rows, "auc": summary([x["auc"] for x in rows]),
                             "auc_inward": summary([x["auc_inward"] for x in rows])}
    if k_null:
        rows = []
        for r in range(k_null):
            a, b = subsets[r][:N], subsets[r][N:2 * N]
            s = score(Xall[a], in_all[a], Xall[b], in_all[b], seed=r)
            rows.append({"rep": r, **s})
            print(f"{'real-real':>12s} rep {r}  AUC {s['auc']:.4f}  inward-only {s['auc_inward']:.4f}", flush=True)
        res["arms"]["real_vs_real"] = {"reps": rows, "auc": summary([x["auc"] for x in rows]),
                                       "auc_inward": summary([x["auc_inward"] for x in rows])}
    names = list(per)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = min(len(per[a]), len(per[b]))
            for key in ("auc", "auc_inward"):
                d = [per[a][r][key] - per[b][r][key] for r in range(m)]
                res["paired_differences"][f"{a} - {b} ({key})"] = summary(d)

    print()
    for name, v in res["arms"].items():
        for key in ("auc", "auc_inward"):
            s = v[key]
            sd = f"{s['std']:.4f}" if s["std"] is not None else "n/a"
            print(f"{name:>12s} {key:>10s}  {s['mean']:.4f} ± {sd}  [{s['min']:.4f}, {s['max']:.4f}]  n={s['n']}")
    for k, s in res["paired_differences"].items():
        sd = f"{s['std']:.4f}" if s["std"] is not None else "n/a"
        print(f"  {k}: {s['mean']:+.4f} ± {sd}  n={s['n']}")
    out = os.environ.get("AUC_INTERVALS_OUT", f"{ROOT}/checkpoints/auc_intervals.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
