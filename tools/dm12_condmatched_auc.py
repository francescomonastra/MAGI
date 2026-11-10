#!/usr/bin/env python
"""B2 on DM1.2: condition-matched classifier AUC of generated against held-out
real crossings, per primary species, with seed-to-seed intervals.

Real arm: the test split of AR-Flow/data/dm12.npz (showers no model trained
on). Each generated file holds one crossing per real test crossing, drawn with
the same crossing type, primary type and E_prim (AR-Flow/scripts/train_dm12.py
sample, tools/dm12_condmatched_cvae.py). Features and classifier are those of
the crossing-level AUC of tools/henabics_closure_dm12.py (log10 E, X, Y, Z,
Vx, Vy, Vz, log10 E_prim, crossing-type one-hot; default gradient boosting,
5-fold cross-validated predictions); replication r uses generation seed r and
fold seed r. The folds are grouped by shower (the closure's are not), since
crossings of one primary are correlated and row-level folds let the classifier
recognise individual showers: with them, val against test already gives 0.54
for mu-. The null compares the val split with the test split.

"Continuum" repeats the test without the uncollided crossings (E = E_prim) of
either arm, as in docs/MAGI_HENABICS_integration_requirements.md §8.

    python tools/dm12_condmatched_auc.py NAME=GLOB[:raw] ... [--null K]
":raw" reads the first-draw (*_raw) columns, before the inward redraw.
Output: checkpoints/dm12_condmatched/auc.json (override with AUC_OUT).
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

ROOT = "/Volumes/T7/MAGI-dev"
D = np.load(f"{ROOT}/AR-Flow/data/dm12.npz")
PDGS = [int(p) for p in D["idx_to_primary_type"]]
N_TYPES = len(D["idx_to_type"])
RTOL = 1e-5


def real_split(split):
    u_r, phi_r, u_v, phi_v = (D[f"{split}_{k}"] for k in ("u_r", "phi_r", "u_v", "phi_v"))
    st, sv = np.sqrt(1 - u_r ** 2), np.sqrt(1 - u_v ** 2)
    P = D["center"] + D["radius"] * np.stack([st * np.cos(phi_r), st * np.sin(phi_r), u_r], 1)
    V = np.stack([sv * np.cos(phi_v), sv * np.sin(phi_v), u_v], 1)
    return {"E": D[f"{split}_E"], "P": P, "V": V, "type": D[f"{split}_type"],
            "ptype": D[f"{split}_ptype"], "E_prim": D[f"{split}_E_prim"], "group": D[f"{split}_group"]}


def gen_file(path, raw):
    g = np.load(path)
    s = "_raw" if raw else ""
    return {"E": g[f"E{s}"], "P": np.stack([g[f"X{s}"], g[f"Y{s}"], g[f"Z{s}"]], 1),
            "V": np.stack([g[f"Vx{s}"], g[f"Vy{s}"], g[f"Vz{s}"]], 1),
            "type": g["type"], "ptype": g["ptype"], "E_prim": g["E_prim"],
            # row i was drawn with the conditioning of real test row i
            "group": D["test_group"]}


def feats(a, p, continuum=False):
    m = a["ptype"] == p
    if continuum:
        m &= np.abs(a["E"] / a["E_prim"] - 1) >= RTOL
    return np.column_stack([np.log10(a["E"][m]), a["P"][m], a["V"][m], np.log10(a["E_prim"][m]),
                            np.eye(N_TYPES)[a["type"][m]]]), a["group"][m]


def auc(real, gen, seed):
    (Xr, gr), (Xg, gg) = real, gen
    rng = np.random.default_rng(seed)
    n = min(len(Xr), len(Xg))
    ir, ig = rng.permutation(len(Xr))[:n], rng.permutation(len(Xg))[:n]
    X, y = np.vstack([Xr[ir], Xg[ig]]), np.r_[np.zeros(n), np.ones(n)]
    # Folds grouped by shower: crossings of one primary are correlated, and a
    # shower seen in training folds would otherwise be recognised when scored.
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), X, y, cv=cv,
                          groups=np.r_[gr[ir], gg[ig]], method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def score(real, gen, seed):
    out = {}
    for p, pdg in enumerate(PDGS):
        out[f"{pdg}"] = auc(feats(real, p), feats(gen, p), seed)
        out[f"{pdg}_continuum"] = auc(feats(real, p, True), feats(gen, p, True), seed)
        m = gen["ptype"] == p
        out[f"{pdg}_outgoing"] = float(np.mean(np.einsum("ij,ij->i", gen["P"][m] - D["center"],
                                                         gen["V"][m]) > 0))
    return out


def summary(v):
    v = np.asarray(v)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "n": len(v)}


def main():
    args = sys.argv[1:]
    k_null = 0
    if "--null" in args:
        i = args.index("--null"); k_null = int(args[i + 1]); del args[i:i + 2]
    real = real_split("test")
    res, per = {}, {}
    for a in args:
        name, pat = a.split("=", 1)
        raw = pat.endswith(":raw")
        pat = pat[:-4] if raw else pat
        files = sorted(glob.glob(pat if os.path.isabs(pat) else f"{ROOT}/{pat}"))
        if not files:
            raise SystemExit(f"no files for {name}: {pat}")
        per[name] = []
        for r, f in enumerate(files):
            s = score(real, gen_file(f, raw), seed=r)
            per[name].append(s)
            print(f"{name:>10s} rep {r}  " + "  ".join(f"{k} {v:.4f}" for k, v in s.items()), flush=True)
    if k_null:
        val = real_split("val")
        per["real_vs_real"] = []
        for r in range(k_null):
            s = score(real, val, seed=r)
            per["real_vs_real"].append(s)
            print(f"{'real-real':>10s} rep {r}  " + "  ".join(f"{k} {v:.4f}" for k, v in s.items()), flush=True)
    print()
    for name, rows in per.items():
        res[name] = {k: summary([x[k] for x in rows]) for k in rows[0]}
        print(f"{name:>12s}  " + "  ".join(
            f"{k} {s['mean']:.4f}±{s['std'] or 0:.4f}" for k, s in res[name].items()))
    names = [n for n in per if n != "real_vs_real"]
    res["paired_differences"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = min(len(per[a]), len(per[b]))
            for k in per[a][0]:
                if k.endswith("outgoing"):
                    continue
                s = summary([per[a][r][k] - per[b][r][k] for r in range(m)])
                res["paired_differences"][f"{a} - {b} ({k})"] = s
                print(f"  {a} - {b} ({k}): {s['mean']:+.4f} ± {s['std'] or 0:.4f}")
    out = os.environ.get("AUC_OUT", f"{ROOT}/checkpoints/dm12_condmatched/auc.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
