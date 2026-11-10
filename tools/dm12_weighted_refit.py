#!/usr/bin/env python
"""Weighted prior refit on DM1.2 (A9): refit the prior of a jointly trained primary-conditioned model on its frozen
encoder, with the events of the muon primaries below E_cut weighted by alpha (all their crossings), and write the run.

    python tools/dm12_weighted_refit.py <joint run> <out run> <alpha> [E_cut MeV=100] [split npz]
"""
import json
import sys

import numpy as np

import magi

src, out, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
ecut = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
R = "/Volumes/T7/MAGI-dev/trained_models"
NAME = "mix_DM1_2_primtruth"
meta = json.load(open(f"{R}/{src}/{NAME}_metadata.json"))
prep, cfg = meta["preprocessing_metadata"], meta["model_config"]
n_types, radius = int(prep["n_types"]), float(prep["radius"])
split = sys.argv[5] if len(sys.argv) > 5 else "/Volumes/T7/MAGI-dev/checkpoints/dm12_val_split.npz"
d = np.load(split, allow_pickle=True)
w = lambda pdg, ep: np.where((pdg == 13) & (ep < ecut), alpha, 1.0).astype(np.float32)
wt, wv = w(d["primary_pdg_train"], d["E_prim_train"]), w(d["primary_pdg"], d["E_prim"])
print(f"{src} -> {out}: alpha {alpha}, E_cut {ecut} MeV; weighted events train {np.mean(wt > 1):.3f}, val {np.mean(wv > 1):.3f}", flush=True)
model = magi.load_task_adaptive_model_for_generation(save_dir=f"{R}/{src}", model_name=NAME, model_config=cfg,
                                                     n_types=n_types, radius=radius, verbose=0)
res = magi.refit_prior(model, d["X_cont_train_s"], d["cond_train"], d["X_cont_val_s"], d["cond_val"],
                       checkpoint_dir=f"/Volumes/T7/MAGI-dev/checkpoints/refit_resume/{out}",
                       sample_weight_train=wt, sample_weight_val=wv)
magi.save_refit_model(model, res.prior, src_dir=f"{R}/{src}", out_dir=f"{R}/{out}", model_name=NAME,
                      n_types=n_types, radius=radius, hidden=res.hidden, n_layers=res.n_layers,
                      val_kl_before=res.val_kl_before, val_kl_after=res.val_kl_after, history=res.history)
best = min(res.history, key=lambda h: h["val_kl"])
print(f"weighted val KL {res.val_kl_before:.4f} -> {res.val_kl_after:.4f} (unweighted at the best epoch "
      f"{best['val_kl_unweighted']:.4f}); saved -> {R}/{out}", flush=True)
