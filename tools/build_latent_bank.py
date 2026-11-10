#!/usr/bin/env python
"""Build the latent bank of a primary-conditioned DM1.2 run (A9): encode the training split the run was fitted on
and write <run>/mix_DM1_2_primtruth_latent_bank.npz, for scripts/generate_geant_source.py --latent-bank.

    python tools/build_latent_bank.py <run dir> <split npz> [K=50]
"""
import json
import sys

import numpy as np

import magi

run, split = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 50
RUN = f"/Volumes/T7/MAGI-dev/trained_models/{run}"
NAME = "mix_DM1_2_primtruth"
meta = json.load(open(f"{RUN}/{NAME}_metadata.json"))
prep, cfg = meta["preprocessing_metadata"], meta["model_config"]
model = magi.load_task_adaptive_model_for_generation(
    save_dir=RUN, model_name=NAME, model_config=cfg, n_types=int(prep["n_types"]),
    radius=float(prep["radius"]), verbose=0)
d = np.load(split, allow_pickle=True)
bank = magi.build_latent_bank(model, d["X_cont_train_s"], d["cond_train"], k_neighbours=K)
out = f"{RUN}/{NAME}_latent_bank.npz"
bank.save(out)
print(f"{run}: latent bank of {len(bank.z_mean):,} training events, {len(bank._groups)} groups, K = {K} -> {out}")
