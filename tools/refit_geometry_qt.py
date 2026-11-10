#!/usr/bin/env python
"""Copy a trained run to <tag>_qtfull and refit its geometry quantile transforms with subsample = all training rows.

    python tools/refit_geometry_qt.py <tag> <iso>

Needs checkpoints/iaps_val/split/<iso>.npz (tools/dump_iaps_splits.sh). Weights are unchanged.
"""
import joblib, shutil, sys, numpy as np
from sklearn.preprocessing import QuantileTransformer
R = "/Volumes/T7/MAGI-dev"; tag, iso = sys.argv[1:3]
src, dst = f"{R}/trained_models/{tag}", f"{R}/trained_models/{tag}_qtfull"
shutil.copytree(src, dst)
f = f"{dst}/mix_{iso}_quantile_transformers.joblib"; q = joblib.load(f)
X = np.load(f"{R}/checkpoints/iaps_val/split/{iso}.npz")["X_cont_train"][:, :4]
for i, k in enumerate(["qt_u_r", "qt_u_v", "qt_phi_r", "qt_phi_v"]):
    phys = q[k].inverse_transform(X[:, [i]])
    q[k] = QuantileTransformer(n_quantiles=10000, subsample=len(phys), output_distribution="normal", random_state=42).fit(phys)
joblib.dump(q, f); print("wrote", f)
