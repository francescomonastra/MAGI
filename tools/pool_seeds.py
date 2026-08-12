#!/usr/bin/env python
"""Does generating from a SEED ENSEMBLE stabilise the energy x geometry coupling?

Three seeds of v0.8.2 give median-E-vs-b at b<10mm of 1.222/1.057/0.495, and
three seeds of v0.8.3 give 1.082/0.455/0.962 - so conditioning the energy head
on geometry and widening the latent did NOT reduce the spread. The loss has no
term covering the detector-aimed subset (~0.5% of events), so the model is
unconstrained there and each run lands somewhere different.

If the spread is training stochasticity rather than bias, pooling equal samples
from several seeds should land near the centre with far less run-to-run scatter
than any single seed - which is a deployable mitigation even without a fix.
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, joblib, tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import magi
from magi.generation.export import NON_TRANSPORT_PARTICLES as NON_TRANSPORT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_v082_v083 import load_and_generate, impact, med_ci, DET1, CUTS, TRAIN_FILE

N_EACH = 700_000
FAMILIES = {
 "v0.8.2 x3": ["trained_models/v0_8_2_priorzone_CR",
               "trained_models/v0_8_2_priorzone_CR_seed7",
               "trained_models/v0_8_2_priorzone_CR_seed13"],
 "v0.8.3 x3": ["trained_models/v0_8_3_geomcond_CR",
               "trained_models/v0_8_3_geomcond_CR_seed7",
               "trained_models/v0_8_3_geomcond_CR_seed13"],
}
NU = {"nu_e","anti_nu_e","nu_mu","anti_nu_mu","nu_tau","anti_nu_tau"}
P,U,E = [],[],[]
for line in open(TRAIN_FILE):
    a=line.split()
    if len(a)<9 or a[1] in NU: continue
    P.append((float(a[3]),float(a[4]),float(a[5])))
    U.append((float(a[6]),float(a[7]),float(a[8]))); E.append(float(a[2]))
P=np.array(P);U=np.array(U);E=np.array(E)
b_r,in_r = impact(P,U,DET1)

pools={}
for fam,dirs in FAMILIES.items():
    Es,bs,ins=[],[],[]
    for d in dirs:
        g=load_and_generate(d,"mix_CR",N_EACH,seed=99)
        Pg=np.column_stack([g["x_gen"],g["y_gen"],g["z_gen"]])
        Ug=np.column_stack([g["vx_gen"],g["vy_gen"],g["vz_gen"]])
        bg,ing=impact(Pg,Ug,DET1)
        Es.append(g["E_gen"]); bs.append(bg); ins.append(ing)
    pools[fam]=dict(E=np.concatenate(Es), b=np.concatenate(bs), ing=np.concatenate(ins))

rng=np.random.default_rng(3)
print()
print("="*78)
print(f"SEED-ENSEMBLE pooling, {N_EACH:,} x 3 per family")
print("="*78)
print(f"{'cut':>10s} {'n_real':>9s} | {'v0.8.2 x3':>22s} | {'v0.8.3 x3':>22s}")
for c in CUTS:
    mr=in_r&(b_r<c)
    if mr.sum()<10: continue
    med_r=float(np.median(E[mr]))
    row=f"{'all' if not np.isfinite(c) else f'b<{c:g}mm':>10s} {mr.sum():9,} |"
    for fam in FAMILIES:
        d=pools[fam]; mg=d["ing"]&(d["b"]<c)
        m,lo,hi=med_ci(d["E"][mg],rng=rng)
        row+=f" {m/med_r:7.3f} [{lo/med_r:.3f},{hi/med_r:.3f}] |"
    print(row)
print()
print("single-seed spread at b<10mm, for reference:")
print("  v0.8.2  0.495 / 1.057 / 1.222")
print("  v0.8.3  0.455 / 0.962 / 1.082")
