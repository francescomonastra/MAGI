"""Synthetic stress test WITH a real energy<->geometry correlation, to gate
the learnable-prior work (docs/v0.8_learnable_prior_plan.md section 7.1).

Why this test exists
--------------------
tools/synthetic_stress_test_small_singlepeak.py uses *noise* geometry, so a
z-conditioned energy head can only ever LEAK energy through z there - it can
never exercise a legitimate energy<->geometry coupling, and so it cannot tell a
good z_cond model from a bad one. This test injects a KNOWN dependence of energy
on a geometry output, so there is a real correlation to reproduce:

  - A hidden per-event class k in {A, B}, 50/50, NOT exposed in cond (single
    particle type). The model can only capture it through the latent z.
  - Class A: geometry u_r ~ N(+2.0, 0.3);  energy ~ sharp peak at logE=-0.5.
  - Class B: geometry u_r ~ N(-2.0, 0.3);  energy ~ sharp peak at logE=-1.8,
    plus a fraction drawn from a fluorescence LINE at logE=-2.0. The line thus
    appears only for low-u_r events - mimicking a material/volume-dependent line
    correlated with geometry (Francesco's physical motivation).
  - u_v / phi_r / phi_v are pure noise (only u_r carries the coupling).

So the real data has a strong u_r<->energy correlation that is NOT explainable by
cond - only by z. The whole point of the learnable prior is that a z-conditioned
energy flow with the fixed N(0,I) prior samples z from prior holes at generation
and distorts the JOINT (u_r, energy) structure and/or the energy marginal, while
the learned coupling prior p(z|cond) samples realistic z and restores it.

The gating A/B
--------------
Run this twice with --energy-flow-condition z_cond:
  --prior gaussian   (expected: energy marginal and/or the conditional
                      energy-by-u_r structure degrade)
  --prior coupling   (expected: both restored)
and compare the reported energy-marginal Wasserstein, the u_r<->energy
correlation, the per-u_r-side conditional energy Wasserstein, and the line's
concentration on the low-u_r side.

Usage:
  python tools/synthetic_stress_test_coupled.py --prior gaussian  --pin-line-width
  python tools/synthetic_stress_test_coupled.py --prior coupling --pin-line-width
"""
import argparse
import numpy as np
import tensorflow as tf
from scipy.stats import wasserstein_distance, pearsonr
import magi

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--continuum-mode", choices=["gaussian", "flow"], default="flow")
parser.add_argument("--energy-flow-condition", choices=["z_cond", "cond"], default="z_cond",
                    help="What the flow-mode energy head conditions on. The coupling "
                         "prior only matters for 'z_cond' (the physically-correct default "
                         "that carries the energy<->geometry coupling through z).")
parser.add_argument("--prior", choices=["gaussian", "coupling"], default="gaussian",
                    help="Latent prior. 'gaussian' = fixed N(0,I) (the mismatch case); "
                         "'coupling' = learned conditional coupling-flow prior p(z|cond).")
parser.add_argument("--prior-n-layers", type=int, default=6)
parser.add_argument("--prior-hidden", type=int, nargs="+", default=[64, 64])
parser.add_argument("--continuum-flow-bins", type=int, default=8)
parser.add_argument("--continuum-flow-transforms", type=int, default=2)
parser.add_argument("--pin-line-width", action="store_true",
                    help="Pin the line width (non-trainable) to the true injected "
                         "resolution (recommended, matches the flow acceptance config).")
args = parser.parse_args()

magi.initialize_environment(seed=42, cpu_only=True)
rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# Correlated synthetic data
# ----------------------------------------------------------------------
N_TOTAL = 30_000

# Hidden class k in {0 (A), 1 (B)}, 50/50, NOT exposed in cond.
klass = (rng.random(N_TOTAL) < 0.5).astype(np.int32)  # 0 -> A, 1 -> B
is_A = klass == 0
is_B = klass == 1
n_A, n_B = int(is_A.sum()), int(is_B.sum())

# Geometry: only u_r carries the coupling; class sets its mean.
UR_MEAN_A, UR_MEAN_B, UR_SIGMA = +2.0, -2.0, 0.30
ur_q = np.empty(N_TOTAL, dtype=np.float32)
ur_q[is_A] = rng.normal(UR_MEAN_A, UR_SIGMA, size=n_A)
ur_q[is_B] = rng.normal(UR_MEAN_B, UR_SIGMA, size=n_B)

# Energy: class A -> sharp peak at -0.5; class B -> sharp peak at -1.8, with a
# fraction of B drawn from a fluorescence line at -2.0 (line only in class B).
PEAK_A_MU, PEAK_B_MU, PEAK_SIGMA = -0.50, -1.80, 0.15
LINE_Y = -2.00
true_line_sigma_y = 0.015
LINE_FRAC_IN_B = 0.20  # fraction of class-B events that are the line

energy_y = np.empty(N_TOTAL, dtype=np.float32)
energy_y[is_A] = rng.normal(PEAK_A_MU, PEAK_SIGMA, size=n_A)

b_idx = np.where(is_B)[0]
b_is_line = rng.random(n_B) < LINE_FRAC_IN_B
energy_y[b_idx[~b_is_line]] = rng.normal(PEAK_B_MU, PEAK_SIGMA, size=int((~b_is_line).sum()))
energy_y[b_idx[b_is_line]] = rng.normal(LINE_Y, true_line_sigma_y, size=int(b_is_line.sum()))

n_line_total = int(b_is_line.sum())
E_all = (10.0 ** energy_y).astype(np.float64)

print(f"N_TOTAL={N_TOTAL}  n_A={n_A}  n_B={n_B}  n_line_total={n_line_total} "
      f"(line only in class B, {LINE_FRAC_IN_B:.0%} of B)")
print(f"class A: u_r~N({UR_MEAN_A},{UR_SIGMA}), logE~N({PEAK_A_MU},{PEAK_SIGMA})")
print(f"class B: u_r~N({UR_MEAN_B},{UR_SIGMA}), logE~N({PEAK_B_MU},{PEAK_SIGMA}) + line@{LINE_Y}")
print(f"REAL corr(u_r, logE) = {pearsonr(ur_q, energy_y)[0]:.4f}")

# Lines / gate targets.
line_energies_mev = np.array([10.0 ** LINE_Y], dtype=np.float64)
line_positions_y = np.log10(line_energies_mev).astype(np.float32)
matched_lines = [{"label": "synthetic_line_0", "origin": "synthetic",
                  "candidate_energy_mev": float(line_energies_mev[0]),
                  "count": int(n_line_total)}]

energy_bins = magi.build_energy_bins(E_all, mode="log_fixed_count", n_bins=256, min_counts=20)
gate_targets = magi.build_gate_targets(E_all, energy_bins, matched_lines)
print("gate_targets mean per column:", gate_targets.mean(axis=0))

# Other geometry columns: pure noise.
uv_q = rng.normal(0, 1, size=N_TOTAL).astype(np.float32)
phi_r_q = rng.normal(0, 1, size=N_TOTAL).astype(np.float32)
phi_v_q = rng.normal(0, 1, size=N_TOTAL).astype(np.float32)

y_cont = np.concatenate(
    [ur_q[:, None], uv_q[:, None], phi_r_q[:, None], phi_v_q[:, None],
     energy_y[:, None], gate_targets.astype(np.float32)],
    axis=1,
).astype(np.float32)

n_types = 1
cond = np.ones((N_TOTAL, n_types), dtype=np.float32)
E_idx_dummy = np.zeros((N_TOTAL,), dtype=np.int32)
dummy_y = np.zeros((N_TOTAL, 1), dtype=np.float32)

n_val = int(0.15 * N_TOTAL)
idx = rng.permutation(N_TOTAL)
val_idx, train_idx = idx[:n_val], idx[n_val:]

def make_ds(idx_sel, batch_size, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(
        ((y_cont[idx_sel], E_idx_dummy[idx_sel], cond[idx_sel]), dummy_y[idx_sel])
    )
    if shuffle:
        ds = ds.shuffle(len(idx_sel), seed=42)
    return ds.batch(batch_size)

train_ds = make_ds(train_idx, 512, shuffle=True)
val_ds = make_ds(val_idx, 512, shuffle=False)

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
print(f"\nTraining: continuum_mode={args.continuum_mode}, "
      f"energy_flow_condition={args.energy_flow_condition}, prior={args.prior}, "
      f"beta={args.beta}, epochs={args.epochs}")

model_kwargs = dict(
    n_types=n_types,
    line_positions_y=line_positions_y,
    latent_dim=8,
    hidden=(128, 128, 64),
    beta=args.beta,
    prior=args.prior,
    prior_n_layers=args.prior_n_layers,
    prior_hidden=tuple(args.prior_hidden),
)
if args.pin_line_width:
    model_kwargs["line_logsigma_init"] = float(np.log(true_line_sigma_y))
    model_kwargs["line_logsigma_trainable"] = False
if args.continuum_mode == "flow":
    model_kwargs.update(
        continuum_mode="flow",
        energy_flow_condition=args.energy_flow_condition,
        continuum_flow_bins=args.continuum_flow_bins,
        continuum_flow_transforms=args.continuum_flow_transforms,
        continuum_flow_y_mean=float(energy_y.mean()),
        continuum_flow_y_scale=float(energy_y.std()),
    )
    print(f"  prior={args.prior} n_layers={args.prior_n_layers} hidden={tuple(args.prior_hidden)}")

model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        gate_focal_gamma=0.0,  # pinned: was the default before v0.8.2 flipped it**model_kwargs)
magi.compile_model(model, learning_rate=2e-4)

history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)
print("\nFinal metrics:",
      {k: round(v[-1], 4) for k, v in history.history.items() if k.startswith("val_")})

# ----------------------------------------------------------------------
# Generate + evaluate the JOINT (u_r, energy) structure
# ----------------------------------------------------------------------
gen_out = model.generate(tf.constant(cond), N_TOTAL)
energy_y_gen = gen_out["energy_y"].numpy()
ur_gen = gen_out["y_cont"].numpy()[:, 0]

real_r = pearsonr(ur_q, energy_y)[0]
gen_r = pearsonr(ur_gen, energy_y_gen)[0]

print("\n================ JOINT energy<->geometry fidelity ================")
print(f"corr(u_r, logE):        real={real_r:+.4f}   gen={gen_r:+.4f}   "
      f"(|gen-real|={abs(gen_r-real_r):.4f})")

# Conditional energy by u_r side. The two sides map to the two energy peaks;
# a broken joint shows up as wrong conditional means and large conditional W1.
def side_stats(ur, ey):
    hi = ey[ur > 0.0]
    lo = ey[ur < 0.0]
    return hi, lo

r_hi, r_lo = side_stats(ur_q, energy_y)
g_hi, g_lo = side_stats(ur_gen, energy_y_gen)
print(f"E[logE | u_r>0]:        real={r_hi.mean():+.3f}   gen={g_hi.mean():+.3f}")
print(f"E[logE | u_r<0]:        real={r_lo.mean():+.3f}   gen={g_lo.mean():+.3f}")
print(f"conditional W1 (u_r>0): {wasserstein_distance(r_hi, g_hi):.4f}")
print(f"conditional W1 (u_r<0): {wasserstein_distance(r_lo, g_lo):.4f}")

# Line concentration: real lines sit only at u_r<0. Where do generated
# line-region events land?
def frac_low_ur_in_line(ur, ey):
    in_line = (ey >= LINE_Y - 0.05) & (ey < LINE_Y + 0.05)
    if in_line.sum() == 0:
        return float("nan"), 0
    return float((ur[in_line] < 0.0).mean()), int(in_line.sum())

r_frac, r_n = frac_low_ur_in_line(ur_q, energy_y)
g_frac, g_n = frac_low_ur_in_line(ur_gen, energy_y_gen)
print(f"\nline-region events with u_r<0: real={r_frac:.3f} (n={r_n})   "
      f"gen={g_frac:.3f} (n={g_n})   [should be ~1.0]")

print("\n================ MARGINAL fidelity ================")
print(f"energy_y marginal W1:   {wasserstein_distance(energy_y, energy_y_gen):.4f}")
print(f"u_r     marginal W1:    {wasserstein_distance(ur_q, ur_gen):.4f}")

# Energy marginal windows (both peaks + line + empty regions).
windows = [
    (-2.30, -2.20, "empty below line"),
    (-2.10, -1.90, "line cluster"),
    (-1.90, -1.70, "peak B core"),
    (-1.70, -1.00, "gap between peaks (~empty)"),
    (-0.70, -0.30, "peak A core"),
    (-0.30, 0.50, "far above peak A (~empty)"),
]
print("\nEnergy marginal windows (real vs generated counts):")
for lo, hi, desc in windows:
    n_real = int(np.sum((energy_y >= lo) & (energy_y < hi)))
    n_gen_w = int(np.sum((energy_y_gen >= lo) & (energy_y_gen < hi)))
    ratio = (n_gen_w / n_real) if n_real > 0 else (float("inf") if n_gen_w > 0 else 1.0)
    print(f"  logE in [{lo:+.2f},{hi:+.2f}) {desc:28s} "
          f"n_real={n_real:6d} n_gen={n_gen_w:6d} ratio={ratio:.3f}")

print("\nDONE")
