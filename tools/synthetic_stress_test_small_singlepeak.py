"""Synthetic stress test reproducing SMALL's specific pathology (see
docs/v0.8_fixing_plan.md): a single sharp unimodal continuum peak (not
multi-modal, unlike tools/synthetic_stress_test_cr_multimodal.py) + a
sparse broad low-energy tail + 2 lines only 0.032 dex apart (mimicking
Cu K-alpha1/Cu K-beta's real separation) sitting inside that sparse tail.

This is the test that confirmed a single-Gaussian continuum can't
represent Small's shape, and later confirmed that a K-component Gaussian
continuum sub-mixture (n_continuum_components>1) doesn't fix it either -
even once component collapse is fixed with w_continuum_balance /
continuum_mu_init, shape fidelity does not improve over K=1. See
docs/v0.8_fixing_plan.md section 7 for the full results table. Current
conclusion: escalate to a conditional Normalizing Flow continuum.

Usage: python tools/synthetic_stress_test_small_singlepeak.py [--continuum-mode gaussian|flow] [--continuum-flow-bins K] [--continuum-flow-transforms T] [--n-continuum-components K] [--epochs N] [--beta B] [--w-continuum-repulsion W] [--continuum-repulsion-margin M]
"""
import argparse
import numpy as np
import tensorflow as tf
import magi

parser = argparse.ArgumentParser()
parser.add_argument("--n-continuum-components", type=int, default=1)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--w-continuum-repulsion", type=float, default=0.3)
parser.add_argument("--continuum-repulsion-margin", type=float, default=0.05)
parser.add_argument("--continuum-mode", choices=["gaussian", "flow"], default="gaussian")
parser.add_argument("--energy-flow-condition", choices=["z_cond", "cond"], default="z_cond",
                    help="What the flow-mode energy head conditions on. 'z_cond' (default) "
                         "preserves energy<->geometry coupling but needs a learnable prior "
                         "for faithful generation; 'cond' is the shape-capacity ablation that "
                         "passes this (noise-geometry) synthetic test at convergence.")
parser.add_argument("--continuum-flow-bins", type=int, default=8)
parser.add_argument("--continuum-flow-transforms", type=int, default=2)
parser.add_argument("--continuum-warp", choices=["affine", "cdf"], default="affine",
                    help="Continuum-flow standardization. 'affine' = (y-mean)/std "
                         "(uniform knots). 'cdf' = monotone empirical-CDF->N(0,1) warp "
                         "(density-proportional knots; brings a far sparse tail inside "
                         "the spline interval). Only used with --continuum-mode flow.")
parser.add_argument("--sparse-tail", action="store_true",
                    help="Make the low-E tail far and rare (mimics real Small: a "
                         "<1%% population many sigma below the bulk, outside [-B,B] "
                         "under affine standardization) so the CDF warp is exercised.")
parser.add_argument("--w-gate-aux", type=float, default=None,
                    help="Override the model's default auxiliary gate-supervision weight.")
parser.add_argument("--pin-line-width", action="store_true",
                    help="Pin the line width (non-trainable) to the true injected "
                         "resolution instead of learning it - tests whether wide "
                         "learned lines are what over-generates the line components.")
args = parser.parse_args()

magi.initialize_environment(seed=42, cpu_only=True)
rng = np.random.default_rng(42)

N_TOTAL = 30_000
LINE_FRAC_TOTAL = 0.03
N_LINES = 2

# Single sharp peak (mimics Small's real bulk: ~0.05-1 MeV, peaked ~0.25 MeV)
# + a sparse broad low-energy tail (mimics Small's near-empty low-E region)
# + 2 closely-spaced lines (0.032 dex apart, mimicking real Cu K-alpha1/
# Cu K-beta separation) sitting inside that sparse tail.
if args.sparse_tail:
    # Far, rare tail: bulk tight at -0.6, tail centered ~4-8 sigma below with a
    # tiny weight, so under affine (y-mean)/std standardization it lands near or
    # beyond [-B,B] and collapses - the regime the CDF warp is meant to fix.
    PEAK_WEIGHT, PEAK_MU, PEAK_SIGMA = 0.985, -0.60, 0.12
    TAIL_WEIGHT, TAIL_MU, TAIL_SIGMA = 0.015, -3.00, 0.45
else:
    PEAK_WEIGHT, PEAK_MU, PEAK_SIGMA = 0.90, -0.60, 0.20
    TAIL_WEIGHT, TAIL_MU, TAIL_SIGMA = 0.10, -2.30, 0.60

cont_w = np.array([PEAK_WEIGHT, TAIL_WEIGHT])
cont_w = cont_w / cont_w.sum() * (1.0 - LINE_FRAC_TOTAL)

n_lines_total = int(round(N_TOTAL * LINE_FRAC_TOTAL))
n_continuum = N_TOTAL - n_lines_total
n_peak = int(round(n_continuum * cont_w[0] / (1.0 - LINE_FRAC_TOTAL)))
n_tail = n_continuum - n_peak

E_peak = 10.0 ** rng.normal(PEAK_MU, PEAK_SIGMA, size=n_peak)
E_tail = 10.0 ** rng.normal(TAIL_MU, TAIL_SIGMA, size=n_tail)
E_continuum = np.concatenate([E_peak, E_tail])

# Lines: 0.032 dex apart, sitting inside the sparse tail region.
line_positions_y_true = np.array([-2.30, -2.268], dtype=np.float64)
line_energies_mev = 10.0 ** line_positions_y_true
true_line_sigma_y = 0.015

n_per_line = n_lines_total // N_LINES
counts_per_line = [n_per_line] * (N_LINES - 1) + [n_lines_total - n_per_line * (N_LINES - 1)]

E_lines = []
for y0, n in zip(line_positions_y_true, counts_per_line):
    E_lines.append(10.0 ** rng.normal(y0, true_line_sigma_y, size=n))
E_lines = np.concatenate(E_lines)

E_all = np.concatenate([E_continuum, E_lines]).astype(np.float64)
rng.shuffle(E_all)

print(f"N_TOTAL={E_all.size}  n_peak={n_peak}  n_tail={n_tail}  "
      f"n_lines_total={n_lines_total}  per-line counts={counts_per_line}")
print(f"line energies (MeV): {line_energies_mev}  separation(dex)="
      f"{line_positions_y_true[1] - line_positions_y_true[0]:.4f}")

energy_bins = magi.build_energy_bins(E_all, mode="log_fixed_count", n_bins=256, min_counts=20)

matched_lines = [
    {"label": f"synthetic_line_{i}", "origin": "synthetic",
     "candidate_energy_mev": float(e), "count": int(c)}
    for i, (e, c) in enumerate(zip(line_energies_mev, counts_per_line))
]

gate_targets = magi.build_gate_targets(E_all, energy_bins, matched_lines)
print("gate_targets mean per column:", gate_targets.mean(axis=0))

energy_y = np.log10(E_all).astype(np.float32)
line_positions_y = np.log10(line_energies_mev).astype(np.float32)

n = E_all.size
ur_q = rng.normal(0, 1, size=n).astype(np.float32)
uv_q = rng.normal(0, 1, size=n).astype(np.float32)
phi_r_q = rng.normal(0, 1, size=n).astype(np.float32)
phi_v_q = rng.normal(0, 1, size=n).astype(np.float32)

y_cont = np.concatenate(
    [ur_q[:, None], uv_q[:, None], phi_r_q[:, None], phi_v_q[:, None],
     energy_y[:, None], gate_targets.astype(np.float32)],
    axis=1,
).astype(np.float32)

n_types = 1
cond = np.ones((n, n_types), dtype=np.float32)
E_idx_dummy = np.zeros((n,), dtype=np.int32)
dummy_y = np.zeros((n, 1), dtype=np.float32)

n_val = int(0.15 * n)
idx = rng.permutation(n)
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

print(f"\nTraining with continuum_mode={args.continuum_mode}, "
      f"n_continuum_components={args.n_continuum_components}, "
      f"beta={args.beta}, w_continuum_repulsion={args.w_continuum_repulsion}, "
      f"continuum_repulsion_margin={args.continuum_repulsion_margin}, epochs={args.epochs}")

model_kwargs = dict(
    n_types=n_types,
    line_positions_y=line_positions_y,
    latent_dim=8,
    hidden=(128, 128, 64),
    beta=args.beta,
    w_continuum_repulsion=args.w_continuum_repulsion,
    continuum_repulsion_margin=args.continuum_repulsion_margin,
)
if args.n_continuum_components != 1:
    model_kwargs["n_continuum_components"] = args.n_continuum_components
if args.w_gate_aux is not None:
    model_kwargs["w_gate_aux"] = args.w_gate_aux
if args.pin_line_width:
    # true injected line sigma in y-space is true_line_sigma_y; ln() -> logsigma
    model_kwargs["line_logsigma_init"] = float(np.log(true_line_sigma_y))
    model_kwargs["line_logsigma_trainable"] = False
if args.continuum_mode == "flow":
    model_kwargs.update(
        continuum_mode="flow",
        energy_flow_condition=args.energy_flow_condition,
        continuum_flow_bins=args.continuum_flow_bins,
        continuum_flow_transforms=args.continuum_flow_transforms,
        # Standardize the flow's working space from the training energy_y so
        # the data bulk sits inside the spline interval [-B, B].
        continuum_flow_y_mean=float(energy_y.mean()),
        continuum_flow_y_scale=float(energy_y.std()),
        continuum_flow_warp=args.continuum_warp,
    )
    if args.continuum_warp == "cdf":
        yk, zk = magi.fit_cdf_warp_knots(energy_y, n_knots=256, eps=1e-4)
        model_kwargs.update(continuum_flow_warp_y_knots=yk, continuum_flow_warp_z_knots=zk)
        print(f"  flow: bins={args.continuum_flow_bins} transforms={args.continuum_flow_transforms} "
              f"warp=cdf ({yk.size} knots, y_knots range [{yk[0]:.2f},{yk[-1]:.2f}])")
    else:
        print(f"  flow: bins={args.continuum_flow_bins} transforms={args.continuum_flow_transforms} "
              f"warp=affine y_mean={energy_y.mean():.3f} y_scale={energy_y.std():.3f}")

model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        gate_focal_gamma=0.0,  # pinned: was the default before v0.8.2 flipped it
        prior="gaussian",  # pinned: was the default before v0.8.2 flipped it**model_kwargs)
magi.compile_model(model, learning_rate=2e-4)

history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)

print("\nFinal metrics:",
      {k: round(v[-1], 4) for k, v in history.history.items() if k.startswith("val_")})

n_gen = n
gen_cond = np.ones((n_gen, n_types), dtype=np.float32)
gen_out = model.generate(tf.constant(gen_cond), n_gen)

energy_y_gen = gen_out["energy_y"].numpy()
comp_idx_gen = gen_out["energy_component_idx"].numpy()
E_gen = 10.0 ** energy_y_gen

print(f"\nlearned line_logsigma (per line) = "
      f"{np.round(model._line_logsigma_clipped().numpy(), 4)} "
      f"(init {-2.0}, true injected line sigma_y = {true_line_sigma_y})")
print("\ngenerated component_idx value counts:", np.bincount(comp_idx_gen, minlength=N_LINES + 1))
true_weights = np.concatenate([[1.0 - LINE_FRAC_TOTAL], np.full(N_LINES, LINE_FRAC_TOTAL / N_LINES)])
gen_weights = np.bincount(comp_idx_gen, minlength=N_LINES + 1) / n_gen
print("true mixture weights:     ", np.round(true_weights, 4))
print("generated mixture weights:", np.round(gen_weights, 4))

recovery = magi.compute_line_integral_recovery(
    E_all, E_gen, matched_lines, energy_bins, energy_component_idx_gen=comp_idx_gen,
)
print("\nPer-line recovery:")
for r in recovery:
    print(f"  {r['label']:18s} n_real={r['n_real']:6d} n_gen={r['n_gen']:6d} "
          f"recovery_ratio={r['recovery_ratio']:.3f} component_fraction={r['component_fraction']:.5f}")

# Shape-fidelity check: does the generated continuum reproduce the sharp
# single-peak shape, or does it smear into a monotonic ramp like Small did?
print("\nContinuum shape check (windowed real vs generated counts, log-E windows):")
windows_log10 = [
    (-3.5, -3.0, "deep sparse tail"),
    (-3.0, -2.5, "sparse tail"),
    (-2.5, -2.35, "just below line cluster"),
    (-2.35, -2.25, "line cluster (both lines)"),
    (-2.25, -1.5, "sparse tail above cluster"),
    (-1.5, -1.0, "peak rising edge"),
    (-1.0, -0.8, "peak shoulder"),
    (-0.8, -0.4, "peak core"),
    (-0.4, -0.2, "peak falling shoulder"),
    (-0.2, 0.2, "peak falling edge"),
    (0.2, 1.0, "far above peak (should be ~empty)"),
]
real_log10 = np.log10(E_all)
gen_log10 = energy_y_gen
for lo, hi, desc in windows_log10:
    n_real = int(np.sum((real_log10 >= lo) & (real_log10 < hi)))
    n_gen_w = int(np.sum((gen_log10 >= lo) & (gen_log10 < hi)))
    ratio = (n_gen_w / n_real) if n_real > 0 else (float("inf") if n_gen_w > 0 else 1.0)
    print(f"  y in [{lo:+.2f},{hi:+.2f}) {desc:32s} n_real={n_real:6d} n_gen={n_gen_w:6d} ratio={ratio:.3f}")

print("\nDONE")
