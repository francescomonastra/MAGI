"""v0.8 architecture demo plots on the single-peak synthetic stress test.

Trains two energy heads on the SAME synthetic dataset (Small's pathology:
single sharp peak + sparse tail + 2 close lines) and plots real-vs-generated
energy spectra side by side:

  - continuum_mode="gaussian"  (the v0.8 K=1 parametric-continuum baseline)
  - continuum_mode="flow"      (the new conditional RQS-flow continuum;
                                energy_flow_condition="cond" ablation +
                                line width pinned to the injected resolution -
                                the config that passes this test at convergence)

Output: Plots/v0_8_flow_vs_gaussian_synthetic.png

This is a SYNTHETIC demonstration of the continuum shape-capacity gain. It
uses noise geometry, so it does NOT exercise the energy<->geometry coupling
(that needs the learnable prior - see docs/v0.8_learnable_prior_plan.md);
it isolates the continuum-shape question the flow was built to solve.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import magi

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--out", type=str, default="Plots/v0_8_flow_vs_gaussian_synthetic.png")
args = parser.parse_args()

magi.initialize_environment(seed=42, cpu_only=True)
rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# Synthetic data (identical construction to
# tools/synthetic_stress_test_small_singlepeak.py)
# ----------------------------------------------------------------------
N_TOTAL = 30_000
LINE_FRAC_TOTAL = 0.03
N_LINES = 2
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

line_positions_y_true = np.array([-2.30, -2.268], dtype=np.float64)
line_energies_mev = 10.0 ** line_positions_y_true
true_line_sigma_y = 0.015
n_per_line = n_lines_total // N_LINES
counts_per_line = [n_per_line] * (N_LINES - 1) + [n_lines_total - n_per_line * (N_LINES - 1)]
E_lines = np.concatenate([
    10.0 ** rng.normal(y0, true_line_sigma_y, size=n)
    for y0, n in zip(line_positions_y_true, counts_per_line)
])

E_all = np.concatenate([E_continuum, E_lines]).astype(np.float64)
rng.shuffle(E_all)

energy_bins = magi.build_energy_bins(E_all, mode="log_fixed_count", n_bins=256, min_counts=20)
matched_lines = [
    {"label": f"synthetic_line_{i}", "origin": "synthetic",
     "candidate_energy_mev": float(e), "count": int(c)}
    for i, (e, c) in enumerate(zip(line_energies_mev, counts_per_line))
]
gate_targets = magi.build_gate_targets(E_all, energy_bins, matched_lines)

energy_y = np.log10(E_all).astype(np.float32)
line_positions_y = np.log10(line_energies_mev).astype(np.float32)

n = E_all.size
noise = lambda: rng.normal(0, 1, size=n).astype(np.float32)
y_cont = np.concatenate(
    [noise()[:, None], noise()[:, None], noise()[:, None], noise()[:, None],
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


def train_and_generate(model_kwargs):
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
    gate_focal_gamma=0.0,  # pinned: was the default before v0.8.2 flipped it
    prior="gaussian",  # pinned: was the default before v0.8.2 flipped it**model_kwargs)
    magi.compile_model(model, learning_rate=2e-4)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=0)
    gen = model.generate(tf.constant(cond), n)
    return 10.0 ** gen["energy_y"].numpy()


common = dict(n_types=n_types, line_positions_y=line_positions_y,
              latent_dim=8, hidden=(128, 128, 64), beta=0.2)

print("Training Gaussian (K=1) baseline ...")
E_gen_gauss = train_and_generate(dict(**common))

print("Training flow continuum ...")
E_gen_flow = train_and_generate(dict(
    **common,
    continuum_mode="flow",
    energy_flow_condition="cond",
    continuum_flow_y_mean=float(energy_y.mean()),
    continuum_flow_y_scale=float(energy_y.std()),
    line_logsigma_init=float(np.log(true_line_sigma_y)),
    line_logsigma_trainable=False,
))

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
os.makedirs(os.path.dirname(args.out), exist_ok=True)
bins = np.linspace(-3.6, 1.2, 120)
real_l = np.log10(E_all)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True, sharey=True)
for ax, E_gen, title, col in [
    (axes[0], E_gen_gauss, "Gaussian continuum (v0.8 K=1 baseline)", "#d1495b"),
    (axes[1], E_gen_flow, "Flow continuum (new)", "#2e86ab"),
]:
    ax.hist(real_l, bins=bins, histtype="stepfilled", color="0.75",
            edgecolor="0.5", label="real", log=True)
    ax.hist(np.log10(E_gen), bins=bins, histtype="step", color=col, lw=2.0,
            label="generated", log=True)
    for ly in line_positions_y_true:
        ax.axvline(ly, color="0.3", ls=":", lw=1.0)
    ax.axvspan(-0.8, -0.4, color="orange", alpha=0.10)  # peak core
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{MeV})$")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.2)
axes[0].set_ylabel("counts / bin")
fig.suptitle(
    "v0.8 energy head on synthetic Small-like spectrum "
    "(sharp peak + sparse tail + 2 close lines)\n"
    "shaded = true peak core; dotted = line positions",
    fontsize=12.5,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(args.out, dpi=140)
print(f"\nSaved figure -> {args.out}")

# quick numeric recap of the key windows
def frac(a, lo, hi):
    return int(np.sum((a >= lo) & (a < hi)))
for lo, hi, desc in [(-0.8, -0.4, "peak core"), (-0.2, 0.2, "peak falling edge"),
                     (0.2, 1.0, "far above peak (real ~empty)")]:
    nr = frac(real_l, lo, hi)
    print(f"{desc:34s} real={nr:6d}  gauss={frac(np.log10(E_gen_gauss),lo,hi):6d}"
          f"  flow={frac(np.log10(E_gen_flow),lo,hi):6d}")
