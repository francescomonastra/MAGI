"""Harder synthetic stress test for CVAE_MixEnergy_ContPhi_TaskAdaptive (v0.8
Part 3) - multi-modal continuum + a cluster of closely-spaced lines sitting
in a genuinely sparse region, mimicking the real CR failure mode that the
original clean synthetic test (single-hump continuum + well-separated,
well-populated lines) did not catch. See docs/v0.8_fixing_plan.md for the
full history and results this script produced.

All physics is generated directly as E (MeV) so real preprocessing.py entry
points (build_energy_bins, build_gate_targets) are exercised faithfully -
these operate in E-space regardless of the model's energy_transform.

Usage:
  # Gaussian-continuum baseline (original behavior):
  python tools/synthetic_stress_test_cr_multimodal.py
  # Flow continuum + learned coupling prior (the v0.8 target config):
  python tools/synthetic_stress_test_cr_multimodal.py \
      --continuum-mode flow --energy-flow-condition z_cond --prior coupling \
      --pin-line-width
"""
import argparse
import numpy as np
import tensorflow as tf
import magi

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--w-continuum-repulsion", type=float, default=0.3)
parser.add_argument("--continuum-repulsion-margin", type=float, default=0.05)
parser.add_argument("--continuum-mode", choices=["gaussian", "flow"], default="gaussian")
parser.add_argument("--energy-flow-condition", choices=["z_cond", "cond"], default="z_cond",
                    help="What the flow-mode energy head conditions on. 'z_cond' (default) "
                         "preserves energy<->geometry coupling and needs a learned prior for "
                         "faithful generation; 'cond' is the shape-capacity ablation.")
parser.add_argument("--continuum-flow-bins", type=int, default=8)
parser.add_argument("--continuum-flow-transforms", type=int, default=2)
parser.add_argument("--prior", choices=["gaussian", "coupling"], default="gaussian",
                    help="Latent prior. 'gaussian' = fixed N(0,I); 'coupling' = learned "
                         "conditional coupling-flow prior p(z|cond).")
parser.add_argument("--prior-n-layers", type=int, default=6)
parser.add_argument("--prior-hidden", type=int, nargs="+", default=[64, 64])
parser.add_argument("--pin-line-width", action="store_true",
                    help="Pin the line width (non-trainable) to the true injected resolution.")
parser.add_argument("--w-flow-line-repulsion", type=float, default=0.0,
                    help="Weight of the flow-line repulsion (flow mode): penalizes the flow "
                         "continuum from spiking at pinned line positions, so the gate routes "
                         "line events to line slots. >0 to enable (fixes line under-generation "
                         "on this multimodal cluster).")
parser.add_argument("--w-gate-aux", type=float, default=None,
                    help="Override the auxiliary gate-supervision weight (default 0.3). "
                         "Higher values route more line-region events to the line slots "
                         "without touching the continuum - the gate-side lever for line "
                         "under-generation.")
parser.add_argument("--gate-class-balance-power", type=float, default=0.0,
                    help="Cycle 2 imbalance-aware gate loss. Per-slot inverse-frequency "
                         "weights w_c = (mass_continuum / mass_c)^power (continuum=1), "
                         "computed from the gate_target column masses. 0 = off (uniform). "
                         "~0.3-0.5 lifts rare-line routing without over-generating.")
parser.add_argument("--gate-class-weight-cap", type=float, default=5.0,
                    help="Cap on per-slot class weights, so rare lines don't blow up "
                         "(mass_cont/mass_line)^power into an unstable, over-routing weight.")
parser.add_argument("--gate-focal-gamma", type=float, default=0.0,
                    help="Cycle 2 focal down-weighting of easy (well-classified) gate "
                         "slots: multiply each slot's CE by (1-p)^gamma. 0 = off. Data-free "
                         "alternative/complement to --gate-class-balance-power.")
parser.add_argument("--continuum-warp", choices=["affine", "cdf"], default="affine",
                    help="Continuum-flow standardization (flow mode). 'cdf' = CDF pre-warp "
                         "(Cycle 1) for parity with the real run.")
parser.add_argument("--line-sigma-y", type=float, default=0.02,
                    help="Injected line width in log10(E). The default 0.02 dex is ~500-1000 eV "
                         "at these energies - far BROADER than the 4 eV lines of the real data, "
                         "so the lines sit on a dense continuum and any proximity labelling "
                         "over-labels. Use ~0.002 for a delta-line regime like the real spectra.")
parser.add_argument("--gate-bandwidth-mode", choices=["bins", "resolution"], default="bins",
                    help="v0.8.1: 'resolution' sets the gate-target bandwidth from the "
                         "physical line width instead of the detection bin width (which "
                         "is 31-10000x too wide on real data and trains the gate to "
                         "over-route by that factor).")
parser.add_argument("--line-frac", type=float, default=0.03,
                    help="Total fraction of events in the 4 lines (default 0.03 = 0.75%%/line). "
                         "Lower values (e.g. 0.008 = 0.2%%/line) mimic real CR's rare Al/Ni "
                         "fluorescence, reproducing the under-routing the class-balance fixes.")
args = parser.parse_args()

magi.initialize_environment(seed=42, cpu_only=True)
rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Synthesize E (MeV): 3-mode continuum (in log10(E) space) + a cluster
#    of 4 closely-spaced lines at 5-10 keV sitting in/near the sparse low
#    tail (mode C) - analogous to real CR/Small's K-lines living where the
#    bulk continuum has already died off.
# ----------------------------------------------------------------------
N_TOTAL = 30_000
LINE_FRAC_TOTAL = args.line_frac
N_LINES = 4

continuum_modes = [
    # (weight within continuum, mu_log10E, sigma_log10E)
    (0.65, -0.30, 0.35),   # bulk, E ~ 0.2-1.1 MeV
    (0.27,  0.60, 0.30),   # secondary bump, E ~ 2-8 MeV
    (0.05, -1.70, 0.30),   # sparse low tail, E ~ 0.01-0.04 MeV
]
cont_w = np.array([m[0] for m in continuum_modes])
cont_w = cont_w / cont_w.sum()

n_lines_total = int(round(N_TOTAL * LINE_FRAC_TOTAL))
n_continuum = N_TOTAL - n_lines_total
n_per_mode = np.round(n_continuum * cont_w).astype(int)
n_per_mode[-1] = n_continuum - n_per_mode[:-1].sum()

E_continuum = []
for (w, mu, sigma), n in zip(continuum_modes, n_per_mode):
    y = rng.normal(mu, sigma, size=n)
    E_continuum.append(10.0 ** y)
E_continuum = np.concatenate(E_continuum)

# Line cluster: y = log10(E) positions, closely spaced (0.1 dex apart),
# sitting 1-2 sigma below mode C's mean (i.e. in mode C's sparse tail, not
# its core) - true physical widths are narrow (sigma_y = 0.02).
line_positions_y_true = np.array([-2.30, -2.20, -2.10, -2.00], dtype=np.float64)
line_energies_mev = 10.0 ** line_positions_y_true
true_line_sigma_y = args.line_sigma_y

n_per_line = n_lines_total // N_LINES
counts_per_line = [n_per_line] * (N_LINES - 1) + [n_lines_total - n_per_line * (N_LINES - 1)]

E_lines = []
for y0, n in zip(line_positions_y_true, counts_per_line):
    y = rng.normal(y0, true_line_sigma_y, size=n)
    E_lines.append(10.0 ** y)
E_lines = np.concatenate(E_lines)

E_all = np.concatenate([E_continuum, E_lines]).astype(np.float64)
rng.shuffle(E_all)  # order doesn't matter downstream, but keep it honest

print(f"N_TOTAL={E_all.size}  n_continuum={n_continuum}  n_lines_total={n_lines_total}  "
      f"per-line counts={counts_per_line}")
print(f"line energies (MeV): {line_energies_mev}")
print(f"E range: [{E_all.min():.6f}, {E_all.max():.6f}] MeV")

# ----------------------------------------------------------------------
# 2. Real preprocessing entry points: energy bins, matched_lines (hand-built
#    since we already know ground truth), gate_targets, log10 energy_y.
# ----------------------------------------------------------------------
energy_bins = magi.build_energy_bins(E_all, mode="log_fixed_count", n_bins=256, min_counts=20)

matched_lines = []
for i, (e_mev, cnt) in enumerate(zip(line_energies_mev, counts_per_line)):
    matched_lines.append({
        "label": f"synthetic_line_{i}",
        "origin": "synthetic",
        "candidate_energy_mev": float(e_mev),
        "count": int(cnt),
    })

if args.gate_bandwidth_mode == "resolution":
    # Per-line FWHM in MeV for the injected dex-width lines: a constant width in
    # log10(E) is a different width in eV at every line energy.
    fwhm_per_line = 2.3548200450309493 * true_line_sigma_y * np.log(10.0) * line_energies_mev
    gate_targets = magi.build_gate_targets(
        E_all, energy_bins, matched_lines,
        bandwidth_mode="resolution", bandwidth_fwhm_mev=fwhm_per_line)
    print(f"gate-target bandwidth: resolution mode, per-line FWHM (eV) = "
          f"{np.round(fwhm_per_line * 1e6, 1)}")
else:
    gate_targets = magi.build_gate_targets(E_all, energy_bins, matched_lines)
    print("gate-target bandwidth: legacy bin-width mode")
true_line_frac = np.array([c / E_all.size for c in counts_per_line])
print("gate-target line mass vs true line fraction: "
      f"{np.round(gate_targets[:, 1:].mean(axis=0) / true_line_frac, 3)}")
print("gate_targets shape:", gate_targets.shape, " mean per column:", gate_targets.mean(axis=0))

energy_y = np.log10(E_all).astype(np.float32)
line_positions_y = np.log10(line_energies_mev).astype(np.float32)

# ----------------------------------------------------------------------
# 3. Dummy geometry columns (uniform noise - not under test here) + single
#    particle type conditioning.
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 4. Train the redesigned model (log10 energy transform is implicit - we
#    already fed log10(E) as energy_y).
# ----------------------------------------------------------------------
print(f"\nTraining with continuum_mode={args.continuum_mode}, "
      f"energy_flow_condition={args.energy_flow_condition}, prior={args.prior}, "
      f"w_continuum_repulsion={args.w_continuum_repulsion}, "
      f"continuum_repulsion_margin={args.continuum_repulsion_margin}, "
      f"beta={args.beta}, epochs={args.epochs}")

model_kwargs = dict(
    n_types=n_types,
    line_positions_y=line_positions_y,
    latent_dim=8,
    hidden=(128, 128, 64),
    beta=args.beta,
    w_continuum_repulsion=args.w_continuum_repulsion,
    continuum_repulsion_margin=args.continuum_repulsion_margin,
    prior=args.prior,
    prior_n_layers=args.prior_n_layers,
    prior_hidden=tuple(args.prior_hidden),
    w_flow_line_repulsion=args.w_flow_line_repulsion,
)
if args.w_gate_aux is not None:
    model_kwargs["w_gate_aux"] = args.w_gate_aux
if args.gate_focal_gamma > 0.0:
    model_kwargs["gate_focal_gamma"] = args.gate_focal_gamma
if args.gate_class_balance_power > 0.0:
    mass = gate_targets.mean(axis=0)  # [continuum, line_1..line_L] empirical masses
    gcw = (mass[0] / np.maximum(mass, 1e-8)) ** args.gate_class_balance_power
    gcw = np.minimum(gcw, args.gate_class_weight_cap)  # cap so rare lines don't blow up
    gcw[0] = 1.0
    model_kwargs["gate_class_weights"] = gcw.astype(np.float32)
    print(f"  gate_class_weights (power={args.gate_class_balance_power}, cap={args.gate_class_weight_cap}): {np.round(gcw, 2)}")
if args.pin_line_width:
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
          f"warp={args.continuum_warp} prior={args.prior} n_layers={args.prior_n_layers} hidden={tuple(args.prior_hidden)}")

model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(**model_kwargs)
magi.compile_model(model, learning_rate=2e-4)

history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)

print("\nFinal metrics:",
      {k: round(v[-1], 4) for k, v in history.history.items() if k.startswith("val_")})

# ----------------------------------------------------------------------
# 5. Generate + diagnostics.
# ----------------------------------------------------------------------
n_gen = n
gen_cond = np.ones((n_gen, n_types), dtype=np.float32)
gen_out = model.generate(tf.constant(gen_cond), n_gen)

energy_y_gen = gen_out["energy_y"].numpy()
comp_idx_gen = gen_out["energy_component_idx"].numpy()
E_gen = 10.0 ** energy_y_gen

print("\ngenerated component_idx value counts:", np.bincount(comp_idx_gen, minlength=N_LINES + 1))
true_weights = np.concatenate([[1.0 - LINE_FRAC_TOTAL], np.full(N_LINES, LINE_FRAC_TOTAL / N_LINES)])
gen_weights = np.bincount(comp_idx_gen, minlength=N_LINES + 1) / n_gen
print("true mixture weights:     ", np.round(true_weights, 4))
print("generated mixture weights:", np.round(gen_weights, 4))

line_logsigma = np.round(model._line_logsigma_clipped().numpy(), 4)
print(f"\nlearned line_logsigma (per line) = {line_logsigma} (true injected sigma_y = {true_line_sigma_y})")

# Diagnostic: is the flow continuum spiking at the line centers (stealing line
# events from the gate) or leaving the peaks to the line component? Report the
# flow's log-density at each line center vs at nearby off-line points.
if args.continuum_mode == "flow":
    diag_cond = model._decode_params(
        tf.constant(np.random.randn(3000, 8).astype(np.float32)),
        tf.ones((3000, 1), np.float32),
        training=False,
    )["flow_cond"]
    print("flow log-density diagnostic (mean over conditioners):")
    for l, ly in enumerate(line_positions_y):
        at = float(tf.reduce_mean(model.continuum_flow.log_prob(
            tf.fill((3000,), float(ly)), diag_cond)))
        off = float(tf.reduce_mean(model.continuum_flow.log_prob(
            tf.fill((3000,), float(ly) - 0.15), diag_cond)))
        print(f"  line {l} (y={ly:+.3f}): flow logp at line={at:+.3f}  "
              f"at line-0.15={off:+.3f}  (spike if at>>off)")

# Window scaled to the injected line width (per line: a constant dex width is a
# different eV width at each energy), continuum-subtracted and size-normalized.
fwhm_ev_per_line = (2.3548200450309493 * true_line_sigma_y * np.log(10.0)
                    * line_energies_mev * 1e6)
recovery = magi.compute_line_integral_recovery(
    E_all, E_gen, matched_lines, energy_bins, energy_component_idx_gen=comp_idx_gen,
    resolution_ev=fwhm_ev_per_line,
)
print("\nPer-line recovery (+/-5 sigma of the injected width, continuum-subtracted):")
for r in recovery:
    rec = "  n/a" if r["recovery_ratio"] is None else f"{r['recovery_ratio']:.3f}"
    flag = "  (low significance)" if r.get("low_significance") else ""
    print(f"  {r['label']:18s} n_real={r['n_real']:6d} (line {r['n_real_line']:7.1f}) "
          f"n_gen={r['n_gen']:6d} (line {r['n_gen_line']:7.1f}) "
          f"recovery_ratio={rec} component_fraction={r['component_fraction']:.5f}{flag}")

# ----------------------------------------------------------------------
# 6. Spurious-hump check: windowed real-vs-generated counts across several
#    E-space windows spanning the sparse line-cluster region and its
#    immediate surroundings (not just on the lines themselves).
# ----------------------------------------------------------------------
print("\nSpurious-hump check (windowed real vs generated counts, log-E windows):")
windows_log10 = [
    (-2.6, -2.4, "just below cluster (sparse gap)"),
    (-2.4, -2.32, "just below first line"),
    (-2.32, -2.28, "line 0 window"),
    (-2.28, -2.22, "between line 0/1"),
    (-2.22, -2.18, "line 1 window"),
    (-2.18, -2.12, "between line 1/2"),
    (-2.12, -2.08, "line 2 window"),
    (-2.08, -2.02, "between line 2/3"),
    (-2.02, -1.98, "line 3 window"),
    (-1.98, -1.85, "just above cluster"),
    (-1.85, -1.55, "mode C core (legitimate continuum)"),
]
real_log10 = np.log10(E_all)
gen_log10 = energy_y_gen
for lo, hi, desc in windows_log10:
    n_real = int(np.sum((real_log10 >= lo) & (real_log10 < hi)))
    n_gen_w = int(np.sum((gen_log10 >= lo) & (gen_log10 < hi)))
    ratio = (n_gen_w / n_real) if n_real > 0 else float("inf") if n_gen_w > 0 else 1.0
    print(f"  y in [{lo:+.2f},{hi:+.2f}) {desc:35s} n_real={n_real:6d} n_gen={n_gen_w:6d} ratio={ratio:.3f}")

print("\nDONE")
