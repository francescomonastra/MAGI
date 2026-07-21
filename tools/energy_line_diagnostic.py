#!/usr/bin/env python3
"""
MAGI energy-line diagnostic
===========================

Purpose
-------
Emission lines in the generated CryoSphere spectrum come out *lower* than in the
training set. This script isolates WHY, by attributing the peak-height loss to
the three mechanisms in the current v0.7.2 energy path, which are otherwise
tangled together:

  (A) label smoothing   -> caps the training TARGET on a pure line bin at (1-s)
                           before the model makes any error at all
      (magi.core.losses.smoothed_categorical_ce)
  (B) sampling temperature -> T>1 flattens the categorical at generation,
                           lowering peaks relative to the continuum
      (model._sample_energy_from_logits, energy_sampling_temperature)
  (C) in-bin uniform sampling -> spreads a near-delta line uniformly across the
                           full bin width, broadening it and dropping peak height
      (magi.generation.sampling.energy_from_idx, mode="uniform")

Design
------
The CORE analysis (spectrum + line detection + A/B/C attribution) needs only the
training .dat and the energy-bin edges, so it always runs. If you also pass a
saved model, the script overlays the real vs generated spectrum and measures the
model's *predicted* per-bin probability, separating "the head under-predicts the
line" from "sampling/reconstruction erodes it".

Nothing here is destructive: it only reads. Outputs a PNG and a printed verdict.

Usage
-----
Core (no model needed):
    python energy_line_diagnostic.py \
        --dat /path/TrainingData/alloutputDSCryoSphere.dat \
        --energy-bins /path/model_x_metadata.json \
        --smoothing 0.1 --local-neighbors --temperature 1.0 \
        --resolution-kev 3.0 --out energy_line_report.png

With the trained model (adds real-vs-generated overlay):
    python energy_line_diagnostic.py ... \
        --model-dir /path/trained_models --model-name magi_v0_7_2 \
        --n-gen 2000000

Notes
-----
- Energy is column index 2 in both the 9-col and 10-col (PrimBool) schemas.
- --energy-bins accepts a MAGI *_metadata.json, a .json list, or a .npy of edges.
- --smoothing / --local-neighbors / --temperature should match what the model was
  TRAINED / GENERATED with; the report shows the effect of the configured values
  and sweeps alternatives so you can see the fix.
"""

import argparse
import json
import os
import sys
import numpy as np

from magi.data.preprocessing import bin_counts, detect_line_bins


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_real_energies(dat_path):
    """Return the Energy column (index 2) from a MAGI detector .dat file.

    Tries magi.data.io.load_detector_table first (handles 9/10/13-col schemas);
    falls back to a whitespace parse of column 2.
    """
    try:
        from magi.data.io import load_detector_table
        df = load_detector_table(dat_path)
        return df["Energy"].to_numpy(dtype=np.float64)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[info] magi.data.io not usable ({exc!r}); parsing column 2 directly.")
        energies = []
        with open(dat_path) as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    energies.append(float(parts[2]))
                except ValueError:
                    # header line or non-numeric; skip
                    continue
        return np.asarray(energies, dtype=np.float64)


def load_energy_bins(path):
    """Load energy-bin edges from a metadata json, a plain json list, or a .npy."""
    if path.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.float64)

    with open(path) as fh:
        obj = json.load(fh)

    if isinstance(obj, list):
        return np.asarray(obj, dtype=np.float64)

    # Search a MAGI metadata dict for anything that looks like energy bin edges.
    candidates = []

    def walk(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if "energy_bin" in str(k).lower() and isinstance(v, (list, tuple)):
                    candidates.append(np.asarray(v, dtype=np.float64))
                walk(v)
        elif isinstance(d, list):
            for v in d:
                walk(v)

    walk(obj)
    edges = [c for c in candidates if c.ndim == 1 and c.size >= 3]
    if not edges:
        raise ValueError(
            f"Could not find energy-bin edges in {path}. "
            "Pass a .npy or .json list of edges via --energy-bins."
        )
    # pick the longest edge array found
    return max(edges, key=lambda a: a.size)


# ----------------------------------------------------------------------------
# Line detection: bin_counts / detect_line_bins now live in
# magi.data.preprocessing (imported above) so they're reusable outside this
# script; behavior is unchanged.
# ----------------------------------------------------------------------------
# Mechanism (A): label-smoothing target cap
# ----------------------------------------------------------------------------
def smoothing_target_cap(smoothing, local_neighbors, n_bins):
    """Max achievable softmax target on a PURE line bin under the smoothing config.

    A perfect model cannot exceed this on the line bin, so any deficit here is
    baked into the loss target, not the model.
    """
    if smoothing <= 0:
        return 1.0
    if local_neighbors:
        # interior bin: center_w = 1 - 0.5s - 0.5s = 1 - s
        return 1.0 - smoothing
    # global: (1-s) on the bin + s/n_bins returned to it
    return (1.0 - smoothing) + smoothing / float(n_bins)


# ----------------------------------------------------------------------------
# Mechanism (B): temperature flattening
# ----------------------------------------------------------------------------
def temperature_reweight(probs, T):
    """softmax(log p / T) proportional to p**(1/T); returns renormalised probs."""
    p = np.clip(probs, 1e-300, None)
    pt = p ** (1.0 / T)
    return pt / pt.sum()


# ----------------------------------------------------------------------------
# Mechanism (C): in-bin uniform smearing
# ----------------------------------------------------------------------------
def inbin_peak_retention(edges, line_idx, resolution):
    """Fraction of a line's counts that land within +/- `resolution` of the line
    centre when sampled UNIFORMLY across its bin, vs a delta.

    Retention = min(1, 2*resolution / bin_width). This is the peak-height ratio
    (uniform / delta) at detector resolution -- the height lost purely to
    in-bin uniform sampling.
    """
    out = {}
    for b in line_idx:
        w = float(edges[b + 1] - edges[b])
        retention = min(1.0, (2.0 * resolution) / w) if w > 0 else 1.0
        out[int(b)] = dict(bin_width=w, retention=retention,
                           broadening_factor=(w / (2.0 * resolution)) if w > 0 else 1.0)
    return out


# ----------------------------------------------------------------------------
# Optional: model-based predicted-bin distribution + generated spectrum
# ----------------------------------------------------------------------------
def model_predicted_and_generated(args, edges, real_energies):
    """Load the trained model, return (p_pred, E_gen) or (None, None) on failure.

    p_pred : model's marginal predicted per-bin probability (avg softmax over
             z~N(0,I) and type mix), at T=1.
    E_gen  : reconstructed generated energies via energy_from_idx(uniform).
    """
    try:
        import tensorflow as tf  # noqa: F401
        from magi.training.checkpointing import load_task_adaptive_model_for_generation
        from magi.generation.sampling import (
            sample_types, one_hot_from_idx, energy_from_idx,
        )
        from magi.data.io import load_detector_table
    except Exception as exc:
        print(f"[info] model path unavailable ({exc!r}); running core analysis only.")
        return None, None

    # Particle-type mix from the training table.
    df = load_detector_table(args.dat)
    names, counts = np.unique(df["ParticleName"].to_numpy(), return_counts=True)
    type_probs = counts / counts.sum()
    n_types = len(names)

    model = load_task_adaptive_model_for_generation(
        save_dir=args.model_dir,
        model_name=args.model_name,
        n_types=n_types,
        energy_bins=edges,
    )

    n_gen = int(args.n_gen)
    rng = np.random.default_rng(0)

    # (1) Predicted per-bin probability, averaged over prior samples + type mix.
    idx = sample_types(n_gen, type_probs, rng=rng)
    cond = one_hot_from_idx(idx, n_types)
    import tensorflow as tf
    z = tf.random.normal((n_gen, model.latent_dim))
    params = model.decode(z, cond)
    logits = params["energy_logits"].numpy()
    p_pred = np.exp(logits - logits.max(axis=1, keepdims=True))
    p_pred = (p_pred / p_pred.sum(axis=1, keepdims=True)).mean(axis=0)

    # (2) Generated spectrum through the real generation path.
    gen_out = model.generate(cond, n_gen)
    e_idx = gen_out["energy_idx"].numpy().astype(np.int32)
    E_gen = energy_from_idx(e_idx, energy_bins=edges, mode=args.energy_mode, rng=rng)
    return p_pred, E_gen


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def make_report(args, edges, real_energies, p_pred, E_gen):
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_bins = centres.size
    counts = bin_counts(real_energies, edges)
    p_real = counts / counts.sum()

    line_idx, local_med = detect_line_bins(
        counts, prominence_factor=args.prominence, window=args.window
    )

    cap = smoothing_target_cap(args.smoothing, args.local_neighbors, n_bins)
    inbin = inbin_peak_retention(edges, line_idx, args.resolution_kev)

    # Temperature effect on the (empirical) line-bin probability.
    p_realT = temperature_reweight(p_real, args.temperature) if args.temperature != 1.0 else p_real

    print("\n" + "=" * 72)
    print("MAGI ENERGY-LINE DIAGNOSTIC")
    print("=" * 72)
    print(f"real events            : {real_energies.size:,}")
    print(f"energy bins            : {n_bins}")
    print(f"detected line bins     : {len(line_idx)}  (prominence>{args.prominence}x local continuum)")
    print(f"smoothing              : {args.smoothing}  local_neighbors={args.local_neighbors}")
    print(f"  -> TARGET cap on a pure line bin (A): {cap:.4f}  "
          f"(best-case peak = {cap*100:.1f}% of a delta target)")
    print(f"temperature            : {args.temperature}")

    print("\nPer-line attribution")
    print("-" * 72)
    header = f"{'E_centre':>10} {'real_frac':>10} {'width':>8} {'A_cap':>7} {'C_keep':>7}"
    if args.temperature != 1.0:
        header += f" {'B_tempR':>8}"
    if p_pred is not None:
        header += f" {'pred/real':>9}"
    print(header)

    verdicts = []
    for b in line_idx:
        c_keep = inbin[int(b)]["retention"]
        row = (f"{centres[b]:10.2f} {p_real[b]:10.4e} "
               f"{edges[b+1]-edges[b]:8.2f} {cap:7.3f} {c_keep:7.3f}")
        b_tempR = None
        if args.temperature != 1.0:
            b_tempR = p_realT[b] / max(p_real[b], 1e-300)
            row += f" {b_tempR:8.3f}"
        pred_ratio = None
        if p_pred is not None:
            pred_ratio = p_pred[b] / max(p_real[b], 1e-300)
            row += f" {pred_ratio:9.3f}"
        print(row)
        verdicts.append((int(b), cap, c_keep, b_tempR, pred_ratio))

    # Combined worst-offender diagnosis.
    print("\nDominant mechanism per line (smaller factor = bigger culprit)")
    print("-" * 72)
    for b, cap_, c_keep, b_tempR, pred_ratio in verdicts:
        factors = {"A_label_smoothing": cap_, "C_inbin_uniform": c_keep}
        if b_tempR is not None:
            factors["B_temperature"] = b_tempR
        if pred_ratio is not None:
            factors["model_head(pred/real)"] = pred_ratio
        worst = min(factors, key=factors.get)
        chain = 1.0
        for f in (cap_, c_keep, (b_tempR if b_tempR is not None else 1.0)):
            chain *= f
        print(f"  E~{centres[b]:8.2f} keV : dominant={worst:22s} "
              f"factors={ {k: round(v,3) for k,v in factors.items()} }  "
              f"predicted_peak~{chain*100:5.1f}% of delta (A*B*C)")

    _plot(args, edges, centres, counts, p_real, line_idx, local_med, E_gen)


def _plot(args, edges, centres, counts, p_real, line_idx, local_med, E_gen):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[info] matplotlib unavailable ({exc!r}); skipping plot.")
        return

    have_gen = E_gen is not None
    fig, axes = plt.subplots(2 if have_gen else 1, 1,
                             figsize=(10, 8 if have_gen else 4.5), squeeze=False)

    ax = axes[0][0]
    ax.step(centres, counts, where="mid", label="real (training)", color="#2b6cb0")
    ax.plot(centres, local_med, "--", color="#a0aec0", lw=1, label="local continuum")
    ax.scatter(centres[line_idx], counts[line_idx], color="#e53e3e", zorder=5,
               label="detected lines")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("counts / bin")
    ax.set_title("Real spectrum with detected emission lines")
    ax.legend(fontsize=8)

    if have_gen:
        gen_counts, _ = np.histogram(E_gen, bins=edges)
        gen_counts = gen_counts.astype(np.float64)
        # scale generated to same total as real for shape comparison
        scale = counts.sum() / max(gen_counts.sum(), 1.0)
        ax2 = axes[1][0]
        ax2.step(centres, counts, where="mid", label="real", color="#2b6cb0")
        ax2.step(centres, gen_counts * scale, where="mid",
                 label="generated (scaled)", color="#dd6b20", alpha=0.8)
        ax2.scatter(centres[line_idx], counts[line_idx], color="#e53e3e", zorder=5)
        ax2.set_yscale("log")
        ax2.set_xlabel("Energy [keV]")
        ax2.set_ylabel("counts / bin")
        ax2.set_title("Real vs generated (line bins highlighted)")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n[ok] wrote {args.out}")


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="MAGI energy-line suppression diagnostic")
    p.add_argument("--dat", required=True, help="training .dat file")
    p.add_argument("--energy-bins", required=True,
                   help="metadata json, json list, or .npy of bin edges")
    p.add_argument("--smoothing", type=float, default=0.0,
                   help="label smoothing mass used in training")
    p.add_argument("--local-neighbors", action="store_true",
                   help="smoothing was local-neighbor mode")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="energy_sampling_temperature used at generation")
    p.add_argument("--resolution-kev", dest="resolution_kev", type=float, default=1.0,
                   help="detector energy resolution (half-width) for the in-bin check")
    p.add_argument("--prominence", type=float, default=3.0,
                   help="line = bin count > this x local continuum")
    p.add_argument("--window", type=int, default=5, help="local-continuum window (bins)")
    # optional model path
    p.add_argument("--model-dir", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--n-gen", type=int, default=1_000_000)
    p.add_argument("--energy-mode", default="uniform", choices=["uniform", "center"])
    p.add_argument("--out", default="energy_line_report.png")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    real_energies = load_real_energies(args.dat)
    edges = load_energy_bins(args.energy_bins)

    p_pred, E_gen = (None, None)
    if args.model_dir and args.model_name:
        p_pred, E_gen = model_predicted_and_generated(args, edges, real_energies)

    make_report(args, edges, real_energies, p_pred, E_gen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
