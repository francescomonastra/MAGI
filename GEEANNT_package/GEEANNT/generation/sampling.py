"""
Sampling utilities for GEEANNT generation.
"""

import numpy as np
import tensorflow as tf


def sample_types(n_samples, type_probs, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.choice(len(type_probs), size=n_samples, p=type_probs)
    return idx.astype(np.int32)


def one_hot_from_idx(idx, n_types):
    return tf.one_hot(idx, depth=n_types, dtype=tf.float32)


def energy_from_idx(idx, energy_bins, mode="uniform", rng=None):
    """
    Reconstruct physical energy from categorical bin indices.
    """
    idx = np.asarray(idx, dtype=np.int32)
    left = energy_bins[idx]
    right = energy_bins[idx + 1]

    if mode == "uniform":
        rng = np.random.default_rng() if rng is None else rng
        u = rng.uniform(0.0, 1.0, size=len(idx))
        return (left + u * (right - left)).astype(np.float32)

    return (0.5 * (left + right)).astype(np.float32)


def generate_latent_outputs(
    model,
    n_samples,
    type_probs,
    n_types,
    idx_to_type=None,
    rng=None,
):
    """
    Sample particle types, build conditioning, and generate raw model outputs.
    """
    gen_type_idx = sample_types(n_samples, type_probs, rng=rng)
    gen_cond = one_hot_from_idx(gen_type_idx, n_types)

    gen_out = model.generate(gen_cond, n_samples)

    out = {
        "gen_type_idx": gen_type_idx,
        "gen_cond": gen_cond,
        "energy_idx_gen": gen_out["energy_idx"].numpy().astype(np.int32),
        "uv_idx_gen": gen_out["uv_idx"].numpy().astype(np.int32),
        "uv_value_gen": gen_out["uv_value"].numpy().astype(np.float32).reshape(-1),
        "y_cont_gen_s": gen_out["y_cont"].numpy().astype(np.float32),
        "params": gen_out.get("params", None),
    }

    if idx_to_type is not None:
        out["idx_to_type"] = idx_to_type
        unique, counts = np.unique(gen_type_idx, return_counts=True)
        out["generated_type_counts"] = {
            idx_to_type[u]: int(c) for u, c in zip(unique, counts)
        }

    return out