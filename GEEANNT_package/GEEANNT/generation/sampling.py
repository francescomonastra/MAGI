"""
Sampling utilities for GEEANNT generation.
Supports:
  - v0.6 legacy models with categorical u_v
  - v0.7 continuous-geometry models with u_r_q/u_v_q and cos/sin phi
  - v0.7.2 continuous-phi models with u_r_q/u_v_q/phi_r_q/phi_v_q
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
    Sample particle types, build conditioning vectors,
    and generate raw model outputs.

    Supports:
      - v0.6 legacy models with categorical u_v
      - v0.7 continuous-geometry models with u_r_q/u_v_q

    Returns
    -------
    dict
        Common keys:
            gen_type_idx
            gen_cond
            energy_idx_gen
            y_cont_gen_s
            params

        v0.6-only keys, if produced by model.generate():
            uv_idx_gen
            uv_value_gen
    """
    gen_type_idx = sample_types(n_samples, type_probs, rng=rng)
    gen_cond = one_hot_from_idx(gen_type_idx, n_types)

    gen_out = model.generate(gen_cond, n_samples)

    out = {
        "gen_type_idx": gen_type_idx,
        "gen_cond": gen_cond,
        "energy_idx_gen": gen_out["energy_idx"].numpy().astype(np.int32),
        "y_cont_gen_s": gen_out["y_cont"].numpy().astype(np.float32),
        "params": gen_out.get("params", None),
    }

    # Legacy v0.6 categorical u_v output
    if "uv_idx" in gen_out:
        out["uv_idx_gen"] = gen_out["uv_idx"].numpy().astype(np.int32)

    if "uv_value" in gen_out:
        out["uv_value_gen"] = (
            gen_out["uv_value"]
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )

    if idx_to_type is not None:
        idx_to_type = {
            int(k): v
            for k, v in idx_to_type.items()
        } if isinstance(idx_to_type, dict) else idx_to_type

        out["idx_to_type"] = idx_to_type

        out["ParticleName"] = np.array(
            [idx_to_type[int(i)] for i in gen_type_idx],
            dtype=object,
        )

        unique, counts = np.unique(gen_type_idx, return_counts=True)

        out["generated_type_counts"] = {
            idx_to_type[int(u)]: int(c)
            for u, c in zip(unique, counts)
        }

    return out