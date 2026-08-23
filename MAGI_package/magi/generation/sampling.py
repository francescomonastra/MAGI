"""
Sampling utilities for MAGI generation.
Supports:
  - v0.6 legacy models with categorical u_v
  - v0.7 continuous-geometry models with u_r_q/u_v_q and cos/sin phi
  - v0.7.2 continuous-phi models with u_r_q/u_v_q/phi_r_q/phi_v_q
  - v0.8 mixture-energy models, which additionally return the raw energy_y
    sample and the index of the mixture component each event was routed to
    (continuum or a specific line)
"""

import numpy as np
import tensorflow as tf


def sample_types(n_samples, type_probs, rng=None):
    """Draw `n_samples` particle-type indices from the categorical `type_probs`.

    Pass the training-set type fractions to reproduce the natural mix, or a
    different vector to deliberately re-weight the generated source.

    Parameters
    ----------
    n_samples : int
        How many events to draw.

    type_probs : array-like of float
        Categorical probabilities over particle types, summing to 1. Use
        `dataset_pack["type_probs"]` for the natural mix.

    rng : np.random.Generator or None
        Generator to draw from. Pass one seeded explicitly for a
        reproducible source file; None (default) uses fresh entropy.

    Returns
    -------
    np.ndarray
        int32 array of shape (n_samples,) holding type indices.
    """
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.choice(len(type_probs), size=n_samples, p=type_probs)
    return idx.astype(np.int32)


def one_hot_from_idx(idx, n_types):
    """One-hot the type indices into the `cond` tensor the models expect.

    Parameters
    ----------
    idx : array-like of int
        Particle-type indices, e.g. from sample_types.

    n_types : int
        One-hot depth. Must equal the `n_types` the model was built with.

    Returns
    -------
    tf.Tensor
        float32 tensor of shape (len(idx), n_types), ready to pass as `cond`.
    """
    return tf.one_hot(idx, depth=n_types, dtype=tf.float32)


def energy_from_idx(idx, energy_bins, mode="uniform", rng=None):
    """
    Reconstruct physical energy from categorical bin indices.

    Only used by the categorical-energy heads (v0.6 - v0.7.2). The v0.8
    mixture head emits a continuous energy directly and never goes through
    this function.

    Parameters
    ----------
    idx : array-like of int
        Bin indices in [0, len(energy_bins) - 2].

    energy_bins : array-like of float
        Bin edges in MeV, as returned by build_energy_bins. Must be the same
        edges the run was trained with - they are saved in the checkpoint
        metadata for exactly this reason.

    mode : str
        "uniform" (default) draws uniformly inside the bin, which avoids the
        comb-like artifact that bin centres would imprint on the spectrum.
        Any other value returns the bin centre.

    rng : np.random.Generator or None
        Generator used when `mode` is "uniform". None uses fresh entropy.

    Returns
    -------
    np.ndarray
        float32 energies in MeV, shape (len(idx),).
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

    The returned keys depend on which head produced them, so downstream code
    should test for a key rather than assume it. reconstruct_generated_features
    does exactly that.

    Parameters
    ----------
    model : keras.Model
        A built MAGI model exposing `.generate(cond, n)`. Typically returned
        by load_task_adaptive_model_for_generation.

    n_samples : int
        How many events to generate. For validation, set this to the real
        event count: the line-recovery metrics compare raw counts, and a
        smaller generated sample silently deflates every ratio.

    type_probs : array-like of float
        Categorical probabilities over particle types, summing to 1.

    n_types : int
        One-hot depth; must match the model's conditioning width.

    idx_to_type : dict[int, str] or None
        Index-to-name map. When given, the result also carries
        "ParticleName", "idx_to_type" and "generated_type_counts", which the
        export helpers need to write a Geant4 source file.

    rng : np.random.Generator or None
        Generator for the type draw. Note this does not seed the model's own
        sampling - use magi.initialize_environment for that.

    Returns
    -------
    dict
        Common keys:
            gen_type_idx
            gen_cond
            y_cont_gen_s
            params

        Categorical-energy models (v0.6/v0.7/v0.7.2), if produced by
        model.generate():
            energy_idx_gen

        v0.8 mixture-energy models, if produced by model.generate():
            energy_y_gen
            energy_component_idx_gen

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
        "y_cont_gen_s": gen_out["y_cont"].numpy().astype(np.float32),
        "params": gen_out.get("params", None),
    }

    if "energy_idx" in gen_out:
        out["energy_idx_gen"] = gen_out["energy_idx"].numpy().astype(np.int32)

    if "energy_y" in gen_out:
        out["energy_y_gen"] = gen_out["energy_y"].numpy().astype(np.float32).reshape(-1)
        out["energy_component_idx_gen"] = gen_out["energy_component_idx"].numpy().astype(np.int32)

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