"""
Export utilities for generated GEEANNT particle events.

These functions allow:
1. conversion of generated physics dictionaries into detector-like tables
2. chunked generation directly to disk for very large event samples
"""

import os
import numpy as np
import pandas as pd

from .sampling import generate_latent_outputs
from .reconstruction import (
    reconstruct_generated_features,
    reconstruct_generated_physics,
)


def generated_physics_to_detector_dataframe(
    generated_data,
    idx_to_type=None,
    include_event_id=False,
    event_id_offset=0,
):
    """
    Convert generated physical particle data into a detector-style dataframe.

    Expected detector schema:
        ParticleName Energy X Y Z Vx Vy Vz

    Optionally:
        EventId ParticleName Energy X Y Z Vx Vy Vz

    Parameters
    ----------
    generated_data : dict
        Dictionary produced by reconstruct_generated_physics().

        Required keys:
            E_gen
            x_gen
            y_gen
            z_gen
            vx_gen
            vy_gen
            vz_gen
            gen_type_idx

    idx_to_type : dict or list, optional
        Mapping from integer type index to particle name.

        Example:
            {0: "gamma", 1: "e-", 2: "proton"}

    include_event_id : bool
        If True, prepend sequential EventId column.

    event_id_offset : int
        Starting index for EventId numbering.

    Returns
    -------
    pd.DataFrame
        Detector-compatible dataframe.
    """
    required = [
        "E_gen",
        "x_gen",
        "y_gen",
        "z_gen",
        "vx_gen",
        "vy_gen",
        "vz_gen",
        "gen_type_idx",
    ]

    for key in required:
        if key not in generated_data:
            raise ValueError(f"Missing required generated key: {key}")

    if idx_to_type is None:
        raise ValueError("idx_to_type mapping is required to recover particle names.")

    particle_names = np.array(
        [idx_to_type[i] for i in generated_data["gen_type_idx"]],
        dtype=object,
    )

    df = pd.DataFrame({
        "ParticleName": particle_names,
        "Energy": generated_data["E_gen"],
        "X": generated_data["x_gen"],
        "Y": generated_data["y_gen"],
        "Z": generated_data["z_gen"],
        "Vx": generated_data["vx_gen"],
        "Vy": generated_data["vy_gen"],
        "Vz": generated_data["vz_gen"],
    })

    if include_event_id:
        df.insert(
            0,
            "EventId",
            np.arange(
                event_id_offset,
                event_id_offset + len(df),
                dtype=np.int64,
            ),
        )

    return df


def generate_detector_table_to_file(
    *,
    model,
    filepath,
    n_events,
    type_probs,
    n_types,
    idx_to_type,
    s_r_mean,
    s_r_std,
    energy_bins,
    u_v_bins,
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    energy_mode="uniform",
    chunk_size=100_000,
    seed=42,
    include_event_id=False,
    float_format="%.8e",
    verbose=1,
):
    """
    Generate a large synthetic particle dataset directly to disk in chunks.

    This avoids exhausting RAM when generating millions of particles.

    Pipeline:
        latent sampling
        -> feature reconstruction
        -> physical reconstruction
        -> detector table export

    Parameters
    ----------
    model : keras.Model
        Trained GEEANNT generative model.

    filepath : str
        Output text file path.

    n_events : int
        Total number of events to generate.

    type_probs : array-like
        Sampling probabilities for particle classes.

    n_types : int
        Number of particle classes.

    idx_to_type : dict or list
        Mapping integer class index -> particle name.

    s_r_mean : float
        Mean used during preprocessing normalization.

    s_r_std : float
        Std used during preprocessing normalization.

    energy_bins : np.ndarray
        Energy bin edges.

    u_v_bins : np.ndarray
        Direction cosine bin edges.

    center : tuple
        Emission sphere center.

    radius : float
        Emission sphere radius.

    energy_mode : str
        Energy reconstruction mode.

        Options:
            "uniform"
            "bin_center"

    chunk_size : int
        Number of events generated per chunk.

    seed : int
        RNG seed.

    include_event_id : bool
        Whether to include EventId column.

    float_format : str
        Formatting for saved floating-point numbers.

    verbose : int
        Print progress if >0.

    Returns
    -------
    str
        Output filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rng = np.random.default_rng(seed)

    # reset output file
    open(filepath, "w").close()

    n_done = 0

    while n_done < n_events:
        n_chunk = min(chunk_size, n_events - n_done)

        if verbose:
            print(
                f"[GEEANNT] Generating chunk "
                f"{n_done} -> {n_done + n_chunk} / {n_events}"
            )

        gen_raw = generate_latent_outputs(
            model=model,
            n_samples=n_chunk,
            type_probs=type_probs,
            n_types=n_types,
            idx_to_type=idx_to_type,
            rng=rng,
        )

        gen_feat = reconstruct_generated_features(
            gen_raw,
            s_r_mean=s_r_mean,
            s_r_std=s_r_std,
            energy_bins=energy_bins,
            u_v_bins=u_v_bins,
            energy_mode=energy_mode,
            rng=rng,
        )

        gen_phys = reconstruct_generated_physics(
            gen_feat,
            center=center,
            radius=radius,
        )

        df_chunk = generated_physics_to_detector_dataframe(
            gen_phys,
            idx_to_type=idx_to_type,
            include_event_id=include_event_id,
            event_id_offset=n_done,
        )

        df_chunk.to_csv(
            filepath,
            mode="a",
            sep="\t",
            header=False,
            index=False,
            float_format=float_format,
        )

        n_done += n_chunk

    if verbose:
        print(f"[GEEANNT] Saved generated detector table to {filepath}")

    return filepath

def generate_detector_input_file(
    *,
    save_dir,
    model_name,
    model_config,
    energy_bins,
    u_v_bins,
    n_types,
    type_weights,
    type_probs,
    idx_to_type,
    s_r_mean,
    s_r_std,
    output_file,
    n_events,
    radius,
    center=(0.0, 0.0, -507.66),
    seed=42,
    chunk_size=100_000,
    energy_mode="uniform",
    verbose=1,
):
    """
    High-level utility to generate a Geant4-ready particle input file.

    This function:
      1. loads a trained task-adaptive GEEANNT model
      2. generates particles in chunks
      3. reconstructs physical coordinates and directions
      4. writes a detector-table text file directly to disk

    Use this when starting from a saved trained model.
    """

    from GEEANNT.training.checkpointing import load_task_adaptive_model_for_generation

    model = load_task_adaptive_model_for_generation(
        save_dir=save_dir,
        model_name=model_name,
        model_config=model_config,
        energy_bins=energy_bins,
        u_v_bins=u_v_bins,
        n_types=n_types,
        type_weights=type_weights,
        radius=radius,
        compile_model_fn=None,
        verbose=verbose,
    )

    return generate_detector_table_to_file(
        model=model,
        filepath=output_file,
        n_events=n_events,
        type_probs=type_probs,
        n_types=n_types,
        idx_to_type=idx_to_type,
        s_r_mean=s_r_mean,
        s_r_std=s_r_std,
        energy_bins=energy_bins,
        u_v_bins=u_v_bins,
        center=center,
        radius=radius,
        energy_mode=energy_mode,
        chunk_size=chunk_size,
        seed=seed,
        include_event_id=False,
        verbose=verbose,
    )