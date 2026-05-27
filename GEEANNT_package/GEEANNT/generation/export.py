"""
Export utilities for generated GEEANNT particle events.

These functions allow:
1. conversion of generated physics dictionaries into detector-like tables
2. chunked generation directly to disk for very large event samples
3. export to either text or compact binary Geant4-ready format
"""

import os
import numpy as np
import pandas as pd
import struct

from .sampling import generate_latent_outputs
from .reconstruction import (
    reconstruct_generated_features,
    reconstruct_generated_physics,
)


GEEANNT_BINARY_MAGIC = b"GNTBIN1\0"
GEEANNT_BINARY_VERSION = 1

PARTICLE_TO_PDG = {
    "gamma": 22,
    "e-": 11,
    "e+": -11,
    "proton": 2212,
    "neutron": 2112,
    "alpha": 1000020040,
    "mu-": 13,
    "mu+": -13,
}


def _ensure_parent_dir(filepath):
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _normalize_output_format(output_format):
    output_format = str(output_format).lower()

    if output_format == "txt":
        return "text"

    if output_format == "bin":
        return "binary"

    if output_format not in ["text", "binary"]:
        raise ValueError(
            "output_format must be one of: 'text', 'txt', 'binary', 'bin'"
        )

    return output_format


def _normalize_idx_to_type(idx_to_type):
    if isinstance(idx_to_type, dict):
        return {int(k): v for k, v in idx_to_type.items()}
    return idx_to_type


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

    idx_to_type = _normalize_idx_to_type(idx_to_type)

    particle_names = np.array(
        [idx_to_type[int(i)] for i in generated_data["gen_type_idx"]],
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


def _write_binary_header(filepath, n_events):
    with open(filepath, "wb") as f:
        f.write(
            struct.pack(
                "<8siQ",
                GEEANNT_BINARY_MAGIC,
                GEEANNT_BINARY_VERSION,
                int(n_events),
            )
        )


def _append_detector_binary_chunk(df, filepath):
    """
    Append one detector dataframe chunk to an existing GEEANNT binary file.
    """
    required_cols = ["ParticleName", "Energy", "X", "Y", "Z", "Vx", "Vy", "Vz"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for binary export: {col}")

    try:
        pdg = np.array(
            [PARTICLE_TO_PDG[str(p)] for p in df["ParticleName"]],
            dtype="<i4",
        )
    except KeyError as exc:
        raise ValueError(
            f"Unsupported particle name for binary export: {exc}. "
            f"Supported names are: {sorted(PARTICLE_TO_PDG.keys())}"
        )

    record_dtype = np.dtype([
        ("pdg", "<i4"),
        ("energy", "<f4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("vx", "<f4"),
        ("vy", "<f4"),
        ("vz", "<f4"),
    ])

    records = np.empty(len(df), dtype=record_dtype)
    records["pdg"] = pdg
    records["energy"] = df["Energy"].to_numpy(dtype=np.float32)
    records["x"] = df["X"].to_numpy(dtype=np.float32)
    records["y"] = df["Y"].to_numpy(dtype=np.float32)
    records["z"] = df["Z"].to_numpy(dtype=np.float32)
    records["vx"] = df["Vx"].to_numpy(dtype=np.float32)
    records["vy"] = df["Vy"].to_numpy(dtype=np.float32)
    records["vz"] = df["Vz"].to_numpy(dtype=np.float32)

    with open(filepath, "ab") as f:
        records.tofile(f)


def save_detector_binary(data, filepath):
    """
    Save generated detector events in compact binary format.

    Header:
        magic      8 bytes   b"GNTBIN1\\0"
        version    int32
        n_events   uint64

    Records:
        pdg        int32
        energy     float32   [MeV]
        x          float32   [mm]
        y          float32   [mm]
        z          float32   [mm]
        vx         float32
        vy         float32
        vz         float32
    """
    _ensure_parent_dir(filepath)

    if isinstance(data, dict):
        df = generated_physics_to_detector_dataframe(data)
    else:
        df = data.copy()

    _write_binary_header(filepath, len(df))
    _append_detector_binary_chunk(df, filepath)

    print(f"Saved binary detector file: {filepath}")


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
    output_format="text",
    verbose=1,
):
    """
    Generate a large synthetic particle dataset directly to disk in chunks.

    Supported formats:
      - text / txt:
          ParticleName Energy X Y Z Vx Vy Vz

      - binary / bin:
          compact GEEANNT binary format
    """
    output_format = _normalize_output_format(output_format)
    idx_to_type = _normalize_idx_to_type(idx_to_type)

    _ensure_parent_dir(filepath)

    rng = np.random.default_rng(seed)

    if output_format == "text":
        open(filepath, "w").close()

    elif output_format == "binary":
        _write_binary_header(filepath, n_events)

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

        if output_format == "text":
            df_chunk.to_csv(
                filepath,
                mode="a",
                sep="\t",
                header=False,
                index=False,
                float_format=float_format,
            )

        elif output_format == "binary":
            _append_detector_binary_chunk(df_chunk, filepath)

        n_done += n_chunk

    if verbose:
        print(
            f"[GEEANNT] Saved generated detector {output_format} file to {filepath}"
        )

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
    output_format="text",
    verbose=1,
):
    """
    Generate a Geant4-ready particle input file from a trained GEEANNT model.
    """
    from GEEANNT.training.checkpointing import load_task_adaptive_model_for_generation

    output_format = _normalize_output_format(output_format)
    idx_to_type = _normalize_idx_to_type(idx_to_type)

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
        output_format=output_format,
        include_event_id=False,
        verbose=verbose,
    )