"""
Export utilities for generated MAGI particle events.

These functions allow:
1. conversion of generated physics dictionaries into detector-like tables
2. chunked generation directly to disk for very large event samples
3. export to either text or compact binary Geant4-ready format
4. filtering of non-transport particles, e.g. neutrinos
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


MAGI_BINARY_MAGIC = b"GNTBIN1\0"
MAGI_BINARY_VERSION = 1

PARTICLE_TO_PDG = {
    "gamma": 22,
    "e-": 11,
    "e+": -11,
    "mu-": 13,
    "mu+": -13,
    "proton": 2212,
    "anti_proton": -2212,
    "neutron": 2112,
    "anti_neutron": -2112,
    "alpha": 1000020040,
}

NON_TRANSPORT_PARTICLES = {
    "nu_e",
    "anti_nu_e",
    "nu_mu",
    "anti_nu_mu",
    "nu_tau",
    "anti_nu_tau",
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


def _filter_non_transport_particles(df, excluded_particle_names=None):
    """
    Remove particles that should not be transported in Geant4 source files.

    By default, neutrinos are removed because they do not contribute to detector
    background in this use case and may not be supported by the current binary
    PDG export table.
    """
    if excluded_particle_names is None:
        excluded_particle_names = NON_TRANSPORT_PARTICLES

    excluded_particle_names = set(excluded_particle_names)

    if len(excluded_particle_names) == 0:
        return df

    return df[~df["ParticleName"].isin(excluded_particle_names)].copy()


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
    """
    Create a binary file and write its header.

    This opens the file in 'wb' mode and therefore resets any existing file.
    """
    with open(filepath, "wb") as f:
        f.write(
            struct.pack(
                "<8siQ",
                MAGI_BINARY_MAGIC,
                MAGI_BINARY_VERSION,
                int(n_events),
            )
        )


def _rewrite_binary_header(filepath, n_events):
    """
    Rewrite only the binary header without deleting the payload.

    Used after chunked writing when the final number of written particles is
    known only after filtering non-transport particles.
    """
    with open(filepath, "r+b") as f:
        f.seek(0)
        f.write(
            struct.pack(
                "<8siQ",
                MAGI_BINARY_MAGIC,
                MAGI_BINARY_VERSION,
                int(n_events),
            )
        )


def _append_detector_binary_chunk(
    df,
    filepath,
    excluded_particle_names=None,
):
    """
    Append one detector dataframe chunk to an existing MAGI binary file.

    Returns
    -------
    int
        Number of records actually written.
    """
    required_cols = ["ParticleName", "Energy", "X", "Y", "Z", "Vx", "Vy", "Vz"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for binary export: {col}")

    df = _filter_non_transport_particles(
        df,
        excluded_particle_names=excluded_particle_names,
    )

    if len(df) == 0:
        return 0

    unsupported = sorted(set(df["ParticleName"].astype(str)) - set(PARTICLE_TO_PDG))
    if unsupported:
        raise ValueError(
            f"Unsupported particle names for binary export: {unsupported}. "
            f"Supported names are: {sorted(PARTICLE_TO_PDG.keys())}. "
            f"Filtered non-transport particles are: {sorted(NON_TRANSPORT_PARTICLES)}."
        )

    pdg = np.array(
        [PARTICLE_TO_PDG[str(p)] for p in df["ParticleName"]],
        dtype="<i4",
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

    return len(records)


def save_detector_binary(
    data,
    filepath,
    excluded_particle_names=None,
):
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

    _write_binary_header(filepath, 0)
    n_written = _append_detector_binary_chunk(
        df,
        filepath,
        excluded_particle_names=excluded_particle_names,
    )
    _rewrite_binary_header(filepath, n_written)

    print(f"Saved binary detector file: {filepath}")
    print(f"Written binary particles: {n_written}")

    return filepath


def generate_detector_table_to_file(
    *,
    model,
    filepath,
    n_events,
    type_probs,
    n_types,
    idx_to_type,
    energy_bins,
    geometry_metadata=None,
    s_r_mean=None,
    s_r_std=None,
    u_v_bins=None,
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    energy_mode="uniform",
    chunk_size=100_000,
    seed=42,
    include_event_id=False,
    float_format="%.8e",
    output_format="text",
    excluded_particle_names=None,
    generate_until_n_written=True,
    max_empty_chunks=100,
    verbose=1,
):
    """
    Generate a large synthetic particle dataset directly to disk in chunks.

    Supported formats:
      - text / txt:
          ParticleName Energy X Y Z Vx Vy Vz

      - binary / bin:
          compact MAGI binary format

    Parameters
    ----------
    excluded_particle_names : set/list or None
        Particle names removed before writing. By default this removes neutrinos.

    generate_until_n_written : bool
        If True, keep generating until n_events transported particles are written.
        This is recommended when filtering neutrinos, so Geant4 receives exactly
        n_events usable source particles.

        If False, generate exactly n_events raw model samples, then write fewer
        particles if some are filtered out.

    max_empty_chunks : int
        Safety stop if filtering removes entire chunks repeatedly.
    """
    output_format = _normalize_output_format(output_format)
    idx_to_type = _normalize_idx_to_type(idx_to_type)

    if excluded_particle_names is None:
        excluded_particle_names = NON_TRANSPORT_PARTICLES
    else:
        excluded_particle_names = set(excluded_particle_names)

    _ensure_parent_dir(filepath)

    rng = np.random.default_rng(seed)

    if output_format == "text":
        open(filepath, "w").close()

    elif output_format == "binary":
        _write_binary_header(filepath, 0)

    n_attempted = 0
    n_written = 0
    n_empty_chunks = 0

    while True:
        if generate_until_n_written:
            if n_written >= n_events:
                break
            n_chunk = min(chunk_size, n_events - n_written)
        else:
            if n_attempted >= n_events:
                break
            n_chunk = min(chunk_size, n_events - n_attempted)

        if verbose:
            print(
                f"[MAGI] Generating chunk "
                f"attempted={n_attempted} written={n_written} / target={n_events}"
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
            energy_bins=energy_bins,
            geometry_metadata=geometry_metadata,
            s_r_mean=s_r_mean,
            s_r_std=s_r_std,
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
            event_id_offset=n_written,
        )

        df_chunk = _filter_non_transport_particles(
            df_chunk,
            excluded_particle_names=excluded_particle_names,
        )

        n_attempted += n_chunk

        if len(df_chunk) == 0:
            n_empty_chunks += 1
            if n_empty_chunks >= max_empty_chunks:
                raise RuntimeError(
                    "Too many empty chunks after particle filtering. "
                    "Check type_probs / idx_to_type / excluded_particle_names."
                )
            continue

        n_empty_chunks = 0

        if generate_until_n_written and len(df_chunk) > (n_events - n_written):
            df_chunk = df_chunk.iloc[:(n_events - n_written)].copy()

        if output_format == "text":
            df_chunk.to_csv(
                filepath,
                mode="a",
                sep="\t",
                header=False,
                index=False,
                float_format=float_format,
            )
            n_written += len(df_chunk)

        elif output_format == "binary":
            n_written_chunk = _append_detector_binary_chunk(
                df_chunk,
                filepath,
                excluded_particle_names=set(),
            )
            n_written += n_written_chunk

    if output_format == "binary":
        _rewrite_binary_header(filepath, n_written)

    if verbose:
        print(
            f"[MAGI] Saved generated detector {output_format} file to {filepath}"
        )
        print(f"[MAGI] Raw generated samples attempted: {n_attempted}")
        print(f"[MAGI] Transport particles written: {n_written}")

    return filepath


def generate_detector_input_file(
    *,
    save_dir,
    model_name,
    model_config,
    energy_bins,
    n_types,
    type_weights,
    type_probs,
    idx_to_type,
    geometry_metadata=None,
    u_v_bins=None,
    s_r_mean=None,
    s_r_std=None,
    output_file,
    n_events,
    radius,
    center=(0.0, 0.0, -507.66),
    seed=42,
    chunk_size=100_000,
    energy_mode="uniform",
    output_format="text",
    excluded_particle_names=None,
    generate_until_n_written=True,
    verbose=1,
):
    """
    Generate a Geant4-ready particle input file from a trained MAGI model.
    """
    from magi.training.checkpointing import load_task_adaptive_model_for_generation

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
        energy_bins=energy_bins,
        geometry_metadata=geometry_metadata,
        s_r_mean=s_r_mean,
        s_r_std=s_r_std,
        u_v_bins=u_v_bins,
        center=center,
        radius=radius,
        energy_mode=energy_mode,
        chunk_size=chunk_size,
        seed=seed,
        output_format=output_format,
        excluded_particle_names=excluded_particle_names,
        generate_until_n_written=generate_until_n_written,
        include_event_id=False,
        verbose=verbose,
    )