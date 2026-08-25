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

    Parameters
    ----------
    generated_data : dict or pd.DataFrame
        Generated events, typically the result of
        reconstruct_generated_physics. Column names are matched against the
        same alias sets save_detector_table accepts.

    idx_to_type : dict[int, str] or None
        Index-to-name map, needed when `generated_data` carries integer type
        indices rather than a ready "ParticleName" column.

    include_event_id : bool
        Prepend an EventId column.

    event_id_offset : int
        First EventId value. Use it to keep IDs unique when concatenating
        several generated batches into one source file.

    Returns
    -------
    pd.DataFrame
        Detector-schema table in output column order, ready for
        save_detector_table.
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
    energy_bins=None,
    geometry_metadata=None,
    s_r_mean=None,
    s_r_std=None,
    u_v_bins=None,

    # v0.8 mixture energy
    energy_head_mode=None,
    energy_transform=None,
    qt_energy=None,
    energy_metadata=None,

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

    Supported energy heads:
      - categorical (v0.6/v0.7/v0.7.2):
          energy_bins is required and generated bin indices are converted back
          to physical energies with energy_mode.

      - mixture (v0.8):
          the model returns a continuous energy target y instead of a bin index,
          so energy_bins is unused and the energy transform must be supplied:
          energy_transform="log10" or a fitted qt_energy (equivalently through
          energy_metadata, mirroring geometry_metadata).

    All arguments are keyword-only.

    Parameters
    ----------
    model : keras.Model
        A loaded MAGI model exposing `.generate(cond, n)`.

    filepath : str
        Output path. Parent directories are created if missing. Any existing
        file is truncated.

    n_events : int
        Number of events. Whether this counts raw samples or written
        particles depends on `generate_until_n_written`.

    type_probs : array-like of float
        Categorical probabilities over particle types, summing to 1.

    n_types : int
        One-hot depth; must match the model's conditioning width.

    idx_to_type : dict[int, str]
        Index-to-name map, required here because the output format is keyed
        by particle name.

    energy_bins : array-like or None
        Bin edges in MeV. Required for categorical heads, unused for mixture.

    geometry_metadata : dict or None
        Checkpoint metadata block supplying the geometry transform name and
        the fitted `qt_*` transformers.

    s_r_mean, s_r_std : float or None
        v0.6 only, for unscaling s_r.

    u_v_bins : array-like or None
        v0.6 only, edges of the categorical u_v binning.

    energy_head_mode : str or None
        "categorical" or "mixture".

    energy_transform : str or None
        Energy transform name for mixture heads, e.g. "log10".

    qt_energy : sklearn QuantileTransformer or None
        Fitted energy transform for mixture heads.

    energy_metadata : dict or None
        Checkpoint metadata block supplying the two entries above.

    center : tuple[float, float, float]
        Sphere centre in mm. Must match the training geometry - the default
        (0, 0, 0) is almost certainly not what your run used.

    radius : float
        Sphere radius in mm. Same caveat as `center`.

    energy_mode : str
        Within-bin draw for categorical heads; see energy_from_idx.

    chunk_size : int
        Events generated and written per iteration. Bounds peak memory, so
        arbitrarily large `n_events` stays feasible.

    seed : int
        Seed for the type and energy draws, making the source file
        reproducible.

    include_event_id : bool
        Text format only: prepend an EventId column.

    float_format : str
        Text format only: printf-style numeric format.

    output_format : str
        "text"/"txt" or "binary"/"bin".

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

    verbose : int
        1 (default) reports chunk progress and the final written count.

    Returns
    -------
    dict
        Summary of the write: the output path, format, the number of raw
        samples attempted and the number of particles actually written.
        These differ whenever filtering removed events.
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
            energy_head_mode=energy_head_mode,
            energy_transform=energy_transform,
            qt_energy=qt_energy,
            energy_metadata=energy_metadata,
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
    energy_bins=None,
    n_types,
    type_weights,
    type_probs,
    idx_to_type,
    geometry_metadata=None,
    u_v_bins=None,
    s_r_mean=None,
    s_r_std=None,

    # v0.8 mixture energy
    energy_head_mode=None,
    energy_transform=None,
    qt_energy=None,
    energy_metadata=None,

    output_file,
    n_events,
    radius,
    center=(0.0, 0.0, 0.0),
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

    energy_bins is only needed by categorical-energy models (v0.6/v0.7/v0.7.2).
    v0.8 mixture-energy models instead need the energy transform, supplied
    through energy_transform / qt_energy / energy_metadata.

    This is the one-call entry point: it loads the checkpoint with
    load_task_adaptive_model_for_generation and then streams the events to
    disk with generate_detector_table_to_file. It is what
    scripts/generate_geant_source.py wraps, and therefore what the companion
    Geant4 project's `/generator/mlScript` macro ultimately drives.

    All arguments are keyword-only.

    Parameters
    ----------
    save_dir : str
        Run directory holding the weights and JSON files.

    model_name : str
        Filename stem used when the run was saved.

    model_config : dict
        The saved architecture config, from `<model_name>_config.json`.

    energy_bins : array-like or None
        Bin edges in MeV. Required for categorical heads only.

    n_types : int
        Number of particle types; must match the checkpoint.

    type_weights : array-like
        Per-type loss weights from the run. Not used for sampling, but
        required to rebuild the model.

    type_probs : array-like of float
        Categorical probabilities over particle types for the generated
        source. Pass the training fractions to reproduce the natural mix.

    idx_to_type : dict[int, str]
        Index-to-name map.

    geometry_metadata : dict or None
        Checkpoint metadata block with the geometry transform and fitted
        `qt_*` transformers, loaded from the run's
        `*_quantile_transformers.joblib`.

    u_v_bins, s_r_mean, s_r_std : optional
        v0.6-only reconstruction parameters.

    energy_head_mode : str or None
        "categorical" or "mixture".

    energy_transform : str or None
        Energy transform name for mixture heads, e.g. "log10".

    qt_energy : sklearn QuantileTransformer or None
        Fitted energy transform for mixture heads.

    energy_metadata : dict or None
        Checkpoint metadata block supplying the two entries above.

    output_file : str
        Path of the Geant4 source file to write.

    n_events : int
        Number of source particles to produce.

    radius : float
        Sphere radius in mm. Must match the training geometry.

    center : tuple[float, float, float]
        Sphere centre in mm. The default is the reference X-IFU cryostat
        crossing sphere; override it for another mass model.

    seed : int
        Seed for the sampling, making the source file reproducible.

    chunk_size : int
        Events generated and written per iteration.

    energy_mode : str
        Within-bin draw for categorical heads.

    output_format : str
        "text"/"txt" or "binary"/"bin".

    excluded_particle_names : set/list or None
        Particle names removed before writing; neutrinos by default.

    generate_until_n_written : bool
        Keep generating until `n_events` transportable particles have been
        written, rather than stopping after `n_events` raw samples. True by
        default, so Geant4 receives exactly the requested count.

    verbose : int
        1 (default) reports load and chunk progress.

    Returns
    -------
    dict
        The write summary from generate_detector_table_to_file.
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
        energy_head_mode=energy_head_mode,
        energy_transform=energy_transform,
        qt_energy=qt_energy,
        energy_metadata=energy_metadata,
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