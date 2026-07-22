"""
Input/output utilities for MAGI datasets.
"""

import numpy as np
import pandas as pd
import os
import re


DEFAULT_COLUMNS_DET = [
    "EventId", "ParticleName", "Energy", "X", "Y", "Z", "Vx", "Vy", "Vz"
]

DEFAULT_COLUMNS_DET_PRIM = [
    "EventId", "ParticleName", "Energy", "PrimBool",
    "X", "Y", "Z", "Vx", "Vy", "Vz",
]

# Diagnostic lineage format: adds raw trackID/parentID and the Geant4
# CreatorProcessName ("Primary", "RadioactiveDecay", "eBrem", "compt", ...),
# needed to define "primary" for sources where PrimBool (ParentID==0) is
# never true, e.g. radioactive decay chains embedded in bulk material.
DEFAULT_COLUMNS_DET_LINEAGE = [
    "EventId", "ParticleName", "Energy", "PrimBool",
    "ParticleId", "ParentParticleId", "CreatorProcessName",
    "X", "Y", "Z", "Vx", "Vy", "Vz",
]


def _peek_ncols(filepath, sep=None):
    """
    Count whitespace/sep-separated fields on the first non-empty,
    non-comment line, without loading the whole file.

    Mirrors pandas' own sep convention (used identically by pd.read_table
    below): sep=None means any whitespace, a single character is a literal
    delimiter, anything longer (e.g. r"\\s+") is treated as a regex.
    """
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if sep is None:
                return len(s.split())
            if len(sep) == 1:
                return len(s.split(sep))
            return len(re.split(sep, s))
    return 0


def load_detector_table(
    filepath,
    columns=None,
    drop_event_id=True,
    sep=None,
    has_primary_flag=None,
    has_lineage_cols=None,
    drop_primary_flag=False,
    drop_lineage_cols=False,
):
    """
    Load a detector event table.

    Supports three schemas, auto-detected from the field count on the first
    data line unless overridden:
        9  cols - legacy: EventId ParticleName Energy X Y Z Vx Vy Vz
        10 cols - PrimBool-tagged: adds PrimBool (ParentID==0)
        13 cols - lineage: adds ParticleId, ParentParticleId,
                  CreatorProcessName (e.g. "Primary", "RadioactiveDecay",
                  "eBrem") for sources where PrimBool alone can't define
                  "primary" (see compute_primary_fraction).

    has_primary_flag, has_lineage_cols : bool or None
        If None (default), inferred from field count (>=10 => PrimBool
        present, >=13 => lineage columns present).
    """
    if columns is None:
        n_cols = _peek_ncols(filepath, sep)

        if has_lineage_cols is None:
            has_lineage_cols = (n_cols >= 13)
        if has_primary_flag is None:
            has_primary_flag = (n_cols >= 10)

        if has_lineage_cols:
            columns = DEFAULT_COLUMNS_DET_LINEAGE
        elif has_primary_flag:
            columns = DEFAULT_COLUMNS_DET_PRIM
        else:
            columns = DEFAULT_COLUMNS_DET

    dtype_map = {}

    for name in columns:
        if name in ("ParticleName", "CreatorProcessName"):
            dtype_map[name] = str
        elif name in ("EventId", "PrimBool", "ParticleId", "ParentParticleId"):
            dtype_map[name] = int
        else:
            dtype_map[name] = float

    df = pd.read_table(
        filepath,
        names=columns,
        dtype=dtype_map,
        sep=sep,
    )

    if drop_event_id and "EventId" in df.columns:
        df = df.drop(columns=["EventId"])

    if drop_primary_flag and "PrimBool" in df.columns:
        df = df.drop(columns=["PrimBool"])

    if drop_lineage_cols:
        lineage_cols = [
            c for c in ("ParticleId", "ParentParticleId", "CreatorProcessName")
            if c in df.columns
        ]
        if lineage_cols:
            df = df.drop(columns=lineage_cols)

    return df


def report_basic_table_checks(df, numeric_cols=None):
    """
    Print basic integrity checks on the loaded dataframe.
    """
    if numeric_cols is None:
        numeric_cols = ["Energy", "X", "Y", "Z", "Vx", "Vy", "Vz"]

    print("N rows:", len(df))
    print(df.head())

    print("\nNaN per column:\n", df.isna().sum())

    for c in numeric_cols:
        if c in df.columns:
            bad = ~np.isfinite(df[c].to_numpy(dtype=float))
            if bad.any():
                print(f"Warning: {bad.sum()} non-finite values in {c}")

    if "PrimBool" in df.columns:
        n_generated = len(df)
        n_primaries = int((df["PrimBool"].to_numpy() == 1).sum())
        print(
            f"\nPrimBool present: n_primaries={n_primaries} "
            f"n_generated={n_generated} "
            f"primary_fraction={(n_primaries / n_generated if n_generated else None)}"
        )

    if "CreatorProcessName" in df.columns:
        print("\nCreatorProcessName counts:\n", df["CreatorProcessName"].value_counts())


def save_normalization_summary(normalization, filepath):
    """
    Save a compute_primary_fraction()/build_physical_features() normalization
    dict as a small standalone JSON file, so it can be read from outside the
    MAGI package (e.g. a Geant4-side analysis notebook in another repo)
    without pulling in the full trained-model metadata.
    """
    import json

    if normalization is None:
        raise ValueError("normalization is None - nothing to save.")

    outdir = os.path.dirname(filepath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(normalization, f, indent=2)

    print(f"Saved normalization summary to: {filepath}")

    return filepath


def load_normalization_summary(filepath):
    """
    Load a normalization dict previously written by save_normalization_summary().
    """
    import json

    with open(filepath) as f:
        return json.load(f)


def save_candidate_energy_lines(payload, filepath):
    """
    Save a candidate-energy-lines payload (as built by
    tools/build_candidate_lines_from_geant4.py - a dict with a "lines" list
    plus mass-model/dataset provenance) as a standalone JSON file, so the
    experiment-specific line table lives as data rather than package source.
    """
    import json

    if payload is None:
        raise ValueError("payload is None - nothing to save.")

    outdir = os.path.dirname(filepath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved candidate energy lines to: {filepath}")

    return filepath


def load_candidate_energy_lines(filepath):
    """
    Load a candidate-energy-lines payload previously written by
    save_candidate_energy_lines() / tools/build_candidate_lines_from_geant4.py.
    """
    import json

    with open(filepath) as f:
        return json.load(f)


def save_detector_table(
    data,
    filepath,
    include_event_id=False,
    sep="\t",
    float_format="%.8e",
):
    """
    Save generated particles in detector input format.

    Expected output:
        ParticleName Energy X Y Z Vx Vy Vz

    Compatible with gen_phys returned by reconstruct_generated_physics().
    """
    import os
    import numpy as np
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        df = data.copy()

    elif isinstance(data, dict):
        required = {
            "ParticleName": ["ParticleName", "particle_name", "particle", "type"],
            "Energy": ["Energy", "E_gen", "energy"],
            "X": ["X", "x_gen", "x"],
            "Y": ["Y", "y_gen", "y"],
            "Z": ["Z", "z_gen", "z"],
            "Vx": ["Vx", "vx_gen", "vx"],
            "Vy": ["Vy", "vy_gen", "vy"],
            "Vz": ["Vz", "vz_gen", "vz"],
        }

        out_dict = {}

        for out_name, candidates in required.items():
            found = None
            for key in candidates:
                if key in data:
                    found = key
                    break

            if found is None:
                raise ValueError(
                    f"Missing required quantity '{out_name}'. "
                    f"Accepted keys: {candidates}. "
                    f"Available keys: {list(data.keys())}"
                )

            arr = data[found]

            if hasattr(arr, "numpy"):
                arr = arr.numpy()

            arr = np.asarray(arr)
            arr = np.squeeze(arr)

            if arr.ndim != 1:
                raise ValueError(
                    f"Quantity '{found}' for output column '{out_name}' "
                    f"must be 1D after squeeze, but has shape {arr.shape}."
                )

            out_dict[out_name] = arr

        df = pd.DataFrame(out_dict)

    else:
        raise TypeError(
            "save_detector_table expects a pandas DataFrame or a dict."
        )

    ordered_cols = [
        "ParticleName",
        "Energy",
        "X",
        "Y",
        "Z",
        "Vx",
        "Vy",
        "Vz",
    ]

    out = df[ordered_cols].copy()

    if include_event_id:
        out.insert(0, "EventId", np.arange(len(out), dtype=int))

    outdir = os.path.dirname(filepath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    out.to_csv(
        filepath,
        sep=sep,
        header=False,
        index=False,
        float_format=float_format,
    )

    print(f"Saved detector table to: {filepath}")
    print("Rows saved:", len(out))
    print("Columns:", list(out.columns))

    return out