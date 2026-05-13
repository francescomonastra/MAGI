"""
Input/output utilities for GEEANNT datasets.
"""

import numpy as np
import pandas as pd


DEFAULT_COLUMNS_DET = [
    "EventId", "ParticleName", "Energy", "X", "Y", "Z", "Vx", "Vy", "Vz"
]


def load_detector_table(
    filepath,
    columns=None,
    drop_event_id=True,
    sep=None,
):
    """
    Load a detector event table.

    Parameters
    ----------
    filepath : str
        Path to the input text table.
    columns : list[str] or None
        Column names. If None, use the default detector layout.
    drop_event_id : bool
        If True, drop the EventId column after loading.
    sep : str or None
        Separator passed to pandas.read_table. If None, pandas infers whitespace.

    Returns
    -------
    pd.DataFrame
        Loaded event table.
    """
    if columns is None:
        columns = DEFAULT_COLUMNS_DET

    dtype_map = {name: float for name in columns}
    if "ParticleName" in columns:
        dtype_map["ParticleName"] = str

    df = pd.read_table(
        filepath,
        names=columns,
        dtype=dtype_map,
        sep=sep,
    )

    if drop_event_id and "EventId" in df.columns:
        df = df.drop(columns=["EventId"])

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


def save_detector_table(
    df,
    filepath,
    include_event_id=False,
    sep="\t",
    float_format="%.8e",
):
    """
    Save generated detector events in GEEANNT detector-table format.

    Output schema (default):
        ParticleName Energy X Y Z Vx Vy Vz

    Optional:
        EventId ParticleName Energy X Y Z Vx Vy Vz

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing generated physical events.

        Accepted column naming:
            lowercase:
                x, y, z, vx, vy, vz
            uppercase:
                X, Y, Z, Vx, Vy, Vz

        Required:
            ParticleName
            Energy
            coordinates
            direction cosines

    filepath : str
        Output path.

    include_event_id : bool
        If True, prepend sequential EventId column.

    sep : str
        Output separator.

    float_format : str
        Floating-point formatting passed to pandas.to_csv().
    """
    required_base = ["ParticleName", "Energy"]

    for col in required_base:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    colmap = {}

    # coordinates
    if all(c in df.columns for c in ["x", "y", "z"]):
        colmap.update({
            "x": "X",
            "y": "Y",
            "z": "Z",
        })
    elif not all(c in df.columns for c in ["X", "Y", "Z"]):
        raise ValueError("Missing position columns (x,y,z) or (X,Y,Z)")

    # directions
    if all(c in df.columns for c in ["vx", "vy", "vz"]):
        colmap.update({
            "vx": "Vx",
            "vy": "Vy",
            "vz": "Vz",
        })
    elif not all(c in df.columns for c in ["Vx", "Vy", "Vz"]):
        raise ValueError("Missing direction columns (vx,vy,vz) or (Vx,Vy,Vz)")

    out = df.copy().rename(columns=colmap)

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

    out = out[ordered_cols]

    if include_event_id:
        out.insert(0, "EventId", np.arange(len(out), dtype=int))

    out.to_csv(
        filepath,
        sep=sep,
        header=False,
        index=False,
        float_format=float_format,
    )

    print(f"Saved detector table to: {filepath}")