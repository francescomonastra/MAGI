"""
Input/output utilities for GEEANNT datasets.
"""

import numpy as np
import pandas as pd
import os


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

    Supports both formats:
        EventId ParticleName Energy X Y Z Vx Vy Vz
    and:
        ParticleName Energy X Y Z Vx Vy Vz
    """
    if columns is None:
        if drop_event_id:
            columns = [
                "EventId", "ParticleName", "Energy",
                "X", "Y", "Z", "Vx", "Vy", "Vz"
            ]
        else:
            columns = [
                "ParticleName", "Energy",
                "X", "Y", "Z", "Vx", "Vy", "Vz"
            ]

    dtype_map = {}

    for name in columns:
        if name == "ParticleName":
            dtype_map[name] = str
        elif name == "EventId":
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