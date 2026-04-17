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