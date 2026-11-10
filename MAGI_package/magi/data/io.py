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

# HENABICS primary-truth format (DM1.2 HENABICS-primary-truth branch): the
# lineage layout plus the primary that produced the crossing, appended at the
# end so no earlier column moves. EventId here is the join key: EventId + the
# per-job offset in a full-simulation merge, the injected group key in a
# post-MAGI run. PrimaryEnergy is in MeV.
PRIMARY_TRUTH_COLS = ["PrimaryTrackId", "PrimaryPDG", "PrimaryEnergy"]
DEFAULT_COLUMNS_DET_PRIMTRUTH = DEFAULT_COLUMNS_DET_LINEAGE + PRIMARY_TRUTH_COLS

# Track G (S1GDML_DM1.2_MultiVerse, branch trackG-multiverse): a 17th column, AncestorInside, written
# by the virtual tracking ball. 0 = first entry (the track and all its ancestors were outside the
# ball before this crossing), 1 = the track, or an ancestor, had been inside. The rows with 0 are the
# sample that is independent of what lies inside the ball; see `select_first_entry`.
FIRST_ENTRY_COL = "AncestorInside"
DEFAULT_COLUMNS_DET_FIRSTENTRY = DEFAULT_COLUMNS_DET_PRIMTRUTH + [FIRST_ENTRY_COL]

# primaries.dat: one row per primary that left at least one record (crossing
# or deposit). Primaries with neither are only in run_summary.json's thrown
# histogram - see load_run_summaries.
DEFAULT_COLUMNS_PRIMARIES = [
    "EventId", "PrimaryTrackId", "PrimaryPDG", "PrimaryEnergy",
    "NCrossings", "NHits",
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
    drop_event_id=None,
    sep=None,
    has_primary_flag=None,
    has_lineage_cols=None,
    drop_primary_flag=False,
    drop_lineage_cols=False,
    has_primary_truth=None,
):
    """
    Load a detector event table.

    Supports four schemas, auto-detected from the field count on the first
    data line unless overridden:
        9  cols - legacy: EventId ParticleName Energy X Y Z Vx Vy Vz
        10 cols - PrimBool-tagged: adds PrimBool (ParentID==0)
        13 cols - lineage: adds ParticleId, ParentParticleId,
                  CreatorProcessName (e.g. "Primary", "RadioactiveDecay",
                  "eBrem") for sources where PrimBool alone can't define
                  "primary" (see compute_primary_fraction).
        16 cols - primary truth (HENABICS): lineage plus PrimaryTrackId,
                  PrimaryPDG, PrimaryEnergy [MeV] of the primary that
                  produced the crossing. EventId is then the join key.

    Parameters
    ----------
    filepath : str
        Path to the whitespace- or `sep`-delimited detector table.

    columns : list[str] or None
        Explicit column names. If None (default) the schema is auto-detected
        from the field count and one of DEFAULT_COLUMNS_DET,
        DEFAULT_COLUMNS_DET_PRIM, DEFAULT_COLUMNS_DET_LINEAGE or
        DEFAULT_COLUMNS_DET_PRIMTRUTH is used. Passing this disables
        auto-detection entirely.

    drop_event_id : bool or None
        Drop the "EventId" column after loading. None (default) drops it
        unless the table carries primary truth: the per-crossing pipeline
        never uses it, but with primary truth it is the group key that ties
        the crossings of one shower together.

    sep : str or None
        Field delimiter, following pandas' convention: None means any run of
        whitespace, a single character is a literal delimiter, and anything
        longer (e.g. r"\\s+") is treated as a regex.

    has_primary_flag : bool or None
        Whether the file carries the PrimBool column (Geant4 ParentID==0).
        If None (default), inferred as `n_cols >= 10`.

    has_lineage_cols : bool or None
        Whether the file carries ParticleId / ParentParticleId /
        CreatorProcessName. If None (default), inferred as `n_cols >= 13`.

    drop_primary_flag : bool
        Drop PrimBool after loading, even when present.

    drop_lineage_cols : bool
        Drop the three lineage columns after loading, even when present.

    has_primary_truth : bool or None
        Whether the file carries PrimaryTrackId / PrimaryPDG / PrimaryEnergy.
        If None (default), inferred as `n_cols >= 16`.

    A file with 17 columns (the MultiVerse tracking ball of Track G) carries the
    first-entry flag `AncestorInside` as its last column; it is read as an
    integer column and can be used with `select_first_entry`.

    Returns
    -------
    pd.DataFrame
        One row per detector crossing, with float energy/position/direction
        columns and string ParticleName. Energy is in MeV and positions in
        mm, matching the Geant4 output convention.

    See Also
    --------
    compute_primary_fraction : uses the lineage columns to define "primary"
        for sources where PrimBool is never true (e.g. decay chains).
    """
    if columns is None:
        n_cols = _peek_ncols(filepath, sep)

        if has_primary_truth is None:
            has_primary_truth = (n_cols >= 16)
        if has_lineage_cols is None:
            has_lineage_cols = (n_cols >= 13)
        if has_primary_flag is None:
            has_primary_flag = (n_cols >= 10)

        if has_primary_truth and n_cols >= 17:
            columns = DEFAULT_COLUMNS_DET_FIRSTENTRY
        elif has_primary_truth:
            columns = DEFAULT_COLUMNS_DET_PRIMTRUTH
        elif has_lineage_cols:
            columns = DEFAULT_COLUMNS_DET_LINEAGE
        elif has_primary_flag:
            columns = DEFAULT_COLUMNS_DET_PRIM
        else:
            columns = DEFAULT_COLUMNS_DET

    dtype_map = {}

    for name in columns:
        if name in ("ParticleName", "CreatorProcessName"):
            dtype_map[name] = str
        elif name in ("EventId", "PrimBool", "ParticleId", "ParentParticleId",
                      "PrimaryTrackId", "PrimaryPDG", FIRST_ENTRY_COL):
            dtype_map[name] = np.int64
        else:
            dtype_map[name] = float

    df = pd.read_table(
        filepath,
        names=columns,
        dtype=dtype_map,
        sep=sep,
    )

    if drop_event_id is None:
        drop_event_id = "PrimaryPDG" not in df.columns
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


def select_first_entry(df, drop_flag=True):
    """
    Keep the first entries of a crossing table written by the MultiVerse tracking ball.

    A first entry is a crossing whose track, and every ancestor of it, was outside the
    ball before the crossing (`AncestorInside == 0`). Only these rows are independent of
    what lies inside the ball, since everything with the flag set is produced again by
    the application run from the injected first entries. A model trained on all the rows
    would count that part twice.

    Parameters
    ----------
    df : pd.DataFrame
        A table loaded with `load_detector_table` from a 17-column file.

    drop_flag : bool
        Drop the `AncestorInside` column after the selection (default). The fraction
        removed is printed, since it is a diagnostic of the sphere (about 1-3 % at a free
        radius, and more where the sphere cuts material).

    Returns
    -------
    pd.DataFrame
        The rows with `AncestorInside == 0`, index reset.

    Raises
    ------
    ValueError
        If the table has no `AncestorInside` column, which means that it was written by a
        run without the tracking ball and is already the all-crossings sample.
    """
    if FIRST_ENTRY_COL not in df.columns:
        raise ValueError(
            f"no {FIRST_ENTRY_COL} column: the table is not a MultiVerse (17-column) crossing "
            "table, so there is no first-entry flag to select on"
        )
    keep = df[FIRST_ENTRY_COL].to_numpy() == 0
    print(f"[MAGI] first entries: {int(keep.sum()):,} of {len(df):,} crossings kept "
          f"({100.0 * (1.0 - keep.mean()):.2f} % flagged as returning)", flush=True)
    out = df[keep].reset_index(drop=True)
    return out.drop(columns=[FIRST_ENTRY_COL]) if drop_flag else out


def load_primaries_table(filepaths, sep=None):
    """
    Load one or more primaries.dat tables (HENABICS primary truth).

    One row per primary that left at least one record, with columns
    DEFAULT_COLUMNS_PRIMARIES: EventId (the join key into the crossing and
    deposit tables), PrimaryTrackId, PrimaryPDG, PrimaryEnergy [MeV],
    NCrossings (the shower multiplicity k at the MAGI sphere) and NHits.

    Parameters
    ----------
    filepaths : str or list[str]
        One path, or several (e.g. one per species) to concatenate. Keys are
        unique within a species' merged file, not across species - keep
        (PrimaryPDG, EventId) as the key when mixing species.

    sep : str or None
        Field delimiter, as in load_detector_table.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(filepaths, (str, os.PathLike)):
        filepaths = [filepaths]
    dtype_map = {c: np.int64 for c in DEFAULT_COLUMNS_PRIMARIES}
    dtype_map["PrimaryEnergy"] = float
    frames = [
        pd.read_table(fp, names=DEFAULT_COLUMNS_PRIMARIES, dtype=dtype_map, sep=sep)
        for fp in filepaths
    ]
    return pd.concat(frames, ignore_index=True)


def load_run_summaries(paths):
    """
    Sum the run_summary.json files written by the DM1.2 RunAction.

    Each file holds, per primary PDG code, a histogram of log10(E/MeV) of
    every primary thrown ("thrown") and of those that left at least one
    record ("recorded"), laid out as [underflow, n bins, overflow]. Summed
    over jobs, thrown is the flux-normalization denominator and thrown -
    recorded gives the k = 0 primaries that primaries.dat does not list.

    Parameters
    ----------
    paths : str or list[str]
        A directory (every job*.json inside is read), a single JSON file, or
        a list of either.

    Returns
    -------
    dict
        {"n_events", "log10E_edges" (n+1 edges), "thrown" {pdg: counts},
        "recorded" {pdg: counts}, "jobs" [per-file job_tag/seeds/source_macro]}
        with counts as int64 arrays of length n+2 (underflow, bins, overflow)
        and pdg as int.
    """
    import glob
    import json

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError(f"no run_summary JSON under {paths}")

    out = {"n_events": 0, "log10E_edges": None, "thrown": {}, "recorded": {}, "jobs": []}
    for fp in files:
        with open(fp) as f:
            s = json.load(f)
        b = s["log10E_bins"]
        edges = np.linspace(b["min"], b["max"], int(b["n"]) + 1)
        if out["log10E_edges"] is None:
            out["log10E_edges"] = edges
        elif not np.allclose(out["log10E_edges"], edges):
            raise ValueError(f"{fp}: log10E binning differs from the other summaries")
        out["n_events"] += int(s["n_events"])
        for key in ("thrown", "recorded"):
            for pdg, counts in s[key].items():
                c = np.asarray(counts, dtype=np.int64)
                acc = out[key].setdefault(int(pdg), np.zeros_like(c))
                acc += c
        out["jobs"].append({
            k: s.get(k) for k in ("job_tag", "source_macro", "seeds", "n_events")
        })
    return out


def report_basic_table_checks(df, numeric_cols=None):
    """
    Print basic integrity checks on the loaded dataframe.

    Prints row count, a head preview, per-column NaN counts, a warning for any
    non-finite numeric value, and - when the corresponding columns are present
    - the measured primary fraction and the CreatorProcessName breakdown.
    Diagnostic only; nothing is returned and the dataframe is not modified.

    Parameters
    ----------
    df : pd.DataFrame
        Table as returned by load_detector_table.

    numeric_cols : list[str] or None
        Columns to check for non-finite values. Defaults to
        ["Energy", "X", "Y", "Z", "Vx", "Vy", "Vz"]. Names absent from `df`
        are skipped silently.

    Returns
    -------
    None
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

    Parameters
    ----------
    normalization : dict
        JSON-serializable normalization summary. Typically the dict returned
        by compute_primary_fraction, carrying the chi / f_primary factors the
        Geant4-side analysis needs to rescale generated fluxes.

    filepath : str
        Destination path. Parent directories are created if missing.

    Returns
    -------
    str
        The `filepath` written, for chaining.

    Raises
    ------
    ValueError
        If `normalization` is None.
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

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    dict
        The normalization summary, exactly as saved.
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

    Parameters
    ----------
    payload : dict
        Dict with a "lines" list - each entry carrying at least
        "energy_mev" and "label" - plus provenance keys recording
        which GDML mass model and fluorescence table it was derived from.

    filepath : str
        Destination path. Parent directories are created if missing.
        By convention these live under CandidateLines/ in the repo.

    Returns
    -------
    str
        The `filepath` written, for chaining.

    Raises
    ------
    ValueError
        If `payload` is None.
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

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    dict
        The payload, exactly as saved. Pass `payload["lines"]` to
        detect_energy_lines or build_gate_targets.
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

    Parameters
    ----------
    data : pd.DataFrame or dict
        Generated events. A DataFrame must already carry the eight output
        columns. A dict is matched key-by-key against a set of accepted
        aliases per column ("Energy"/"E_gen"/"energy", "X"/"x_gen"/"x", ...),
        so the raw output of reconstruct_generated_physics can be passed
        directly. Values may be TensorFlow tensors; they are converted and
        squeezed to 1D.

    filepath : str
        Destination path. Parent directories are created if missing.

    include_event_id : bool
        Prepend a 0-based integer EventId column, matching the 9-column
        input schema that load_detector_table reads back.

    sep : str
        Field delimiter. Tab by default.

    float_format : str
        printf-style format for the numeric columns. The default "%.8e"
        preserves full float32 precision through the text round-trip.

    Returns
    -------
    pd.DataFrame
        The exact table written, with columns in output order.

    Raises
    ------
    ValueError
        If a required quantity is missing from a dict input, or a value is
        not 1D after squeezing.
    TypeError
        If `data` is neither a DataFrame nor a dict.

    See Also
    --------
    generate_detector_table_to_file : streams large runs in chunks instead of
        materializing the whole table in memory.
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