"""
Preprocessing and physical feature construction for MAGI.
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing
from scipy.signal import find_peaks


# Default "what counts as primary" criterion per source type. Cosmic rays:
# the GPS-thrown particle reaching the CryoSphere unmodified (ParentID==0,
# collapsed into PrimBool by the Geant4 writer). Radioactive decay sources
# (K-40, Th-232, Ra-226, ...): PrimBool is always 0, since the primary track
# is the decaying nucleus/ion itself and never propagates anywhere - what
# should count as "primary" there is a decay-line emission that reached the
# CryoSphere (however many times it Compton-scattered en route, since that
# doesn't change CreatorProcessName or the track's identity), as opposed to
# e.g. bremsstrahlung/annihilation photons genuinely produced along the way.
SOURCE_TYPE_PRIMARY_DEFAULTS = {
    "cosmic_ray": ("PrimBool", 1),
    "radioactive": ("CreatorProcessName", "RadioactiveDecay"),
}


# Minimal, mass-model-independent default candidate line for detect_energy_lines()
# when no experiment-specific table is supplied - just the one line that's true
# regardless of detector/mass-model (positron annihilation). Real usage should
# pass an experiment-specific table loaded via
# magi.data.load_candidate_energy_lines(<path>)["lines"], generated for the
# actual GDML mass model by tools/build_candidate_lines_from_geant4.py (which
# sources fluorescence/decay-line energies from Geant4's own data, not values
# hand-typed into this file - see that script's docstring for why).
DEFAULT_CANDIDATE_ENERGY_LINES = [
    {"label": "e+e- annihilation", "energy_kev": 511.00, "energy_mev": 511.00 / 1000.0, "origin": "instrumental"},
]


def compute_primary_fraction(df, primary_col="PrimBool", primary_value=1):
    """
    Count primaries vs. total CryoSphere crossings for flux normalization.

    primary_col / primary_value : which column, and which value in it,
        marks a row as "primary". Defaults match the cosmic-ray convention
        (PrimBool == 1); pass e.g. primary_col="CreatorProcessName",
        primary_value="RadioactiveDecay" for radioactive sources, or use
        source_type="radioactive" in build_physical_features instead of
        setting these directly. See SOURCE_TYPE_PRIMARY_DEFAULTS.

    Returns None if `primary_col` is not present (e.g. legacy 9-column
    data, or a lineage column requested on a file that doesn't have it),
    so downstream code can treat "no normalization info" uniformly.
    """
    if primary_col not in df.columns:
        return None

    is_prim = df[primary_col].to_numpy()
    n_generated = int(len(is_prim))
    n_primaries = int(np.sum(is_prim == primary_value))

    per_species = {}
    if "ParticleName" in df.columns:
        for name, sub in df.groupby("ParticleName"):
            tot = int(len(sub))
            pri = int(np.sum(sub[primary_col].to_numpy() == primary_value))
            per_species[str(name)] = {
                "n_generated": tot,
                "n_primaries": pri,
                "primary_fraction": (pri / tot if tot else None),
            }

    return {
        "primary_col": primary_col,
        "primary_value": primary_value,
        "n_generated": n_generated,
        "n_primaries": n_primaries,
        "primary_fraction": (n_primaries / n_generated if n_generated else None),
        "per_species": per_species,
    }


def build_physical_features(
    df,
    center=(0.0, 0.0, -507.66),
    radius=100.0,
    eps=1e-6,
    drop_invalid_energy=True,
    source_type="cosmic_ray",
    primary_col=None,
    primary_value=None,
):
    """
    Build the base physical feature representation used by MAGI.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns:
        ParticleName, Energy, X, Y, Z, Vx, Vy, Vz

    center : tuple[float, float, float]
        Sphere center in mm.

    radius : float
        Sphere radius in mm.

    eps : float
        Numerical stability constant.

    drop_invalid_energy : bool
        If True, remove rows with E <= 0 or non-finite energy.

    source_type : str
        Selects the default "primary" criterion for compute_primary_fraction
        via SOURCE_TYPE_PRIMARY_DEFAULTS: "cosmic_ray" (PrimBool==1, default)
        or "radioactive" (CreatorProcessName=="RadioactiveDecay"). Ignored if
        primary_col/primary_value are given explicitly.

    primary_col, primary_value : str or None
        Explicit override for compute_primary_fraction's criterion, taking
        precedence over source_type. Use this for a source type not covered
        by SOURCE_TYPE_PRIMARY_DEFAULTS.

    Returns
    -------
    dict
        Dictionary containing cleaned dataframe, physical features,
        projected coordinates, normalized directions and diagnostics.
    """
    C = np.asarray(center, dtype=np.float64)
    R = float(radius)

    df = df.copy()

    if primary_col is None or primary_value is None:
        default_col, default_value = SOURCE_TYPE_PRIMARY_DEFAULTS.get(
            source_type, ("PrimBool", 1)
        )
        if primary_col is None:
            primary_col = default_col
        if primary_value is None:
            primary_value = default_value

    # Computed before any row filtering so n_generated matches the raw
    # crossing count (NCryoSphereCR in the notebook), None if primary_col absent.
    normalization = compute_primary_fraction(
        df, primary_col=primary_col, primary_value=primary_value
    )

    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    v = df[["Vx", "Vy", "Vz"]].to_numpy(dtype=np.float64)
    E = df["Energy"].to_numpy(dtype=np.float64)

    if drop_invalid_energy:
        mask_posE = (E > 0) & np.isfinite(E)
        if not mask_posE.all():
            df = df.loc[mask_posE].reset_index(drop=True)
            xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
            v = df[["Vx", "Vy", "Vz"]].to_numpy(dtype=np.float64)
            E = df["Energy"].to_numpy(dtype=np.float64)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    # Raw geometry checks
    rvec = xyz - C[None, :]
    r = np.linalg.norm(rvec, axis=1)
    vnorm = np.linalg.norm(v, axis=1)

    # Project positions onto the sphere
    rhat = rvec / (r[:, None] + 1e-12)
    rvec_proj = R * rhat
    xyz_proj = C[None, :] + rvec_proj

    x_proj = xyz_proj[:, 0]
    y_proj = xyz_proj[:, 1]
    z_proj = xyz_proj[:, 2]
    r_proj = np.linalg.norm(rvec_proj, axis=1)

    # Renormalize directions
    vhat = v / (vnorm[:, None] + 1e-12)
    vx_hat = vhat[:, 0]
    vy_hat = vhat[:, 1]
    vz_hat = vhat[:, 2]
    vnorm_hat = np.linalg.norm(vhat, axis=1)

    # Position features
    u_r = np.clip(rhat[:, 2], -1.0 + eps, 1.0 - eps)
    s_r = np.arctanh(u_r)

    phi_r = np.arctan2(rhat[:, 1], rhat[:, 0])
    cphi_r = np.cos(phi_r)
    sphi_r = np.sin(phi_r)

    # Direction features
    u_v = np.clip(vz_hat, -1.0 + eps, 1.0 - eps)

    phi_v = np.arctan2(vy_hat, vx_hat)
    cphi_v = np.cos(phi_v)
    sphi_v = np.sin(phi_v)

    logE = np.log10(E)

    features = pd.DataFrame({
        "ParticleName": df["ParticleName"].astype(str).to_numpy(),
        "Energy": E.astype(np.float64),
        "logE": logE.astype(np.float64),

        # Radial position features
        "u_r": u_r.astype(np.float64),
        "s_r": s_r.astype(np.float64),

        # Angular position features
        "phi_r": phi_r.astype(np.float64),
        "cphi_r": cphi_r.astype(np.float64),
        "sphi_r": sphi_r.astype(np.float64),

        # Direction cosine feature
        "u_v": u_v.astype(np.float64),

        # Angular direction features
        "phi_v": phi_v.astype(np.float64),
        "cphi_v": cphi_v.astype(np.float64),
        "sphi_v": sphi_v.astype(np.float64),
    })

    if "PrimBool" in df.columns:
        features["PrimBool"] = df["PrimBool"].to_numpy()

    if primary_col in df.columns and primary_col not in features.columns:
        features[primary_col] = df[primary_col].to_numpy()

    return {
        "dataframe": df,
        "features": features,
        "normalization": normalization,
        "center": C,
        "radius": R,
        "raw": {
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "r": r,
            "vnorm": vnorm,
        },
        "projected": {
            "x": x_proj,
            "y": y_proj,
            "z": z_proj,
            "r": r_proj,
        },
        "normalized_direction": {
            "vx": vx_hat,
            "vy": vy_hat,
            "vz": vz_hat,
            "vnorm": vnorm_hat,
        },
        "derived": {
            "u_r": u_r,
            "s_r": s_r,
            "u_v": u_v,
            "cphi_r": cphi_r,
            "sphi_r": sphi_r,
            "cphi_v": cphi_v,
            "sphi_v": sphi_v,
            "phi_r": phi_r,
            "phi_v": phi_v,
            "logE": logE,
        },
    }


def print_physical_summary(prep):
    """
    Print a compact summary of the preprocessed dataset.
    """
    raw = prep["raw"]
    proj = prep["projected"]
    normdir = prep["normalized_direction"]
    feat = prep["features"]
    R = prep["radius"]

    print("\n--- RAW sphere check ---")
    print(
        "r: mean =", raw["r"].mean(),
        "std =", raw["r"].std(),
        "min =", raw["r"].min(),
        "max =", raw["r"].max(),
    )
    print("Mean absolute deviation from R:", np.mean(np.abs(raw["r"] - R)))

    print("\n--- RAW direction norm check ---")
    print(
        "||v||: mean =", raw["vnorm"].mean(),
        "std =", raw["vnorm"].std(),
        "min =", raw["vnorm"].min(),
        "max =", raw["vnorm"].max(),
    )
    print("Mean absolute deviation from 1:", np.mean(np.abs(raw["vnorm"] - 1.0)))

    print("\n--- PROJECTED sphere check ---")
    print(
        "r_proj: mean =", proj["r"].mean(),
        "std =", proj["r"].std(),
        "min =", proj["r"].min(),
        "max =", proj["r"].max(),
    )
    print("Mean absolute deviation from R:", np.mean(np.abs(proj["r"] - R)))

    print("\n--- RENORMALIZED direction check ---")
    print(
        "||vhat||: mean =", normdir["vnorm"].mean(),
        "std =", normdir["vnorm"].std(),
        "min =", normdir["vnorm"].min(),
        "max =", normdir["vnorm"].max(),
    )
    print("Mean absolute deviation from 1:", np.mean(np.abs(normdir["vnorm"] - 1.0)))

    print("\n--- Energy check ---")
    print(
        "E: min =", feat["Energy"].min(),
        "max =", feat["Energy"].max(),
        "#E<=0 =", np.sum(feat["Energy"] <= 0),
    )

    print("\n--- u_r check ---")
    print(
        "u_r: min =", feat["u_r"].min(),
        "max =", feat["u_r"].max(),
        "mean =", feat["u_r"].mean(),
        "std =", feat["u_r"].std(),
    )

    print("\n--- u_v check ---")
    print(
        "u_v: min =", feat["u_v"].min(),
        "max =", feat["u_v"].max(),
        "mean =", feat["u_v"].mean(),
        "std =", feat["u_v"].std(),
    )
    print("fraction u_v > 0 =", np.mean(feat["u_v"] > 0))


def build_energy_bins(
    E,
    mode,
    bin_width=0.5,
    n_bins=512,
    min_counts=20,
):
    """
    Build energy bin edges according to the selected mode.

    Parameters
    ----------
    E : array-like
        Physical energy values in MeV.

    mode : str
        One of:
          - "fixed_width"
          - "min_counts"
          - "log_fixed_count"

    bin_width : float
        Fixed bin width in MeV for "fixed_width".

    n_bins : int
        Initial / target number of bins.

    min_counts : int
        Minimum counts per bin for "min_counts".

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    E = np.asarray(E, dtype=np.float64)

    if E.ndim != 1 or len(E) == 0:
        raise ValueError("E must be a non-empty 1D array.")

    if mode == "fixed_width":
        E_min, E_max = E.min(), E.max()
        bins = np.arange(E_min, E_max + bin_width, bin_width, dtype=np.float64)

        if len(bins) < 2:
            bins = np.array([E_min, E_max + bin_width], dtype=np.float64)

        if bins[-1] <= E_max:
            bins = np.append(bins, bins[-1] + bin_width)

    elif mode == "log_fixed_count":
        if np.any(E <= 0):
            raise ValueError("All energies must be > 0 for log_fixed_count mode.")

        logE = np.log10(E)
        bins_log = np.linspace(logE.min(), logE.max(), n_bins + 1, dtype=np.float64)
        bins = np.power(10.0, bins_log)

    elif mode == "min_counts":
        hist, edges = np.histogram(E, bins=n_bins)

        hist = hist.astype(np.int64)
        edges = list(edges.astype(np.float64))

        i = 0
        while i < len(hist):
            if hist[i] < min_counts and len(hist) > 1:
                if i == 0:
                    hist[1] += hist[0]
                    hist = np.delete(hist, 0)
                    edges.pop(1)

                elif i == len(hist) - 1:
                    hist[i - 1] += hist[i]
                    hist = np.delete(hist, i)
                    edges.pop(i)

                else:
                    if hist[i - 1] <= hist[i + 1]:
                        hist[i - 1] += hist[i]
                        hist = np.delete(hist, i)
                        edges.pop(i)
                        i -= 1
                    else:
                        hist[i + 1] += hist[i]
                        hist = np.delete(hist, i)
                        edges.pop(i + 1)
            else:
                i += 1

        bins = np.array(edges, dtype=np.float64)

    else:
        raise ValueError(
            "Invalid energy binning mode. Use "
            "'fixed_width', 'min_counts', or 'log_fixed_count'."
        )

    return bins


def bin_counts(E, edges):
    """
    Histogram E into edges.

    Parameters
    ----------
    E : array-like
        Physical energy values in MeV.
    edges : array-like
        Bin edges, as returned by build_energy_bins.

    Returns
    -------
    np.ndarray
        Per-bin counts, float64.
    """
    counts, _ = np.histogram(E, bins=edges)
    return counts.astype(np.float64)


def detect_line_bins(counts, prominence_factor=3.0, window=5):
    """
    Return indices of bins that stand out as lines above a local continuum.

    A bin is a line if its count exceeds `prominence_factor` x the local
    median (rolling window, excluding the bin itself), intersected with
    scipy.signal.find_peaks for a cleaner peak set.

    Parameters
    ----------
    counts : array-like
        Per-bin counts, as returned by bin_counts.
    prominence_factor : float
        A bin is flagged as a line if count > prominence_factor * local_median.
    window : int
        Rolling window (in bins) used to estimate the local continuum.

    Returns
    -------
    lines : np.ndarray
        Indices of bins flagged as lines.
    local_med : np.ndarray
        Local-median continuum estimate, same length as counts.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.size
    local_med = np.empty(n)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        neigh = np.concatenate([counts[lo:i], counts[i + 1:hi]])
        local_med[i] = np.median(neigh) if neigh.size else 0.0

    gate = counts > prominence_factor * np.maximum(local_med, 1.0)

    peaks, _ = find_peaks(counts)
    peak_mask = np.zeros(n, dtype=bool)
    peak_mask[peaks] = True
    lines = np.where(gate & peak_mask)[0]

    return lines, local_med


def detect_energy_lines(
    E,
    energy_bins=None,
    binning_mode="log_fixed_count",
    bin_width=0.5,
    n_bins=512,
    min_counts=20,
    prominence_factor=3.0,
    window=5,
    candidate_lines=None,
    match_tolerance_bins=2.0,
    resolution_mev=None,
):
    """
    Detect statistically significant spectral lines in an energy spectrum and
    match them against a table of physically expected candidate lines.

    Parameters
    ----------
    E : array-like
        Physical energy values in MeV.
    energy_bins : array-like or None
        Bin edges to use. If None, they are built from E via
        build_energy_bins(E, mode=binning_mode, ...).
    binning_mode, bin_width, n_bins, min_counts : see build_energy_bins.
        Only used when energy_bins is None.
    prominence_factor, window : see detect_line_bins.
    candidate_lines : list of dict or None
        Candidate line table to match against. Defaults to
        DEFAULT_CANDIDATE_ENERGY_LINES (just the universal 511 keV
        annihilation line) - for real, mass-model-specific matching, pass a
        table loaded via magi.data.load_candidate_energy_lines(<path>)["lines"].
        Each entry must have "energy_mev".
    match_tolerance_bins : float
        Matching tolerance for a candidate line, expressed as a multiple of
        the local bin width at that candidate's energy. Bin-width-relative
        (rather than a fixed MeV tolerance) because build_energy_bins
        supports non-uniform binning (e.g. log_fixed_count), where bin width
        varies by orders of magnitude across the spectrum.
    resolution_mev : float or None
        If given, the matching tolerance is widened to at least this value
        (detector energy resolution) - matching can never be tighter than
        the instrument's own resolution allows.

    Returns
    -------
    dict
        n_events, energy_bins, binning_mode, prominence_factor, window,
        match_tolerance_bins, resolution_mev,
        detected_peaks (all statistically-flagged bins),
        matched_lines (candidate<->detected-peak matches),
        unmatched_detected_peaks (significant bins with no nearby candidate -
            possible unknown/uncatalogued lines),
        unmatched_candidates (expected candidates not statistically detected -
            e.g. source decay lines Compton-degraded before crossing the
            detector),
        n_candidate_lines, n_detected_peaks, n_matched.
        All values are native Python types (no numpy scalars), so the result
        can be passed directly to save_normalization_summary.
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 1 or len(E) == 0:
        raise ValueError("E must be a non-empty 1D array.")

    if energy_bins is None:
        edges = build_energy_bins(
            E, mode=binning_mode, bin_width=bin_width,
            n_bins=n_bins, min_counts=min_counts,
        )
        used_mode = binning_mode
    else:
        edges = np.asarray(energy_bins, dtype=np.float64)
        used_mode = "external"

    if candidate_lines is None:
        candidate_lines = DEFAULT_CANDIDATE_ENERGY_LINES

    counts = bin_counts(E, edges)
    peak_idx, local_med = detect_line_bins(
        counts, prominence_factor=prominence_factor, window=window
    )
    centres = 0.5 * (edges[:-1] + edges[1:])

    detected_peaks = [
        {
            "bin_index": int(b),
            "energy_mev": float(centres[b]),
            "count": float(counts[b]),
            "local_median": float(local_med[b]),
        }
        for b in peak_idx
    ]

    matched_lines = []
    matched_peak_bins = set()
    unmatched_candidates = []

    for cand in candidate_lines:
        E_c = float(cand["energy_mev"])
        bin_i = int(np.clip(np.searchsorted(edges, E_c) - 1, 0, len(edges) - 2))
        local_width = float(edges[bin_i + 1] - edges[bin_i])
        tol = match_tolerance_bins * local_width
        if resolution_mev is not None:
            tol = max(tol, float(resolution_mev))

        best_peak = None
        best_dist = None
        for peak in detected_peaks:
            if peak["bin_index"] in matched_peak_bins:
                continue
            dist = abs(peak["energy_mev"] - E_c)
            if dist <= tol and (best_dist is None or dist < best_dist):
                best_peak, best_dist = peak, dist

        if best_peak is not None:
            matched_peak_bins.add(best_peak["bin_index"])
            matched_lines.append({
                "label": cand["label"],
                "origin": cand["origin"],
                "candidate_energy_mev": E_c,
                "detected_energy_mev": best_peak["energy_mev"],
                "bin_index": best_peak["bin_index"],
                "count": best_peak["count"],
                "delta_mev": best_peak["energy_mev"] - E_c,
            })
        else:
            unmatched_candidates.append({
                "label": cand["label"],
                "origin": cand["origin"],
                "candidate_energy_mev": E_c,
            })

    unmatched_detected_peaks = [
        peak for peak in detected_peaks if peak["bin_index"] not in matched_peak_bins
    ]

    return {
        "n_events": int(E.size),
        "energy_bins": [float(e) for e in edges],
        "binning_mode": used_mode,
        "prominence_factor": float(prominence_factor),
        "window": int(window),
        "match_tolerance_bins": float(match_tolerance_bins),
        "resolution_mev": (float(resolution_mev) if resolution_mev is not None else None),
        "detected_peaks": detected_peaks,
        "matched_lines": matched_lines,
        "unmatched_detected_peaks": unmatched_detected_peaks,
        "unmatched_candidates": unmatched_candidates,
        "n_candidate_lines": len(candidate_lines),
        "n_detected_peaks": len(detected_peaks),
        "n_matched": len(matched_lines),
    }


def print_detected_energy_lines(result):
    """
    Print a compact summary of a detect_energy_lines() result.
    """
    print("\n--- Energy line detection ---")
    print(
        "events:", result["n_events"],
        " bins:", len(result["energy_bins"]) - 1,
        " mode:", result["binning_mode"],
    )
    print(
        "candidate lines:", result["n_candidate_lines"],
        " detected peaks:", result["n_detected_peaks"],
        " matched:", result["n_matched"],
    )

    print("\nMatched lines:")
    for m in result["matched_lines"]:
        print(
            f"  {m['label']:20s} ({m['origin']:22s}) "
            f"expected={m['candidate_energy_mev']:.6f} MeV "
            f"detected={m['detected_energy_mev']:.6f} MeV "
            f"delta={m['delta_mev']:+.6f} MeV  count={m['count']:.0f}"
        )

    print("\nExpected-but-undetected candidates (likely Compton-smeared / below prominence):")
    for c in result["unmatched_candidates"]:
        print(
            f"  {c['label']:20s} ({c['origin']:22s}) "
            f"expected={c['candidate_energy_mev']:.6f} MeV"
        )

    print("\nUnmatched detected peaks (possible unknown/uncatalogued lines):")
    for p in result["unmatched_detected_peaks"]:
        print(f"  bin={p['bin_index']:5d}  E={p['energy_mev']:.6f} MeV  count={p['count']:.0f}")


def build_gate_targets(
    E,
    energy_bins,
    matched_lines,
    match_tolerance_bins=2.0,
    resolution_mev=None,
    continuum_floor=0.15,
    max_bandwidth_frac_of_spacing=0.5,
):
    """
    Soft per-event target distribution over {continuum, line_1..line_L},
    based on each event's physical proximity (in MeV) to the known matched
    line positions - NOT on anything the model itself learns.

    This exists to fix a real degeneracy in the v0.8 mixture energy head
    (CVAE_MixEnergy_ContPhi_TaskAdaptive): since the true energy is fed into
    the encoder (as part of y_cont) and reconstructed by the decoder, the
    encoder can leak "which line this event is near" into the latent z, and
    the decoder's per-sample continuum mean can just chase that value -
    reconstructing well without the discrete gate ever learning to route
    events to the fixed line components. Supervising the gate directly from
    physical proximity (independent of z) breaks that degeneracy.

    Parameters
    ----------
    E : array-like
        Physical energy values in MeV (real training data, NOT generated).
    energy_bins : array-like
        Bin edges, used only to compute the local bin width for the
        per-line Gaussian bandwidth - same convention as
        detect_energy_lines's matching tolerance.
    matched_lines : list of dict
        detect_energy_lines(...)["matched_lines"] entries (must have
        "candidate_energy_mev"), in the same order used to build the
        model's line_positions_y.
    match_tolerance_bins : float
        Gaussian bandwidth per line, as a multiple of the local bin width
        at that line's energy.
    resolution_mev : float or None
        If given, widens the bandwidth to at least this value.
    continuum_floor : float
        Minimum unnormalized weight given to the continuum slot for every
        event, so events far from every line get a target close to
        all-continuum rather than an arbitrary near-zero-everywhere vector.
        Also controls how strongly the auxiliary gate loss pulls toward the
        line components vs. continuum - the default (0.15, paired with
        CVAE_MixEnergy_ContPhi_TaskAdaptive's default w_gate_aux=0.3) was
        tuned against a synthetic continuum+lines dataset with known
        mixture weights (see the v0.8 verification notebook cells); a
        smaller floor over-weights the lines relative to their true rate.
    max_bandwidth_frac_of_spacing : float or None
        Caps each line's bandwidth at this fraction of the distance to its
        nearest OTHER matched line, so two close lines can't bleed gate
        weight onto each other. Without this, `match_tolerance_bins *
        local_bin_width` can exceed the line spacing outright in sparse
        regions with fixed-count binning - confirmed on real CryoSphere-CR
        data, where Ni K-beta (8.265 keV) and Cu K-beta (8.905 keV) are
        0.032 dex apart but the uncapped bandwidth there was ~0.034 dex,
        i.e. wider than the gap itself, so the far-more-abundant Ni K-beta
        (13,385 real events) bled substantial gate_target weight onto Cu
        K-beta's slot (1,915 real events), training the model to
        over-generate the rarer line ~10x. Set to None to restore the old
        unbounded behavior. No effect when only one line is matched.

    Returns
    -------
    np.ndarray
        Shape (len(E), 1 + len(matched_lines)), float32, each row summing
        to 1 - column 0 is the continuum target, columns 1..L are the line
        targets, in matched_lines order.
    """
    E = np.asarray(E, dtype=np.float64)
    edges = np.asarray(energy_bins, dtype=np.float64)
    n = E.size
    n_lines = len(matched_lines)

    line_positions = np.array(
        [float(m["candidate_energy_mev"]) for m in matched_lines], dtype=np.float64
    )

    weights = np.zeros((n, n_lines), dtype=np.float64)
    for l, line in enumerate(matched_lines):
        E_c = line_positions[l]
        bin_i = int(np.clip(np.searchsorted(edges, E_c) - 1, 0, len(edges) - 2))
        local_width = float(edges[bin_i + 1] - edges[bin_i])
        bandwidth = match_tolerance_bins * local_width
        if resolution_mev is not None:
            bandwidth = max(bandwidth, float(resolution_mev))

        if max_bandwidth_frac_of_spacing is not None and n_lines > 1:
            nearest_spacing = np.min(np.abs(np.delete(line_positions, l) - E_c))
            bandwidth = min(bandwidth, max_bandwidth_frac_of_spacing * nearest_spacing)

        weights[:, l] = np.exp(-0.5 * ((E - E_c) / bandwidth) ** 2)

    continuum_w = np.full(n, float(continuum_floor))
    total = continuum_w + weights.sum(axis=1)

    targets = np.empty((n, n_lines + 1), dtype=np.float32)
    targets[:, 0] = (continuum_w / total).astype(np.float32)
    targets[:, 1:] = (weights / total[:, None]).astype(np.float32)

    return targets


def _fit_quantile_column(
    values,
    n_quantiles=10000,
    random_state=42,
    output_distribution="normal",
):
    """
    Fit a QuantileTransformer to one 1D feature array.

    Returns
    -------
    transformed : np.ndarray
        Transformed values as float32.

    transformer : sklearn.preprocessing.QuantileTransformer
        Fitted transformer.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)

    qt = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=min(int(n_quantiles), len(values)),
        output_distribution=output_distribution,
        random_state=random_state,
    )

    transformed = qt.fit_transform(values).reshape(-1).astype(np.float32)

    return transformed, qt


def transform_quantile_values(transformer, values):
    """
    Map new scalar values through an already-fitted QuantileTransformer
    (transform-only, no re-fit) - e.g. mapping a handful of physical line
    energies into an existing energy y-space fitted on the full spectrum.

    Parameters
    ----------
    transformer : sklearn.preprocessing.QuantileTransformer
        A transformer previously returned by _fit_quantile_column.
    values : array-like
        New values to map through the transform.

    Returns
    -------
    np.ndarray
        Transformed values as float32.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return transformer.transform(values).reshape(-1).astype(np.float32)


def build_feature_dataframe(
    prep,
    energy_binning_mode="log_fixed_count",
    e_min_cut=None,
    e_max_cut=None,
    bin_width=0.5,
    n_bins=512,
    min_counts=20,
    geometry_transform="arctanh_uv_discrete",
    n_quantiles=10000,
    random_state=42,
    energy_transform="none",
    energy_n_quantiles=10000,
    energy_random_state=42,
):
    """
    Build the final feature dataframe used before train/val/test splitting.

    Parameters
    ----------
    prep : dict
        Output of build_physical_features().

    energy_binning_mode : str
        Energy binning mode:
        - "fixed_width"
        - "min_counts"
        - "log_fixed_count"

    e_min_cut : float or None
        Optional lower energy cut in MeV.

    e_max_cut : float or None
        Optional upper energy cut in MeV.

    bin_width : float
        Bin width for fixed-width energy binning.

    n_bins : int
        Number of bins for fixed-count/log binning.

    min_counts : int
        Minimum counts per bin for min-counts mode.

    geometry_transform : str
        Geometry preprocessing mode.

        Supported:
        - "arctanh_uv_discrete":
            legacy mode.
            Uses s_r = arctanh(u_r), keeps u_v raw for later discretization.

        - "quantile_u_r_u_v":
            v0.7 mode.
            Uses QuantileTransformer on u_r and u_v.
            Continuous variables:
            u_r_q, u_v_q, cphi_r, sphi_r, cphi_v, sphi_v

        - "quantile_u_r_u_v_phi_r_phi_v":
            v0.7.2 mode.
            Uses QuantileTransformer on u_r, u_v, phi_r, phi_v.
            Continuous variables:
            u_r_q, u_v_q, phi_r_q, phi_v_q

    n_quantiles : int
        Number of quantiles for QuantileTransformer.

    random_state : int
        Random state for QuantileTransformer.

    energy_transform : str
        Continuum energy transform for the v0.8 mixture-density energy head
        (unused by every other model version - default "none" leaves `feat`
        byte-identical to prior behavior). One of:
        - "none": no energy_y column is built (default).
        - "log10": energy_y = log10(Energy). No fitted transformer.
        - "quantile": fit a QuantileTransformer on the full Energy array
            (via _fit_quantile_column), add "qt_energy" to the returned
            quantile_transformers dict. Fit on all real energies (not a
            continuum-only subset - there is no per-event line/continuum
            label to split on).

    energy_n_quantiles, energy_random_state :
        Passed to _fit_quantile_column when energy_transform == "quantile".

    Returns
    -------
    dict
        Dictionary containing:
        - feat : pd.DataFrame
        - energy_bins : np.ndarray
        - n_energy_bins : int
        - mask_energy : np.ndarray
        - filtered_prep : dict
        - geometry_transform : str
        - quantile_transformers : dict
        - geometry_metadata : dict
        - energy_transform : str
    """
    df = prep["dataframe"].copy()
    feat0 = prep["features"].copy()

    E = feat0["Energy"].to_numpy(dtype=np.float64)

    mask_energy = np.isfinite(E)

    if e_min_cut is not None:
        mask_energy &= (E >= e_min_cut)

    if e_max_cut is not None:
        mask_energy &= (E <= e_max_cut)

    df = df.loc[mask_energy].reset_index(drop=True)
    feat0 = feat0.loc[mask_energy].reset_index(drop=True)

    E = feat0["Energy"].to_numpy(dtype=np.float64)

    energy_bins = build_energy_bins(
        E,
        mode=energy_binning_mode,
        bin_width=bin_width,
        n_bins=n_bins,
        min_counts=min_counts,
    )

    energy_idx = np.digitize(E, energy_bins) - 1
    energy_idx = np.clip(energy_idx, 0, len(energy_bins) - 2)

    n_energy_bins = len(energy_bins) - 1

    geometry_transform = str(geometry_transform)

    quantile_transformers = {}

    geometry_metadata = {
        "geometry_transform": geometry_transform,
    }

    if geometry_transform == "arctanh_uv_discrete":
        geom_cols = {
            "s_r": feat0["s_r"].to_numpy(dtype=np.float32),
            "u_v": feat0["u_v"].to_numpy(dtype=np.float32),
        }

        geometry_metadata.update({
            "X_cont_cols": [
                "s_r",
                "cphi_r",
                "sphi_r",
                "cphi_v",
                "sphi_v",
            ],
            "u_v_mode": "raw_for_discretization",
        })

    elif geometry_transform == "quantile_u_r_u_v":
        u_r_q, qt_u_r = _fit_quantile_column(
            feat0["u_r"].to_numpy(dtype=np.float64),
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        u_v_q, qt_u_v = _fit_quantile_column(
            feat0["u_v"].to_numpy(dtype=np.float64),
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        geom_cols = {
            "u_r_q": u_r_q,
            "u_v_q": u_v_q,
        }

        quantile_transformers = {
            "qt_u_r": qt_u_r,
            "qt_u_v": qt_u_v,
        }

        geometry_metadata.update({
            "quantile_output_distribution": "normal",
            "quantile_n_quantiles": min(int(n_quantiles), len(feat0)),
            "random_state": random_state,
            "X_cont_cols": [
                "u_r_q",
                "u_v_q",
                "cphi_r",
                "sphi_r",
                "cphi_v",
                "sphi_v",
            ],
        })

    elif geometry_transform == "quantile_u_r_u_v_phi_r_phi_v":
        u_r_q, qt_u_r = _fit_quantile_column(
            feat0["u_r"].to_numpy(dtype=np.float64),
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        u_v_q, qt_u_v = _fit_quantile_column(
            feat0["u_v"].to_numpy(dtype=np.float64),
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        phi_r_q, qt_phi_r = _fit_quantile_column(
            feat0["phi_r"].to_numpy(dtype=np.float64),
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        phi_v_q, qt_phi_v = _fit_quantile_column(
            feat0["phi_v"].to_numpy(dtype=np.float64),
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        geom_cols = {
            "u_r_q": u_r_q,
            "u_v_q": u_v_q,
            "phi_r_q": phi_r_q,
            "phi_v_q": phi_v_q,
        }

        quantile_transformers = {
            "qt_u_r": qt_u_r,
            "qt_u_v": qt_u_v,
            "qt_phi_r": qt_phi_r,
            "qt_phi_v": qt_phi_v,
        }

        geometry_metadata.update({
            "quantile_output_distribution": "normal",
            "quantile_n_quantiles": min(int(n_quantiles), len(feat0)),
            "random_state": random_state,
            "phi_domain": "[-pi, pi]",
            "phi_periodic": True,
            "phi_shift_applied": False,
            "X_cont_cols": [
                "u_r_q",
                "u_v_q",
                "phi_r_q",
                "phi_v_q",
            ],
        })

    else:
        raise ValueError(
            "Invalid geometry_transform. Use "
            "'arctanh_uv_discrete', 'quantile_u_r_u_v', or 'quantile_u_r_u_v_phi_r_phi_v'."
        )

    energy_cols = {}

    if energy_transform == "none":
        pass

    elif energy_transform == "log10":
        energy_cols["energy_y"] = np.log10(E).astype(np.float32)

    elif energy_transform == "quantile":
        energy_y, qt_energy = _fit_quantile_column(
            E,
            n_quantiles=energy_n_quantiles,
            random_state=energy_random_state,
            output_distribution="normal",
        )
        energy_cols["energy_y"] = energy_y
        quantile_transformers["qt_energy"] = qt_energy

    else:
        raise ValueError(
            "Invalid energy_transform. Use 'none', 'log10', or 'quantile'."
        )

    feat_dict = {
        "ParticleName": feat0["ParticleName"].astype(str).to_numpy(),
        "E_idx": energy_idx.astype(np.int32),
    }

    if geometry_transform in [
        "arctanh_uv_discrete",
        "quantile_u_r_u_v",
    ]:
        feat_dict.update({
            "cphi_r": feat0["cphi_r"].to_numpy(dtype=np.float32),
            "sphi_r": feat0["sphi_r"].to_numpy(dtype=np.float32),
            "cphi_v": feat0["cphi_v"].to_numpy(dtype=np.float32),
            "sphi_v": feat0["sphi_v"].to_numpy(dtype=np.float32),
        })

    feat_dict.update(geom_cols)
    feat_dict.update(energy_cols)

    if "PrimBool" in feat0.columns:
        feat_dict["PrimBool"] = feat0["PrimBool"].to_numpy()

    normalization = prep.get("normalization")
    primary_col = normalization.get("primary_col") if normalization else None
    if primary_col and primary_col in feat0.columns and primary_col not in feat_dict:
        feat_dict[primary_col] = feat0[primary_col].to_numpy()

    feat = pd.DataFrame(feat_dict)

    filtered_prep = {
        **prep,
        "dataframe": df,
        "features": feat0,
    }

    return {
        "feat": feat,
        "energy_bins": energy_bins,
        "n_energy_bins": n_energy_bins,
        "mask_energy": mask_energy,
        "filtered_prep": filtered_prep,
        "geometry_transform": geometry_transform,
        "quantile_transformers": quantile_transformers,
        "geometry_metadata": geometry_metadata,
        "energy_transform": energy_transform,
        "normalization": normalization,
        "energy_config": {
            "mode": energy_binning_mode,
            "e_min_cut": e_min_cut,
            "e_max_cut": e_max_cut,
            "bin_width": bin_width,
            "n_bins": n_bins,
            "min_counts": min_counts,
        },
    }


def report_feature_dataframe(feature_pack):
    """
    Print checks and summaries for the final feature dataframe.
    """
    feat = feature_pack["feat"]
    energy_bins = feature_pack["energy_bins"]
    geometry_transform = feature_pack.get("geometry_transform", "unknown")

    print("\nfeat shape:", feat.shape)
    print("geometry_transform:", geometry_transform)

    print("\nFirst rows:")
    print(feat.head())

    print("\n--- Finite check ---")

    numeric_cols = [
        c for c in feat.columns
        if c not in ["ParticleName"]
    ]

    arr = feat[numeric_cols].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(arr).all(axis=1)
    print("Rows with NaN/Inf:", np.sum(bad))

    print("\n--- Feature ranges ---")
    for c in numeric_cols:
        print(f"{c:>10s} : {feat[c].min()}  {feat[c].max()}")

    if {"cphi_r", "sphi_r", "cphi_v", "sphi_v"}.issubset(feat.columns):
        print("\n--- Unit circle checks ---")
        cr = np.sqrt(
            feat["cphi_r"].to_numpy()**2
            + feat["sphi_r"].to_numpy()**2
        )
        cv = np.sqrt(
            feat["cphi_v"].to_numpy()**2
            + feat["sphi_v"].to_numpy()**2
        )

        print(
            "position: mean =", cr.mean(),
            "std =", cr.std(),
            "min =", cr.min(),
            "max =", cr.max(),
        )
        print(
            "direction: mean =", cv.mean(),
            "std =", cv.std(),
            "min =", cv.min(),
            "max =", cv.max(),
        )

    if "u_v" in feat.columns:
        print("\n--- u_v checks ---")
        print("u_v: min =", feat["u_v"].min(), "max =", feat["u_v"].max())
        print("u_v mean =", feat["u_v"].mean(), "std =", feat["u_v"].std())
        print("fraction u_v > 0 =", np.mean(feat["u_v"] > 0))

    q_cols = [c for c in ["u_r_q", "u_v_q", "phi_r_q", "phi_v_q"] if c in feat.columns]

    if len(q_cols) > 0:
        print("\n--- Quantile geometry checks ---")
        for c in q_cols:
            print(
                f"{c}: mean = {feat[c].mean():.6f}, "
                f"std = {feat[c].std():.6f}, "
                f"min = {feat[c].min():.6f}, "
                f"max = {feat[c].max():.6f}"
            )

    print("\n--- Energy bin population ---")
    hist_counts, _ = np.histogram(
        feature_pack["filtered_prep"]["features"]["Energy"].to_numpy(dtype=np.float64),
        bins=energy_bins,
    )
    print("Min counts per bin:", hist_counts.min())
    print("Max counts per bin:", hist_counts.max())
    print("Mean counts per bin:", hist_counts.mean())
    print("Empty bins:", np.sum(hist_counts == 0))


def fit_quantile_geometry_transforms(
    feat,
    derived=None,
    n_quantiles=10000,
    random_state=42,
    include_phi=False,
):
    """
    Standalone helper for exploratory tests.

    Prefer using build_feature_dataframe(...) for production preprocessing.
    """
    feat = feat.copy()

    if "u_r" in feat.columns:
        u_r = np.asarray(feat["u_r"], dtype=np.float64)
    elif derived is not None and "u_r" in derived:
        u_r = np.asarray(derived["u_r"], dtype=np.float64)
    else:
        raise ValueError("u_r must be present in feat or derived.")

    if "u_v" in feat.columns:
        u_v = np.asarray(feat["u_v"], dtype=np.float64)
    elif derived is not None and "u_v" in derived:
        u_v = np.asarray(derived["u_v"], dtype=np.float64)
    else:
        raise ValueError("u_v must be present in feat or derived.")

    u_r_q, qt_u_r = _fit_quantile_column(
        u_r,
        n_quantiles=n_quantiles,
        random_state=random_state,
        output_distribution="normal",
    )

    u_v_q, qt_u_v = _fit_quantile_column(
        u_v,
        n_quantiles=n_quantiles,
        random_state=random_state,
        output_distribution="normal",
    )

    feat["u_r_q"] = u_r_q
    feat["u_v_q"] = u_v_q

    transformers = {
        "qt_u_r": qt_u_r,
        "qt_u_v": qt_u_v,
    }

    x_cont_cols = [
        "u_r_q",
        "u_v_q",
        "cphi_r",
        "sphi_r",
        "cphi_v",
        "sphi_v",
    ]

    geometry_transform = "quantile_u_r_u_v"

    if include_phi:
        if "phi_r" in feat.columns:
            phi_r = np.asarray(feat["phi_r"], dtype=np.float64)
        elif derived is not None and "phi_r" in derived:
            phi_r = np.asarray(derived["phi_r"], dtype=np.float64)
        else:
            raise ValueError("phi_r must be present in feat or derived.")

        if "phi_v" in feat.columns:
            phi_v = np.asarray(feat["phi_v"], dtype=np.float64)
        elif derived is not None and "phi_v" in derived:
            phi_v = np.asarray(derived["phi_v"], dtype=np.float64)
        else:
            raise ValueError("phi_v must be present in feat or derived.")

        phi_r_q, qt_phi_r = _fit_quantile_column(
            phi_r,
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        phi_v_q, qt_phi_v = _fit_quantile_column(
            phi_v,
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        feat["phi_r_q"] = phi_r_q
        feat["phi_v_q"] = phi_v_q

        transformers["qt_phi_r"] = qt_phi_r
        transformers["qt_phi_v"] = qt_phi_v

        x_cont_cols = [
            "u_r_q",
            "u_v_q",
            "phi_r_q",
            "phi_v_q",
        ]

        geometry_transform = "quantile_u_r_u_v_phi_r_phi_v"

    metadata = {
        "geometry_transform": geometry_transform,
        "quantile_output_distribution": "normal",
        "quantile_n_quantiles": min(int(n_quantiles), len(feat)),
        "random_state": random_state,
        "phi_domain": "[-pi, pi]",
        "phi_periodic": bool(include_phi),
        "phi_shift_applied": False,
        "X_cont_cols": x_cont_cols,
    }

    return feat, transformers, metadata