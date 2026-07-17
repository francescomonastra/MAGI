"""
Preprocessing and physical feature construction for GEEANNT.
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing


def compute_primary_fraction(df, primary_col="PrimBool"):
    """
    Count primaries vs. total CryoSphere crossings for flux normalization.

    Returns None if `primary_col` is not present (legacy 9-column data),
    so downstream code can treat "no normalization info" uniformly.
    """
    if primary_col not in df.columns:
        return None

    is_prim = df[primary_col].to_numpy()
    n_generated = int(len(is_prim))
    n_primaries = int(np.sum(is_prim == 1))

    per_species = {}
    if "ParticleName" in df.columns:
        for name, sub in df.groupby("ParticleName"):
            tot = int(len(sub))
            pri = int(np.sum(sub[primary_col].to_numpy() == 1))
            per_species[str(name)] = {
                "n_generated": tot,
                "n_primaries": pri,
                "primary_fraction": (pri / tot if tot else None),
            }

    return {
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
):
    """
    Build the base physical feature representation used by GEEANNT.

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

    Returns
    -------
    dict
        Dictionary containing cleaned dataframe, physical features,
        projected coordinates, normalized directions and diagnostics.
    """
    C = np.asarray(center, dtype=np.float64)
    R = float(radius)

    df = df.copy()

    # Computed before any row filtering so n_generated matches the raw
    # crossing count (NCryoSphereCR in the notebook), None if PrimBool absent.
    normalization = compute_primary_fraction(df)

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

    if "PrimBool" in feat0.columns:
        feat_dict["PrimBool"] = feat0["PrimBool"].to_numpy()

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
        "normalization": prep.get("normalization"),
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