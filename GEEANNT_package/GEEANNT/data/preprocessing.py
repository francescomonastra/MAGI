"""
Preprocessing and physical feature construction for GEEANNT.
"""

import numpy as np
import pandas as pd


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
        Sphere center.
    radius : float
        Sphere radius.
    eps : float
        Numerical stability constant.
    drop_invalid_energy : bool
        If True, remove rows with E <= 0 or non-finite energy.

    Returns
    -------
    dict
        Dictionary containing:
        - cleaned dataframe
        - raw/projected coordinates
        - normalized directions
        - engineered features
        - diagnostic arrays
    """
    C = np.asarray(center, dtype=np.float64)
    R = float(radius)

    df = df.copy()

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
        "s_r": s_r.astype(np.float64),
        "cphi_r": cphi_r.astype(np.float64),
        "sphi_r": sphi_r.astype(np.float64),
        "u_v": u_v.astype(np.float64),
        "cphi_v": cphi_v.astype(np.float64),
        "sphi_v": sphi_v.astype(np.float64),
    })

    return {
        "dataframe": df,
        "features": features,
        "center": C,
        "radius": R,
        "raw": {
            "x": x, "y": y, "z": z,
            "vx": vx, "vy": vy, "vz": vz,
            "r": r,
            "vnorm": vnorm,
        },
        "projected": {
            "x": x_proj, "y": y_proj, "z": z_proj,
            "r": r_proj,
        },
        "normalized_direction": {
            "vx": vx_hat, "vy": vy_hat, "vz": vz_hat,
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
    print("r: mean =", raw["r"].mean(), "std =", raw["r"].std(),
          "min =", raw["r"].min(), "max =", raw["r"].max())
    print("Mean absolute deviation from R:", np.mean(np.abs(raw["r"] - R)))

    print("\n--- RAW direction norm check ---")
    print("||v||: mean =", raw["vnorm"].mean(), "std =", raw["vnorm"].std(),
          "min =", raw["vnorm"].min(), "max =", raw["vnorm"].max())
    print("Mean absolute deviation from 1:", np.mean(np.abs(raw["vnorm"] - 1.0)))

    print("\n--- PROJECTED sphere check ---")
    print("r_proj: mean =", proj["r"].mean(), "std =", proj["r"].std(),
          "min =", proj["r"].min(), "max =", proj["r"].max())
    print("Mean absolute deviation from R:", np.mean(np.abs(proj["r"] - R)))

    print("\n--- RENORMALIZED direction check ---")
    print("||vhat||: mean =", normdir["vnorm"].mean(), "std =", normdir["vnorm"].std(),
          "min =", normdir["vnorm"].min(), "max =", normdir["vnorm"].max())
    print("Mean absolute deviation from 1:", np.mean(np.abs(normdir["vnorm"] - 1.0)))

    print("\n--- Energy check ---")
    print("E: min =", feat["Energy"].min(), "max =", feat["Energy"].max(),
          "#E<=0 =", np.sum(feat["Energy"] <= 0))

    print("\n--- Check u_v ---")
    print("u_v: min =", feat["u_v"].min(),
          "max =", feat["u_v"].max(),
          "mean =", feat["u_v"].mean(),
          "std =", feat["u_v"].std())
    print("fraction u_v > 0 =", np.mean(feat["u_v"] > 0))


def build_energy_bins(E, mode, bin_width=0.5, n_bins=512, min_counts=20):
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
        raise ValueError("Invalid energy binning mode.")

    return bins


def build_feature_dataframe(
    prep,
    energy_binning_mode="log_fixed_count",
    e_min_cut=None,
    e_max_cut=None,
    bin_width=0.5,
    n_bins=512,
    min_counts=20,
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

    Returns
    -------
    dict
        Dictionary containing:
        - feat : pd.DataFrame
        - energy_bins : np.ndarray
        - n_energy_bins : int
        - mask_energy : np.ndarray
        - filtered_prep : dict
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

    feat = pd.DataFrame({
        "ParticleName": feat0["ParticleName"].astype(str).to_numpy(),
        "E_idx": energy_idx.astype(np.int32),
        "s_r": feat0["s_r"].to_numpy(dtype=np.float32),
        "cphi_r": feat0["cphi_r"].to_numpy(dtype=np.float32),
        "sphi_r": feat0["sphi_r"].to_numpy(dtype=np.float32),
        "u_v": feat0["u_v"].to_numpy(dtype=np.float32),
        "cphi_v": feat0["cphi_v"].to_numpy(dtype=np.float32),
        "sphi_v": feat0["sphi_v"].to_numpy(dtype=np.float32),
    })

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

    print("\nfeat shape:", feat.shape)
    print("\nFirst rows:")
    print(feat.head())

    print("\n--- Finite check ---")
    arr = feat[["s_r", "cphi_r", "sphi_r", "u_v", "cphi_v", "sphi_v"]].to_numpy()
    bad = ~np.isfinite(arr).all(axis=1)
    print("Rows with NaN/Inf:", np.sum(bad))

    print("\n--- Feature ranges ---")
    for c in ["s_r", "cphi_r", "sphi_r", "u_v", "cphi_v", "sphi_v"]:
        print(f"{c:>8s} : {feat[c].min()}  {feat[c].max()}")

    print("\n--- Unit circle checks ---")
    cr = np.sqrt(feat["cphi_r"].to_numpy()**2 + feat["sphi_r"].to_numpy()**2)
    cv = np.sqrt(feat["cphi_v"].to_numpy()**2 + feat["sphi_v"].to_numpy()**2)

    print("position: mean =", cr.mean(), "std =", cr.std(), "min =", cr.min(), "max =", cr.max())
    print("direction: mean =", cv.mean(), "std =", cv.std(), "min =", cv.min(), "max =", cv.max())

    print("\n--- u_v checks ---")
    print("u_v: min =", feat["u_v"].min(), "max =", feat["u_v"].max())
    print("u_v mean =", feat["u_v"].mean(), "std =", feat["u_v"].std())
    print("fraction u_v > 0 =", np.mean(feat["u_v"] > 0))

    print("\n--- Energy bin population ---")
    hist_counts, _ = np.histogram(
        feature_pack["filtered_prep"]["features"]["Energy"].to_numpy(dtype=np.float64),
        bins=energy_bins
    )
    print("Min counts per bin:", hist_counts.min())
    print("Max counts per bin:", hist_counts.max())
    print("Mean counts per bin:", hist_counts.mean())
    print("Empty bins:", np.sum(hist_counts == 0))