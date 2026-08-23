"""
Dataset construction utilities for MAGI.

This module supports three geometry modes:

1. Legacy discrete-u_v mode:
   - s_r continuous
   - u_v discretized into quantile bins
   - continuous columns:
       s_r, cphi_r, sphi_r, cphi_v, sphi_v

2. v0.7 continuous-geometry mode:
   - u_r_q continuous
   - u_v_q continuous
   - phi encoded as cos/sin
   - continuous columns:
       u_r_q, u_v_q, cphi_r, sphi_r, cphi_v, sphi_v

3. v0.7.2 continuous-phi mode:
   - u_r_q continuous
   - u_v_q continuous
   - phi_r_q continuous
   - phi_v_q continuous
   - continuous columns:
       u_r_q, u_v_q, phi_r_q, phi_v_q
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Legacy pipeline: particle filtering + u_v discretization
# =============================================================================

def filter_particle_types_and_discretize_uv(
    feat,
    prob_threshold=1e-5,
    uv_nbins=128,
    uv_eps=1e-6,
    normalization=None,
):
    """
    Legacy dataset builder.

    Filter rare particle types and discretize u_v.

    Expected columns:
        ParticleName, E_idx, s_r, cphi_r, sphi_r, u_v, cphi_v, sphi_v

    This is the v0.6 path, kept for reloading old runs. New work should use
    filter_particle_types_continuous_geometry.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature frame from build_feature_dataframe, in the legacy
        discrete-u_v column layout listed above.

    prob_threshold : float
        Drop any ParticleName whose population fraction is below this. Rare
        species cannot be conditioned on reliably and destabilize the type
        one-hot; 1e-5 is the notebook default.

    uv_nbins : int
        Requested number of u_v bins. The effective count
        ("uv_nbins_eff" in the result) can be lower if the data does not
        populate them all.

    uv_eps : float
        Margin used to keep u_v strictly inside [-1, 1] before binning,
        avoiding an edge bin that only ever receives boundary values.

    normalization : dict or None
        Dataset-level primary/secondary crossing counts from
        build_feature_dataframe(...)["normalization"], carried through
        unchanged (not a per-row model input) so it survives to
        checkpointing metadata.

    Returns
    -------
    dict
        Dataset pack with "dataset_mode" == "discrete_uv" and keys:
        "feat", "cont_cols", "X_cont_raw", "E_idx", "u_v_idx", "y_type",
        "type_names", "type_to_idx", "idx_to_type", "type_probs", "n_types",
        "u_v_bins", "u_v_centers", "u_v_values", "uv_nbins_requested",
        "uv_nbins_eff", "normalization". Pass it to split_feature_data.
    """
    feat = feat.copy()

    type_counts_full = feat["ParticleName"].value_counts()
    type_probs_full = type_counts_full / type_counts_full.sum()

    valid_types = type_probs_full[type_probs_full >= prob_threshold].index
    feat = feat[feat["ParticleName"].isin(valid_types)].reset_index(drop=True)

    type_counts = feat["ParticleName"].value_counts()
    type_names = type_counts.index.to_list()
    n_types = len(type_names)

    type_to_idx = {t: i for i, t in enumerate(type_names)}
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    type_probs = (type_counts / type_counts.sum()).to_numpy(dtype=np.float64)

    u_v_values = feat["u_v"].to_numpy(dtype=np.float64)

    u_v_bins = np.quantile(u_v_values, np.linspace(0.0, 1.0, uv_nbins + 1))
    u_v_bins[0] = min(u_v_bins[0], u_v_values.min()) - uv_eps
    u_v_bins[-1] = max(u_v_bins[-1], u_v_values.max()) + uv_eps
    u_v_bins = np.unique(u_v_bins)

    uv_nbins_eff = len(u_v_bins) - 1
    if uv_nbins_eff < 2:
        raise ValueError("u_v binning collapsed to fewer than 2 bins.")

    u_v_idx = np.digitize(u_v_values, u_v_bins) - 1
    u_v_idx = np.clip(u_v_idx, 0, uv_nbins_eff - 1).astype(np.int32)

    u_v_centers = 0.5 * (u_v_bins[:-1] + u_v_bins[1:])

    cont_cols = ["s_r", "cphi_r", "sphi_r", "cphi_v", "sphi_v"]
    X_cont_raw = feat[cont_cols].to_numpy(dtype=np.float32)

    E_idx = feat["E_idx"].to_numpy(dtype=np.int32)
    y_type = feat["ParticleName"].map(type_to_idx).to_numpy(dtype=np.int32)

    return {
        "dataset_mode": "discrete_uv",
        "feat": feat,
        "cont_cols": cont_cols,
        "X_cont_raw": X_cont_raw,
        "E_idx": E_idx,
        "u_v_idx": u_v_idx,
        "y_type": y_type,
        "type_names": type_names,
        "type_to_idx": type_to_idx,
        "idx_to_type": idx_to_type,
        "type_probs": type_probs,
        "n_types": n_types,
        "u_v_bins": u_v_bins,
        "u_v_centers": u_v_centers,
        "u_v_values": u_v_values,
        "uv_nbins_requested": uv_nbins,
        "uv_nbins_eff": uv_nbins_eff,
        "normalization": normalization,
    }


def report_discretized_features(dataset_pack):
    """
    Report diagnostics for the legacy discrete-u_v dataset.

    Printed sanity check only; nothing is returned.

    Parameters
    ----------
    dataset_pack : dict
        Result of filter_particle_types_and_discretize_uv.

    Returns
    -------
    None
    """
    feat = dataset_pack["feat"]
    X_cont_raw = dataset_pack["X_cont_raw"]
    cont_cols = dataset_pack["cont_cols"]
    u_v_bins = dataset_pack["u_v_bins"]
    u_v_idx = dataset_pack["u_v_idx"]
    u_v_values = dataset_pack["u_v_values"]
    u_v_centers = dataset_pack["u_v_centers"]

    print("Dataset mode:", dataset_pack.get("dataset_mode", "discrete_uv"))
    print("Original/kept particle types:", dataset_pack["n_types"])
    print("Types:", dataset_pack["type_names"])

    print("\nEmpirical probabilities:")
    for t, p in zip(dataset_pack["type_names"], dataset_pack["type_probs"]):
        print(f"{t:>12s}  p={p:.6e}")

    print("\n--- u_v binning ---")
    print("Requested bins:", dataset_pack["uv_nbins_requested"])
    print("Effective bins:", dataset_pack["uv_nbins_eff"])
    print("u_v min/max in data:", u_v_values.min(), u_v_values.max())
    print("u_v bin edges min/max:", u_v_bins[0], u_v_bins[-1])

    hist_uv, _ = np.histogram(u_v_values, bins=u_v_bins)
    print("Min counts per bin:", hist_uv.min())
    print("Max counts per bin:", hist_uv.max())
    print("Mean counts per bin:", hist_uv.mean())
    print("Empty bins:", np.sum(hist_uv == 0))

    print("\nShapes:")
    print("X_cont_raw:", X_cont_raw.shape)
    print("E_idx     :", dataset_pack["E_idx"].shape)
    print("u_v_idx   :", u_v_idx.shape)
    print("y_type    :", dataset_pack["y_type"].shape)

    print("\n--- Continuous feature ranges ---")
    for i, c in enumerate(cont_cols):
        print(f"{c:>8s} : {X_cont_raw[:, i].min()}  {X_cont_raw[:, i].max()}")

    print("\n--- Continuous means/std ---")
    for i, c in enumerate(cont_cols):
        print(
            f"{c:>8s} : "
            f"mean={X_cont_raw[:, i].mean(): .6f}  "
            f"std={X_cont_raw[:, i].std(): .6f}"
        )

    u_v_reco = u_v_centers[u_v_idx]
    print("\n--- u_v reconstruction from bins ---")
    print("u_v reco: min =", u_v_reco.min(), "max =", u_v_reco.max())
    print("mean abs diff =", np.mean(np.abs(u_v_reco - u_v_values)))


# =============================================================================
# v0.7 pipeline: particle filtering + fully continuous geometry
# =============================================================================

def filter_particle_types_continuous_geometry(
    feat,
    prob_threshold=1e-5,
    cont_cols=None,
    normalization=None,
):
    """
    Continuous-geometry dataset builder.

    Supports:

    v0.7:
        ParticleName, E_idx,
        u_r_q, u_v_q, cphi_r, sphi_r, cphi_v, sphi_v

    v0.7.2:
        ParticleName, E_idx,
        u_r_q, u_v_q, phi_r_q, phi_v_q

    This is the current path, used by every v0.7+ head including the v0.8
    mixture head. Which of the two layouts a run used is recorded as
    "continuous_geometry_version" and must match at generation time.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature frame from build_feature_dataframe, already carrying the
        quantile-transformed geometry columns from
        fit_quantile_geometry_transforms.

    prob_threshold : float
        Drop any ParticleName whose population fraction is below this.
        1e-5 is the notebook default.

    cont_cols : tuple[str, ...] or None
        Explicit continuous-feature columns, in model input order. If None
        (default) the layout is inferred from which columns are present,
        preferring v0.7.2 (continuous phi) over v0.7 (cos/sin phi).

    normalization : dict or None
        Dataset-level primary/secondary crossing counts from
        build_feature_dataframe(...)["normalization"], carried through
        unchanged (not a per-row model input) so it survives to
        checkpointing metadata.

    Returns
    -------
    dict
        Dataset pack with "dataset_mode" == "continuous_geometry" and keys:
        "continuous_geometry_version", "feat", "cont_cols", "X_cont_raw",
        "E_idx", "y_type", "type_names", "type_to_idx", "idx_to_type",
        "type_probs", "n_types", "normalization". Pass it to
        split_feature_data.

    Raises
    ------
    ValueError
        If neither the v0.7 nor the v0.7.2 column set is present and
        `cont_cols` was not given explicitly.
    """
    feat = feat.copy()

    if cont_cols is None:
        if {"u_r_q", "u_v_q", "phi_r_q", "phi_v_q"}.issubset(feat.columns):
            cont_cols = ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q")
            continuous_geometry_version = "v0.7.2_continuous_phi"
        elif {"u_r_q", "u_v_q", "cphi_r", "sphi_r", "cphi_v", "sphi_v"}.issubset(feat.columns):
            cont_cols = ("u_r_q", "u_v_q", "cphi_r", "sphi_r", "cphi_v", "sphi_v")
            continuous_geometry_version = "v0.7_cos_sin_phi"
        else:
            raise ValueError(
                "Could not infer continuous geometry columns. "
                "Expected either v0.7 columns "
                "('u_r_q','u_v_q','cphi_r','sphi_r','cphi_v','sphi_v') "
                "or v0.7.2 columns "
                "('u_r_q','u_v_q','phi_r_q','phi_v_q')."
            )
    else:
        cont_cols = tuple(cont_cols)
        if set(cont_cols) == {"u_r_q", "u_v_q", "phi_r_q", "phi_v_q"}:
            continuous_geometry_version = "v0.7.2_continuous_phi"
        elif set(cont_cols) == {"u_r_q", "u_v_q", "cphi_r", "sphi_r", "cphi_v", "sphi_v"}:
            continuous_geometry_version = "v0.7_cos_sin_phi"
        else:
            continuous_geometry_version = "custom_continuous_geometry"

    missing = [c for c in ["ParticleName", "E_idx", *cont_cols] if c not in feat.columns]
    if missing:
        raise ValueError(f"Missing required columns for continuous geometry: {missing}")

    type_counts_full = feat["ParticleName"].value_counts()
    type_probs_full = type_counts_full / type_counts_full.sum()

    valid_types = type_probs_full[type_probs_full >= prob_threshold].index
    feat = feat[feat["ParticleName"].isin(valid_types)].reset_index(drop=True)

    type_counts = feat["ParticleName"].value_counts()
    type_names = type_counts.index.to_list()
    n_types = len(type_names)

    type_to_idx = {t: i for i, t in enumerate(type_names)}
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    type_probs = (type_counts / type_counts.sum()).to_numpy(dtype=np.float64)

    X_cont_raw = feat[list(cont_cols)].to_numpy(dtype=np.float32)
    E_idx = feat["E_idx"].to_numpy(dtype=np.int32)
    y_type = feat["ParticleName"].map(type_to_idx).to_numpy(dtype=np.int32)

    return {
        "dataset_mode": "continuous_geometry",
        "continuous_geometry_version": continuous_geometry_version,
        "feat": feat,
        "cont_cols": list(cont_cols),
        "X_cont_raw": X_cont_raw,
        "E_idx": E_idx,
        "y_type": y_type,
        "type_names": type_names,
        "type_to_idx": type_to_idx,
        "idx_to_type": idx_to_type,
        "type_probs": type_probs,
        "n_types": n_types,
        "normalization": normalization,
    }

def report_continuous_geometry_features(dataset_pack):
    """
    Report diagnostics for the v0.7 continuous-geometry dataset.

    Printed sanity check only; nothing is returned.

    Parameters
    ----------
    dataset_pack : dict
        Result of filter_particle_types_continuous_geometry.

    Returns
    -------
    None
    """
    feat = dataset_pack["feat"]
    X_cont_raw = dataset_pack["X_cont_raw"]
    cont_cols = dataset_pack["cont_cols"]

    print("Dataset mode:", dataset_pack.get("dataset_mode", "continuous_geometry"))
    print("Continuous geometry version:", dataset_pack.get("continuous_geometry_version", "unknown"))
    print("Kept particle types:", dataset_pack["n_types"])
    print("Types:", dataset_pack["type_names"])

    print("\nEmpirical probabilities:")
    for t, p in zip(dataset_pack["type_names"], dataset_pack["type_probs"]):
        print(f"{t:>12s}  p={p:.6e}")

    print("\nShapes:")
    print("X_cont_raw:", X_cont_raw.shape)
    print("E_idx     :", dataset_pack["E_idx"].shape)
    print("y_type    :", dataset_pack["y_type"].shape)

    print("\n--- Continuous feature ranges ---")
    for i, c in enumerate(cont_cols):
        x = X_cont_raw[:, i]
        print(f"{c:>8s} : {x.min()}  {x.max()}")

    print("\n--- Continuous means/std ---")
    for i, c in enumerate(cont_cols):
        x = X_cont_raw[:, i]
        print(f"{c:>8s} : mean={x.mean(): .6f}  std={x.std(): .6f}")

    print("\nFirst rows:")
    print(feat.head())


# =============================================================================
# Shared split / scaling / conditioning utilities
# =============================================================================

def split_feature_data(
    dataset_pack,
    test_size_total=0.30,
    val_size_from_temp=0.50,
    random_state=42,
):
    """
    Split raw feature matrices into train/val/test.

    Works for both:
      - dataset_mode == "discrete_uv"
      - dataset_mode == "continuous_geometry"

    The split is stratified on particle type, so every species keeps its
    population fraction in all three splits - the conditioning is a type
    one-hot, and a species missing from validation cannot be monitored.

    Parameters
    ----------
    dataset_pack : dict
        Result of filter_particle_types_continuous_geometry (or the legacy
        filter_particle_types_and_discretize_uv).

    test_size_total : float
        Fraction held out of train, later halved into val and test. The
        default 0.30 gives a 70/15/15 split.

    val_size_from_temp : float
        Fraction of that held-out part that becomes test rather than val.
        0.50 (default) splits it evenly.

    random_state : int
        Seed for both train_test_split calls. Fixing it is what makes a run
        reproducible end-to-end; the multi-seed acceptance harness varies it.

    Returns
    -------
    dict
        `dataset_pack` extended with "idx_train"/"idx_val"/"idx_test",
        "X_cont_{train,val,test}", "E_{train,val,test}",
        "y_{train,val,test}", and - when the source columns exist -
        "E_{train,val,test}_raw" (physical MeV) and, in discrete_uv mode,
        "u_v_{train,val,test}" plus their "_raw" counterparts. Pass it to
        scale_continuous_features.
    """
    X_cont_raw = dataset_pack["X_cont_raw"]
    E_idx = dataset_pack["E_idx"]
    y_type = dataset_pack["y_type"]
    feat = dataset_pack["feat"]

    dataset_mode = dataset_pack.get("dataset_mode", "discrete_uv")

    idx_all = np.arange(len(X_cont_raw))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all,
        y_type,
        test_size=test_size_total,
        random_state=random_state,
        stratify=y_type,
    )

    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=val_size_from_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    split_pack = {
        **dataset_pack,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "X_cont_train": X_cont_raw[idx_train],
        "X_cont_val": X_cont_raw[idx_val],
        "X_cont_test": X_cont_raw[idx_test],
        "E_train": E_idx[idx_train],
        "E_val": E_idx[idx_val],
        "E_test": E_idx[idx_test],
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    if dataset_mode == "discrete_uv":
        u_v_idx = dataset_pack["u_v_idx"]
        split_pack.update({
            "u_v_train": u_v_idx[idx_train],
            "u_v_val": u_v_idx[idx_val],
            "u_v_test": u_v_idx[idx_test],
        })

        if "u_v" in feat.columns:
            u_v_raw = feat["u_v"].to_numpy()
            split_pack.update({
                "u_v_train_raw": u_v_raw[idx_train],
                "u_v_val_raw": u_v_raw[idx_val],
                "u_v_test_raw": u_v_raw[idx_test],
            })

    if "Energy" in feat.columns:
        E_raw = feat["Energy"].to_numpy()
        split_pack["E_train_raw"] = E_raw[idx_train]
        split_pack["E_val_raw"] = E_raw[idx_val]
        split_pack["E_test_raw"] = E_raw[idx_test]

    return split_pack


def frac_per_type(y_idx, n_types):
    """Population fraction of each particle-type index (sums to 1)."""
    c = np.bincount(y_idx, minlength=n_types).astype(np.float64)
    return c / c.sum()


def report_split_summary(split_pack, n_types):
    """Print train/val/test shapes and the per-type fractions in each split.

    Printed sanity check only - confirms the split did not skew the particle-type
    mix, which matters because the conditioning is a type one-hot.

    Parameters
    ----------
    split_pack : dict
        Result of split_feature_data.

    n_types : int
        Number of particle types, i.e. `dataset_pack["n_types"]`.

    Returns
    -------
    None
    """
    print(
        "Train:", split_pack["X_cont_train"].shape,
        "Val:", split_pack["X_cont_val"].shape,
        "Test:", split_pack["X_cont_test"].shape,
    )

    print("\nType distribution check:")
    print("Train:", frac_per_type(split_pack["y_train"], n_types))
    print("Val  :", frac_per_type(split_pack["y_val"], n_types))
    print("Test :", frac_per_type(split_pack["y_test"], n_types))


def scale_continuous_features(split_pack, scale_cols=(0,)):
    """
    Scale selected continuous features using TRAIN only.

    For v0.7 continuous geometry, use:
        scale_cols=()

    because u_r_q and u_v_q are already quantile-normalized.

    The scaler is fit on the training split only and then applied to val and
    test, so no validation statistics leak into training. It is returned in
    the pack and saved with the checkpoint, because generation must invert
    exactly this transform.

    Parameters
    ----------
    split_pack : dict
        Result of split_feature_data.

    scale_cols : tuple[int, ...]
        Column indices of X_cont to standardize. The default (0,) scales only
        the first column, which is s_r in the legacy layout. Pass an empty
        tuple for v0.7+ quantile geometry, where every geometry column is
        already standardized by its own transform.

    Returns
    -------
    dict
        `split_pack` extended with "X_cont_{train,val,test}_s" (float32),
        "scaler" (a fitted sklearn StandardScaler, or None when
        `scale_cols` is empty), "scaler_sr" (alias kept for older notebooks)
        and "scale_cols". When column 0 is scaled, also "s_r_mean" and
        "s_r_std". Pass it to build_conditioning_and_weights.
    """
    scale_cols = tuple(scale_cols)

    if len(scale_cols) == 0:
        return {
            **split_pack,
            "X_cont_train_s": split_pack["X_cont_train"].copy().astype(np.float32),
            "X_cont_val_s": split_pack["X_cont_val"].copy().astype(np.float32),
            "X_cont_test_s": split_pack["X_cont_test"].copy().astype(np.float32),
            "scaler": None,
            "scaler_sr": None,
            "scale_cols": (),
        }

    scaler = StandardScaler()
    scaler.fit(split_pack["X_cont_train"][:, list(scale_cols)])

    def apply_selected_scaler(X_in):
        """Standardize only the `scale_cols` columns, leaving the rest as-is.

        Closes over the scaler fitted on the training split above, so val and
        test go through the identical transform.
        """
        X_out = X_in.copy()
        X_out[:, list(scale_cols)] = scaler.transform(
            X_in[:, list(scale_cols)]
        ).astype(np.float32)
        return X_out

    X_cont_train_s = apply_selected_scaler(split_pack["X_cont_train"])
    X_cont_val_s = apply_selected_scaler(split_pack["X_cont_val"])
    X_cont_test_s = apply_selected_scaler(split_pack["X_cont_test"])

    out = {
        **split_pack,
        "X_cont_train_s": X_cont_train_s,
        "X_cont_val_s": X_cont_val_s,
        "X_cont_test_s": X_cont_test_s,
        "scaler": scaler,
        "scaler_sr": scaler,
        "scale_cols": scale_cols,
    }

    if 0 in scale_cols:
        idx0 = list(scale_cols).index(0)
        out["s_r_mean"] = float(scaler.mean_[idx0])
        out["s_r_std"] = float(np.sqrt(scaler.var_[idx0]))

    return out


def report_scaled_features(scaled_pack):
    """Print the per-column mean/std of the scaled training features.

    Printed sanity check only. Columns left out of `scale_cols` are reported
    unscaled, which is expected for the quantile-transformed geometry columns -
    they are already standardized by their own transform.

    Parameters
    ----------
    scaled_pack : dict
        Result of scale_continuous_features.

    Returns
    -------
    None
    """
    cont_cols = scaled_pack.get("cont_cols", None)

    if scaled_pack.get("scaler", None) is None:
        print("No StandardScaler applied.")
        print("scale_cols:", scaled_pack["scale_cols"])

        print("\n--- Train continuous means/std ---")
        X = scaled_pack["X_cont_train_s"]
        for i in range(X.shape[1]):
            name = cont_cols[i] if cont_cols is not None else f"col_{i}"
            print(f"{name:>10s}: mean={X[:, i].mean(): .6f}  std={X[:, i].std(): .6f}")
        return

    scaler = scaled_pack["scaler"]

    print("Scaler mean:", scaler.mean_)
    print("Scaler std :", np.sqrt(scaler.var_))
    print("scale_cols :", scaled_pack["scale_cols"])

    print("\n--- RAW ranges ---")
    for i in range(scaled_pack["X_cont_train"].shape[1]):
        name = cont_cols[i] if cont_cols is not None else f"col_{i}"
        x = scaled_pack["X_cont_train"][:, i]
        print(f"{name:>10s} RAW train: {x.min()}  {x.max()}")

    print("\n--- SCALED ranges ---")
    for i in range(scaled_pack["X_cont_train_s"].shape[1]):
        name = cont_cols[i] if cont_cols is not None else f"col_{i}"
        x = scaled_pack["X_cont_train_s"][:, i]
        print(f"{name:>10s} scaled train: {x.min()}  {x.max()}")

    print("\n--- Means/std AFTER scaling ---")
    for i in range(scaled_pack["X_cont_train_s"].shape[1]):
        name = cont_cols[i] if cont_cols is not None else f"col_{i}"
        x = scaled_pack["X_cont_train_s"][:, i]
        print(f"{name:>10s}: mean={x.mean(): .6f}  std={x.std(): .6f}")


def to_one_hot(y_idx, n_types):
    """One-hot encode integer particle-type indices as a float32 tensor.

    Parameters
    ----------
    y_idx : array-like of int
        Particle-type indices, as produced by the dataset builders.

    n_types : int
        One-hot depth, i.e. `dataset_pack["n_types"]`.

    Returns
    -------
    tf.Tensor
        float32 tensor of shape (len(y_idx), n_types).
    """
    y_idx = np.asarray(y_idx, dtype=np.int32)
    return tf.one_hot(y_idx, depth=n_types, dtype=tf.float32)


def build_conditioning_and_weights(
    scaled_pack,
    n_types,
    idx_to_type=None,
    alpha=0.5,
):
    """Build the particle-type conditioning one-hots and per-type loss weights.

    The conditioning vector `cond` is the type one-hot; the model is conditioned
    on it at both encode and decode time, so generation can be steered to a
    chosen type mix.

    Type weights counteract the strongly uneven particle-type populations:

        w_t = (1 / freq_t) ** alpha,  renormalized to mean 1

    alpha = 0 leaves the natural proportions untouched, alpha = 1 fully balances
    the types, and the default 0.5 is the compromise the notebooks use - enough
    to keep rare species from being ignored, not so much that the common ones
    are under-fit.

    Parameters
    ----------
    scaled_pack : dict
        Result of scale_continuous_features.

    n_types : int
        Number of particle types, i.e. `dataset_pack["n_types"]`.

    idx_to_type : dict[int, str] or None
        Index-to-name map. If given it is copied into the result, so the
        reporting helpers can label the classes; otherwise whatever the
        upstream pack already carried is kept.

    alpha : float
        Exponent in the weight formula above, in [0, 1]. See the discussion
        of the extremes in the summary; 0.5 is the default.

    Returns
    -------
    dict
        `scaled_pack` extended with "cond_train"/"cond_val"/"cond_test"
        (float32 one-hot tensors), "type_weights" (a float32 tf constant of
        length n_types, mean 1), "counts_train" and "freq_train". Pass it to
        build_tf_datasets.
    """
    cond_train = to_one_hot(scaled_pack["y_train"], n_types)
    cond_val = to_one_hot(scaled_pack["y_val"], n_types)
    cond_test = to_one_hot(scaled_pack["y_test"], n_types)

    counts_train = np.bincount(
        scaled_pack["y_train"],
        minlength=n_types,
    ).astype(np.float64)

    freq_train = counts_train / counts_train.sum()

    w = (1.0 / np.maximum(freq_train, 1e-12)) ** alpha
    w = w / np.mean(w)

    out = {
        **scaled_pack,
        "cond_train": cond_train,
        "cond_val": cond_val,
        "cond_test": cond_test,
        "type_weights": tf.constant(w.astype(np.float32)),
        "counts_train": counts_train,
        "freq_train": freq_train,
    }

    if idx_to_type is not None:
        out["idx_to_type"] = idx_to_type

    return out


def report_conditioning(condition_pack, n_types):
    """Print the conditioning shapes, class fractions, counts and type weights.

    Printed sanity check only; nothing is returned.

    Parameters
    ----------
    condition_pack : dict
        Result of build_conditioning_and_weights.

    n_types : int
        Number of particle types. Class fractions are only printed when the
        pack carries an "idx_to_type" map to label them with.

    Returns
    -------
    None
    """
    print("\ncond_train shape:", condition_pack["cond_train"].shape)
    print("cond_val shape  :", condition_pack["cond_val"].shape)
    print("cond_test shape :", condition_pack["cond_test"].shape)

    print("\nExample one-hot rows:")
    print(condition_pack["cond_train"][:5])

    if "idx_to_type" in condition_pack:
        print("\nClass fractions in train:")
        for i in range(n_types):
            print(
                f"{condition_pack['idx_to_type'][i]}:",
                condition_pack["cond_train"][:, i].numpy().mean(),
            )

    print("\nClass counts:", condition_pack["counts_train"])
    print("Class frequencies:", condition_pack["freq_train"])
    print("Class weights:", condition_pack["type_weights"].numpy())


def make_dummy_targets(n):
    """Placeholder y for tf.data.

    The models compute their own losses inside train_step from the inputs, so
    Keras never uses the target tensor - but the dataset element must still be
    an (inputs, targets) pair.
    """
    return tf.zeros((n, 1), dtype=tf.float32)


def build_tf_datasets(
    condition_pack,
    batch_size=4096,
    shuffle_buffer_cap=200_000,
):
    """
    Build TensorFlow datasets.

    Legacy discrete-u_v output batch:
        (X_cont, E_idx, u_v_idx, cond), dummy

    v0.7 continuous-geometry output batch:
        (X_cont, E_idx, cond), dummy

    Only the training dataset is shuffled. The targets are dummies: every
    model computes its own loss inside train_step from the inputs (see
    make_dummy_targets).

    Parameters
    ----------
    condition_pack : dict
        Result of build_conditioning_and_weights.

    batch_size : int
        Examples per batch. 4096 is the notebook default - these are small
        per-event feature vectors, and large batches keep the CPU-only
        training throughput reasonable.

    shuffle_buffer_cap : int
        Upper bound on the shuffle buffer, which is otherwise the training
        set size. Caps memory on multi-million-event sources; the buffer is
        still far larger than the batch, so shuffling stays effective.

    Returns
    -------
    dict
        `condition_pack` extended with "train_ds", "val_ds", "test_ds"
        (batched, prefetched tf.data.Dataset) and "batch_size". Pass
        "train_ds"/"val_ds" to fit_model or train_single_run.

    See Also
    --------
    report_tf_datasets : prints the resulting element layout, which differs
        between the two dataset modes.
    """
    dataset_mode = condition_pack.get("dataset_mode", "discrete_uv")

    X_cont_train_s = condition_pack["X_cont_train_s"].astype(np.float32)
    X_cont_val_s = condition_pack["X_cont_val_s"].astype(np.float32)
    X_cont_test_s = condition_pack["X_cont_test_s"].astype(np.float32)

    E_train = condition_pack["E_train"].astype(np.int32)
    E_val = condition_pack["E_val"].astype(np.int32)
    E_test = condition_pack["E_test"].astype(np.int32)

    cond_train = condition_pack["cond_train"]
    cond_val = condition_pack["cond_val"]
    cond_test = condition_pack["cond_test"]

    shuffle_buffer = min(len(X_cont_train_s), shuffle_buffer_cap)

    if dataset_mode == "discrete_uv":
        u_v_train = condition_pack["u_v_train"].astype(np.int32)
        u_v_val = condition_pack["u_v_val"].astype(np.int32)
        u_v_test = condition_pack["u_v_test"].astype(np.int32)

        train_inputs = (X_cont_train_s, E_train, u_v_train, cond_train)
        val_inputs = (X_cont_val_s, E_val, u_v_val, cond_val)
        test_inputs = (X_cont_test_s, E_test, u_v_test, cond_test)

    elif dataset_mode == "continuous_geometry":
        train_inputs = (X_cont_train_s, E_train, cond_train)
        val_inputs = (X_cont_val_s, E_val, cond_val)
        test_inputs = (X_cont_test_s, E_test, cond_test)

    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_inputs, make_dummy_targets(len(X_cont_train_s)))
    ).shuffle(
        shuffle_buffer,
        reshuffle_each_iteration=True,
    ).batch(
        batch_size,
        drop_remainder=False,
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_inputs, make_dummy_targets(len(X_cont_val_s)))
    ).batch(
        batch_size,
        drop_remainder=False,
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_inputs, make_dummy_targets(len(X_cont_test_s)))
    ).batch(
        batch_size,
        drop_remainder=False,
    ).prefetch(tf.data.AUTOTUNE)

    return {
        **condition_pack,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "batch_size": batch_size,
    }


def report_tf_datasets(dataset_tf_pack):
    """Print batch counts and the shapes of one batch, per dataset mode.

    Worth running once after build_tf_datasets: the element layout differs
    between the legacy discrete-u_v models and the continuous-geometry ones, and
    a mismatch surfaces here rather than as an unpacking error inside train_step.

    Parameters
    ----------
    dataset_tf_pack : dict
        Result of build_tf_datasets.

    Returns
    -------
    None
    """
    dataset_mode = dataset_tf_pack.get("dataset_mode", "discrete_uv")

    print("\nDatasets ready.")
    print("Dataset mode:", dataset_mode)
    print("Train batches:", tf.data.experimental.cardinality(dataset_tf_pack["train_ds"]).numpy())
    print("Val batches  :", tf.data.experimental.cardinality(dataset_tf_pack["val_ds"]).numpy())
    print("Test batches :", tf.data.experimental.cardinality(dataset_tf_pack["test_ds"]).numpy())

    if dataset_mode == "discrete_uv":
        for (xb_cont, eb, uvb, cb), _ in dataset_tf_pack["train_ds"].take(1):
            print("\nBatch check:")
            print("X_cont batch shape :", xb_cont.shape)
            print("E_idx batch shape  :", eb.shape)
            print("u_v_idx batch shape:", uvb.shape)
            print("cond batch shape   :", cb.shape)

    elif dataset_mode == "continuous_geometry":
        for (xb_cont, eb, cb), _ in dataset_tf_pack["train_ds"].take(1):
            print("\nBatch check:")
            print("X_cont batch shape:", xb_cont.shape)
            print("E_idx batch shape :", eb.shape)
            print("cond batch shape  :", cb.shape)


def report_energy_binning_diagnostics(
    energy_bins,
    E_idx,
    E_values,
    idx_train,
    energy_binning_mode,
    bin_width=None,
    n_bins=None,
    min_counts=None,
):
    """Print diagnostics for the chosen energy binning.

    Covers bin count and widths, how the training events distribute across bins,
    and empty/underpopulated bins. Use it to check that the binning resolves the
    spectral features you care about before committing to a long run.

    Parameters
    ----------
    energy_bins : dict
        Result of build_energy_bins, carrying at least the bin edges.

    E_idx : array-like of int
        Per-event bin index, over the whole dataset.

    E_values : array-like of float
        Per-event physical energy in MeV, over the whole dataset.

    idx_train : array-like of int
        Row indices of the training split, so the occupancy report reflects
        what the model will actually see.

    energy_binning_mode : str
        Which mode produced the binning: "fixed_width", "min_counts" or
        "log_fixed_count". Only used to label the report and to decide which
        of the three parameters below are meaningful.

    bin_width : float or None
        Bin width in MeV, for "fixed_width".

    n_bins : int or None
        Bin count, for "log_fixed_count".

    min_counts : int or None
        Minimum events per bin, for "min_counts".

    Returns
    -------
    None
    """
    E_train_values = E_values[idx_train].astype(np.float64)
    E_train_idx = E_idx[idx_train].astype(np.int32)

    n_energy_bins = len(energy_bins) - 1
    energy_bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])
    energy_bin_widths = np.diff(energy_bins)

    print("Energy binning mode:", energy_binning_mode)
    print("Number of energy bins:", n_energy_bins)

    print("\nEnergy train min/max:")
    print(E_train_values.min(), E_train_values.max())

    print("\nFirst bin:", energy_bins[0], energy_bins[1])
    print("Last bin :", energy_bins[-2], energy_bins[-1])

    print("\nBin width summary:")
    print("min width :", energy_bin_widths.min())
    print("max width :", energy_bin_widths.max())
    print("mean width:", energy_bin_widths.mean())

    counts = np.bincount(E_train_idx, minlength=n_energy_bins)
    empty_bins = np.sum(counts == 0)

    print("\n--- Energy bin occupancy check ---")
    print("min counts:", counts.min())
    print("max counts:", counts.max())
    print("mean counts:", counts.mean())
    print("empty bins:", empty_bins)
    print("empty bin fraction:", empty_bins / n_energy_bins)

    if energy_binning_mode == "fixed_width":
        print("\n--- Fixed-width mode diagnostics ---")
        print("Requested bin width:", bin_width)

    elif energy_binning_mode == "min_counts":
        print("\n--- Minimum-count mode diagnostics ---")
        print("Initial requested bins:", n_bins)
        print("Minimum counts threshold:", min_counts)
        print("Final number of bins after merging:", n_energy_bins)

    elif energy_binning_mode == "log_fixed_count":
        print("\n--- Log-fixed-count mode diagnostics ---")
        logE_train_values = np.log10(E_train_values)
        log_energy_bin_edges = np.log10(energy_bins)
        log_energy_bin_widths = np.diff(log_energy_bin_edges)

        print("Requested bins:", n_bins)
        print("log10(E) min/max:", logE_train_values.min(), logE_train_values.max())
        print("log-bin width min :", log_energy_bin_widths.min())
        print("log-bin width max :", log_energy_bin_widths.max())
        print("log-bin width mean:", log_energy_bin_widths.mean())

    plt.figure(figsize=(8, 4))
    plt.hist(E_train_idx, bins=np.arange(n_energy_bins + 1) - 0.5, histtype="step")
    plt.xlabel("energy bin index")
    plt.ylabel("counts")
    plt.title(f"Energy bin occupancy ({energy_binning_mode})")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(E_train_values, bins=energy_bins, histtype="step", density=True, label="train energy")
    plt.xlabel("Energy")
    plt.ylabel("density")
    plt.title(f"Energy histogram with selected bins ({energy_binning_mode})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(energy_bin_centers, energy_bin_widths, linewidth=1.5)
    plt.xlabel("energy bin center")
    plt.ylabel("bin width")
    plt.title(f"Energy bin widths ({energy_binning_mode})")
    plt.grid(True, alpha=0.3)
    plt.show()