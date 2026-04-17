"""
Dataset construction utilities for GEEANNT.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def filter_particle_types_and_discretize_uv(
    feat,
    prob_threshold=1e-5,
    uv_nbins=128,
    uv_eps=1e-6,
):
    """
    Filter rare particle types and discretize u_v.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature dataframe containing:
        ParticleName, E_idx, s_r, cphi_r, sphi_r, u_v, cphi_v, sphi_v
    prob_threshold : float
        Minimum empirical particle-type probability to keep a class.
    uv_nbins : int
        Requested number of u_v quantile bins.
    uv_eps : float
        Small padding applied to first/last u_v bin edge.

    Returns
    -------
    dict
        Dictionary containing filtered dataframe, mappings, matrices and u_v bins.
    """
    feat = feat.copy()

    # ------------------------------------------------------
    # Filter rare particle types
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # Discretize u_v
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # Continuous matrix
    # ------------------------------------------------------
    cont_cols = ["s_r", "cphi_r", "sphi_r", "cphi_v", "sphi_v"]
    X_cont_raw = feat[cont_cols].to_numpy(dtype=np.float32)

    # ------------------------------------------------------
    # Targets
    # ------------------------------------------------------
    E_idx = feat["E_idx"].to_numpy(dtype=np.int32)
    y_type = feat["ParticleName"].map(type_to_idx).to_numpy(dtype=np.int32)

    return {
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
    }


def report_discretized_features(dataset_pack):
    feat = dataset_pack["feat"]
    X_cont_raw = dataset_pack["X_cont_raw"]
    cont_cols = dataset_pack["cont_cols"]
    u_v_bins = dataset_pack["u_v_bins"]
    u_v_idx = dataset_pack["u_v_idx"]
    u_v_values = dataset_pack["u_v_values"]
    u_v_centers = dataset_pack["u_v_centers"]

    print("Original particle types:", feat["ParticleName"].nunique())
    print("Kept particle types    :", dataset_pack["n_types"])
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

    print("\n--- Continuous feature ranges (RAW) ---")
    for i, c in enumerate(cont_cols):
        print(f"{c:>8s} : {X_cont_raw[:, i].min()}  {X_cont_raw[:, i].max()}")

    print("\n--- Continuous means/std ---")
    for i, c in enumerate(cont_cols):
        print(f"{c:>8s} : mean={X_cont_raw[:, i].mean(): .6f}  std={X_cont_raw[:, i].std(): .6f}")

    u_v_reco = u_v_centers[u_v_idx]
    print("\n--- u_v reconstruction from bins ---")
    print("u_v reco: min =", u_v_reco.min(), "max =", u_v_reco.max())
    print("mean abs diff =", np.mean(np.abs(u_v_reco - u_v_values)))


def split_feature_data(
    dataset_pack,
    test_size_total=0.30,
    val_size_from_temp=0.50,
    random_state=42,
):
    """
    Split raw feature matrices into train/val/test.
    """
    X_cont_raw = dataset_pack["X_cont_raw"]
    E_idx = dataset_pack["E_idx"]
    u_v_idx = dataset_pack["u_v_idx"]
    y_type = dataset_pack["y_type"]
    feat = dataset_pack["feat"]

    idx_all = np.arange(len(X_cont_raw))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all,
        y_type,
        test_size=test_size_total,
        random_state=random_state,
        stratify=y_type
    )

    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=val_size_from_temp,
        random_state=random_state,
        stratify=y_temp
    )

    split_pack = {
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "X_cont_train": X_cont_raw[idx_train],
        "X_cont_val": X_cont_raw[idx_val],
        "X_cont_test": X_cont_raw[idx_test],
        "E_train": E_idx[idx_train],
        "E_val": E_idx[idx_val],
        "E_test": E_idx[idx_test],
        "u_v_train": u_v_idx[idx_train],
        "u_v_val": u_v_idx[idx_val],
        "u_v_test": u_v_idx[idx_test],
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "u_v_train_raw": feat["u_v"].to_numpy()[idx_train],
        "u_v_val_raw": feat["u_v"].to_numpy()[idx_val],
        "u_v_test_raw": feat["u_v"].to_numpy()[idx_test],
    }

    # optional diagnostic raw energy if present in feat
    if "Energy" in feat.columns:
        E_raw = feat["Energy"].to_numpy()
        split_pack["E_train_raw"] = E_raw[idx_train]
        split_pack["E_val_raw"] = E_raw[idx_val]
        split_pack["E_test_raw"] = E_raw[idx_test]

    return split_pack


def frac_per_type(y_idx, n_types):
    c = np.bincount(y_idx, minlength=n_types).astype(np.float64)
    return c / c.sum()


def report_split_summary(split_pack, n_types):
    print("Train:", split_pack["X_cont_train"].shape,
          "Val:", split_pack["X_cont_val"].shape,
          "Test:", split_pack["X_cont_test"].shape)

    print("\nType distribution check:")
    print("Train:", frac_per_type(split_pack["y_train"], n_types))
    print("Val  :", frac_per_type(split_pack["y_val"], n_types))
    print("Test :", frac_per_type(split_pack["y_test"], n_types))


def scale_continuous_features(split_pack, scale_cols=(0,)):
    """
    Scale selected continuous features using TRAIN only.
    """
    scaler = StandardScaler()
    scaler.fit(split_pack["X_cont_train"][:, list(scale_cols)])

    def apply_selected_scaler(X_in):
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
        "scaler_sr": scaler,
        "scale_cols": tuple(scale_cols),
    }

    if 0 in scale_cols:
        out["s_r_mean"] = float(scaler.mean_[list(scale_cols).index(0)])
        out["s_r_std"] = float(np.sqrt(scaler.var_[list(scale_cols).index(0)]))

    return out


def report_scaled_features(scaled_pack):
    scaler = scaled_pack["scaler_sr"]
    print("Scaler mean:", scaler.mean_)
    print("Scaler std :", np.sqrt(scaler.var_))

    print("\n--- RAW ranges ---")
    print("s_r RAW train:", scaled_pack["X_cont_train"][:, 0].min(), scaled_pack["X_cont_train"][:, 0].max())

    print("\n--- SCALED ranges ---")
    print("s_r scaled train:", scaled_pack["X_cont_train_s"][:, 0].min(), scaled_pack["X_cont_train_s"][:, 0].max())

    print("\n--- Means/std AFTER scaling (train only) ---")
    print("s_r : mean =", scaled_pack["X_cont_train_s"][:, 0].mean(),
          "std =", scaled_pack["X_cont_train_s"][:, 0].std())

    print("\n--- Angular features remain unscaled ---")
    print("cphi_r: mean =", scaled_pack["X_cont_train_s"][:, 1].mean(),
          "std =", scaled_pack["X_cont_train_s"][:, 1].std())
    print("sphi_r: mean =", scaled_pack["X_cont_train_s"][:, 2].mean(),
          "std =", scaled_pack["X_cont_train_s"][:, 2].std())
    print("cphi_v: mean =", scaled_pack["X_cont_train_s"][:, 3].mean(),
          "std =", scaled_pack["X_cont_train_s"][:, 3].std())
    print("sphi_v: mean =", scaled_pack["X_cont_train_s"][:, 4].mean(),
          "std =", scaled_pack["X_cont_train_s"][:, 4].std())


def to_one_hot(y_idx, n_types):
    y_idx = np.asarray(y_idx, dtype=np.int32)
    return tf.one_hot(y_idx, depth=n_types, dtype=tf.float32)


def build_conditioning_and_weights(scaled_pack, n_types, idx_to_type=None, alpha=0.5):
    cond_train = to_one_hot(scaled_pack["y_train"], n_types)
    cond_val = to_one_hot(scaled_pack["y_val"], n_types)
    cond_test = to_one_hot(scaled_pack["y_test"], n_types)

    counts_train = np.bincount(scaled_pack["y_train"], minlength=n_types).astype(np.float64)
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
    print("\ncond_train shape:", condition_pack["cond_train"].shape)
    print("cond_val shape  :", condition_pack["cond_val"].shape)
    print("cond_test shape :", condition_pack["cond_test"].shape)

    print("\nExample one-hot rows:")
    print(condition_pack["cond_train"][:5])

    if "idx_to_type" in condition_pack:
        print("\nClass fractions in train:")
        for i in range(n_types):
            print(f"{condition_pack['idx_to_type'][i]}:",
                  condition_pack["cond_train"][:, i].numpy().mean())

    print("\nClass counts:", condition_pack["counts_train"])
    print("Class frequencies:", condition_pack["freq_train"])
    print("Class weights:", condition_pack["type_weights"].numpy())


def make_dummy_targets(n):
    return tf.zeros((n, 1), dtype=tf.float32)


def build_tf_datasets(condition_pack, batch_size=4096, shuffle_buffer_cap=200_000):
    X_cont_train_s = condition_pack["X_cont_train_s"].astype(np.float32)
    X_cont_val_s = condition_pack["X_cont_val_s"].astype(np.float32)
    X_cont_test_s = condition_pack["X_cont_test_s"].astype(np.float32)

    E_train = condition_pack["E_train"].astype(np.int32)
    E_val = condition_pack["E_val"].astype(np.int32)
    E_test = condition_pack["E_test"].astype(np.int32)

    u_v_train = condition_pack["u_v_train"].astype(np.int32)
    u_v_val = condition_pack["u_v_val"].astype(np.int32)
    u_v_test = condition_pack["u_v_test"].astype(np.int32)

    cond_train = condition_pack["cond_train"]
    cond_val = condition_pack["cond_val"]
    cond_test = condition_pack["cond_test"]

    shuffle_buffer = min(len(X_cont_train_s), shuffle_buffer_cap)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (
            (X_cont_train_s, E_train, u_v_train, cond_train),
            make_dummy_targets(len(X_cont_train_s))
        )
    ).shuffle(shuffle_buffer, reshuffle_each_iteration=True).batch(
        batch_size, drop_remainder=False
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (
            (X_cont_val_s, E_val, u_v_val, cond_val),
            make_dummy_targets(len(X_cont_val_s))
        )
    ).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (
            (X_cont_test_s, E_test, u_v_test, cond_test),
            make_dummy_targets(len(X_cont_test_s))
        )
    ).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return {
        **condition_pack,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "batch_size": batch_size,
    }


def report_tf_datasets(dataset_tf_pack):
    print("\nDatasets ready.")
    print("Train batches:", tf.data.experimental.cardinality(dataset_tf_pack["train_ds"]).numpy())
    print("Val batches  :", tf.data.experimental.cardinality(dataset_tf_pack["val_ds"]).numpy())
    print("Test batches :", tf.data.experimental.cardinality(dataset_tf_pack["test_ds"]).numpy())

    for (xb_cont, eb, uvb, cb), _ in dataset_tf_pack["train_ds"].take(1):
        print("\nBatch check:")
        print("X_cont batch shape :", xb_cont.shape)
        print("E_idx batch shape  :", eb.shape)
        print("u_v_idx batch shape:", uvb.shape)
        print("cond batch shape   :", cb.shape)


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