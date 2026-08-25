"""
Reconstruction utilities for generated and real physical variables.
"""

import numpy as np


def renorm_cos_sin(c, s, eps=1e-12):
    """Project a predicted (cos, sin) pair back onto the unit circle.

    The angle heads are only softly regularized towards norm 1, so the raw
    output has to be renormalized before an angle can be read off it.

    Parameters
    ----------
    c, s : array-like of float
        Raw predicted cosine and sine components, elementwise paired.

    eps : float
        Floor on the norm, guarding against division by zero when a
        predicted pair collapses to the origin.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (cos, sin) pair rescaled to unit norm.
    """
    r = np.sqrt(c * c + s * s) + eps
    return c / r, s / r


def u_from_s(s):
    """Inverse of the s = atanh(u) reparametrization: maps R back to (-1, 1).

    u_v is bounded and piles up at the edges, which a Gaussian head fits badly;
    the model works in s-space and this brings the sample back.

    Parameters
    ----------
    s : array-like of float
        Unbounded s-space values, as produced by the model's head.

    Returns
    -------
    np.ndarray
        The corresponding u = tanh(s) values, in (-1, 1).
    """
    return np.tanh(s)


def _inverse_quantile(transformer, x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    return transformer.inverse_transform(x).reshape(-1)


def reconstruct_generated_features(
    gen_pack,
    energy_bins=None,
    energy_mode="uniform",
    rng=None,

    # v0.6 legacy
    s_r_mean=None,
    s_r_std=None,
    u_v_bins=None,

    # v0.7 quantile geometry
    qt_u_r=None,
    qt_u_v=None,
    qt_phi_r=None,
    qt_phi_v=None,
    geometry_mode=None,
    geometry_metadata=None,

    # v0.8 mixture energy
    energy_head_mode=None,
    qt_energy=None,
    energy_transform=None,
    energy_metadata=None,
):
    """
    Convert raw generated model outputs into physical feature arrays.

    Supports:
      - v0.6 legacy:
          y_cont = [s_r_scaled, cphi_r, sphi_r, cphi_v, sphi_v]
          u_v from generated categorical bin

      - v0.7 continuous geometry:
          y_cont = [u_r_q, u_v_q, cphi_r, sphi_r, cphi_v, sphi_v]
          u_r/u_v recovered through QuantileTransformer inverse_transform

      - v0.7.2 continuous geometry with quantile phi_r/phi_v:
          y_cont = [u_r_q, u_v_q, phi_r_q, phi_v_q]
          u_r/u_v recovered through QuantileTransformer inverse_transform
          phi_r/phi_v recovered through QuantileTransformer inverse_transform

    Energy reconstruction is an independent axis from geometry
    (energy_head_mode, alongside geometry_mode):
      - "categorical" (v0.6/v0.7/v0.7.2): energy_bins + energy_from_idx.
      - "mixture" (v0.8): inverse the energy transform T on the model's raw
          energy_y_gen sample - via qt_energy.inverse_transform if T was a
          QuantileTransformer, or 10**y if T was log10 (energy_transform="log10").

    The transforms passed here MUST be the ones fitted during training, not
    refitted on generated data. In practice pass `geometry_metadata` and
    `energy_metadata` straight from the loaded checkpoint and leave the
    individual transformer arguments alone.

    Parameters
    ----------
    gen_pack : dict
        Result of generate_latent_outputs, carrying at least "y_cont_gen_s"
        plus whichever energy keys the head produced.

    energy_bins : array-like or None
        Bin edges in MeV. Required for `energy_head_mode="categorical"`,
        ignored for "mixture".

    energy_mode : str
        Passed to energy_from_idx for categorical heads: "uniform" (default)
        draws inside the bin, anything else takes the bin centre.

    rng : np.random.Generator or None
        Generator for the within-bin energy draw. Categorical heads only.

    s_r_mean, s_r_std : float or None
        v0.6 only. Mean/std used to unscale s_r, from
        `scaled_pack["s_r_mean"]` / `["s_r_std"]`.

    u_v_bins : array-like or None
        v0.6 only. Edges of the categorical u_v binning.

    qt_u_r, qt_u_v, qt_phi_r, qt_phi_v : sklearn QuantileTransformer or None
        Fitted geometry transforms, inverted to recover physical u_r, u_v,
        phi_r and phi_v. Values found in `geometry_metadata` take precedence
        over these.

    geometry_mode : str or None
        One of "arctanh_uv_discrete", "quantile_u_r_u_v",
        "quantile_u_r_u_v_phi_r_phi_v". If None it is inferred from which
        transformers were supplied, but passing it explicitly (or via
        metadata) is safer - a mismatch here silently produces wrong angles
        rather than an error.

    geometry_metadata : dict or None
        Checkpoint metadata block. When given, its "geometry_transform" and
        "qt_*" entries override the corresponding arguments above.

    energy_head_mode : str or None
        "categorical" or "mixture". Selects which energy path runs.

    qt_energy : sklearn QuantileTransformer or None
        Fitted energy transform, for mixture heads whose transform was a
        quantile transform.

    energy_transform : str or None
        Name of the energy transform for mixture heads, e.g. "log10", in
        which case energy is recovered as 10**y.

    energy_metadata : dict or None
        Checkpoint metadata block supplying `energy_transform` / `qt_energy`.

    Returns
    -------
    dict
        Physical feature arrays: "E" (MeV), "u_r", "u_v", "phi_r", "phi_v",
        and - for v0.8 packs - the pass-through "energy_component_idx_gen"
        identifying which mixture component each event came from, which the
        line-recovery metric uses. Pass the result to
        reconstruct_generated_physics.
    """
    from .sampling import energy_from_idx

    y_cont_gen_s = gen_pack["y_cont_gen_s"]
    y_cont_gen = y_cont_gen_s.copy()

    if geometry_metadata is not None:
        geometry_mode = geometry_metadata.get(
            "geometry_transform",
            geometry_metadata.get("geometry_mode", geometry_mode),
        )

        qt_u_r = geometry_metadata.get("qt_u_r", qt_u_r)
        qt_u_v = geometry_metadata.get("qt_u_v", qt_u_v)
        qt_phi_r = geometry_metadata.get("qt_phi_r", qt_phi_r)
        qt_phi_v = geometry_metadata.get("qt_phi_v", qt_phi_v)

    if geometry_mode is None:
        if (
            qt_u_r is not None
            and qt_u_v is not None
            and qt_phi_r is not None
            and qt_phi_v is not None
        ):
            geometry_mode = "quantile_u_r_u_v_phi_r_phi_v"
        elif qt_u_r is not None and qt_u_v is not None:
            geometry_mode = "quantile_u_r_u_v"
        else:
            geometry_mode = "legacy_sr_discrete_uv"

    if energy_metadata is not None:
        energy_head_mode = energy_metadata.get("energy_head_mode", energy_head_mode)
        qt_energy = energy_metadata.get("qt_energy", qt_energy)
        energy_transform = energy_metadata.get("energy_transform", energy_transform)

    if energy_head_mode is None:
        energy_head_mode = "mixture" if "energy_y_gen" in gen_pack else "categorical"

    if energy_head_mode == "categorical":
        if energy_bins is None:
            raise ValueError(
                "energy_bins is required for categorical energy reconstruction."
            )
        E_gen = energy_from_idx(
            gen_pack["energy_idx_gen"],
            energy_bins=energy_bins,
            mode=energy_mode,
            rng=rng,
        )
    elif energy_head_mode == "mixture":
        if qt_energy is not None:
            E_gen = _inverse_quantile(qt_energy, gen_pack["energy_y_gen"])
        elif energy_transform == "log10":
            E_gen = 10.0 ** gen_pack["energy_y_gen"]
        else:
            raise ValueError(
                "qt_energy or energy_transform='log10' is required for "
                "mixture energy reconstruction."
            )
    else:
        raise ValueError(f"Unknown energy_head_mode: {energy_head_mode}")

    logE_gen = np.log10(E_gen)

    # ======================================================
    # v0.7.2 quantile geometry + quantile phi
    # ======================================================
    if geometry_mode in [
        "quantile_u_r_u_v_phi_r_phi_v",
        "continuous_phi",
        "quantile_phi",
    ]:

        if qt_u_r is None or qt_u_v is None or qt_phi_r is None or qt_phi_v is None:
            raise ValueError(
                "qt_u_r, qt_u_v, qt_phi_r and qt_phi_v are required "
                "for v0.7.2 quantile-phi reconstruction."
            )

        u_r_q_gen = y_cont_gen[:, 0]
        u_v_q_gen = y_cont_gen[:, 1]
        phi_r_q_gen = y_cont_gen[:, 2]
        phi_v_q_gen = y_cont_gen[:, 3]

        u_r_gen = _inverse_quantile(qt_u_r, u_r_q_gen)
        u_v_gen = _inverse_quantile(qt_u_v, u_v_q_gen)
        phi_r_gen = _inverse_quantile(qt_phi_r, phi_r_q_gen)
        phi_v_gen = _inverse_quantile(qt_phi_v, phi_v_q_gen)

        u_r_gen = np.clip(u_r_gen, -1.0 + 1e-6, 1.0 - 1e-6)
        u_v_gen = np.clip(u_v_gen, -1.0 + 1e-6, 1.0 - 1e-6)

        # Wrap angles back to [-pi, pi]
        phi_r_gen = (phi_r_gen + np.pi) % (2.0 * np.pi) - np.pi
        phi_v_gen = (phi_v_gen + np.pi) % (2.0 * np.pi) - np.pi

        cphi_r_gen = np.cos(phi_r_gen)
        sphi_r_gen = np.sin(phi_r_gen)
        cphi_v_gen = np.cos(phi_v_gen)
        sphi_v_gen = np.sin(phi_v_gen)

        s_r_gen = np.arctanh(u_r_gen)

    # ======================================================
    # v0.7 quantile geometry
    # ======================================================
    elif geometry_mode in ["quantile_u_r_u_v", "continuous_geometry"]:

        if qt_u_r is None or qt_u_v is None:
            raise ValueError(
                "qt_u_r and qt_u_v are required for quantile geometry reconstruction."
            )

        u_r_q_gen = y_cont_gen[:, 0]
        u_v_q_gen = y_cont_gen[:, 1]
        

        u_r_gen = _inverse_quantile(qt_u_r, u_r_q_gen)
        u_v_gen = _inverse_quantile(qt_u_v, u_v_q_gen)

        u_r_gen = np.clip(u_r_gen, -1.0 + 1e-6, 1.0 - 1e-6)
        u_v_gen = np.clip(u_v_gen, -1.0 + 1e-6, 1.0 - 1e-6)

        cphi_r_gen = y_cont_gen[:, 2]
        sphi_r_gen = y_cont_gen[:, 3]
        cphi_v_gen = y_cont_gen[:, 4]
        sphi_v_gen = y_cont_gen[:, 5]

        phi_r_q_gen = None
        phi_v_q_gen = None
        phi_r_gen = np.arctan2(sphi_r_gen, cphi_r_gen)
        phi_v_gen = np.arctan2(sphi_v_gen, cphi_v_gen)

        s_r_gen = np.arctanh(u_r_gen)

    # ======================================================
    # v0.6 legacy geometry
    # ======================================================
    elif geometry_mode in ["legacy_sr_discrete_uv", "legacy"]:

        if s_r_mean is None or s_r_std is None:
            raise ValueError(
                "s_r_mean and s_r_std are required for legacy reconstruction."
            )

        if u_v_bins is None:
            raise ValueError(
                "u_v_bins is required for legacy reconstruction."
            )

        y_cont_gen[:, 0] = y_cont_gen_s[:, 0] * s_r_std + s_r_mean

        s_r_gen = y_cont_gen[:, 0]
        u_r_gen = np.tanh(s_r_gen)

        cphi_r_gen = y_cont_gen[:, 1]
        sphi_r_gen = y_cont_gen[:, 2]
        cphi_v_gen = y_cont_gen[:, 3]
        sphi_v_gen = y_cont_gen[:, 4]

        u_v_gen = gen_pack["uv_value_gen"].copy()
        u_v_gen = np.clip(u_v_gen, u_v_bins[0], u_v_bins[-1])

        u_r_q_gen = None
        u_v_q_gen = None
        phi_r_q_gen = None
        phi_v_q_gen = None
        phi_r_gen = np.arctan2(sphi_r_gen, cphi_r_gen)
        phi_v_gen = np.arctan2(sphi_v_gen, cphi_v_gen)

    else:
        raise ValueError(f"Unknown geometry_mode: {geometry_mode}")

    particle_names_gen = None
    if "gen_type_idx" in gen_pack and "idx_to_type" in gen_pack:
        particle_names_gen = np.array(
            [gen_pack["idx_to_type"][int(i)] for i in gen_pack["gen_type_idx"]]
        )

    return {
        **gen_pack,
        "geometry_mode": geometry_mode,

        "y_cont_gen": y_cont_gen,

        "u_r_q_gen": u_r_q_gen if geometry_mode != "legacy_sr_discrete_uv" else None,
        "u_v_q_gen": u_v_q_gen if geometry_mode != "legacy_sr_discrete_uv" else None,
        "phi_r_q_gen": phi_r_q_gen,
        "phi_v_q_gen": phi_v_q_gen,
        "phi_r_gen": phi_r_gen,
        "phi_v_gen": phi_v_gen,

        "s_r_gen": s_r_gen,
        "u_r_gen": u_r_gen,
        "u_v_gen": u_v_gen,

        "phi_r_gen": phi_r_gen,
        "phi_v_gen": phi_v_gen,

        "cphi_r_gen": cphi_r_gen,
        "sphi_r_gen": sphi_r_gen,
        "cphi_v_gen": cphi_v_gen,
        "sphi_v_gen": sphi_v_gen,

        "E_gen": E_gen,
        "logE_gen": logE_gen,
        "ParticleName": particle_names_gen,
    }


def reconstruct_generated_physics(
    reco_pack,
    center=(0.0, 0.0, 0.0),
    radius=100.0,
    eps=1e-6,
):
    """
    Reconstruct x, y, z, vx, vy, vz from generated feature variables.

    Applies the sphere-surface coordinate transforms in magi.core.geometry:
    (u_r, phi_r) place the crossing point on the sphere, (u_v, phi_v) give
    the direction. The geometry is currently sphere-only, so `center` and
    `radius` must match the virtual surface used in the Geant4 run.

    Parameters
    ----------
    reco_pack : dict
        Result of reconstruct_generated_features.

    center : tuple[float, float, float]
        Sphere centre in mm. The default is (0.0, 0.0, 0.0).

    radius : float
        Sphere radius in mm.

    eps : float
        Numerical margin keeping u_r/u_v strictly inside (-1, 1) before the
        trigonometric inversion, so an event exactly at a pole does not
        produce a NaN.

    Returns
    -------
    dict
        Physical event arrays "E", "x", "y", "z", "vx", "vy", "vz" (energies
        in MeV, positions in mm, directions unit-normalized), plus the
        feature arrays carried through. Pass it to
        generated_physics_to_detector_dataframe or save_detector_table.
    """
    C = np.asarray(center, dtype=np.float64)
    R = float(radius)

    u_r_gen = np.clip(reco_pack["u_r_gen"], -1.0 + eps, 1.0 - eps)
    u_v_gen = np.clip(reco_pack["u_v_gen"], -1.0 + eps, 1.0 - eps)

    cphi_r_gen = reco_pack["cphi_r_gen"]
    sphi_r_gen = reco_pack["sphi_r_gen"]
    cphi_v_gen = reco_pack["cphi_v_gen"]
    sphi_v_gen = reco_pack["sphi_v_gen"]

    cpr_gen, spr_gen = renorm_cos_sin(cphi_r_gen, sphi_r_gen)
    sin_tr_gen = np.sqrt(np.maximum(0.0, 1.0 - u_r_gen * u_r_gen))

    x_gen = C[0] + R * sin_tr_gen * cpr_gen
    y_gen = C[1] + R * sin_tr_gen * spr_gen
    z_gen = C[2] + R * u_r_gen

    cpv_gen, spv_gen = renorm_cos_sin(cphi_v_gen, sphi_v_gen)
    sin_tv_gen = np.sqrt(np.maximum(0.0, 1.0 - u_v_gen * u_v_gen))

    vx_gen = sin_tv_gen * cpv_gen
    vy_gen = sin_tv_gen * spv_gen
    vz_gen = u_v_gen

    vnorm = np.sqrt(vx_gen * vx_gen + vy_gen * vy_gen + vz_gen * vz_gen) + 1e-12
    vx_gen /= vnorm
    vy_gen /= vnorm
    vz_gen /= vnorm

    rvec_gen = np.vstack([x_gen, y_gen, z_gen]).T - C[None, :]
    r_gen = np.linalg.norm(rvec_gen, axis=1)

    return {
        **reco_pack,
        "x_gen": x_gen,
        "y_gen": y_gen,
        "z_gen": z_gen,
        "vx_gen": vx_gen,
        "vy_gen": vy_gen,
        "vz_gen": vz_gen,
        "r_gen": r_gen,
        "phi_r_norm_gen": np.sqrt(cpr_gen * cpr_gen + spr_gen * spr_gen),
        "phi_v_norm_gen": np.sqrt(cpv_gen * cpv_gen + spv_gen * spv_gen),
    }


def reconstruct_real_test_physics(
    X_cont_test,
    E_test_raw,
    center=(0.0, 0.0, 0.0),
    radius=100.0,
    eps=1e-6,

    # v0.6 legacy
    u_v_test_raw=None,
    u_v_bins=None,

    # v0.7 quantile geometry
    qt_u_r=None,
    qt_u_v=None,
    geometry_mode=None,

    # v0.7.2 quantile geometry with quantile phi
    qt_phi_r=None,
    qt_phi_v=None,
    geometry_metadata=None,
):
    """
    Reconstruct physical quantities from real test-set features.

    Supports:
      - v0.6:
          X_cont_test = [s_r, cphi_r, sphi_r, cphi_v, sphi_v]

      - v0.7:
          X_cont_test = [u_r_q, u_v_q, cphi_r, sphi_r, cphi_v, sphi_v]
    
      - v0.7.2:
          X_cont_test = [u_r_q, u_v_q, phi_r_q, phi_v_q]

    This is the real-data counterpart of reconstruct_generated_physics: it
    runs the held-out test split back through the same inverse transforms, so
    the validation metrics compare like with like rather than comparing
    generated physical values against real quantile-space ones.

    Parameters
    ----------
    X_cont_test : np.ndarray
        Test-split continuous features, in one of the layouts above. Use the
        unscaled `split_pack["X_cont_test"]` for quantile geometry.

    E_test_raw : array-like of float
        Test-split physical energies in MeV, i.e. `split_pack["E_test_raw"]`.

    center : tuple[float, float, float]
        Sphere centre in mm; must match the training geometry.

    radius : float
        Sphere radius in mm.

    eps : float
        Numerical margin keeping u_r/u_v strictly inside (-1, 1).

    u_v_test_raw : array-like or None
        v0.6 only. Raw u_v values, bypassing the categorical binning.

    u_v_bins : array-like or None
        v0.6 only. Edges of the categorical u_v binning.

    qt_u_r, qt_u_v, qt_phi_r, qt_phi_v : sklearn QuantileTransformer or None
        The same fitted transforms used at training time, inverted here.
        Entries in `geometry_metadata` take precedence.

    geometry_mode : str or None
        Which layout X_cont_test is in. Inferred from the supplied
        transformers when None.

    geometry_metadata : dict or None
        Checkpoint metadata block supplying `geometry_transform` and the
        `qt_*` transformers.

    Returns
    -------
    dict
        Real physical arrays in the same key layout as
        reconstruct_generated_physics, so the two can be passed directly to
        compute_wasserstein_scores or build_real_generated_featureframes.
    """
    C = np.asarray(center, dtype=np.float64)
    R = float(radius)

    if geometry_mode is None:
        if (
            qt_u_r is not None
            and qt_u_v is not None
            and qt_phi_r is not None
            and qt_phi_v is not None
        ):
            geometry_mode = "quantile_u_r_u_v_phi_r_phi_v"
        elif qt_u_r is not None and qt_u_v is not None:
            geometry_mode = "quantile_u_r_u_v"
        else:
            geometry_mode = "legacy_sr_discrete_uv"

    E_real = E_test_raw.copy()
    logE_real = np.log10(E_real)

    # ======================================================
    # v0.7.2 quantile geometry + quantile phi
    # ======================================================
    if geometry_mode in [
        "quantile_u_r_u_v_phi_r_phi_v",
        "continuous_phi",
        "quantile_phi",
    ]:

        if qt_u_r is None or qt_u_v is None or qt_phi_r is None or qt_phi_v is None:
            raise ValueError(
                "qt_u_r, qt_u_v, qt_phi_r and qt_phi_v are required "
                "for v0.7.2 quantile-phi real reconstruction."
            )

        u_r_q_real = X_cont_test[:, 0]
        u_v_q_real = X_cont_test[:, 1]
        phi_r_q_real = X_cont_test[:, 2]
        phi_v_q_real = X_cont_test[:, 3]

        u_r_real = _inverse_quantile(qt_u_r, u_r_q_real)
        u_v_real = _inverse_quantile(qt_u_v, u_v_q_real)
        phi_r_real = _inverse_quantile(qt_phi_r, phi_r_q_real)
        phi_v_real = _inverse_quantile(qt_phi_v, phi_v_q_real)

        u_r_real = np.clip(u_r_real, -1.0 + eps, 1.0 - eps)
        u_v_real = np.clip(u_v_real, -1.0 + eps, 1.0 - eps)

        phi_r_real = (phi_r_real + np.pi) % (2.0 * np.pi) - np.pi
        phi_v_real = (phi_v_real + np.pi) % (2.0 * np.pi) - np.pi

        s_r_real = np.arctanh(u_r_real)

        cphi_r_real = np.cos(phi_r_real)
        sphi_r_real = np.sin(phi_r_real)
        cphi_v_real = np.cos(phi_v_real)
        sphi_v_real = np.sin(phi_v_real)

    # ======================================================
    # v0.7 quantile geometry
    # ======================================================
    elif geometry_mode in ["quantile_u_r_u_v", "continuous_geometry"]:

        if qt_u_r is None or qt_u_v is None:
            raise ValueError(
                "qt_u_r and qt_u_v are required for quantile real reconstruction."
            )

        u_r_q_real = X_cont_test[:, 0]
        u_v_q_real = X_cont_test[:, 1]

        u_r_real = _inverse_quantile(qt_u_r, u_r_q_real)
        u_v_real = _inverse_quantile(qt_u_v, u_v_q_real)

        u_r_real = np.clip(u_r_real, -1.0 + eps, 1.0 - eps)
        u_v_real = np.clip(u_v_real, -1.0 + eps, 1.0 - eps)

        s_r_real = np.arctanh(u_r_real)

        cphi_r_real = X_cont_test[:, 2].copy()
        sphi_r_real = X_cont_test[:, 3].copy()
        cphi_v_real = X_cont_test[:, 4].copy()
        sphi_v_real = X_cont_test[:, 5].copy()

        phi_r_q_real = None
        phi_v_q_real = None
        phi_r_real = np.arctan2(sphi_r_real, cphi_r_real)
        phi_v_real = np.arctan2(sphi_v_real, cphi_v_real)

    # ======================================================
    # v0.6 legacy geometry
    # ======================================================
    elif geometry_mode in ["legacy_sr_discrete_uv", "legacy"]:

        if u_v_test_raw is None:
            raise ValueError(
                "u_v_test_raw is required for legacy real reconstruction."
            )

        if u_v_bins is None:
            raise ValueError(
                "u_v_bins is required for legacy real reconstruction."
            )

        s_r_real = X_cont_test[:, 0]
        u_r_real = np.tanh(s_r_real)

        cphi_r_real = X_cont_test[:, 1].copy()
        sphi_r_real = X_cont_test[:, 2].copy()
        cphi_v_real = X_cont_test[:, 3].copy()
        sphi_v_real = X_cont_test[:, 4].copy()

        u_v_real = np.clip(
            u_v_test_raw.copy(),
            u_v_bins[0] + eps,
            u_v_bins[-1] - eps,
        )

        u_r_q_real = None
        u_v_q_real = None
        phi_r_q_real = None
        phi_v_q_real = None
        phi_r_real = np.arctan2(sphi_r_real, cphi_r_real)
        phi_v_real = np.arctan2(sphi_v_real, cphi_v_real)

    else:
        raise ValueError(f"Unknown geometry_mode: {geometry_mode}")

    cphi_r_real, sphi_r_real = renorm_cos_sin(cphi_r_real, sphi_r_real)
    cphi_v_real, sphi_v_real = renorm_cos_sin(cphi_v_real, sphi_v_real)

    sin_tr_real = np.sqrt(np.maximum(0.0, 1.0 - u_r_real**2))

    x_real = C[0] + R * sin_tr_real * cphi_r_real
    y_real = C[1] + R * sin_tr_real * sphi_r_real
    z_real = C[2] + R * u_r_real

    sin_tv_real = np.sqrt(np.maximum(0.0, 1.0 - u_v_real**2))

    vx_real = sin_tv_real * cphi_v_real
    vy_real = sin_tv_real * sphi_v_real
    vz_real = u_v_real

    vnorm = np.sqrt(vx_real**2 + vy_real**2 + vz_real**2) + 1e-12
    vx_real /= vnorm
    vy_real /= vnorm
    vz_real /= vnorm

    return {
        "geometry_mode": geometry_mode,

        "E_real": E_real,
        "logE_real": logE_real,

        "u_r_q_real": u_r_q_real if geometry_mode != "legacy_sr_discrete_uv" else None,
        "u_v_q_real": u_v_q_real if geometry_mode != "legacy_sr_discrete_uv" else None,
        "phi_r_q_real": phi_r_q_real,
        "phi_v_q_real": phi_v_q_real,
        "phi_r_real": phi_r_real,
        "phi_v_real": phi_v_real,

        "s_r_real": s_r_real,
        "u_r_real": u_r_real,
        "u_v_real": u_v_real,

        "cphi_r_real": cphi_r_real,
        "sphi_r_real": sphi_r_real,
        "cphi_v_real": cphi_v_real,
        "sphi_v_real": sphi_v_real,

        "x_real": x_real,
        "y_real": y_real,
        "z_real": z_real,

        "vx_real": vx_real,
        "vy_real": vy_real,
        "vz_real": vz_real,
    }