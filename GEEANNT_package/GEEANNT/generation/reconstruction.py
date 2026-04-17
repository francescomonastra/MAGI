"""
Reconstruction utilities for generated and real physical variables.
"""

import numpy as np


def renorm_cos_sin(c, s, eps=1e-12):
    r = np.sqrt(c * c + s * s) + eps
    return c / r, s / r


def u_from_s(s):
    return np.tanh(s)


def reconstruct_generated_features(
    gen_pack,
    s_r_mean,
    s_r_std,
    energy_bins,
    u_v_bins,
    energy_mode="uniform",
    rng=None,
):
    """
    Convert raw generated model outputs into physical feature arrays.
    """
    from .sampling import energy_from_idx

    y_cont_gen_s = gen_pack["y_cont_gen_s"]

    y_cont_gen = y_cont_gen_s.copy()
    y_cont_gen[:, 0] = y_cont_gen_s[:, 0] * s_r_std + s_r_mean

    s_r_gen = y_cont_gen[:, 0]
    cphi_r_gen = y_cont_gen[:, 1]
    sphi_r_gen = y_cont_gen[:, 2]
    cphi_v_gen = y_cont_gen[:, 3]
    sphi_v_gen = y_cont_gen[:, 4]

    E_gen = energy_from_idx(
        gen_pack["energy_idx_gen"],
        energy_bins=energy_bins,
        mode=energy_mode,
        rng=rng,
    )

    logE_gen = np.log10(E_gen)
    u_v_gen = gen_pack["uv_value_gen"].copy()
    u_v_gen = np.clip(u_v_gen, u_v_bins[0], u_v_bins[-1])

    return {
        **gen_pack,
        "y_cont_gen": y_cont_gen,
        "s_r_gen": s_r_gen,
        "cphi_r_gen": cphi_r_gen,
        "sphi_r_gen": sphi_r_gen,
        "cphi_v_gen": cphi_v_gen,
        "sphi_v_gen": sphi_v_gen,
        "E_gen": E_gen,
        "logE_gen": logE_gen,
        "u_v_gen": u_v_gen,
    }


def reconstruct_generated_physics(
    reco_pack,
    center=(0.0, 0.0, -507.66),
    radius=100.0,
    eps=1e-6,
):
    """
    Reconstruct x, y, z, vx, vy, vz from generated feature variables.
    """
    C = np.asarray(center, dtype=np.float64)
    R = float(radius)

    s_r_gen = reco_pack["s_r_gen"]
    cphi_r_gen = reco_pack["cphi_r_gen"]
    sphi_r_gen = reco_pack["sphi_r_gen"]
    cphi_v_gen = reco_pack["cphi_v_gen"]
    sphi_v_gen = reco_pack["sphi_v_gen"]
    u_v_gen = np.clip(reco_pack["u_v_gen"], -1.0 + eps, 1.0 - eps)

    u_r_gen = u_from_s(s_r_gen)

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
        "u_r_gen": u_r_gen,
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
    u_v_test_raw,
    u_v_bins,
    center=(0.0, 0.0, -507.66),
    radius=100.0,
    eps=1e-6,
):
    """
    Reconstruct physical quantities from real test-set features.
    """
    C = np.asarray(center, dtype=np.float64)
    R = float(radius)

    E_real = E_test_raw.copy()
    logE_real = np.log10(E_real)

    s_r_real = X_cont_test[:, 0]
    u_r_real = np.tanh(s_r_real)

    cphi_r_real = X_cont_test[:, 1].copy()
    sphi_r_real = X_cont_test[:, 2].copy()
    cphi_r_real, sphi_r_real = renorm_cos_sin(cphi_r_real, sphi_r_real)

    sin_tr_real = np.sqrt(np.maximum(0.0, 1.0 - u_r_real**2))

    x_real = C[0] + R * sin_tr_real * cphi_r_real
    y_real = C[1] + R * sin_tr_real * sphi_r_real
    z_real = C[2] + R * u_r_real

    u_v_real = np.clip(u_v_test_raw.copy(), u_v_bins[0] + eps, u_v_bins[-1] - eps)

    cphi_v_real = X_cont_test[:, 3].copy()
    sphi_v_real = X_cont_test[:, 4].copy()
    cphi_v_real, sphi_v_real = renorm_cos_sin(cphi_v_real, sphi_v_real)

    sin_tv_real = np.sqrt(np.maximum(0.0, 1.0 - u_v_real**2))

    vx_real = sin_tv_real * cphi_v_real
    vy_real = sin_tv_real * sphi_v_real
    vz_real = u_v_real

    return {
        "E_real": E_real,
        "logE_real": logE_real,
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