"""A3: the direction in the local frame of the crossing point (mu, psi), inward by construction."""
import numpy as np
import pandas as pd

import magi
from magi.core.geometry import direction_to_local, local_to_direction

CENTER, R = (0.0, 0.0, 5.0), 105.0


def _unit(rng, n):
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _table(rng, n):
    """Crossings on the sphere with isotropic directions: about half point outward."""
    pos = np.asarray(CENTER) + R * _unit(rng, n)
    v = _unit(rng, n)
    return pd.DataFrame({
        "ParticleName": rng.choice(["gamma", "e-"], n),
        "Energy": 10.0 ** rng.uniform(-3, 1, n),
        "X": pos[:, 0], "Y": pos[:, 1], "Z": pos[:, 2],
        "Vx": v[:, 0], "Vy": v[:, 1], "Vz": v[:, 2],
    })


def _inward_cosine(x, y, z, vx, vy, vz):
    r = np.stack([x, y, z], 1) - np.asarray(CENTER)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    return -np.einsum("ij,ij->i", r, np.stack([vx, vy, vz], 1))


def test_round_trip_and_sign_including_the_poles():
    rng = np.random.default_rng(0)
    n = 5000
    u_r = np.concatenate([rng.uniform(-1, 1, n - 2), [1.0, -1.0]])
    phi_r = rng.uniform(-np.pi, np.pi, n)
    v = _unit(rng, n)
    u_v, phi_v = v[:, 2], np.arctan2(v[:, 1], v[:, 0])
    mu, psi = direction_to_local(u_r, phi_r, u_v, phi_v)
    s = np.sqrt(1 - u_r ** 2)
    r_hat = np.stack([s * np.cos(phi_r), s * np.sin(phi_r), u_r], 1)
    np.testing.assert_allclose(mu, -np.einsum("ij,ij->i", r_hat, v), atol=1e-12)
    inward = mu > 0
    np.testing.assert_allclose(local_to_direction(u_r[inward], phi_r[inward], mu[inward], psi[inward]),
                               v[inward], atol=1e-10)
    # an outward mu is clipped to the tangent plane, never past it
    w = local_to_direction(u_r[~inward], phi_r[~inward], mu[~inward], psi[~inward])
    assert np.all(-np.einsum("ij,ij->i", r_hat[~inward], w) >= -1e-12)


def _features(rng, n=20000):
    df = _table(rng, n)
    prep = magi.build_physical_features(df, center=CENTER, radius=R)
    fp = magi.build_feature_dataframe(prep, geometry_transform="quantile_u_r_mu_phi_r_psi",
                                      n_quantiles=2000, energy_transform="log10")
    return df, fp


def test_outward_crossings_are_dropped_and_columns_are_local():
    rng = np.random.default_rng(1)
    df, fp = _features(rng)
    n_out = int(np.sum(_inward_cosine(*(df[c].to_numpy() for c in ("X", "Y", "Z", "Vx", "Vy", "Vz"))) <= 0))
    meta = fp["geometry_metadata"]
    assert meta["n_outward_dropped"] == n_out and len(fp["feat"]) == len(df) - n_out
    assert meta["X_cont_cols"] == ["u_r_q", "mu_q", "phi_r_q", "psi_q"]
    assert set(fp["quantile_transformers"]) == {"qt_u_r", "qt_mu", "qt_phi_r", "qt_psi"}
    assert len(fp["filtered_prep"]["features"]) == len(fp["feat"])


def test_real_reconstruction_recovers_the_kept_crossings():
    rng = np.random.default_rng(2)
    _, fp = _features(rng)
    X = fp["feat"][fp["geometry_metadata"]["X_cont_cols"]].to_numpy(dtype=np.float64)
    f0 = fp["filtered_prep"]["features"]
    out = magi.reconstruct_real_test_physics(
        X, f0["Energy"].to_numpy(), center=CENTER, radius=R,
        geometry_metadata={**fp["geometry_metadata"], **fp["quantile_transformers"]})
    assert out["geometry_mode"] == "quantile_u_r_mu_phi_r_psi"
    np.testing.assert_allclose(out["u_r_real"], f0["u_r"], atol=1e-4)
    v_true = np.stack([np.sqrt(1 - f0["u_v"] ** 2) * np.cos(f0["phi_v"]),
                       np.sqrt(1 - f0["u_v"] ** 2) * np.sin(f0["phi_v"]), f0["u_v"]], 1)
    v = np.stack([out["vx_real"], out["vy_real"], out["vz_real"]], 1)
    assert np.percentile(np.linalg.norm(v - v_true, axis=1), 99) < 1e-3


def test_generated_directions_are_inward_even_in_the_tails():
    rng = np.random.default_rng(3)
    _, fp = _features(rng)
    n = 50000
    y = rng.standard_normal((n, 4)) * 2.5   # far into the tails of every quantile transform
    y[:10, 1] = -8.0                         # mu at its lowest quantile: the grazing edge
    gen = {"y_cont_gen_s": y.astype(np.float32), "energy_y_gen": np.full(n, -1.0)}
    feat = magi.reconstruct_generated_features(
        gen, geometry_metadata={**fp["geometry_metadata"], **fp["quantile_transformers"]},
        energy_head_mode="mixture", energy_transform="log10")
    phys = magi.reconstruct_generated_physics(feat, center=CENTER, radius=R)
    mu = _inward_cosine(phys["x_gen"], phys["y_gen"], phys["z_gen"],
                        phys["vx_gen"], phys["vy_gen"], phys["vz_gen"])
    assert np.all(mu > 0)
    np.testing.assert_allclose(np.linalg.norm(
        np.stack([phys["vx_gen"], phys["vy_gen"], phys["vz_gen"]], 1), axis=1), 1.0, atol=1e-12)
