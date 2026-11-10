"""
Geometry utilities for MAGI.
"""

import numpy as np
import tensorflow as tf
from .losses import normalize_2d_pair


# ==========================================================
# Legacy v0.6 geometry: s_r = arctanh(u_r)
# ==========================================================

def u_r_from_sr(s_r):
    """
    Reconstruct u_r = cos(theta_r) from s_r = arctanh(u_r).
    """
    return tf.tanh(tf.squeeze(s_r, axis=1))


def xy_from_sr_phi(s_r, phi_pair, sphere_R):
    """
    Legacy v0.6 helper.

    Reconstruct x and y coordinates on a sphere from:
      - s_r = arctanh(u_r)
      - phi_pair = (cos(phi_r), sin(phi_r))
    """
    u_r = u_r_from_sr(s_r)

    phi_pair = normalize_2d_pair(phi_pair)
    c = phi_pair[:, 0]
    s = phi_pair[:, 1]

    sin_theta = tf.sqrt(tf.maximum(0.0, 1.0 - u_r * u_r))

    x = sphere_R * sin_theta * c
    y = sphere_R * sin_theta * s

    return x, y


def xyz_from_sr_phi(s_r, phi_pair, sphere_R, center_z=0.0):
    """
    Legacy v0.6 helper.

    Reconstruct x, y, z coordinates on a sphere from:
      - s_r = arctanh(u_r)
      - phi_pair = (cos(phi_r), sin(phi_r))
    """
    u_r = u_r_from_sr(s_r)

    phi_pair = normalize_2d_pair(phi_pair)
    c = phi_pair[:, 0]
    s = phi_pair[:, 1]

    sin_theta = tf.sqrt(tf.maximum(0.0, 1.0 - u_r * u_r))

    x = sphere_R * sin_theta * c
    y = sphere_R * sin_theta * s
    z = center_z + sphere_R * u_r

    return x, y, z


# ==========================================================
# Shared direction geometry
# ==========================================================

def vxvy_from_uv_phi(uv, phi_pair):
    """
    Reconstruct vx and vy from:
      - uv = cos(theta_v)
      - phi_pair = (cos(phi_v), sin(phi_v))

    Used by legacy v0.6.
    Can also be used in NumPy/TensorFlow diagnostics if uv is already physical.
    """
    phi_pair = normalize_2d_pair(phi_pair)

    c = phi_pair[:, 0]
    s = phi_pair[:, 1]

    uv = tf.clip_by_value(uv, -1.0 + 1e-6, 1.0 - 1e-6)
    sin_theta = tf.sqrt(tf.maximum(0.0, 1.0 - tf.square(uv)))

    vx = sin_theta * c
    vy = sin_theta * s

    return vx, vy


def vxyz_from_uv_phi(uv, phi_pair):
    """
    Reconstruct vx, vy, vz from:
      - uv = cos(theta_v)
      - phi_pair = (cos(phi_v), sin(phi_v))
    """
    phi_pair = normalize_2d_pair(phi_pair)

    c = phi_pair[:, 0]
    s = phi_pair[:, 1]

    uv = tf.clip_by_value(uv, -1.0 + 1e-6, 1.0 - 1e-6)
    sin_theta = tf.sqrt(tf.maximum(0.0, 1.0 - tf.square(uv)))

    vx = sin_theta * c
    vy = sin_theta * s
    vz = uv

    return vx, vy, vz


# ==========================================================
# Legacy v0.6 categorical u_v helper
# ==========================================================

def u_v_from_logits(logits, uv_bin_centers):
    """
    Legacy v0.6 helper.

    Compute expected continuous u_v from predicted categorical logits.
    Not used by v0.7 continuous-geometry models.
    """
    probs = tf.nn.softmax(logits, axis=1)
    centers = tf.reshape(uv_bin_centers, [1, -1])
    uv_exp = tf.reduce_sum(probs * centers, axis=1)

    return uv_exp


# ==========================================================
# v0.7 physical helpers
# ==========================================================

def xy_from_ur_phi(u_r, phi_pair, sphere_R):
    """
    Reconstruct x and y coordinates on a sphere from physical u_r.

    This is useful after inverse-transforming u_r_q -> u_r.
    """
    u_r = tf.squeeze(u_r, axis=-1) if len(u_r.shape) == 2 else u_r
    u_r = tf.clip_by_value(u_r, -1.0 + 1e-6, 1.0 - 1e-6)

    phi_pair = normalize_2d_pair(phi_pair)
    c = phi_pair[:, 0]
    s = phi_pair[:, 1]

    sin_theta = tf.sqrt(tf.maximum(0.0, 1.0 - tf.square(u_r)))

    x = sphere_R * sin_theta * c
    y = sphere_R * sin_theta * s

    return x, y


def xyz_from_ur_phi(u_r, phi_pair, sphere_R, center_z=0.0):
    """
    Reconstruct x, y, z coordinates on a sphere from physical u_r.

    This is useful after inverse-transforming u_r_q -> u_r.
    """
    u_r = tf.squeeze(u_r, axis=-1) if len(u_r.shape) == 2 else u_r
    u_r = tf.clip_by_value(u_r, -1.0 + 1e-6, 1.0 - 1e-6)

    phi_pair = normalize_2d_pair(phi_pair)
    c = phi_pair[:, 0]
    s = phi_pair[:, 1]

    sin_theta = tf.sqrt(tf.maximum(0.0, 1.0 - tf.square(u_r)))

    x = sphere_R * sin_theta * c
    y = sphere_R * sin_theta * s
    z = center_z + sphere_R * u_r

    return x, y, z

# ==========================================================
# v1.0 local direction frame (A3)
# ==========================================================
def local_frame(u_r, phi_r):
    """
    Outward normal r_hat and tangent basis (e_theta, e_phi) at the crossing
    point (u_r, phi_r) of the sphere, in numpy.

    The basis stays orthonormal at the poles (sin(theta) = 0), so the local
    azimuth is defined for every crossing. Same frame as AR-Flow's
    `arflow.transforms.local_frame`.
    """
    u_r = np.asarray(u_r, dtype=np.float64)
    phi_r = np.asarray(phi_r, dtype=np.float64)
    s = np.sqrt(np.clip(1.0 - u_r * u_r, 0.0, None))
    c, n = np.cos(phi_r), np.sin(phi_r)
    r_hat = np.stack([s * c, s * n, u_r], -1)
    e_th = np.stack([u_r * c, u_r * n, -s], -1)
    e_ph = np.stack([-n, c, np.zeros_like(c)], -1)
    return r_hat, e_th, e_ph


def direction_to_local(u_r, phi_r, u_v, phi_v):
    """
    Global direction (u_v, phi_v) -> local (mu, psi) at the crossing point.

    mu = -r_hat . v is the cosine to the inward normal (> 0 for an inward
    crossing) and psi the azimuth of v around it, in [-pi, pi]. In this frame
    the inward condition is mu > 0 at every point of the sphere, which the
    global u_v (the cosine to the z axis) cannot express.
    """
    r_hat, e_th, e_ph = local_frame(u_r, phi_r)
    u_v = np.asarray(u_v, dtype=np.float64)
    phi_v = np.asarray(phi_v, dtype=np.float64)
    s = np.sqrt(np.clip(1.0 - u_v * u_v, 0.0, None))
    v = np.stack([s * np.cos(phi_v), s * np.sin(phi_v), u_v], -1)
    mu = -np.einsum("ij,ij->i", r_hat, v)
    psi = np.arctan2(np.einsum("ij,ij->i", v, e_ph), np.einsum("ij,ij->i", v, e_th))
    return mu, psi


def local_to_direction(u_r, phi_r, mu, psi):
    """
    Inverse of direction_to_local: the unit direction vector, shape (n, 3).

    mu is clipped to [0, 1], so the returned direction never points out of
    the sphere.
    """
    r_hat, e_th, e_ph = local_frame(u_r, phi_r)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)[:, None]
    psi = np.asarray(psi, dtype=np.float64)[:, None]
    t = np.sqrt(1.0 - mu * mu)
    v = -mu * r_hat + t * (np.cos(psi) * e_th + np.sin(psi) * e_ph)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
