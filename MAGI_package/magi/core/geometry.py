"""
Geometry utilities for MAGI.
"""

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