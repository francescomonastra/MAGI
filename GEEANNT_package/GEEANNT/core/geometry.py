"""
Geometry utilities for GEEANNT.

This module contains differentiable geometry helpers used by the model
to reconstruct physical quantities from learned latent variables.
"""

import tensorflow as tf
from .losses import normalize_2d_pair


def xy_from_sr_phi(s_r, phi_pair, sphere_R):
    """
    Reconstruct x and y coordinates on a sphere from:
      - s_r  : transformed radial variable
      - phi_pair : 2D angular pair (cos(phi_r), sin(phi_r))

    Parameters
    ----------
    s_r : tf.Tensor
        Tensor of shape (batch, 1).
    phi_pair : tf.Tensor
        Tensor of shape (batch, 2).
    sphere_R : float
        Sphere radius.

    Returns
    -------
    x, y : tf.Tensor
        Reconstructed Cartesian coordinates.
    """
    u_r = tf.tanh(tf.squeeze(s_r, axis=1))

    phi_pair = normalize_2d_pair(phi_pair)
    c = phi_pair[:, 0]
    s = phi_pair[:, 1]

    sin_theta = tf.sqrt(tf.maximum(0.0, 1.0 - u_r * u_r))

    x = sphere_R * sin_theta * c
    y = sphere_R * sin_theta * s
    return x, y


def xyz_from_sr_phi(s_r, phi_pair, sphere_R, center_z=0.0):
    """
    Reconstruct x, y, z coordinates on a sphere from:
      - s_r
      - phi_pair

    Parameters
    ----------
    s_r : tf.Tensor
        Tensor of shape (batch, 1).
    phi_pair : tf.Tensor
        Tensor of shape (batch, 2).
    sphere_R : float
        Sphere radius.
    center_z : float
        Optional z-offset of the sphere center.

    Returns
    -------
    x, y, z : tf.Tensor
        Reconstructed Cartesian coordinates.
    """
    u_r = tf.tanh(tf.squeeze(s_r, axis=1))

    phi_pair = normalize_2d_pair(phi_pair)
    c = phi_pair[:, 0]
    s = phi_pair[:, 1]

    sin_theta = tf.sqrt(tf.maximum(0.0, 1.0 - u_r * u_r))

    x = sphere_R * sin_theta * c
    y = sphere_R * sin_theta * s
    z = center_z + sphere_R * u_r

    return x, y, z


def vxvy_from_uv_phi(uv, phi_pair):
    """
    Reconstruct vx and vy from:
      - uv      = cos(theta_v)
      - phi_pair = (cos(phi_v), sin(phi_v))

    Parameters
    ----------
    uv : tf.Tensor
        Tensor of shape (batch,).
    phi_pair : tf.Tensor
        Tensor of shape (batch, 2).

    Returns
    -------
    vx, vy : tf.Tensor
        Reconstructed direction components.
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
      - uv      = cos(theta_v)
      - phi_pair = (cos(phi_v), sin(phi_v))

    Parameters
    ----------
    uv : tf.Tensor
        Tensor of shape (batch,).
    phi_pair : tf.Tensor
        Tensor of shape (batch, 2).

    Returns
    -------
    vx, vy, vz : tf.Tensor
        Reconstructed direction components.
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


def u_r_from_sr(s_r):
    """
    Reconstruct u_r = cos(theta_r) from the transformed radial variable s_r.

    Parameters
    ----------
    s_r : tf.Tensor
        Tensor of shape (batch, 1).

    Returns
    -------
    tf.Tensor
        Tensor of shape (batch,).
    """
    return tf.tanh(tf.squeeze(s_r, axis=1))


def u_v_from_logits(logits, uv_bin_centers):
    """
    Compute the expected continuous u_v value from predicted categorical logits.

    Parameters
    ----------
    logits : tf.Tensor
        Tensor of shape (batch, n_uv_bins).
    uv_bin_centers : tf.Tensor
        Tensor of shape (n_uv_bins,).

    Returns
    -------
    tf.Tensor
        Expected u_v of shape (batch,).
    """
    probs = tf.nn.softmax(logits, axis=1)
    centers = tf.reshape(uv_bin_centers, [1, -1])
    uv_exp = tf.reduce_sum(probs * centers, axis=1)
    return uv_exp