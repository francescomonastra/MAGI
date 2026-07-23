"""
Loss functions and low-level tensor utilities for MAGI.
"""

import math

import tensorflow as tf

_LOG_2PI = float(math.log(2.0 * math.pi))


def gaussian_nll(x, mu, log_sigma):
    """
    Gaussian negative log-likelihood (up to an additive constant).

    Parameters
    ----------
    x : tf.Tensor
        True values.
    mu : tf.Tensor
        Predicted mean.
    log_sigma : tf.Tensor
        Predicted log standard deviation.

    Returns
    -------
    tf.Tensor
        Element-wise Gaussian NLL.
    """
    sigma2 = tf.exp(2.0 * log_sigma)
    return 0.5 * ((tf.square(x - mu) / sigma2) + 2.0 * log_sigma)


def gaussian_mixture_nll(y, gate_logits, comp_mu, comp_logsigma):
    """
    Mixture-of-Gaussians negative log-likelihood (up to an additive
    constant, consistent with gaussian_nll's convention of dropping the
    log(2*pi) term - that constant is identical across every component, so
    it factors out of the log-sum-exp as a shared additive shift and can be
    dropped the same way here without affecting gradients).

    Parameters
    ----------
    y : tf.Tensor
        True values, shape (batch,).
    gate_logits : tf.Tensor
        Unnormalized component logits, shape (batch, K).
    comp_mu : tf.Tensor
        Per-component means, shape (batch, K).
    comp_logsigma : tf.Tensor
        Per-component log standard deviations, shape (batch, K).

    Returns
    -------
    tf.Tensor
        Per-sample mixture NLL, shape (batch,).
    """
    log_pi = tf.nn.log_softmax(gate_logits, axis=-1)
    sigma2 = tf.exp(2.0 * comp_logsigma)
    log_normal = -0.5 * (tf.square(y[:, None] - comp_mu) / sigma2 + 2.0 * comp_logsigma)
    return -tf.reduce_logsumexp(log_pi + log_normal, axis=-1)


def gaussian_logpdf(y, mu, log_sigma):
    """
    Fully normalized Gaussian log density (with the log(2*pi) constant),
    shape broadcast of the inputs.

    Unlike gaussian_nll (which drops the shared constant), this keeps the
    constant so the value is a true log density. Required when a Gaussian
    component is mixed in a log-sum-exp with a normalizing flow's log_prob
    (which is fully normalized) - the constant no longer factors out
    because the flow term does not carry it.
    """
    sigma2 = tf.exp(2.0 * log_sigma)
    return -0.5 * (tf.square(y - mu) / sigma2 + _LOG_2PI) - log_sigma


def flow_line_mixture_nll(
    y, gate_logits, flow_log_prob, line_mu, line_logsigma
):
    """
    Negative log-likelihood of a mixture whose continuum component is a
    normalizing flow and whose remaining components are fixed-position
    Gaussian lines:

        p(y) = pi_0 * f_flow(y)  +  sum_l pi_l * N(y; line_mu[l], sigma)

    All component log densities are fully normalized (the flow's log_prob
    and gaussian_logpdf both carry the log(2*pi) constant), so they combine
    correctly in a single log-sum-exp - do NOT substitute gaussian_nll /
    gaussian_mixture_nll here (those drop the constant, which is only valid
    when every component is Gaussian and shares it).

    Parameters
    ----------
    y : tf.Tensor
        True values, shape (batch,).
    gate_logits : tf.Tensor
        Unnormalized logits over (1 + n_lines) slots - column 0 is the
        continuum (flow) slot, columns 1..n_lines are the lines, matching
        the {continuum, line_1..line_L} ordering used everywhere else.
    flow_log_prob : tf.Tensor
        The continuum flow's log density at y, shape (batch,).
    line_mu : tf.Tensor
        Fixed line positions in y-space, shape (n_lines,).
    line_logsigma : tf.Tensor
        Per-line log std, shape (n_lines,) (a scalar 0-D tensor is also
        accepted and broadcasts to every line).

    Returns
    -------
    tf.Tensor
        Per-sample mixture NLL, shape (batch,).
    """
    log_pi = tf.nn.log_softmax(gate_logits, axis=-1)

    line_logdens = gaussian_logpdf(
        y[:, None], line_mu[None, :], tf.reshape(line_logsigma, (1, -1))
    )  # (batch, n_lines)
    comp_logdens = tf.concat(
        [flow_log_prob[:, None], line_logdens], axis=-1
    )  # (batch, 1 + n_lines)

    return -tf.reduce_logsumexp(log_pi + comp_logdens, axis=-1)


def normalize_2d_pair(pair, eps=1e-12):
    """
    Normalize a 2D vector pair (c, s) to unit norm.

    Parameters
    ----------
    pair : tf.Tensor
        Tensor of shape (batch, 2).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    tf.Tensor
        Normalized tensor of shape (batch, 2).
    """
    norm = tf.sqrt(tf.reduce_sum(tf.square(pair), axis=1, keepdims=True) + eps)
    return pair / norm


def angular_loss_2d(true_pair, pred_pair):
    """
    Angular loss between two 2D direction pairs.

    Both true and predicted pairs are normalized before comparison.
    The loss is:
        1 - cos(delta_angle)

    Parameters
    ----------
    true_pair : tf.Tensor
        True angular pair of shape (batch, 2).
    pred_pair : tf.Tensor
        Predicted angular pair of shape (batch, 2).

    Returns
    -------
    tf.Tensor
        Per-sample angular loss.
    """
    true_pair = normalize_2d_pair(true_pair)
    pred_pair = normalize_2d_pair(pred_pair)

    cos_delta = tf.reduce_sum(true_pair * pred_pair, axis=1)
    cos_delta = tf.clip_by_value(cos_delta, -1.0, 1.0)

    return 1.0 - cos_delta


def smoothed_categorical_ce(labels, logits, n_classes, smoothing=0.0, local_neighbors=False):
    """
    Smoothed categorical cross-entropy.

    If smoothing <= 0, this reduces to standard sparse categorical CE.

    Two smoothing modes are supported:

    1) global smoothing:
       smoothing mass is spread uniformly over all classes

    2) local neighbor smoothing:
       smoothing mass is assigned only to adjacent bins
       (useful for ordered categorical variables like binned u_v)

    Parameters
    ----------
    labels : tf.Tensor
        Integer class labels of shape (batch,).
    logits : tf.Tensor
        Logits of shape (batch, n_classes).
    n_classes : int
        Number of classes.
    smoothing : float
        Total smoothing mass.
    local_neighbors : bool
        If True, distribute smoothing only to left/right neighboring bins.

    Returns
    -------
    tf.Tensor
        Per-sample cross-entropy.
    """
    labels = tf.cast(labels, tf.int32)

    if smoothing <= 0.0:
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )

    onehot = tf.one_hot(labels, depth=n_classes, dtype=tf.float32)

    if not local_neighbors:
        target = onehot * (1.0 - smoothing) + smoothing / tf.cast(n_classes, tf.float32)

    else:
        left_idx = tf.maximum(labels - 1, 0)
        right_idx = tf.minimum(labels + 1, n_classes - 1)

        left = tf.one_hot(left_idx, depth=n_classes, dtype=tf.float32)
        right = tf.one_hot(right_idx, depth=n_classes, dtype=tf.float32)

        same_left = tf.cast(tf.equal(left_idx, labels), tf.float32)
        same_right = tf.cast(tf.equal(right_idx, labels), tf.float32)

        left_w = 0.5 * smoothing * (1.0 - same_left)
        right_w = 0.5 * smoothing * (1.0 - same_right)
        center_w = 1.0 - left_w - right_w

        target = (
            onehot * tf.expand_dims(center_w, 1) +
            left * tf.expand_dims(left_w, 1) +
            right * tf.expand_dims(right_w, 1)
        )

    log_probs = tf.nn.log_softmax(logits, axis=1)
    return -tf.reduce_sum(target * log_probs, axis=1)