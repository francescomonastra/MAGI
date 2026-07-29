"""Round-trip identity checks for the two invertible mechanisms the v0.8 head
relies on: the coupling prior's affine-coupling bijection (core/priors.py) and
the continuum flow's CDF pre-warp (core/flows.py). Both must be exact
inverses of themselves - a break here corrupts every downstream NLL/sample
silently, since neither raises on its own if the math is subtly wrong.
"""
import numpy as np
import tensorflow as tf

from magi.core.priors import ConditionalCouplingPrior
from magi.core.flows import ConditionalRQSFlow
from magi.data.preprocessing import fit_cdf_warp_knots


def test_coupling_prior_forward_inverse_identity():
    rng = np.random.default_rng(0)
    prior = ConditionalCouplingPrior(
        latent_dim=6, cond_dim=3, n_layers=4, hidden=(8, 8), log_scale_clamp=3.0,
    )
    batch = 32
    cond = tf.constant(rng.normal(size=(batch, 3)).astype(np.float32))

    u = tf.constant(rng.normal(size=(batch, 6)).astype(np.float32))
    z = prior._forward(u, cond)
    u_rec, _ = prior._inverse(z, cond)
    np.testing.assert_allclose(u.numpy(), u_rec.numpy(), atol=1e-4)

    z2 = tf.constant(rng.normal(size=(batch, 6)).astype(np.float32))
    u2, _ = prior._inverse(z2, cond)
    z2_rec = prior._forward(u2, cond)
    np.testing.assert_allclose(z2.numpy(), z2_rec.numpy(), atol=1e-4)


def test_coupling_prior_log_prob_finite_and_matches_change_of_variables():
    rng = np.random.default_rng(1)
    prior = ConditionalCouplingPrior(latent_dim=4, cond_dim=2, n_layers=2, hidden=(8,))
    cond = tf.constant(rng.normal(size=(16, 2)).astype(np.float32))
    z = prior.sample(cond)
    lp = prior.log_prob(z, cond)
    assert np.all(np.isfinite(lp.numpy()))

    u, sum_log_det = prior._inverse(z, cond)
    log_base = -0.5 * np.sum(u.numpy() ** 2 + np.log(2 * np.pi), axis=1)
    np.testing.assert_allclose(lp.numpy(), log_base - sum_log_det.numpy(), atol=1e-4)


def test_cdf_warp_forward_inverse_identity():
    rng = np.random.default_rng(2)
    y = rng.normal(loc=1.0, scale=2.0, size=5000)
    y_knots, z_knots = fit_cdf_warp_knots(y, n_knots=128)

    flow = ConditionalRQSFlow(
        feat_dim=1, y_mean=0.0, y_scale=1.0, n_bins=4, n_transforms=1,
        warp_mode="cdf", warp_y_knots=y_knots, warp_z_knots=z_knots,
    )

    y_test = tf.constant(rng.normal(loc=1.0, scale=2.0, size=64).astype(np.float32))
    y_test = tf.clip_by_value(y_test, float(y_knots[0]), float(y_knots[-1]))
    w, _ = flow._warp_forward(y_test)
    y_rec = flow._warp_inverse(w)
    np.testing.assert_allclose(y_test.numpy(), y_rec.numpy(), atol=1e-3)

    w_test = tf.constant(rng.normal(size=64).astype(np.float32))
    w_test = tf.clip_by_value(w_test, float(z_knots[0]), float(z_knots[-1]))
    y_from_w = flow._warp_inverse(w_test)
    w_rec, _ = flow._warp_forward(y_from_w)
    np.testing.assert_allclose(w_test.numpy(), w_rec.numpy(), atol=1e-3)


def test_rqs_flow_sample_and_log_prob_are_finite():
    rng = np.random.default_rng(3)
    flow = ConditionalRQSFlow(
        feat_dim=4, y_mean=0.0, y_scale=1.0, n_bins=6, n_transforms=2,
        warp_mode="affine",
    )
    feat = tf.constant(rng.normal(size=(20, 4)).astype(np.float32))
    y = flow.sample(feat)
    assert np.all(np.isfinite(y.numpy()))
    lp = flow.log_prob(y, feat)
    assert np.all(np.isfinite(lp.numpy()))
