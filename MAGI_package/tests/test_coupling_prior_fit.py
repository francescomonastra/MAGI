"""Coupled-synthetic smoke test for ConditionalCouplingPrior: can it actually
learn a real cond->z coupling, or does it collapse to the unconditional
N(0, I) base it starts at? This is the mechanism behind v0.8's headline
result (real cross-correlations reproduced within +/-0.04/0.02, see
docs/v0.8_v072_comparison.md section 4) - injects a strong, known
per-type mean shift and checks the fitted log-density both improves sharply
from init and clears the density a naive unconditional N(0, I) would give.
Deliberately tiny (latent_dim=4, ~200 Adam steps) so this runs in seconds,
not a claim about convergence quality on the real 8-D/z_cond architecture.
"""
import numpy as np
import tensorflow as tf

from magi.core.priors import ConditionalCouplingPrior

LATENT_DIM = 4
SHIFT = 3.0


def _synthetic_batch(rng, n):
    cond_idx = rng.integers(0, 2, size=n)
    cond = np.eye(2, dtype=np.float32)[cond_idx]
    mu = np.where(cond_idx[:, None] == 0, SHIFT, -SHIFT).astype(np.float32)
    z = mu + rng.normal(size=(n, LATENT_DIM)).astype(np.float32)
    return tf.constant(z), tf.constant(cond)


def test_coupling_prior_fits_injected_mean_shift():
    rng = np.random.default_rng(0)
    prior = ConditionalCouplingPrior(
        latent_dim=LATENT_DIM, cond_dim=2, n_layers=4, hidden=(32, 32),
    )
    z0, cond0 = _synthetic_batch(rng, 512)
    init_log_prob = float(tf.reduce_mean(prior.log_prob(z0, cond0)))

    # log p(z|cond) under the TRUE generative process (unit-variance Gaussian
    # centered on the injected per-type shift) - the ceiling this prior is
    # trying to approach.
    true_mu = np.where(cond0.numpy().argmax(1)[:, None] == 0, SHIFT, -SHIFT)
    true_log_prob = float(np.mean(
        -0.5 * np.sum((z0.numpy() - true_mu) ** 2 + np.log(2 * np.pi), axis=1)
    ))
    # log p(z) under a naive unconditional N(0, I), ignoring cond entirely -
    # what the prior starts at (zero-initialized coupling layers = identity).
    unconditional_log_prob = float(np.mean(
        -0.5 * np.sum(z0.numpy() ** 2 + np.log(2 * np.pi), axis=1)
    ))
    np.testing.assert_allclose(init_log_prob, unconditional_log_prob, atol=1e-3)

    optimizer = tf.keras.optimizers.Adam(1e-2)
    for _ in range(200):
        z, cond = _synthetic_batch(rng, 256)
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(prior.log_prob(z, cond))
        grads = tape.gradient(loss, prior.trainable_variables)
        optimizer.apply_gradients(zip(grads, prior.trainable_variables))

    z_eval, cond_eval = _synthetic_batch(rng, 2000)
    fitted_log_prob = float(tf.reduce_mean(prior.log_prob(z_eval, cond_eval)))

    # Must have moved substantially off the unconditional starting point...
    assert fitted_log_prob > unconditional_log_prob + 5.0
    # ...and land close to the true (cond-aware) density, not just improve.
    assert abs(fitted_log_prob - true_log_prob) < 1.0
