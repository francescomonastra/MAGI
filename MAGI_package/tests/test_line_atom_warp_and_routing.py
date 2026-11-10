"""The 511 keV spike on CryoSphere-CR (v0.9.4).

Geant4 writes line photons at the exact catalogued energy, so a line is an
atom in the empirical energy CDF. fit_cdf_warp_knots put several quantile
knots on it and min_gap nudged them ~1e-6 apart, so the flow's fixed warp
alone mapped the atom's probability mass into a few-eV sliver: an untrained
flow already emitted a spike at 511 keV. The fixes under test:
  - fit_cdf_warp_knots(exclude_y=line positions) drops line events before the
    fit, and warns whenever an atom is left in;
  - flow_line_routed_nll trains each event on its own component only
    (energy_line_routing="supervised"), so the flow never sees a line photon;
  - line_sampling="delta" emits line events at exactly the line position.
"""
import warnings

import numpy as np
import pytest
import tensorflow as tf
from scipy.special import ndtr

from magi.core.losses import flow_line_mixture_nll, flow_line_routed_nll
from magi.data.preprocessing import fit_cdf_warp_knots

LINE_Y = -0.29158  # log10(0.510999 MeV)


def _spectrum_with_atom(n=100_000, atom_frac=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n_atom = int(n * atom_frac)
    return np.r_[rng.normal(0.0, 0.5, n - n_atom), np.full(n_atom, LINE_Y)]


def _warp_mass_near(yk, zk, y0, half=5e-6):
    """N(0,1) probability the warp sends into [y0 - half, y0 + half]."""
    return ndtr(np.interp(y0 + half, yk, zk)) - ndtr(np.interp(y0 - half, yk, zk))


def test_atom_builds_a_spike_and_warns():
    y = _spectrum_with_atom()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yk, zk = fit_cdf_warp_knots(y, n_knots=256, eps=1e-4)
    assert any("atom" in str(x.message) for x in w)
    assert _warp_mass_near(yk, zk, LINE_Y) > 0.01


def test_exclude_y_removes_the_spike_without_warning():
    y = _spectrum_with_atom()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yk, zk = fit_cdf_warp_knots(y, n_knots=256, eps=1e-4, exclude_y=[LINE_Y])
    assert not any("atom" in str(x.message) for x in w)
    assert _warp_mass_near(yk, zk, LINE_Y) < 1e-4
    assert np.all(np.diff(yk) > 0)


def test_routed_nll_matches_mixture_when_routing_is_unambiguous():
    """Continuum event far from the line: the line density is ~0, so the
    marginal mixture reduces to the continuum term and both losses agree."""
    y = tf.constant([0.3])
    gate = tf.constant([[1.5, -0.5]])
    flp = tf.constant([-0.7])
    mu, ls = tf.constant([LINE_Y]), tf.constant([-12.0])
    routed = flow_line_routed_nll(y, gate, flp, mu, ls, tf.constant([[1.0, 0.0]]))
    mixture = flow_line_mixture_nll(y, gate, flp, mu, ls)
    np.testing.assert_allclose(routed.numpy(), mixture.numpy(), rtol=1e-6)


def test_routed_nll_gives_flow_no_credit_for_line_events():
    """A line-routed event's loss must not depend on the flow's density at
    all - that is the whole point: no gradient rewards a spike there."""
    y = tf.constant([LINE_Y])
    gate = tf.constant([[0.0, 1.0]])
    mu, ls = tf.constant([LINE_Y]), tf.constant([-12.0])
    route = tf.constant([[0.0, 1.0]])
    flp = tf.Variable([5.0])
    with tf.GradientTape() as tape:
        loss = flow_line_routed_nll(y, gate, flp, mu, ls, route)
    g = tape.gradient(loss, flp)
    assert g is None or float(tf.reduce_sum(tf.abs(g))) == 0.0
    assert np.isfinite(loss.numpy()).all()


def test_routed_nll_is_finite_with_far_line():
    """A continuum event 1 decade from a 1e-6-wide line: the line log density
    is hugely negative, and the zero route weight must not turn it into NaN."""
    y = tf.constant([1.0])
    out = flow_line_routed_nll(y, tf.constant([[0.0, 0.0]]), tf.constant([-1.0]),
                               tf.constant([LINE_Y]), tf.constant([-30.0]),
                               tf.constant([[1.0, 0.0]]))
    assert np.isfinite(out.numpy()).all()


@pytest.mark.parametrize("kw", [{"energy_line_routing": "bogus"}, {"line_sampling": "bogus"}])
def test_model_rejects_unknown_modes(kw):
    from magi.core.model import CVAE_MixEnergy_ContPhi_TaskAdaptive
    with pytest.raises(ValueError):
        CVAE_MixEnergy_ContPhi_TaskAdaptive(n_types=2, line_positions_y=[LINE_Y], **kw)
