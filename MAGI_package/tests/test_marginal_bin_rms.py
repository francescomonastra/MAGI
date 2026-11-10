"""Per-bin RMS of the marginals (magi.validation.marginal_bin_rms)."""
import numpy as np

import magi


def test_two_samples_of_one_distribution_give_about_one():
    rng = np.random.default_rng(0)
    out = [magi.marginal_bin_rms(rng.normal(size=200_000), rng.normal(size=150_000))["rms_z"]
           for _ in range(5)]
    assert 0.8 < np.mean(out) < 1.25


def test_a_shifted_sample_is_far_above_one():
    rng = np.random.default_rng(1)
    r = magi.marginal_bin_rms(rng.normal(size=100_000), rng.normal(0.1, 1.0, size=100_000))
    assert r["rms_z"] > 3.0 and r["rms_rel"] > 0.05


def test_sizes_do_not_matter_and_a_peak_is_located():
    rng = np.random.default_rng(2)
    real = rng.normal(size=300_000)
    gen = np.concatenate([rng.normal(size=100_000), np.full(3000, 0.5)])   # an excess spike
    r = magi.marginal_bin_rms(real, gen, n_bins=40)
    assert r["worst_bin"]["lo"] <= 0.5 <= r["worst_bin"]["hi"] and r["worst_bin"]["z"] < -5


def test_explicit_edges_keep_every_generated_value():
    r = magi.marginal_bin_rms(np.linspace(0, 1, 1000), np.linspace(-5, 5, 1000), edges=[0.0, 0.5, 1.0])
    assert r["n_bins"] == 2


def test_a_smeared_atom_on_a_bin_edge_is_removed_by_a_window():
    rng = np.random.default_rng(3)
    real = np.concatenate([rng.normal(size=100_000), np.full(5000, 0.0)])
    gen = np.concatenate([rng.normal(size=100_000), rng.normal(0.0, 1e-6, size=5000)])
    assert magi.marginal_bin_rms(real, gen, n_bins=40)["rms_z"] > 3.0
    assert magi.marginal_bin_rms(real, gen, n_bins=40, exclude_windows=[(-0.05, 0.05)])["rms_z"] < 1.5
