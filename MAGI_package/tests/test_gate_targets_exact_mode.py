"""bandwidth_mode='exact' for magi.build_gate_targets (v0.8.2 Phase C follow-up
on the Cu Kalpha1 overshoot, docs/v0.8.1_line_truth.md section 14.2).

Simulated fluorescence lines are exactly monoenergetic - every Cu Kalpha1
event in the real training data shares the identical float64 energy, and
continuum events nearby are sparse singletons (verified directly against
TrainingData/alloutputDSCryoSphereCR.dat). bandwidth_mode='resolution' uses
the *detector's* resolution as a Gaussian kernel to decide which real events
are "line" events, which hands significant weight to real continuum a few eV
away and blurs two lines that sit closer together than a few resolution
widths (Cu Kalpha1/Kalpha2, 21 eV apart). 'exact' assigns membership by a
tight numerical match instead - there is no physical width to approximate.
"""
import numpy as np

import magi


def _lines(*energies_mev):
    return [{"candidate_energy_mev": float(e), "label": f"L{i}"}
            for i, e in enumerate(energies_mev)]


def test_exact_event_gets_full_line_weight_default_floor():
    E_c = 0.00800571
    E = np.array([E_c], dtype=np.float64)
    targets = magi.build_gate_targets(E, [0.0, 1.0], _lines(E_c), bandwidth_mode="exact")
    # continuum_floor defaults to 0.02 -> line weight 1/(1+0.02).
    assert targets.shape == (1, 2)
    np.testing.assert_allclose(targets[0, 1], 1.0 / 1.02, atol=1e-6)
    np.testing.assert_allclose(targets[0, 0], 0.02 / 1.02, atol=1e-6)


def test_off_line_event_is_pure_continuum_even_close_by():
    """An event a few eV from the line - which 'resolution' mode (sigma ~1.7 eV
    for 4 eV FWHM) would still weight heavily - must be pure continuum under
    'exact', since it did not come from the line-emission process."""
    E_c = 0.00800571
    off_by_2ev = E_c + 2e-6  # 2 eV, well inside a 'resolution'-mode kernel
    E = np.array([off_by_2ev], dtype=np.float64)
    targets = magi.build_gate_targets(E, [0.0, 1.0], _lines(E_c), bandwidth_mode="exact")
    assert targets[0, 0] == 1.0
    assert targets[0, 1] == 0.0


def test_close_doublet_no_cross_talk():
    """Cu Kalpha1/Kalpha2 analogue: two lines 21 eV apart, plus continuum
    events scattered between and around them. Each line's true events must
    get full credit and zero leakage onto the other line's slot."""
    e1, e2 = 0.00800571, 0.00798467  # ~21 eV apart, matches real CR data
    n_line1, n_line2 = 50, 30
    rng = np.random.default_rng(0)
    continuum = e2 - 5e-6 + rng.uniform(0, 1, size=200) * (e1 - e2 + 10e-6)
    E = np.concatenate([
        np.full(n_line1, e1), np.full(n_line2, e2), continuum,
    ])
    targets = magi.build_gate_targets(
        E, [e2 - 1e-5, e1 + 1e-5], _lines(e1, e2), bandwidth_mode="exact")

    line1_rows = targets[:n_line1]
    line2_rows = targets[n_line1:n_line1 + n_line2]
    cont_rows = targets[n_line1 + n_line2:]

    np.testing.assert_allclose(line1_rows[:, 1], 1.0 / 1.02, atol=1e-6)
    np.testing.assert_allclose(line1_rows[:, 2], 0.0, atol=1e-9)
    np.testing.assert_allclose(line2_rows[:, 2], 1.0 / 1.02, atol=1e-6)
    np.testing.assert_allclose(line2_rows[:, 1], 0.0, atol=1e-9)
    # Continuum between the lines must be pure continuum on both slots,
    # unlike 'resolution' mode where a Gaussian kernel would hand it
    # significant weight on whichever line it's nearer to.
    np.testing.assert_allclose(cont_rows[:, 0], 1.0, atol=1e-9)
    np.testing.assert_allclose(cont_rows[:, 1:], 0.0, atol=1e-9)


def test_resolution_mode_would_leak_but_exact_does_not():
    """Direct A/B on the same synthetic data: 'resolution' mode hands a
    continuum event 2 eV from the line non-trivial weight; 'exact' does not."""
    E_c = 0.00800571
    E = np.array([E_c + 2e-6], dtype=np.float64)
    resolution_targets = magi.build_gate_targets(
        E, [0.0, 1.0], _lines(E_c), bandwidth_mode="resolution",
        bandwidth_fwhm_mev=4e-6)
    exact_targets = magi.build_gate_targets(
        E, [0.0, 1.0], _lines(E_c), bandwidth_mode="exact")
    assert resolution_targets[0, 1] > 0.05  # non-trivial leakage under 'resolution'
    assert exact_targets[0, 1] == 0.0


def test_invalid_bandwidth_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="bandwidth_mode"):
        magi.build_gate_targets(
            np.array([1.0]), [0.0, 2.0], _lines(1.0), bandwidth_mode="bogus")
