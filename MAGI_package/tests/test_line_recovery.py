"""compute_line_integral_recovery on synthetic real/generated spectra with a
known injected line, so the metric's correctness doesn't depend on a real
trained checkpoint. Regression coverage for the normalization + resolution
window + continuum-subtraction fix in docs/v0.8.1_line_truth.md section 2.
"""
import numpy as np

from magi.validation.metrics import compute_line_integral_recovery

LINE_E_MEV = 1.0
RESOLUTION_EV = 10.0
SIGMA_MEV = RESOLUTION_EV * 1e-6 / 2.354820045


def _continuum(rng, n):
    return 10.0 ** rng.uniform(-3.0, 1.0, size=n)


def _line(rng, n, center=LINE_E_MEV, sigma=SIGMA_MEV):
    return rng.normal(loc=center, scale=sigma, size=n)


def _matched_line(label="test_line", energy=LINE_E_MEV):
    return [{"label": label, "origin": "synthetic", "candidate_energy_mev": energy}]


def _energy_bins():
    return np.logspace(-3, 1, 51)


def test_matching_line_and_continuum_recovers_near_one():
    rng = np.random.default_rng(0)
    n_cont_real, n_line_real = 200_000, 2000
    E_real = np.concatenate([_continuum(rng, n_cont_real), _line(rng, n_line_real)])

    # Half the real sample size, same proportional line strength - this is
    # exactly the N_real != N_gen case the normalization fix targets.
    n_cont_gen, n_line_gen = 100_000, 1000
    E_gen = np.concatenate([_continuum(rng, n_cont_gen), _line(rng, n_line_gen)])

    result = compute_line_integral_recovery(
        E_real, E_gen, _matched_line(), _energy_bins(), resolution_ev=RESOLUTION_EV,
    )[0]

    assert result["recovery_ratio"] is not None
    assert 0.8 <= result["recovery_ratio"] <= 1.2
    assert not result["sideband_contaminated"]
    assert result["overlaps_lines"] == []


def test_under_generated_line_is_detected():
    rng = np.random.default_rng(1)
    E_real = np.concatenate([_continuum(rng, 200_000), _line(rng, 2000)])
    # Line under-generated ~5x relative to its proportional share.
    E_gen = np.concatenate([_continuum(rng, 100_000), _line(rng, 100)])

    result = compute_line_integral_recovery(
        E_real, E_gen, _matched_line(), _energy_bins(), resolution_ev=RESOLUTION_EV,
    )[0]

    assert result["recovery_ratio"] is not None
    assert result["recovery_ratio"] < 0.5


def test_sideband_neighbour_nulls_recovery_not_a_blended_one():
    """A real neighbour line just outside the window (in the side-bands)
    contaminates the continuum estimate and must null the ratio; one well
    inside the window (blended, same unresolvable peak) must not - the two
    bugs fixed in docs/v0.8.1_line_truth.md section 9/10.4 note 2."""
    rng = np.random.default_rng(2)
    tol_mev = 5.0 * SIGMA_MEV
    sideband_neighbour = LINE_E_MEV + 2.5 * tol_mev  # inside side-bands, outside window
    blended_neighbour = LINE_E_MEV + 0.1 * tol_mev   # inside the window itself

    E_real = np.concatenate([
        _continuum(rng, 200_000), _line(rng, 2000),
        _line(rng, 500, center=sideband_neighbour),
    ])
    E_gen = np.concatenate([
        _continuum(rng, 100_000), _line(rng, 1000),
        _line(rng, 250, center=sideband_neighbour),
    ])
    neighbour_lines = _matched_line() + [
        {"label": "sideband_neighbour", "origin": "synthetic",
         "candidate_energy_mev": sideband_neighbour},
    ]

    contaminated = compute_line_integral_recovery(
        E_real, E_gen, _matched_line(), _energy_bins(), resolution_ev=RESOLUTION_EV,
        neighbour_lines=neighbour_lines,
    )[0]
    assert contaminated["sideband_contaminated"]
    assert contaminated["recovery_ratio"] is None

    blended_lines = _matched_line() + [
        {"label": "blended_neighbour", "origin": "synthetic",
         "candidate_energy_mev": blended_neighbour},
    ]
    blended = compute_line_integral_recovery(
        E_real, E_gen, _matched_line(), _energy_bins(), resolution_ev=RESOLUTION_EV,
        neighbour_lines=blended_lines,
    )[0]
    assert not blended["sideband_contaminated"]
    assert blended["overlaps_lines"] == ["blended_neighbour"]
    assert blended["recovery_ratio"] is not None
