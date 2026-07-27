"""
Validation metrics for generated vs real samples.
"""

import numpy as np
from scipy.stats import wasserstein_distance


def _maybe_wasserstein(out, name, real_pack, gen_pack, real_key, gen_key):
    if real_key in real_pack and gen_key in gen_pack:
        if real_pack[real_key] is not None and gen_pack[gen_key] is not None:
            out[name] = wasserstein_distance(real_pack[real_key], gen_pack[gen_key])


def compute_wasserstein_scores(real_pack, gen_pack):
    scores = {
        "logE": wasserstein_distance(real_pack["logE_real"], gen_pack["logE_gen"]),

        "u_r": wasserstein_distance(real_pack["u_r_real"], gen_pack["u_r_gen"]),
        "u_v": wasserstein_distance(real_pack["u_v_real"], gen_pack["u_v_gen"]),

        "x": wasserstein_distance(real_pack["x_real"], gen_pack["x_gen"]),
        "y": wasserstein_distance(real_pack["y_real"], gen_pack["y_gen"]),
        "z": wasserstein_distance(real_pack["z_real"], gen_pack["z_gen"]),

        "cphi_r": wasserstein_distance(real_pack["cphi_r_real"], gen_pack["cphi_r_gen"]),
        "sphi_r": wasserstein_distance(real_pack["sphi_r_real"], gen_pack["sphi_r_gen"]),
        "cphi_v": wasserstein_distance(real_pack["cphi_v_real"], gen_pack["cphi_v_gen"]),
        "sphi_v": wasserstein_distance(real_pack["sphi_v_real"], gen_pack["sphi_v_gen"]),

        "vx": wasserstein_distance(real_pack["vx_real"], gen_pack["vx_gen"]),
        "vy": wasserstein_distance(real_pack["vy_real"], gen_pack["vy_gen"]),
        "vz": wasserstein_distance(real_pack["vz_real"], gen_pack["vz_gen"]),
    }

    _maybe_wasserstein(scores, "phi_r", real_pack, gen_pack, "phi_r_real", "phi_r_gen")
    _maybe_wasserstein(scores, "phi_v", real_pack, gen_pack, "phi_v_real", "phi_v_gen")

    _maybe_wasserstein(scores, "u_r_q", real_pack, gen_pack, "u_r_q_real", "u_r_q_gen")
    _maybe_wasserstein(scores, "u_v_q", real_pack, gen_pack, "u_v_q_real", "u_v_q_gen")
    _maybe_wasserstein(scores, "phi_r_q", real_pack, gen_pack, "phi_r_q_real", "phi_r_q_gen")
    _maybe_wasserstein(scores, "phi_v_q", real_pack, gen_pack, "phi_v_q_real", "phi_v_q_gen")

    return scores

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493   # 1 / (2 sqrt(2 ln 2))


def compute_line_integral_recovery(
    E_real,
    E_gen,
    matched_lines,
    energy_bins,
    energy_component_idx_gen=None,
    match_tolerance_bins=2.0,
    resolution_mev=None,
    n_continuum_components=1,
    resolution_ev=None,
    window_sigma=5.0,
    continuum_subtract=True,
    sideband_scale=(2.0, 3.0),
):
    """
    Per-line real-vs-generated recovery check for the v0.8 mixture energy
    head: for each matched candidate line, compare how many real vs
    generated events fall within a small window around its energy.

    Three things make the raw in-window count ratio a poor line metric, and
    all three are corrected here (see docs/v0.8.1_improvement_plan.md and the
    v0.8.1 assessment):

    1. **Sample size.** Generation is routinely capped below the real event
       count (e.g. 1e6 generated vs 3.44e6 real for CryoSphere-CR), so a raw
       n_gen/n_real ratio is deflated by exactly that factor. `recovery_ratio`
       is therefore normalized by N_real/N_gen. The uncorrected value is still
       reported as `recovery_ratio_raw` for continuity with earlier runs.
    2. **Window width.** The legacy window is a multiple of the local bin
       width, which on a log grid spanning CR's ~9 decades is ~750 eV at
       9 keV - about 190x the pinned 4 eV line width, so the count is
       dominated by continuum and adjacent lines' windows overlap. Pass
       `resolution_ev` (the detector FWHM the line widths are pinned to) to
       use a resolution-scaled window of +/- `window_sigma` sigma instead.
    3. **Continuum under the line.** Even a narrow window contains continuum.
       With `continuum_subtract=True` the local continuum is estimated from
       symmetric side-bands just outside the window and subtracted from both
       the real and the generated count, so the ratio compares *line* events.

    Uses the same bin-width-relative tolerance convention as
    magi.data.preprocessing.detect_energy_lines (deliberately not reusing
    that function directly - it statistically *detects* peaks, this one
    just counts events near an already-known line position, real vs
    generated).

    Parameters
    ----------
    E_real, E_gen : array-like
        Real and generated energies, in MeV.
    matched_lines : list of dict
        detect_energy_lines(...)["matched_lines"] entries, in the same
        order used to build the model's line_positions_y, so that
        energy_component_idx_gen == n_continuum_components + i corresponds
        to matched_lines[i].
    energy_bins : array-like
        Bin edges (only used to compute the local bin width for the
        tolerance window - the same edges detect_energy_lines used).
    energy_component_idx_gen : array-like of int or None
        If given (from generation/sampling.py's dual-mode output), also
        reports each line's routing fraction - the fraction of generated
        events the model itself assigned to that line's mixture component -
        as an independent cross-check against the windowed count ratio.
        Large disagreement between the two suggests the model routes to a
        line component but samples far from its center (line_logsigma
        converged too large), or that the line is pinned away from where the
        real events actually are.
    match_tolerance_bins : float
        Window half-width, as a multiple of the local bin width at the
        line's energy. Used only when `resolution_ev` is None.
    resolution_mev : float or None
        If given, widens the window to at least this value.
    resolution_ev : float or None
        Detector energy resolution (FWHM, in eV) that the model's line widths
        are pinned to. When given, the window half-width becomes
        `window_sigma * sigma_E` with `sigma_E = resolution_ev / 2.3548`,
        i.e. it tracks the physical line width instead of the binning.
    window_sigma : float
        Window half-width in units of the line sigma, when `resolution_ev`
        is given. 5 sigma captures ~1 - 6e-7 of a Gaussian line.
    continuum_subtract : bool
        Subtract the locally estimated continuum from both counts.
    sideband_scale : (float, float)
        Inner and outer edge of the continuum side-bands, in units of the
        window half-width, on each side of the line. The default (2, 3)
        gives two bands whose total width equals the window width, so the
        in-window continuum estimate is just the side-band count.

    Returns
    -------
    list of dict
        One entry per line: label, origin, candidate_energy_mev, n_real,
        n_gen (raw in-window counts), n_real_line / n_gen_line
        (continuum-subtracted), gen_scale, recovery_ratio (size-normalized
        and, when enabled, continuum-subtracted - target ~= 1),
        recovery_ratio_window (size-normalized, no subtraction),
        recovery_ratio_raw (the legacy uncorrected n_gen/n_real),
        window_half_width_mev, window_mode, overlaps_lines (labels whose
        centre falls inside this window - the ratio is not trustworthy when
        this is non-empty), real_fraction, real_fraction_line, and, if
        energy_component_idx_gen was given, component_fraction plus
        component_recovery (routing fraction / real line fraction, a
        window-free cross-check).
    """
    E_real = np.asarray(E_real, dtype=np.float64)
    E_gen = np.asarray(E_gen, dtype=np.float64)
    edges = np.asarray(energy_bins, dtype=np.float64)

    comp_idx = (
        np.asarray(energy_component_idx_gen)
        if energy_component_idx_gen is not None
        else None
    )

    gen_scale = (E_real.size / E_gen.size) if E_gen.size else float("nan")
    sb_in, sb_out = float(sideband_scale[0]), float(sideband_scale[1])

    # Window half-widths first, so overlaps between adjacent lines can be
    # flagged before any counting.
    centres = [float(line["candidate_energy_mev"]) for line in matched_lines]
    tols = []
    for E_c in centres:
        if resolution_ev is not None:
            sigma_e = float(resolution_ev) * 1e-6 * FWHM_TO_SIGMA   # eV -> MeV
            tol = float(window_sigma) * sigma_e
            mode = f"{window_sigma:g}sigma@{resolution_ev:g}eV"
        else:
            bin_i = int(np.clip(np.searchsorted(edges, E_c) - 1, 0, len(edges) - 2))
            tol = match_tolerance_bins * float(edges[bin_i + 1] - edges[bin_i])
            mode = f"{match_tolerance_bins:g}bins"
        if resolution_mev is not None:
            tol = max(tol, float(resolution_mev))
        tols.append((tol, mode))

    def _count(E, lo, hi):
        return int(np.sum((E >= lo) & (E <= hi)))

    results = []
    for i, line in enumerate(matched_lines):
        E_c = centres[i]
        tol, mode = tols[i]

        n_real = _count(E_real, E_c - tol, E_c + tol)
        n_gen = _count(E_gen, E_c - tol, E_c + tol)

        # Local continuum from the two side-bands, whose combined width is
        # (sb_out - sb_in) * 2 * tol per side; scale it to the window width.
        n_real_cont = n_gen_cont = 0.0
        if continuum_subtract and sb_out > sb_in:
            band_w = (sb_out - sb_in) * tol
            sb_real = (_count(E_real, E_c - sb_out * tol, E_c - sb_in * tol)
                       + _count(E_real, E_c + sb_in * tol, E_c + sb_out * tol))
            sb_gen = (_count(E_gen, E_c - sb_out * tol, E_c - sb_in * tol)
                      + _count(E_gen, E_c + sb_in * tol, E_c + sb_out * tol))
            scale_to_window = (2.0 * tol) / (2.0 * band_w)
            n_real_cont = sb_real * scale_to_window
            n_gen_cont = sb_gen * scale_to_window

        n_real_line = max(n_real - n_real_cont, 0.0)
        n_gen_line = max(n_gen - n_gen_cont, 0.0)

        overlaps = [
            matched_lines[j]["label"]
            for j in range(len(centres))
            if j != i and abs(centres[j] - E_c) <= tol
        ]

        if continuum_subtract:
            recovery = (n_gen_line * gen_scale / n_real_line) if n_real_line > 0 else None
        else:
            recovery = (n_gen * gen_scale / n_real) if n_real else None

        entry = {
            "label": line["label"],
            "origin": line["origin"],
            "candidate_energy_mev": E_c,
            "window_half_width_mev": tol,
            "window_mode": mode,
            "overlaps_lines": overlaps,

            "n_real": n_real,
            "n_gen": n_gen,
            "n_real_continuum": float(n_real_cont),
            "n_gen_continuum": float(n_gen_cont),
            "n_real_line": float(n_real_line),
            "n_gen_line": float(n_gen_line),

            "gen_scale": float(gen_scale),
            "recovery_ratio": recovery,
            "recovery_ratio_window": (n_gen * gen_scale / n_real) if n_real else None,
            "recovery_ratio_raw": (n_gen / n_real if n_real else None),

            "real_fraction": (n_real / E_real.size if E_real.size else None),
            "real_fraction_line": (n_real_line / E_real.size if E_real.size else None),
        }

        if comp_idx is not None:
            comp_frac = float(np.mean(comp_idx == (n_continuum_components + i)))
            entry["component_fraction"] = comp_frac
            real_frac_line = entry["real_fraction_line"]
            entry["component_recovery"] = (
                comp_frac / real_frac_line if real_frac_line else None
            )

        results.append(entry)

    return results


def report_generated_constraints(gen_pack, radius=100.0):
      print("--- CONSTRAINT CHECKS ---")

      print("\nSphere constraint:")
      print("r_gen: mean", gen_pack["r_gen"].mean(),
            "std", gen_pack["r_gen"].std(),
            "min", gen_pack["r_gen"].min(),
            "max", gen_pack["r_gen"].max())
      print("mean |r-R| =", np.mean(np.abs(gen_pack["r_gen"] - radius)))

      vnorm = np.sqrt(gen_pack["vx_gen"]**2 + gen_pack["vy_gen"]**2 + gen_pack["vz_gen"]**2)
      print("\nDirection norm:")
      print("||v||: mean", vnorm.mean(),
            "std", vnorm.std(),
            "min", vnorm.min(),
            "max", vnorm.max())
      print("mean ||v||-1| =", np.mean(np.abs(vnorm - 1.0)))

      print("\nUnit circle checks:")
      print("phi_r norm: mean", gen_pack["phi_r_norm_gen"].mean(),
            "std", gen_pack["phi_r_norm_gen"].std(),
            "min", gen_pack["phi_r_norm_gen"].min(),
            "max", gen_pack["phi_r_norm_gen"].max())
      print("phi_v norm: mean", gen_pack["phi_v_norm_gen"].mean(),
            "std", gen_pack["phi_v_norm_gen"].std(),
            "min", gen_pack["phi_v_norm_gen"].min(),
            "max", gen_pack["phi_v_norm_gen"].max())

      print("\nPhysical variables:")
      print("E_gen: min", gen_pack["E_gen"].min(), "max", gen_pack["E_gen"].max())
      print("u_r_gen min/max:", gen_pack["u_r_gen"].min(), gen_pack["u_r_gen"].max())
      print("u_v_gen min/max:", gen_pack["u_v_gen"].min(), gen_pack["u_v_gen"].max())
      print("fraction u_v_gen > 0:", np.mean(gen_pack["u_v_gen"] > 0))

      print("\nQuantile variables:")

      for key in ["u_r_q_gen", "u_v_q_gen", "phi_r_q_gen", "phi_v_q_gen"]:
            if key in gen_pack and gen_pack[key] is not None:
                  print(
                        f"{key}: min={gen_pack[key].min()} "
                        f"max={gen_pack[key].max()} "
                        f"mean={gen_pack[key].mean()} "
                        f"std={gen_pack[key].std()}"
                  )

      print(
            "fraction |u_r| > 1:",
            np.mean(np.abs(gen_pack["u_r_gen"]) > 1.0)
      )

      print(
            "fraction |u_v| > 1:",
            np.mean(np.abs(gen_pack["u_v_gen"]) > 1.0)
      )