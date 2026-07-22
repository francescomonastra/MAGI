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

def compute_line_integral_recovery(
    E_real,
    E_gen,
    matched_lines,
    energy_bins,
    energy_component_idx_gen=None,
    match_tolerance_bins=2.0,
    resolution_mev=None,
    n_continuum_components=1,
):
    """
    Per-line real-vs-generated recovery check for the v0.8 mixture energy
    head: for each matched candidate line, compare how many real vs
    generated events fall within a small window around its energy.

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
        converged too large).
    match_tolerance_bins : float
        Window half-width, as a multiple of the local bin width at the
        line's energy.
    resolution_mev : float or None
        If given, widens the window to at least this value.

    Returns
    -------
    list of dict
        One entry per line: label, origin, candidate_energy_mev, n_real,
        n_gen, recovery_ratio (n_gen / n_real, target ~= 1), real_fraction
        (n_real / len(E_real)), and component_fraction (if
        energy_component_idx_gen was given).
    """
    E_real = np.asarray(E_real, dtype=np.float64)
    E_gen = np.asarray(E_gen, dtype=np.float64)
    edges = np.asarray(energy_bins, dtype=np.float64)

    comp_idx = (
        np.asarray(energy_component_idx_gen)
        if energy_component_idx_gen is not None
        else None
    )

    results = []
    for i, line in enumerate(matched_lines):
        E_c = float(line["candidate_energy_mev"])
        bin_i = int(np.clip(np.searchsorted(edges, E_c) - 1, 0, len(edges) - 2))
        local_width = float(edges[bin_i + 1] - edges[bin_i])
        tol = match_tolerance_bins * local_width
        if resolution_mev is not None:
            tol = max(tol, float(resolution_mev))

        n_real = int(np.sum(np.abs(E_real - E_c) <= tol))
        n_gen = int(np.sum(np.abs(E_gen - E_c) <= tol))

        entry = {
            "label": line["label"],
            "origin": line["origin"],
            "candidate_energy_mev": E_c,
            "n_real": n_real,
            "n_gen": n_gen,
            "recovery_ratio": (n_gen / n_real if n_real else None),
            "real_fraction": (n_real / E_real.size if E_real.size else None),
        }

        if comp_idx is not None:
            entry["component_fraction"] = float(
                np.mean(comp_idx == (n_continuum_components + i))
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