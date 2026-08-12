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
    """1-D Wasserstein distance per physical variable, real vs generated.

    Computed on the reconstructed physical quantities (logE, u_r, u_v, the
    angles, and Cartesian position/direction), plus the quantile-space columns
    when the packs carry them. Distances are in each variable's own units and
    are NOT normalized, so compare them across runs on the same source rather
    than across variables.

    Parameters
    ----------
    real_pack : dict
        Real physical arrays, from reconstruct_real_test_physics. Must use
        that function's "<name>_real" key convention.

    gen_pack : dict
        Generated physical arrays, from reconstruct_generated_physics, using
        its "<name>_gen" key convention.

    Returns
    -------
    dict[str, float]
        Wasserstein distance per variable name, in that variable's own units.

    Raises
    ------
    KeyError
        If either pack is missing one of the twelve core variables (logE,
        u_r, u_v, x, y, z, cphi_r, sphi_r, cphi_v, sphi_v, vx, vy, vz). Only
        the six optional ones - phi_r, phi_v and the four quantile-space
        columns - are skipped when absent.

    Notes
    -----
    This helper is tied to the "<name>_real"/"<name>_gen" convention of the
    two reconstruct_*_physics functions. Code that carries plain physical
    arrays instead (as tools/acceptance_v0_8.py does, keying them "E",
    "logE", "u_r", ...) should call scipy.stats.wasserstein_distance
    directly rather than repacking to satisfy this signature.
    """
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
    min_significance=3.0,
    neighbour_lines=None,
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
    resolution_ev : float, sequence of float, or None
        Detector energy resolution (FWHM, in eV) that the model's line widths
        are pinned to. When given, the window half-width becomes
        `window_sigma * sigma_E` with `sigma_E = resolution_ev / 2.3548`,
        i.e. it tracks the physical line width instead of the binning. A
        sequence of length n_lines gives a per-line width.
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
    min_significance : float
        A continuum-subtracted line count must exceed this many Poisson sigma
        of the counts that went into it to be reported; otherwise the line is
        marked `low_significance` and `recovery_ratio` is None. Without this a
        window dominated by continuum returns meaningless ratios (0.0, 6.7,
        ...) driven by side-band noise - which is what a too-wide window on a
        weak line produces.
    neighbour_lines : list of dict or None
        Every line known to be physically present in the real spectrum, as
        dicts with "label" and "energy_mev" - normally the whole candidate
        table. Used ONLY to detect side-band contamination (see
        `sideband_contaminated` below). Defaults to `matched_lines` itself,
        which is the wrong reference whenever a real line exists in the data
        but is not modelled: on CryoSphere-Small, Cu Kalpha2 has 63 real
        events (below the 100-event modelling floor, so it is absent from
        matched_lines) yet still sits 21 eV from Cu Kalpha1 and still lands
        in its side-bands, deflating Kalpha1's continuum-subtracted count by
        ~2x. Passing the full candidate table catches that; passing only the
        modelled lines does not.

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
    # Contamination reference: every real line, not just the modelled ones.
    if neighbour_lines is None:
        neighbours = [(line["label"], float(line["candidate_energy_mev"]))
                      for line in matched_lines]
    else:
        neighbours = [(n["label"],
                       float(n["energy_mev"] if "energy_mev" in n
                             else n["candidate_energy_mev"]))
                      for n in neighbour_lines]
    res_ev = (None if resolution_ev is None else
              np.broadcast_to(np.asarray(resolution_ev, dtype=np.float64).reshape(-1),
                              (len(centres),)))
    tols = []
    for i_c, E_c in enumerate(centres):
        if res_ev is not None:
            sigma_e = float(res_ev[i_c]) * 1e-6 * FWHM_TO_SIGMA   # eV -> MeV
            tol = float(window_sigma) * sigma_e
            mode = f"{window_sigma:g}sigma@{res_ev[i_c]:g}eV"
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

        # Two distinct ways a neighbouring real line interferes, which need
        # different verdicts - both measured against `neighbours` (every line
        # real in the DATA, not only the modelled ones: a line below the
        # modelling floor still sits in the spectrum. Small's Cu Kalpha2, 63
        # events, deflates Cu Kalpha1's subtracted count ~2x exactly this way).
        #
        # 1. BLENDED - neighbour inside the window (|d| <= tol). It is not
        #    resolvable from this line at this resolution, so both lines' events
        #    are in n_real_line together and one model component is expected to
        #    produce all of them. The continuum estimate is untouched, so the
        #    ratio stays meaningful; it just describes the pair, not one line.
        #    CryoSphere Al Kalpha1/Kalpha2 (0.5 eV apart, vs an 8.5 eV window).
        # 2. SIDE-BAND CONTAMINATED - neighbour outside the window but within
        #    the side-bands (tol < |d| <= sb_out*tol). Its real events are
        #    counted as this line's "local continuum" and subtracted from the
        #    line itself, which is fatal to the ratio. CryoSphere Cu
        #    Kalpha1/Kalpha2 (21 eV apart, window 8.5 eV, side-bands
        #    17-25.5 eV): this reported "recovery=0.000" for a line that, by
        #    component routing, was landing exactly on its pinned position
        #    (mean 8005.70 eV vs pinned 8005.71 eV, std 1.64 eV).
        #
        # Only (2) nulls the recovery. Conflating them would discard Al
        # Kalpha1's perfectly usable measurement along with Cu Kalpha1's
        # broken one.
        blended = []
        contaminants = []
        for lbl, e_n in neighbours:
            d = abs(e_n - E_c)
            if d <= 1e-12:
                continue
            if d <= tol:
                blended.append(lbl)
            elif continuum_subtract and d <= sb_out * tol:
                contaminants.append(lbl)
        overlaps = blended + contaminants
        sideband_contaminated = bool(contaminants)

        # Poisson significance of the subtracted real line count. A window
        # dominated by continuum gives a small difference of two large noisy
        # numbers - report nothing rather than a spurious ratio.
        sigma_line = np.sqrt(max(n_real, 0) + max(n_real_cont, 0.0))
        significance = (n_real_line / sigma_line) if sigma_line > 0 else 0.0
        low_significance = bool(continuum_subtract and significance < min_significance)

        if continuum_subtract:
            recovery = (None if (low_significance or sideband_contaminated or n_real_line <= 0)
                        else n_gen_line * gen_scale / n_real_line)
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
            "line_significance": float(significance),
            "low_significance": low_significance,
            "sideband_contaminated": sideband_contaminated,
            "blended_lines": blended,
            "sideband_contaminants": contaminants,
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
    """Print how well generated events satisfy the geometric constraints.

    Three things that must hold by construction, not by fit: every crossing
    should sit on the sphere of the given `radius`, every direction should be a
    unit vector, and each angle's (cos, sin) pair should have norm 1. Large
    deviations point at a reconstruction/transform mismatch rather than at an
    under-trained model.

    Parameters
    ----------
    gen_pack : dict
        Generated physical arrays, from reconstruct_generated_physics.

    radius : float
        Expected sphere radius in mm. Must match the `radius` used during
        reconstruction; a mismatch shows up here as a uniform offset.

    Returns
    -------
    None
    """
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

# ---------------------------------------------------------------------------
# v0.8.3: energy-vs-aiming coupling
# ---------------------------------------------------------------------------

def energy_vs_impact_parameter(
    P_real, U_real, E_real,
    P_gen, U_gen, E_gen,
    detector_centre,
    cuts=(np.inf, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0),
    verbose=True,
):
    """Median energy as a function of how nearly a crossing points at the detector.

    This is the metric that would have caught the v0.8.2 CR deficit. Every
    marginal was right - energy decades within 3%, muon fraction 1.000, the
    fraction of rays intersecting the detector 1.027 - while the JOINT was
    wrong: real median energy is flat in the impact parameter `b`, but the
    generated median fell to 0.47x truth at b < 10 mm and 0.23x at b < 1 mm.
    A marginal-only validation suite cannot see that.

    `b` is the closest approach of the straight ray (position, direction) to
    `detector_centre`, computed only for rays travelling toward it. Positions
    and the centre must share units and frame; energies just have to be
    mutually consistent.

    Returns one row per cut with the real and generated counts, the aimed
    fraction ratio (geometry alone) and the median-energy ratio (the joint).
    A healthy model has BOTH columns near 1 at every cut; v0.8.2 had the first
    near 1 and the second falling.
    """
    P_real = np.asarray(P_real, float); U_real = np.asarray(U_real, float)
    P_gen = np.asarray(P_gen, float); U_gen = np.asarray(U_gen, float)
    E_real = np.asarray(E_real, float); E_gen = np.asarray(E_gen, float)
    D = np.asarray(detector_centre, float)

    def _b(P, U):
        w = D - P
        t = np.einsum("ij,ij->i", w, U)
        b = np.linalg.norm(w - t[:, None] * U, axis=1)
        return b, t > 0

    b_r, in_r = _b(P_real, U_real)
    b_g, in_g = _b(P_gen, U_gen)

    rows = []
    base_r = base_g = None
    for c in cuts:
        m_r = in_r & (b_r < c)
        m_g = in_g & (b_g < c)
        if m_r.sum() < 10 or m_g.sum() < 10:
            continue
        f_r, f_g = m_r.mean(), m_g.mean()
        if base_r is None:
            base_r, base_g = f_r, f_g
        med_r = float(np.median(E_real[m_r]))
        med_g = float(np.median(E_gen[m_g]))
        rows.append({
            "cut_mm": c,
            "n_real": int(m_r.sum()),
            "n_gen": int(m_g.sum()),
            "aimed_frac_ratio": (f_g / base_g) / (f_r / base_r) if base_r else np.nan,
            "median_E_real": med_r,
            "median_E_gen": med_g,
            "median_E_ratio": med_g / med_r if med_r else np.nan,
        })

    if verbose:
        print("energy vs impact parameter to the detector")
        print("  a healthy model keeps median_E_ratio ~1 at EVERY cut;")
        print("  a falling trend is the energy<->geometry coupling error (v0.8.2)")
        print(f"  {'cut':>10s} {'n_real':>9s} {'n_gen':>9s} "
              f"{'medE_real':>10s} {'medE_gen':>10s} {'ratio':>7s}")
        for r in rows:
            lab = "all" if not np.isfinite(r["cut_mm"]) else f"b<{r['cut_mm']:g}mm"
            print(f"  {lab:>10s} {r['n_real']:9,} {r['n_gen']:9,} "
                  f"{r['median_E_real']:10.4g} {r['median_E_gen']:10.4g} "
                  f"{r['median_E_ratio']:7.3f}")
        if rows:
            worst = min(rows, key=lambda r: r["median_E_ratio"])
            lab = "all" if not np.isfinite(worst["cut_mm"]) else f"b<{worst['cut_mm']:g}mm"
            print(f"  worst median_E_ratio = {worst['median_E_ratio']:.3f} at {lab}"
                  f"   (v0.8.2 CR reference: 0.225 at b<1mm)")
    return rows
