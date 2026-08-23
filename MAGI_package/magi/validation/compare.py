"""
Comparison plots between generated and real distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compare_hist_with_residuals(
    real,
    gen,
    name,
    bins=200,
    range_=None,
    eps_res=1e-12,
    ratio_clip=None,
    savepath=None,
    dpi=300,
    show=True
):
    """Overlay real vs generated histograms with a residual panel underneath.

    Residual = gen_density / real_density - 1, so 0 means a perfect match. Bins
    where the real density is ~0 are floored by `eps_res` and will show large
    residuals; `ratio_clip` bounds the residual axis so a handful of empty-bin
    spikes do not flatten the rest of the panel.

    Parameters
    ----------
    real, gen : array-like of float
        The two 1-D samples to compare. They need not be the same length -
        both histograms are density-normalized.

    name : str
        Variable name, used for the axis label and title.

    bins : int
        Number of histogram bins, shared by both samples.

    range_ : tuple[float, float] or None
        Histogram range. None (default) lets numpy pick from the combined
        data. Set it explicitly when comparing plots across runs.

    eps_res : float
        Floor applied to the real density before dividing, so empty real bins
        give a large-but-finite residual instead of inf.

    ratio_clip : float or None
        Symmetric limit on the residual axis, e.g. 1.0 for +/-100%. None
        leaves the axis unbounded.

    savepath : str or None
        Where to write the figure. None does not save.

    dpi : int
        Save resolution.

    show : bool
        Call plt.show(). Set False in scripted runs that only save.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, so a caller can annotate it further before saving.
    """
    real = np.asarray(real)
    gen = np.asarray(gen)

    real_hist, bin_edges = np.histogram(real, bins=bins, range=range_, density=True)
    gen_hist, _ = np.histogram(gen, bins=bin_edges, density=True)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    residuals = gen_hist / np.maximum(real_hist, eps_res) - 1.0

    if ratio_clip is not None:
        residuals = np.clip(residuals, -ratio_clip, ratio_clip)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05}
    )

    ax_top.hist(real, bins=bin_edges, density=True, histtype="step", label="real")
    ax_top.hist(gen, bins=bin_edges, density=True, histtype="step", label="generated")
    ax_top.set_ylabel("density")
    ax_top.set_title(name)
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    ax_bot.axhline(0.0, linestyle="--", linewidth=1)
    ax_bot.bar(bin_centers, residuals, width=bin_widths, alpha=0.7)
    ax_bot.set_xlabel(name)
    ax_bot.set_ylabel("(gen/real)-1")
    ax_bot.grid(True, alpha=0.3)

    return _save_and_show(fig=fig, savepath=savepath, dpi=dpi, show=show)


def report_final_ranges(real_pack, gen_pack):
    """Print min/max of every reconstructed quantity, real vs generated.

    A coarse but effective check: a generated range that overshoots the real one
    usually means a Gaussian head extrapolating past the support of its quantile
    transform.

    Parameters
    ----------
    real_pack : dict
        Real physical arrays, from reconstruct_real_test_physics.

    gen_pack : dict
        Generated physical arrays, from reconstruct_generated_physics.

    Returns
    -------
    None
    """
    print("\n--- RANGE CHECK ---")
    print("logE real/gen:",
          real_pack["logE_real"].min(), real_pack["logE_real"].max(), "|",
          gen_pack["logE_gen"].min(), gen_pack["logE_gen"].max())
    print("u_v  real/gen:",
          real_pack["u_v_real"].min(), real_pack["u_v_real"].max(), "|",
          gen_pack["u_v_gen"].min(), gen_pack["u_v_gen"].max())

    print("\nPosition:")
    print("x real/gen:", real_pack["x_real"].min(), real_pack["x_real"].max(), "|",
          gen_pack["x_gen"].min(), gen_pack["x_gen"].max())
    print("y real/gen:", real_pack["y_real"].min(), real_pack["y_real"].max(), "|",
          gen_pack["y_gen"].min(), gen_pack["y_gen"].max())
    print("z real/gen:", real_pack["z_real"].min(), real_pack["z_real"].max(), "|",
          gen_pack["z_gen"].min(), gen_pack["z_gen"].max())

    print("\nDirection:")
    print("cphi_v real/gen:", real_pack["cphi_v_real"].min(), real_pack["cphi_v_real"].max(), "|",
          gen_pack["cphi_v_gen"].min(), gen_pack["cphi_v_gen"].max())
    print("sphi_v real/gen:", real_pack["sphi_v_real"].min(), real_pack["sphi_v_real"].max(), "|",
          gen_pack["sphi_v_gen"].min(), gen_pack["sphi_v_gen"].max())
    print("vx real/gen:", real_pack["vx_real"].min(), real_pack["vx_real"].max(), "|",
          gen_pack["vx_gen"].min(), gen_pack["vx_gen"].max())
    print("vy real/gen:", real_pack["vy_real"].min(), real_pack["vy_real"].max(), "|",
          gen_pack["vy_gen"].min(), gen_pack["vy_gen"].max())
    print("vz real/gen:", real_pack["vz_real"].min(), real_pack["vz_real"].max(), "|",
          gen_pack["vz_gen"].min(), gen_pack["vz_gen"].max())


def report_norm_checks(real_pack, gen_pack):
    """Print the mean |(cos, sin)| for phi_r and phi_v, real vs generated.

    These should sit at 1: the angle heads predict a 2-D vector that is only
    softly regularized onto the unit circle, so a mean below 1 means the
    regularizer is losing and the reconstructed angles are less reliable.

    Only meaningful for v0.7 (cos/sin phi) heads. The v0.7.2+ heads predict
    phi directly through a quantile transform and have no norm to check.

    Parameters
    ----------
    real_pack : dict
        Real physical arrays, from reconstruct_real_test_physics.

    gen_pack : dict
        Generated physical arrays, from reconstruct_generated_physics.

    Returns
    -------
    None
    """
    norm_phi_r_real = np.sqrt(real_pack["cphi_r_real"]**2 + real_pack["sphi_r_real"]**2)
    norm_phi_r_gen = np.sqrt(gen_pack["cphi_r_gen"]**2 + gen_pack["sphi_r_gen"]**2)

    norm_phi_v_real = np.sqrt(real_pack["cphi_v_real"]**2 + real_pack["sphi_v_real"]**2)
    norm_phi_v_gen = np.sqrt(gen_pack["cphi_v_gen"]**2 + gen_pack["sphi_v_gen"]**2)

    print("\n--- COS/SIN NORM CHECK ---")
    print("phi_r real/gen:", norm_phi_r_real.mean(), norm_phi_r_gen.mean())
    print("phi_v real/gen:", norm_phi_v_real.mean(), norm_phi_v_gen.mean())


def build_real_generated_featureframes(real_pack, gen_pack):
    """Pack the real and generated arrays into DataFrames for the plot helpers.

    Parameters
    ----------
    real_pack : dict
        Real physical arrays, from reconstruct_real_test_physics.

    gen_pack : dict
        Generated physical arrays, from reconstruct_generated_physics.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_real, df_gen, df_both), where df_both is the concatenation with a
        "sample" column of "real"/"generated" - the form plot_pairwise_sample
        and the correlation/covariance plots expect.
    """
    real_dict = {
        "sample": "real",

        "logE": real_pack["logE_real"],

        "u_r": real_pack["u_r_real"],
        "u_v": real_pack["u_v_real"],

        "cphi_r": real_pack["cphi_r_real"],
        "sphi_r": real_pack["sphi_r_real"],

        "cphi_v": real_pack["cphi_v_real"],
        "sphi_v": real_pack["sphi_v_real"],

        "x": real_pack["x_real"],
        "y": real_pack["y_real"],
        "z": real_pack["z_real"],

        "vx": real_pack["vx_real"],
        "vy": real_pack["vy_real"],
        "vz": real_pack["vz_real"],
    }

    gen_dict = {
        "sample": "generated",

        "logE": gen_pack["logE_gen"],

        "u_r": gen_pack["u_r_gen"],
        "u_v": gen_pack["u_v_gen"],

        "cphi_r": gen_pack["cphi_r_gen"],
        "sphi_r": gen_pack["sphi_r_gen"],

        "cphi_v": gen_pack["cphi_v_gen"],
        "sphi_v": gen_pack["sphi_v_gen"],

        "x": gen_pack["x_gen"],
        "y": gen_pack["y_gen"],
        "z": gen_pack["z_gen"],

        "vx": gen_pack["vx_gen"],
        "vy": gen_pack["vy_gen"],
        "vz": gen_pack["vz_gen"],
    }

    # --------------------------------------------------
    # Optional angular variables
    # --------------------------------------------------

    if "phi_r_real" in real_pack:
        real_dict["phi_r"] = real_pack["phi_r_real"]

    if "phi_r_gen" in gen_pack:
        gen_dict["phi_r"] = gen_pack["phi_r_gen"]

    if "phi_v_real" in real_pack:
        real_dict["phi_v"] = real_pack["phi_v_real"]

    if "phi_v_gen" in gen_pack:
        gen_dict["phi_v"] = gen_pack["phi_v_gen"]

    # --------------------------------------------------
    # Optional quantile-space variables v0.7 / v0.7.2
    # --------------------------------------------------

    for key_real, key_gen, out_key in [
        ("u_r_q_real", "u_r_q_gen", "u_r_q"),
        ("u_v_q_real", "u_v_q_gen", "u_v_q"),
        ("phi_r_q_real", "phi_r_q_gen", "phi_r_q"),
        ("phi_v_q_real", "phi_v_q_gen", "phi_v_q"),
    ]:
        if key_real in real_pack and real_pack[key_real] is not None:
            real_dict[out_key] = real_pack[key_real]

        if key_gen in gen_pack and gen_pack[key_gen] is not None:
            gen_dict[out_key] = gen_pack[key_gen]

    # --------------------------------------------------
    # Backward compatibility v0.6
    # --------------------------------------------------

    if "s_r_real" in real_pack:
        real_dict["s_r"] = real_pack["s_r_real"]

    if "s_r_gen" in gen_pack:
        gen_dict["s_r"] = gen_pack["s_r_gen"]

    df_real = pd.DataFrame(real_dict)
    df_gen = pd.DataFrame(gen_dict)

    df_compare = pd.concat(
        [df_real, df_gen],
        ignore_index=True,
    )

    return df_real, df_gen, df_compare

def _save_and_show(fig=None, savepath=None, dpi=300, bbox_inches="tight", show=True):
    """
    Save and/or show a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or None
        Figure to save/show. If None, uses current figure.
    savepath : str or None
        Output path. If None, the figure is not saved.
    dpi : int
        Save resolution.
    bbox_inches : str
        Bounding box mode for saving.
    show : bool
        If True, call plt.show().
    """
    if fig is None:
        fig = plt.gcf()

    if savepath is not None:
        fig.savefig(
            savepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
        )

    if show:
        plt.show()

    return fig