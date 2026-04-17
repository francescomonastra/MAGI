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
):
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

    plt.show()


def report_final_ranges(real_pack, gen_pack):
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
    norm_phi_r_real = np.sqrt(real_pack["cphi_r_real"]**2 + real_pack["sphi_r_real"]**2)
    norm_phi_r_gen = np.sqrt(gen_pack["cphi_r_gen"]**2 + gen_pack["sphi_r_gen"]**2)

    norm_phi_v_real = np.sqrt(real_pack["cphi_v_real"]**2 + real_pack["sphi_v_real"]**2)
    norm_phi_v_gen = np.sqrt(gen_pack["cphi_v_gen"]**2 + gen_pack["sphi_v_gen"]**2)

    print("\n--- COS/SIN NORM CHECK ---")
    print("phi_r real/gen:", norm_phi_r_real.mean(), norm_phi_r_gen.mean())
    print("phi_v real/gen:", norm_phi_v_real.mean(), norm_phi_v_gen.mean())


def build_real_generated_featureframes(real_pack, gen_pack):

    df_real = pd.DataFrame({
        "sample": "real",
        "logE": real_pack["logE_real"],
        "s_r": real_pack["s_r_real"],
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
    })

    df_gen = pd.DataFrame({
        "sample": "generated",
        "logE": gen_pack["logE_gen"],
        "s_r": gen_pack["s_r_gen"],
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
    })

    return df_real, df_gen, pd.concat([df_real, df_gen], ignore_index=True)