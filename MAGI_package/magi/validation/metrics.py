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