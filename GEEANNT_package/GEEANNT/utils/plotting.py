"""
Plotting utilities for GEEANNT.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, keys=None, show_available=True):
    """
    Plot selected metrics from a Keras History object.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history returned by model.fit().
    keys : list[str] or None
        Metrics to plot. If None, a default list for the CVAE model is used.
    show_available : bool
        If True, print all available history keys before plotting.
    """
    h = history.history

    if show_available:
        print("\nAvailable history keys:")
        print(list(h.keys()))

    if keys is None:
        keys = [
            "loss",
            "rec",
            "kl",
            "energy_ce",
            "sr_nll",
            "phi_r_mse",
            "xy_mse",
            "u_r_mse",
            "uv_ce",
            "phi_v_mse",
            "phi_v_ang",
            "phi_v_loss",
            "vxy_mse",
            "sigma_reg",
            "phi_reg",
            "phi_r_reg",
        ]

    if "loss" in h:
        n_ep = len(h["loss"])
    else:
        n_ep = len(next(iter(h.values())))

    epochs = np.arange(1, n_ep + 1)
    has_lr = ("learning_rate" in h)

    for k in keys:
        if k not in h:
            continue

        plt.figure(figsize=(7, 4))

        plt.plot(
            epochs,
            h[k],
            label=k,
            linewidth=2
        )

        val_k = "val_" + k
        if val_k in h:
            plt.plot(
                epochs,
                h[val_k],
                label=val_k,
                linewidth=2
            )

        if k == "loss" and has_lr:
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax2.plot(
                epochs,
                h["learning_rate"],
                linestyle="--",
                label="learning_rate"
            )

            ax2.set_ylabel("learning_rate")
            ax2.legend(loc="upper right")

        plt.xlabel("epoch")
        plt.ylabel(k)
        plt.title(k)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()