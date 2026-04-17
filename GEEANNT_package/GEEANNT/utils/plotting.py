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


def plot_dist(data, name, bins=200, range_=None, density=True, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, range=range_, density=density, histtype="step")
    plt.xlabel(name)
    plt.ylabel("density")
    plt.title(name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dist_by_class(df, value_col, class_col="ParticleName", selected_class=None,
                       bins=200, range_=None, density=True, figsize=(7, 4)):
    if selected_class is None:
        data = df[value_col].to_numpy()
        title = value_col
    else:
        data = df.loc[df[class_col] == selected_class, value_col].to_numpy()
        title = f"{value_col} ({selected_class})"

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, range=range_, density=density, histtype="step")
    plt.xlabel(value_col)
    plt.ylabel("density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, cols, method="pearson", figsize=(8, 6), cmap="coolwarm"):
    corr = df[cols].corr(method=method)

    plt.figure(figsize=figsize)
    im = plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, label=f"{method} correlation")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title(f"{method.capitalize()} correlation matrix")
    plt.tight_layout()
    plt.show()

    return corr

def plot_covariance_matrix(df, cols, figsize=(8, 6), cmap="viridis"):
    cov = df[cols].cov()

    plt.figure(figsize=figsize)
    im = plt.imshow(cov, cmap=cmap)
    plt.colorbar(im, label="covariance")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Covariance matrix")
    plt.tight_layout()
    plt.show()

    return cov

def plot_pairwise_sample(df, cols, class_col=None, sample_size=5000, diag_kind="hist"):
    import seaborn as sns

    data = df.copy()
    if sample_size is not None and len(data) > sample_size:
        data = data.sample(sample_size, random_state=0)

    use_cols = cols.copy()
    if class_col is not None:
        use_cols = use_cols + [class_col]

    g = sns.pairplot(
        data[use_cols],
        hue=class_col,
        diag_kind=diag_kind,
        corner=False,
        plot_kws={"alpha": 0.35, "s": 20},
    )
    return g

def plot_pairgrid_physics(
    df,
    cols,
    class_col=None,
    sample_size=None,
    lower_mode="scatter",
    diag_mode="hist",
    bins=30,
    contour_levels=8,
    figsize_scale=2.8,
    alpha_scatter=0.25,
    scatter_size=12,
    cmap="viridis",
    corr_methods=("pearson", "spearman", "kendall"),
    show_legend=True,
    random_state=0,
):
    """
    General pair-grid plot for physics feature inspection.

    Features:
    - diagonal: 1D histograms
    - lower triangle: scatter / contour / scatter+contour / hist2d
    - upper triangle: correlation summary (Pearson, Spearman, Kendall)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list[str]
        Variables to include in the grid.
    class_col : str or None
        Optional column used as hue/class.
    sample_size : int or None
        Optional random sample size for plotting.
    lower_mode : str
        One of:
        - "scatter"
        - "contour"
        - "scatter_contour"
        - "hist2d"
    diag_mode : str
        Currently supports:
        - "hist"
    bins : int
        Histogram / hist2d bin count.
    contour_levels : int or array-like
        Number of contour levels or explicit levels.
    figsize_scale : float
        Per-panel size scaling. Total figure size scales with len(cols).
    alpha_scatter : float
        Scatter transparency.
    scatter_size : float
        Scatter marker size.
    cmap : str
        Colormap for density-based panels.
    corr_methods : tuple[str]
        Correlation methods to display in upper panels.
        Supported: "pearson", "spearman", "kendall"
    show_legend : bool
        Whether to show legend when class_col is provided.
    random_state : int
        Random seed used for sampling.

    Returns
    -------
    seaborn.axisgrid.PairGrid
        The created PairGrid.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import Ellipse

    # ------------------------------------------------------
    # Input checks
    # ------------------------------------------------------
    data = df.copy()

    use_cols = list(cols)
    if class_col is not None:
        use_cols = use_cols + [class_col]

    data = data[use_cols].dropna()

    if sample_size is not None and len(data) > sample_size:
        data = data.sample(sample_size, random_state=random_state)

    n_vars = len(cols)
    fig_side = max(6, figsize_scale * n_vars)

    # ------------------------------------------------------
    # Correlation matrices
    # ------------------------------------------------------
    corr_dict = {}
    for method in corr_methods:
        if method not in ("pearson", "spearman", "kendall"):
            raise ValueError(f"Unsupported correlation method: {method}")
        corr_dict[method] = data[cols].corr(method=method)

    # ------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------
    def _diag_hist(x, color=None, **kwargs):
        ax = plt.gca()
        x = np.asarray(x)
        finite = np.isfinite(x)
        x = x[finite]

        if len(x) == 0:
            return

        ax.hist(
            x,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color=color,
        )
        ax.grid(True, alpha=0.2)

    def _lower_scatter(x, y, color=None, **kwargs):
        ax = plt.gca()
        ax.scatter(
            x, y,
            s=scatter_size,
            alpha=alpha_scatter,
            color=color,
            linewidths=0,
        )
        ax.grid(True, alpha=0.2)

    def _lower_hist2d(x, y, color=None, **kwargs):
        ax = plt.gca()
        h = ax.hist2d(
            x, y,
            bins=bins,
            density=True,
            cmap=cmap,
        )
        ax.grid(True, alpha=0.2)

    def _lower_contour(x, y, color=None, **kwargs):
        ax = plt.gca()

        x = np.asarray(x)
        y = np.asarray(y)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if len(x) < 10:
            return

        H, xedges, yedges = np.histogram2d(
            x, y,
            bins=bins,
            density=True
        )

        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xc, yc, indexing="xy")
        Z = H.T

        # avoid contour warnings on empty histograms
        if np.all(Z <= 0):
            return

        ax.contour(X, Y, Z, levels=contour_levels, colors="black", linewidths=0.8)
        ax.grid(True, alpha=0.2)

    def _lower_scatter_contour(x, y, color=None, **kwargs):
        ax = plt.gca()

        x = np.asarray(x)
        y = np.asarray(y)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if len(x) == 0:
            return

        ax.scatter(
            x, y,
            s=scatter_size,
            alpha=alpha_scatter,
            color=color,
            linewidths=0,
        )

        if len(x) >= 10:
            H, xedges, yedges = np.histogram2d(
                x, y,
                bins=bins,
                density=True
            )
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xc, yc, indexing="xy")
            Z = H.T

            if np.any(Z > 0):
                ax.contour(X, Y, Z, levels=contour_levels, colors="black", linewidths=0.8)

        ax.grid(True, alpha=0.2)

    def _upper_corr(x, y, color=None, **kwargs):
        ax = plt.gca()
        ax.set_axis_off()
        ax.set_autoscale_on(False)

        col_x = x.name
        col_y = y.name

        lines = []
        color_value = 0.0

        if "pearson" in corr_dict:
            rp = corr_dict["pearson"].loc[col_x, col_y]
            lines.append(f"P {rp:.2f}")
            color_value = rp

        if "spearman" in corr_dict:
            rs = corr_dict["spearman"].loc[col_x, col_y]
            lines.append(f"S {rs:.2f}")

        if "kendall" in corr_dict:
            rk = corr_dict["kendall"].loc[col_x, col_y]
            lines.append(f"K {rk:.2f}")

        cmap_obj = mpl.colormaps["coolwarm"]
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        facecolor = cmap_obj(norm(color_value))

        ellipse = Ellipse(
            (0.5, 0.5),
            width=0.62,
            height=0.62,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor="none",
            alpha=0.45,
            zorder=1
        )
        ax.add_patch(ellipse)

        y0 = 0.62
        dy = 0.14
        for i, txt in enumerate(lines):
            ax.text(
                0.5,
                y0 - i * dy,
                txt,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="black",
                zorder=2,
            )

    # ------------------------------------------------------
    # Build grid
    # ------------------------------------------------------
    with sns.axes_style("ticks"), sns.plotting_context("notebook", font_scale=1.0):
        g = sns.PairGrid(
            data[use_cols],
            vars=cols,
            hue=class_col,
            diag_sharey=False,
            height=figsize_scale,
            aspect=1.0,
        )

        # diagonal
        if diag_mode == "hist":
            g.map_diag(_diag_hist)
        else:
            raise ValueError(f"Unsupported diag_mode: {diag_mode}")

        # lower triangle
        if lower_mode == "scatter":
            g.map_lower(_lower_scatter)
        elif lower_mode == "contour":
            g.map_lower(_lower_contour)
        elif lower_mode == "scatter_contour":
            g.map_lower(_lower_scatter_contour)
        elif lower_mode == "hist2d":
            g.map_lower(_lower_hist2d)
        else:
            raise ValueError(f"Unsupported lower_mode: {lower_mode}")

        # upper triangle
        g.map_upper(_upper_corr)

        # formatting
        for ax in g.axes.flatten():
            if ax is None:
                continue
            ax.tick_params(axis="both", labelsize=9)

        if class_col is not None and show_legend:
            g.add_legend()

        g.fig.set_size_inches(fig_side, fig_side)
        g.fig.tight_layout()

    return g