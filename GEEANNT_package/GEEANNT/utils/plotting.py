"""
Plotting utilities for GEEANNT.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_plot_theme(theme="light"):
    """
    Set global plotting theme.

    Parameters
    ----------
    theme : str
        "light" or "dark"
    """

    if theme == "dark":
        plt.style.use("dark_background")

        mpl.rcParams.update({
            "figure.facecolor": "#121212",
            "axes.facecolor": "#121212",
            "savefig.facecolor": "#121212",

            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",

            "xtick.color": "white",
            "ytick.color": "white",

            "text.color": "white",

            "grid.color": "#444444",

            "legend.facecolor": "#1e1e1e",
            "legend.edgecolor": "white",

            "lines.linewidth": 2,
        })

    elif theme == "light":
        plt.style.use("default")
        mpl.rcParams.update(mpl.rcParamsDefault)

    else:
        raise ValueError("theme must be either 'light' or 'dark'")


def plot_history(history, keys=None, show_available=True):
    """
    Plot selected metrics from a Keras History object.
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
    has_lr = "learning_rate" in h

    for k in keys:
        if k not in h:
            continue

        fig, ax1 = plt.subplots(figsize=(7, 4))

        ax1.plot(epochs, h[k], label=k, linewidth=2)

        val_k = "val_" + k
        if val_k in h:
            ax1.plot(epochs, h[val_k], label=val_k, linewidth=2)

        ax1.set_xlabel("epoch")
        ax1.set_ylabel(k)
        ax1.set_title(k)
        ax1.grid(True, alpha=0.3)

        if k == "loss" and has_lr:
            ax2 = ax1.twinx()

            ax2.plot(
                epochs,
                h["learning_rate"],
                linestyle="--",
                label="learning_rate",
                color="tab:orange",
            )

            ax2.set_ylabel("learning_rate")
            ax2.legend(loc="upper right")

        ax1.legend(loc="best")
        fig.tight_layout()
        plt.show()


def plot_dist(data, name, bins=200, range_=None, density=True, figsize=(7, 4), xscale="linear", yscale="linear", savepath=None, dpi=300, show=True):
    fig = plt.figure(figsize=figsize)
    plt.hist(
        data,
        bins=bins,
        range=range_,
        density=density,
        histtype="step",
        linewidth=2,
    )
    plt.xlabel(name)
    plt.ylabel("density")
    plt.title(name)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save_and_show(fig=fig, savepath=savepath, dpi=dpi, show=show)


def plot_dist_by_class(
    df,
    value_col,
    class_col="ParticleName",
    selected_class=None,
    bins=200,
    range_=None,
    density=True,
    figsize=(7, 4),
):
    if selected_class is None:
        data = df[value_col].to_numpy()
        title = value_col
    else:
        data = df.loc[df[class_col] == selected_class, value_col].to_numpy()
        title = f"{value_col} ({selected_class})"

    plt.figure(figsize=figsize)
    plt.hist(
        data,
        bins=bins,
        range=range_,
        density=density,
        histtype="step",
        linewidth=2,
    )
    plt.xlabel(value_col)
    plt.ylabel("density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, cols, method="pearson", figsize=(8, 6), cmap="coolwarm", savepath=None, dpi=300, show=True):
    corr = df[cols].corr(method=method)

    fig = plt.figure(figsize=figsize)
    im = plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, label=f"{method} correlation")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title(f"{method.capitalize()} correlation matrix")
    plt.tight_layout()
    return _save_and_show(fig=fig, savepath=savepath, dpi=dpi, show=show), corr


def plot_covariance_matrix(df, cols, figsize=(8, 6), cmap="viridis", savepath=None, dpi=300, show=True):
    cov = df[cols].cov()

    fig = plt.figure(figsize=figsize)
    im = plt.imshow(cov, cmap=cmap)
    plt.colorbar(im, label="covariance")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Covariance matrix")
    plt.tight_layout()
    return _save_and_show(fig=fig, savepath=savepath, dpi=dpi, show=show), cov


def plot_pairwise_sample(
    df,
    cols,
    class_col=None,
    sample_size=5000,
    diag_kind="hist",
    theme=None,
    palette="deep",
):
    import seaborn as sns

    if theme is not None:
        set_plot_theme(theme)

    data = df.copy()

    if sample_size is not None and len(data) > sample_size:
        data = data.sample(sample_size, random_state=0)

    use_cols = cols.copy()
    if class_col is not None:
        use_cols = use_cols + [class_col]

    g = sns.pairplot(
        data[use_cols],
        hue=class_col,
        palette=palette,
        diag_kind=diag_kind,
        corner=False,
        plot_kws={"alpha": 0.35, "s": 20},
    )

    g.fig.patch.set_facecolor(plt.rcParams["figure.facecolor"])

    for ax in g.axes.flatten():
        if ax is None:
            continue
        ax.set_facecolor(plt.rcParams["axes.facecolor"])
        ax.tick_params(colors=plt.rcParams["xtick.color"])
        ax.xaxis.label.set_color(plt.rcParams["axes.labelcolor"])
        ax.yaxis.label.set_color(plt.rcParams["axes.labelcolor"])

        for spine in ax.spines.values():
            spine.set_color(plt.rcParams["axes.edgecolor"])
    plt.show()
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
    corr_cmap="coolwarm",
    palette="deep",
    contour_color=None,
    background_color=None,
    figure_color=None,
    grid_color=None,
    text_color=None,
    axis_color=None,
    corr_methods=("pearson", "spearman", "kendall"),
    show_legend=True,
    random_state=0,
    theme=None,
):
    """
    General pair-grid plot for physics feature inspection.

    Useful dark settings:

    plot_pairgrid_physics(
        df,
        cols,
        class_col="ParticleName",
        theme="dark",
        palette="mako",
        cmap="magma",
        corr_cmap="coolwarm",
        background_color="#0b0f19",
        figure_color="#0b0f19",
        grid_color="#2a2f3a",
        contour_color="white",
    )
    """
    import seaborn as sns
    from matplotlib.patches import Ellipse

    if theme is not None:
        set_plot_theme(theme)

    if text_color is None:
        text_color = plt.rcParams["text.color"]

    if contour_color is None:
        contour_color = text_color

    if background_color is None:
        background_color = plt.rcParams["axes.facecolor"]

    if figure_color is None:
        figure_color = plt.rcParams["figure.facecolor"]

    if grid_color is None:
        grid_color = plt.rcParams["grid.color"]

    if axis_color is None:
        axis_color = plt.rcParams["axes.edgecolor"]

    data = df.copy()

    use_cols = list(cols)
    if class_col is not None:
        use_cols = use_cols + [class_col]

    data = data[use_cols].dropna()

    if sample_size is not None and len(data) > sample_size:
        data = data.sample(sample_size, random_state=random_state)

    n_vars = len(cols)
    fig_side = max(6, figsize_scale * n_vars)

    corr_dict = {}
    for method in corr_methods:
        if method not in ("pearson", "spearman", "kendall"):
            raise ValueError(f"Unsupported correlation method: {method}")
        corr_dict[method] = data[cols].corr(method=method)

    grid_alpha = 0.25

    def _diag_hist(x, color=None, **kwargs):
        ax = plt.gca()
        x = np.asarray(x)
        x = x[np.isfinite(x)]

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

        ax.grid(True, alpha=grid_alpha, color=grid_color)

    def _lower_scatter(x, y, color=None, **kwargs):
        ax = plt.gca()

        ax.scatter(
            x,
            y,
            s=scatter_size,
            alpha=alpha_scatter,
            color=color,
            linewidths=0,
        )

        ax.grid(True, alpha=grid_alpha, color=grid_color)

    def _lower_hist2d(x, y, color=None, **kwargs):
        ax = plt.gca()

        ax.hist2d(
            x,
            y,
            bins=bins,
            density=True,
            cmap=cmap,
        )

        ax.grid(True, alpha=grid_alpha, color=grid_color)

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
            x,
            y,
            bins=bins,
            density=True,
        )

        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xc, yc, indexing="xy")
        Z = H.T

        if np.all(Z <= 0):
            return

        ax.contour(
            X,
            Y,
            Z,
            levels=contour_levels,
            colors=contour_color,
            linewidths=0.8,
        )

        ax.grid(True, alpha=grid_alpha, color=grid_color)

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
            x,
            y,
            s=scatter_size,
            alpha=alpha_scatter,
            color=color,
            linewidths=0,
        )

        if len(x) >= 10:
            H, xedges, yedges = np.histogram2d(
                x,
                y,
                bins=bins,
                density=True,
            )

            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xc, yc, indexing="xy")
            Z = H.T

            if np.any(Z > 0):
                ax.contour(
                    X,
                    Y,
                    Z,
                    levels=contour_levels,
                    colors=contour_color,
                    linewidths=0.8,
                )

        ax.grid(True, alpha=grid_alpha, color=grid_color)

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

        cmap_obj = mpl.colormaps[corr_cmap]
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        facecolor = cmap_obj(norm(color_value))

        ellipse = Ellipse(
            (0.5, 0.5),
            width=0.62,
            height=0.62,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor="none",
            alpha=0.55,
            zorder=1,
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
                color=text_color,
                zorder=2,
            )

    context = sns.plotting_context("notebook", font_scale=1.0)

    with plt.rc_context(context):
        g = sns.PairGrid(
            data[use_cols],
            vars=cols,
            hue=class_col,
            palette=palette,
            diag_sharey=False,
            height=figsize_scale,
            aspect=1.0,
        )

        if diag_mode == "hist":
            g.map_diag(_diag_hist)
        else:
            raise ValueError(f"Unsupported diag_mode: {diag_mode}")

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

        g.map_upper(_upper_corr)

        g.fig.patch.set_facecolor(figure_color)

        for ax in g.axes.flatten():
            if ax is None:
                continue

            ax.set_facecolor(background_color)

            ax.tick_params(
                axis="both",
                labelsize=9,
                colors=text_color,
            )

            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)

            for spine in ax.spines.values():
                spine.set_color(axis_color)

        if class_col is not None and show_legend:
            g.add_legend()

            if g.legend is not None:
                legend_facecolor = plt.rcParams["legend.facecolor"]
                legend_edgecolor = plt.rcParams["legend.edgecolor"]

                if legend_facecolor == "inherit":
                    legend_facecolor = figure_color

                if legend_edgecolor == "inherit":
                    legend_edgecolor = axis_color

                g.legend.get_frame().set_facecolor(legend_facecolor)
                g.legend.get_frame().set_edgecolor(legend_edgecolor)

                for txt in g.legend.texts:
                    txt.set_color(text_color)

                if g.legend.get_title() is not None:
                    g.legend.get_title().set_color(text_color)

        g.fig.set_size_inches(fig_side, fig_side)
        g.fig.tight_layout()
    plt.show()
    return g

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