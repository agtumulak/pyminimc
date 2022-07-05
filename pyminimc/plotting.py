"""
Plotting utilities
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from matplotlib import cm


def plot_pdf(s):
    """
    Plots bivariate PDF of in alpha and beta

    Parameters
    ----------
    s : pd.Series
        Bivariate PDF in alpha and beta to plot. MultiIndex must be beta
        followed by alpha.
    """
    plt.contourf(
        s.index.unique("beta"),
        s.index.unique("alpha"),
        np.log(s).unstack(),
        levels=100,
    )
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha$")
    plt.title(r"$\log p_{\alpha, \beta} (\alpha, \beta)$")
    plt.colorbar()
    plt.show()


def compare_bivariate_pdf(title, *series):
    """
    Compares PDFs in alpha and beta from multiple series.

    Parameters
    ----------
    title : string
        Plot title
    s1, s2, ... : sequence of PDFs in alpha and beta
    """
    nrows, ncols = 2, 2
    min_log_density = np.log(min(s[s > 0].min() for s in series))
    min_beta = max(s.index.unique("beta").min() for s in series)
    max_beta = min(s.index.unique("beta").max() for s in series)
    min_alpha = max(s.index.unique("alpha").min() for s in series)
    max_alpha = min(s.index.unique("alpha").max() for s in series)
    f, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            k = ncols * i + j
            if k >= len(series):
                break
            s = series[k]
            cm = ax.contourf(
                s.index.unique("beta"),
                s.index.unique("alpha"),
                np.log(s).unstack(),
                levels=np.linspace(min_log_density, 0, 100),
            )
            if j == 0:
                ax.set_ylabel(r"$\alpha$")
            if i == 1:
                ax.set_xlabel(r"$\beta$")
            ax.set_xlim(min_beta, max_beta)
            ax.set_ylim(min_alpha, max_alpha)
            ax.set_title(s.name)
    f.colorbar(cm, ax=axs, location="right")
    plt.show()


def compare_univariate_pdf(title, *series, axis="beta"):
    """
    Compares PDFs in beta from multiple series.

    Parameters
    ----------
    title : string
        Plot title
    s1, s2, ... : sequence of PDFs in beta
        Beta PDFs to plot
    axis : {'alpha', 'beta'}, optional
        The axis corresponding to the abscissa
    """
    for s in series:
        s.plot(label=s.name)
    plt.xlabel(rf"$\{axis}$")
    plt.ylabel(rf"$p_{{\{axis}}} (\{axis})$")
    plt.xlim(
        max(s.index.min() for s in series), min(s.index.max() for s in series)
    )
    plt.legend()
    plt.title(title)
    plt.show()


def inspect_visually(
    true_df: pd.DataFrame, truncated_df: pd.DataFrame, value_name: str
) -> None:
    """
    Display interactive dashboard for inspecting reference and truncated
    DataFrames

    Parameters
    ----------
    true_df
        Reference DataFrame
    truncated_df
        Approximated DataFrame which to be compared against reference DataFrame
    value_name
        Name of quantity being compared. Not provided in the DataFrames so it
        must be passed explicitly to have proper plot labels.
    """
    col_names = true_df.columns.names
    fig, axs = plt.subplots(2, 3)

    def on_click(event):
        if (
            event.inaxes
            and event.inaxes != event.canvas.figure.axes[5]
            and event.button is MouseButton.LEFT
        ):
            ax = event.canvas.figure.axes[5]
            ax.clear()
            true_s = true_df.iloc[:, round(event.xdata)]
            ax.plot(
                true_s,
                range(len(true_s)),
                linestyle="solid",
                label="true",
            )
            truncated_s = truncated_df.iloc[:, round(event.xdata)]
            ax.plot(
                truncated_s,
                range(len(truncated_s)),
                linestyle="dashed",
                label="truncated",
            )
            ax.legend()
            ax.set_xlim(right=true_s.max())
            ax.set_ylim(bottom=0)
            ax.set_xlabel(f"{value_name}")
            ax.set_ylabel("CDF Index")
            ax.set_title(
                f"{col_names[0]}={true_s.name[0]:5.3E}, "
                f"{col_names[1]}={true_s.name[1]}"
            )
            ax.grid()
            plt.draw()

    # plot true values
    ax = axs[0, 0]
    pcm = ax.imshow(true_df, interpolation="none", aspect="auto")
    ax.set_ylabel("CDF Index")
    ax.set_xlabel(f"{col_names[0]}, {col_names[1]} Index")
    ax.set_title(f"{value_name}")
    fig.colorbar(pcm, ax=ax)

    # plot log(abs(true values))
    ax = axs[0, 1]
    pcm = ax.imshow(
        np.log10(np.abs(true_df)), interpolation="none", aspect="auto"
    )
    ax.set_ylabel("CDF Index")
    ax.set_xlabel(f"{col_names[0]}, {col_names[1]} Index")
    ax.set_title(f"log10(abs({value_name}))")
    fig.colorbar(pcm, ax=ax)

    # plot nonmonotonic values
    ax = axs[0, 2]
    ax.imshow(
        truncated_df.diff() < 0,
        aspect="auto",
        cmap="gray",
    )
    ax.set_ylabel("CDF Index")
    ax.set_xlabel(f"{col_names[0]}, {col_names[1]} Index")
    ax.set_title(f"nonmonotonic entries after SVD")

    # plot absolute error
    ax = axs[1, 0]
    cmap = cm.get_cmap("viridis").with_extremes(over="red")
    log_abs_err = np.log10(np.abs(true_df - truncated_df))
    worst_idx = np.unravel_index(np.argmax(log_abs_err), log_abs_err.shape)
    pcm = ax.imshow(
        log_abs_err, interpolation="none", aspect="auto", cmap=cmap, vmax=-1.5
    )
    ax.set_ylabel("CDF Index")
    ax.set_xlabel(f"{col_names[0]}, {col_names[1]} Index")
    ax.set_title(
        f"log abs. err. "
        f"(worst: {log_abs_err.iloc[worst_idx]:.3f} at {worst_idx})"
    )
    fig.colorbar(pcm, ax=ax, extend="max")

    # plot relative error
    ax = axs[1, 1]
    cmap = cm.get_cmap("viridis").with_extremes(over="red")
    log_abs_rel_err = np.log10(np.abs((true_df - truncated_df) / true_df))
    worst_idx = np.unravel_index(
        np.argmax(log_abs_rel_err), log_abs_rel_err.shape
    )
    pcm = ax.imshow(
        log_abs_rel_err,
        interpolation="none",
        aspect="auto",
        cmap=cmap,
        vmax=0.5,
    )
    ax.set_ylabel("CDF Index")
    ax.set_xlabel(f"{col_names[0]}, {col_names[1]} Index")
    ax.set_title(
        f"log abs. rel. err. "
        f"(worst: {log_abs_rel_err.iloc[worst_idx]:.1f} at {worst_idx}"
    )
    fig.colorbar(pcm, ax=ax, extend="max")

    # plot
    plt.connect("button_press_event", on_click)
    plt.show()


