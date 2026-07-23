r"""Plot sensor designs: positions, scores, costs, and overlay on the field.

The overlap between two designs is **not** the right metric: on a diffusive
field, different placements can carry the same information. What matters is
the score, not the coincidence of indices.
"""

import matplotlib.pyplot as plt
import numpy as np

from cboed.viz.style import COLORS


def plot_sensor_positions(x, designs, m=None, ax=None, title=""):
    """Plot the sensor positions selected by each strategy, one row per design.

    Each row shows the selected positions as tick markers, colored by
    selection rank (early picks vs. late picks), with the insertion order
    annotated above each marker.

    Parameters
    ----------
    x : array_like, shape (n,)
        Spatial grid.
    designs : dict[str, array_like]
        One entry per design, e.g.
        ``{"iEIG>= (19)": indices, "cEIG>= (20)": indices, ...}``. Insertion
        order of the dict sets the row order; within each row, color encodes
        the rank of selection.
    m : int, optional
        Truncate each design to its first `m` sensors.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if not given.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The parent figure of `ax`.
    """
    ax = ax or plt.subplots(figsize=(8, 0.7 * len(designs) + 1.2))[1]
    x = np.asarray(x)

    for row, (_label, idx) in enumerate(designs.items()):
        idx = np.asarray(idx)[:m]
        ax.scatter(
            x[idx],
            np.full(len(idx), row),
            c=np.arange(len(idx)),
            cmap="viridis_r",
            s=60,
            marker="|",
            linewidths=2.5,
            zorder=3,
        )
        for rank, j in enumerate(idx):
            ax.annotate(
                str(rank + 1),
                (x[j], row),
                fontsize=5,
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color="0.4",
            )

    ax.set_yticks(range(len(designs)))
    ax.set_yticklabels(list(designs), fontsize=8)
    ax.set_xlabel("$x$")
    ax.set_ylim(-0.5, len(designs) - 0.5)
    ax.grid(axis="x", alpha=0.2)
    if title:
        ax.set_title(f"{title}   (the number is the insertion order)", fontsize=10)
    return ax.figure


def plot_greedy_comparison(ms, scores, title=""):
    """Plot criterion score as a function of ``m``, one curve per greedy strategy.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets.
    scores : dict[str, array_like]
        One score curve per strategy, shape ``(M,)`` each, e.g.
        ``{"naive": ..., "batch": ..., "schur": ...}``. All curves must be
        evaluated with the same criterion -- comparing scores from different
        criteria would be meaningless.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    The naive greedy is the **oracle** for the Schur greedy: they must
    produce the same design. A discrepancy is a bug, not a result.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, s in scores.items():
        ax.plot(ms, s, "o-", ms=3, lw=1.6, label=label)
    ax.set_xlabel("number of sensors $m$")
    ax.set_ylabel("score (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_greedy_cost(ms, costs, title=""):
    """Plot the cost of each greedy strategy as a function of ``m``, log scale.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets.
    costs : dict[str, array_like]
        Cost per strategy, shape ``(M,)`` each -- number of black-box
        criterion evaluations for ``naive``/``batch``, flops for ``schur``.
        Units are heterogeneous across strategies, hence the log scale: only
        the order of magnitude is comparable.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, c in costs.items():
        ax.semilogy(ms, c, "o-", ms=3, lw=1.6, label=label)
    ax.set_xlabel("number of sensors $m$")
    ax.set_ylabel("cost (log scale)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_design_on_field(x, field, designs, title=""):
    """Plot sensor positions overlaid on a reference field.

    Answers "where is the information taken from?": the field is drawn as a
    filled curve, and each design's sensors are drawn as vertical lines at a
    height that decreases with the design's position in `designs`, so
    overlapping designs remain distinguishable.

    Parameters
    ----------
    x : array_like, shape (n,)
        Spatial grid.
    field : array_like, shape (n,)
        A reference field: prior ``std``, ``|E[u]|``, or the diagonal of
        ``Sigma_Y - Sigma_signal`` (where the non-Gaussianity resides).
    designs : dict[str, array_like]
        One entry per design (indices into `x`). The first entry is colored
        as incremental, every other entry as conservative -- only two colors
        are used regardless of how many designs are given.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.asarray(x)
    ax.plot(x, np.asarray(field), color="0.4", lw=1.5, zorder=1)
    ax.fill_between(x, 0, np.asarray(field), color="0.85", zorder=0)

    ymax = float(np.max(field))
    for k, (label, idx) in enumerate(designs.items()):
        idx = np.asarray(idx)
        ax.vlines(
            x[idx],
            0,
            ymax * (0.9 - 0.1 * k),
            lw=1.2,
            alpha=0.75,
            color=COLORS["incremental"] if k == 0 else COLORS["conservative"],
            label=label,
        )
    ax.set_xlabel("$x$")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
