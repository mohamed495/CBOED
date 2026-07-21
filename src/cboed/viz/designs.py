r"""Designs: sensor positions, comparison of greedy strategies.

The overlap between two designs is **not** the right metric: on a diffusive
field, different placements can carry the same information. What matters is
the score, not the coincidence of indices.
"""

import matplotlib.pyplot as plt
import numpy as np

from cboed.viz.style import COLORS


def plot_sensor_positions(x, designs, m=None, ax=None, title=""):
    """Positions selected by each strategy, one row per design.

    Parameters
    ----------
    designs : dict[str, array]
        ``{"iEIG>= (19)": indices, "cEIG>= (20)": indices, ...}``. Insertion
        order is respected: color encodes rank.
    m : int or None
        Truncate to ``m`` sensors.
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
    """Score as a function of ``m``, one curve per greedy strategy.

    Parameters
    ----------
    scores : dict[str, array]
        ``{"naive": ..., "batch": ..., "schur": ...}`` -- **all evaluated
        with the same criterion**. Comparing scores from different criteria
        would be meaningless.

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
    """Cost of each greedy strategy as a function of ``m``, log scale.

    Parameters
    ----------
    costs : dict[str, array]
        Cost per strategy -- number of black-box criterion evaluations for
        ``naive``/``batch``, flops for ``schur``. Heterogeneous units, hence
        the log scale: only the order of magnitude is comparable.
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
    """Sensors overlaid on the field -- where is the information taken from?

    Parameters
    ----------
    field : (n,)
        A reference field: prior ``std``, ``|E[u]|``, or the diagonal of
        ``Sigma_Y - Sigma_signal`` (where the non-Gaussianity resides).
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
