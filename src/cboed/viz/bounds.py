r"""Bounds as a function of budget.

⚠️ **Direction trap.** The same pair ``(A, B)`` gives **opposite** bounds
depending on the strategy:

===============================  ==========================  ==================
                                 ``(signal, Y|theta)``       ``(Y, noise)``
===============================  ==========================  ==================
**Incremental** (Cor. 1)         **LOWER** bound (15)         **UPPER** bound (16)
**Conservative** (Cor. 2)        **UPPER** bound (18)         **LOWER** bound (17)
===============================  ==========================  ==================

Legends therefore encode both axes -- never ``lb``/``ub`` alone.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cboed.viz.style import COLORS


def plot_bounds_vs_m(
    ms,
    inc_low,
    inc_up,
    cons_low=None,
    cons_up=None,
    mc=None,
    mc_ms=None,
    truth=None,
    ax=None,
    title="",
):
    """Bounds as a function of ``m``.

    Parameters
    ----------
    ms : (M,)
        Budgets. The bounds can be computed at any ``m`` at no extra cost (the
        greedy telescopes): plot a continuous curve, not five points.
    inc_low, inc_up : (M,)
        Incremental bounds (Cor. 1).
    cons_low, cons_up : (M,) or None
        Conservative bounds (Cor. 2).
    mc : (R, K) or None
        ``R`` repetitions of an MC estimator at budgets ``mc_ms`` -- boxplots.
    truth : (M,) or None
        Exact EIG, if computable (linear-Gaussian case).
    """
    ax = ax or plt.subplots(figsize=(7, 4.2))[1]

    ax.fill_between(
        ms, inc_low, inc_up, color=COLORS["incremental"], alpha=0.25, label="incremental (Cor. 1)"
    )
    ax.plot(ms, inc_low, color=COLORS["incremental"], lw=1.2)
    ax.plot(ms, inc_up, color=COLORS["incremental"], lw=1.2)

    if cons_low is not None:
        ax.fill_between(
            ms,
            cons_low,
            cons_up,
            color=COLORS["conservative"],
            alpha=0.20,
            label="conservative (Cor. 2)",
        )
        ax.plot(ms, cons_low, color=COLORS["conservative"], lw=1.2, ls="--")
        ax.plot(ms, cons_up, color=COLORS["conservative"], lw=1.2, ls="--")

    if mc is not None:
        ax.boxplot(
            np.asarray(mc),
            positions=np.asarray(mc_ms),
            widths=0.8,
            manage_ticks=False,
            showfliers=False,
        )

    if truth is not None:
        ax.plot(ms, truth, color=COLORS["exact"], lw=2.0, label="exact EIG")

    ax.set_xlabel("number of sensors $m$")
    ax.set_ylabel("information gain (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_two_strategies(ms, per_strategy, title=""):
    """Two designs, four bounds each -- the protocol from the paper's §2.

    Parameters
    ----------
    per_strategy : dict[str, dict]
        ``{"iEIG>= (19)": {"inc_low":..., "inc_up":..., "cons_low":..., "cons_up":...}}``

    Notes
    -----
    Duality §2: maximizing (19) is equivalent to maximizing the conservative
    upper bound. The two designs **must** differ -- this is not an
    inconsistency, it is the point of the figure.
    """
    fig, axes = plt.subplots(
        1, len(per_strategy), figsize=(6 * len(per_strategy), 4.2), sharey=True, squeeze=False
    )
    for ax, (label, b) in zip(axes[0], per_strategy.items(), strict=True):
        plot_bounds_vs_m(
            ms, b["inc_low"], b["inc_up"], b.get("cons_low"), b.get("cons_up"), ax=ax, title=label
        )
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_width_vs_m(ms, widths, title=""):
    """Bound width (``upper - lower``) per strategy.

    Prop. 1: the sub-optimality constant **grows** with ``m`` for the
    incremental strategy, and **shrinks** for the conservative one. The
    widths follow suit, and their crossing point is the true transition
    between the two strategies -- unlike ``crossover()``, which is always
    ``p//2 + 1`` regardless of the spectrum.
    """
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for label, w in widths.items():
        ax.plot(
            ms,
            w,
            lw=1.8,
            label=label,
            color=COLORS.get("incremental" if "inc" in label.lower() else "conservative"),
        )
    ax.set_xlabel("number of sensors $m$")
    ax.set_ylabel("width (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_gap_vs_parameter(values, gaps, xlabel=r"$\lambda$", mc_floor=None, title=""):
    """Gap as a function of a model parameter.

    Parameters
    ----------
    mc_floor : float or None
        Monte-Carlo floor -- typically the ``gap`` measured at ``lambda=0``,
        where the gap is theoretically zero (Rem. 2.2). Plotted as a dotted
        line: nothing below it is a real effect.

    Notes
    -----
    The gap measures the **non-Gaussianity** of ``Y`` and ``Y|theta`` (Rem.
    2.2 + Cramér-Rao), not the non-linearity. ``lambda`` drives the latter;
    the former is a consequence of it, and since the advective term
    ``lambda u d_x u`` is quadratic in ``u``, the field's amplitude matters
    just as much as ``lambda``.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(values, gaps, "o-", color=COLORS["Sigma_signal"], lw=1.8)
    if mc_floor is not None:
        ax.axhline(mc_floor, color="0.5", ls=":", lw=1.2, label="MC floor")
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("gap (nats)")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


#: Validated categorical palette (dataviz skill, references/palette.md) -- fixed
#: order, never recycled: slots 1-4 (blue, green, magenta, yellow), the only
#: four that pass the CVD check across every pair. One color per series, plus
#: a line style (solid/dashed) that encodes the family (incremental vs.
#: conservative) -- double encoding was explicitly requested (4 series, hue
#: alone is not enough to quickly distinguish the two families). All series
#: share the same ``x`` position (no offset): when two series are close in
#: value, their boxes overlap -- that is the information (the bounds are
#: tightening), not a display defect.
_SERIES_STYLE = {
    "inc_low": ("#2a78d6", "inc_lb", "-"),
    "inc_up": ("#008300", "inc_ub", "-"),
    "cons_low": ("#e87ba4", "cons_lb", "--"),
    "cons_up": ("#eda100", "cons_ub", "--"),
}


def _boxplot_series(ax, ms, values, color, label, linestyle="-"):
    """Line (median across repetitions) + boxplot at each ``m`` -- one series.

    ``values`` : ``(n_repeats, n_budgets)``. At ``n_repeats = 1``, the median
    passes exactly through the points and each box degenerates into a line --
    expected, not a bug.
    """
    values = np.atleast_2d(values)
    ax.plot(
        ms,
        np.median(values, axis=0),
        color=color,
        lw=2.0,
        ls=linestyle,
        zorder=2,
        solid_capstyle="round",
        dash_capstyle="round",
    )
    width = (ms[1] - ms[0]) * 0.3 if len(ms) > 1 else 0.6
    line_props = dict(color=color, linewidth=1.4)
    ax.boxplot(
        values,
        positions=ms,
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops=line_props,
        whiskerprops=line_props,
        capprops=line_props,
        boxprops=dict(edgecolor=color, facecolor=color, alpha=0.12, linewidth=1.2),
        zorder=3,
    )
    return Line2D([0], [0], color=color, lw=2.0, ls=linestyle, label=label)


def plot_bounds_boxplot_vs_m(ms, inc_low, inc_up, cons_low=None, cons_up=None, ax=None, title=""):
    """Like :func:`plot_bounds_vs_m`, but a boxplot (over repetitions) at
    each ``m`` rather than a continuous band -- 4 overlaid series, one color
    and one line style each (see ``_SERIES_STYLE``), as in the NumPy
    prototype. The area between the lower and upper bound of each family
    (the certified gap, Prop. 1) is hatched -- present for **both** the
    incremental and the conservative family, not just one of them.

    Parameters
    ----------
    inc_low, inc_up : array ``(n_repeats, n_budgets)``
        Incremental bounds (Cor. 1), one value per repetition and per budget.
    cons_low, cons_up : array ``(n_repeats, n_budgets)`` or None
        Conservative bounds (Cor. 2), same shape.
    """
    ax = ax or plt.subplots(figsize=(6.5, 4.5))[1]
    ms = np.asarray(ms)

    series = {"inc_low": inc_low, "inc_up": inc_up}
    if cons_low is not None:
        series["cons_low"] = cons_low
        series["cons_up"] = cons_up

    med = {key: np.median(np.atleast_2d(arr), axis=0) for key, arr in series.items()}
    ax.fill_between(
        ms,
        med["inc_low"],
        med["inc_up"],
        facecolor="none",
        edgecolor="0.35",
        hatch="////",
        linewidth=0.0,
        zorder=1,
    )
    if "cons_low" in med:
        ax.fill_between(
            ms,
            med["cons_low"],
            med["cons_up"],
            facecolor="none",
            edgecolor="0.35",
            hatch="\\\\\\\\",
            linewidth=0.0,
            zorder=1,
        )

    handles = [_boxplot_series(ax, ms, arr, *_SERIES_STYLE[key]) for key, arr in series.items()]

    ax.set_xticks(ms)
    ax.set_xticklabels(ms)
    ax.set_xlabel("number of sensors $m$")
    ax.set_ylabel("information gain (nats)")
    ax.legend(handles=handles, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_two_strategies_boxplot(ms, per_strategy, title=""):
    """Boxplot version of :func:`plot_two_strategies` -- same layout.

    Parameters
    ----------
    per_strategy : dict[str, dict]
        ``{"iEIG>= (19)": {"inc_low": (n_repeats, n_budgets), ...}}`` -- same
        keys as :func:`plot_two_strategies`, but arrays of repetitions.
    """
    fig, axes = plt.subplots(
        1, len(per_strategy), figsize=(6.5 * len(per_strategy), 4.2), sharey=True, squeeze=False
    )
    for ax, (label, b) in zip(axes[0], per_strategy.items(), strict=True):
        plot_bounds_boxplot_vs_m(
            ms, b["inc_low"], b["inc_up"], b.get("cons_low"), b.get("cons_up"), ax=ax, title=label
        )
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
