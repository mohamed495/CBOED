r"""Plot certified EIG bounds as a function of the sensor budget.

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
    """Plot certified incremental/conservative EIG bounds as a function of ``m``.

    Draws the incremental band (Cor. 1) as a filled region with solid edges,
    optionally overlays the conservative band (Cor. 2) with dashed edges, and
    optionally overlays Monte-Carlo boxplots and/or the exact EIG.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets. The bounds can be computed at any ``m`` at no extra
        cost (the greedy telescopes), so this is plotted as a continuous
        curve rather than a handful of points.
    inc_low, inc_up : array_like, shape (M,)
        Incremental lower/upper bounds (Cor. 1), one value per budget in
        `ms`.
    cons_low, cons_up : array_like, shape (M,), optional
        Conservative lower/upper bounds (Cor. 2), same shape as `inc_low`.
        Omit both to plot only the incremental band.
    mc : array_like, shape (R, K), optional
        ``R`` repetitions of a Monte-Carlo EIG estimator at the ``K`` budgets
        given by `mc_ms`, drawn as boxplots.
    mc_ms : array_like, shape (K,), optional
        Budgets at which `mc` was evaluated. Required if `mc` is given.
    truth : array_like, shape (M,), optional
        Exact EIG at each budget in `ms`, if computable (linear-Gaussian
        case).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if not given.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The parent figure of `ax`.
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
    """Plot bounds for several designs side by side, one panel per design.

    Each panel is produced by :func:`plot_bounds_vs_m`, sharing the y-axis so
    the panels are directly comparable -- the protocol from the paper's §2.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets, shared across all designs.
    per_strategy : dict[str, dict]
        One entry per design, e.g.
        ``{"iEIG>= (19)": {"inc_low":..., "inc_up":..., "cons_low":..., "cons_up":...}}``.
        Each inner dict is passed as keyword arguments to
        :func:`plot_bounds_vs_m`.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with one axes per entry of `per_strategy`.

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
    """Plot the bound width (``upper - lower``) per strategy, as a function of ``m``.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets.
    widths : dict[str, array_like]
        One width curve per strategy, each of shape ``(M,)``. Keys containing
        ``"inc"`` (case-insensitive) are colored as incremental, others as
        conservative.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Prop. 1: the sub-optimality constant **grows** with ``m`` for the
    incremental strategy, and **shrinks** for the conservative one. The
    widths follow suit, and their crossing point is the true transition
    between the two strategies -- unlike ``crossover()``, which is always
    ``p // 2 + 1`` regardless of the spectrum.
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
    r"""Plot the certified gap as a function of a model parameter (e.g. ``lambda``).

    Parameters
    ----------
    values : array_like, shape (N,)
        Values of the model parameter (x-axis).
    gaps : array_like, shape (N,)
        Certified gap at each value in `values`.
    xlabel : str, optional
        Label for the x-axis.
    mc_floor : float, optional
        Monte-Carlo floor -- typically the gap measured at ``lambda=0``,
        where the gap is theoretically zero (Rem. 2.2). Plotted as a dotted
        horizontal line: nothing below it is a real effect.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure

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
    """Draw one series as a median line plus a boxplot at each budget in ``ms``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    ms : array_like, shape (M,)
        Sensor budgets; the x-position of each boxplot.
    values : array_like, shape (R, M)
        One row per repetition, one column per budget in `ms`. At ``R = 1``,
        the median passes exactly through the points and each box
        degenerates into a line -- expected, not a bug.
    color : str
        Color used for the median line and the boxplot (edge, whiskers,
        caps, median, and a low-alpha fill).
    label : str
        Legend label carried by the returned proxy handle.
    linestyle : str, optional
        Line style of the median line, e.g. ``"-"`` (incremental) or
        ``"--"`` (conservative).

    Returns
    -------
    handle : matplotlib.lines.Line2D
        Proxy artist for the legend (the boxplot itself is not a suitable
        legend handle).
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
    """Plot bounds as boxplots over repetitions at each ``m``, instead of a continuous band.

    Like :func:`plot_bounds_vs_m`, but draws a boxplot (over repetitions) at
    each ``m`` rather than a continuous band: 4 overlaid series, one color
    and one line style each (see ``_SERIES_STYLE``), as in the NumPy
    prototype. The area between the lower and upper bound of each family
    (the certified gap, Prop. 1) is hatched -- present for **both** the
    incremental and the conservative family, not just one of them.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets.
    inc_low, inc_up : array_like, shape (R, M)
        Incremental bounds (Cor. 1), one row per repetition and one column
        per budget in `ms`.
    cons_low, cons_up : array_like, shape (R, M), optional
        Conservative bounds (Cor. 2), same shape. Omit both to plot only the
        incremental family.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if not given.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The parent figure of `ax`.

    Notes
    -----
    Colors and line styles are fixed per series in ``_SERIES_STYLE``: a
    validated categorical palette (dataviz skill, references/palette.md),
    slots 1-4 (blue, green, magenta, yellow), the only four that pass the
    CVD check across every pair. One color per series, plus a line style
    (solid/dashed) that encodes the family (incremental vs. conservative) --
    double encoding was explicitly requested (4 series, hue alone is not
    enough to quickly distinguish the two families). All series share the
    same ``x`` position (no offset): when two series are close in value,
    their boxes overlap -- that is the information (the bounds are
    tightening), not a display defect.

    Examples
    --------
    >>> import numpy as np
    >>> ms = np.array([1, 2, 3, 4])
    >>> inc_low = np.random.rand(5, 4) + 1.0
    >>> inc_up = inc_low + np.random.rand(5, 4) + 0.5
    >>> fig = plot_bounds_boxplot_vs_m(ms, inc_low, inc_up)
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
    """Plot boxplot bounds for several designs side by side.

    Boxplot version of :func:`plot_two_strategies` -- same layout. Each panel
    is produced by :func:`plot_bounds_boxplot_vs_m`, sharing the y-axis.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets, shared across all designs.
    per_strategy : dict[str, dict]
        ``{"iEIG>= (19)": {"inc_low": (R, M), ...}}`` -- same keys as
        :func:`plot_two_strategies`, but arrays of repetitions.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with one axes per entry of `per_strategy`.
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
