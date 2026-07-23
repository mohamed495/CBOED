r"""Plot the generalized spectrum and sub-optimality constants of Proposition 1.

The identity that structures everything:

.. math::
    \sum_i (\ln\alpha_i + \ln\beta_i)

The sub-optimality constants use this same spectrum but read it in two
different ways:
- the incremental one sums the first ``m`` modes, so the guarantee
  loosens as the budget grows;
- the conservative one sums the first ``d-m`` modes, so the guarantee
  tightens as the budget grows, giving a bound independent of the choice
  of sensors.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cboed.viz.style import COLORS


def plot_alpha_spectrum(alpha, beta=None, effective_rank=None, ax=None, title=""):
    """Plot the generalized spectrum ``alpha_i`` (and optionally ``beta_i``), log scale.

    Parameters
    ----------
    alpha : array_like, shape (p,)
        Generalized eigenvalues ``alpha_i`` (``Sigma_Y`` vs ``Sigma_signal``).
    beta : array_like, shape (p,), optional
        Generalized eigenvalues ``beta_i`` (``Sigma_{Y|theta}`` vs
        ``Sigma_noise``). Omitted if not available.
    effective_rank : int, optional
        Mode index beyond which the spectrum is considered numerically flat;
        marked with a vertical line and annotated.
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
    ``alpha_i, beta_i >= 1`` (Prop. 1): the line at 1 marks the floor. A mode
    at ``alpha_i = 1`` does **not** contribute to the gap -- it is Gaussian.

    A **concentrated** spectrum (a few modes >> 1, the rest at 1) means the
    non-Gaussianity lives in few directions.
    """
    ax = ax or plt.subplots(figsize=(6, 3.4))[1]
    alpha = np.asarray(alpha)
    idx = np.arange(1, len(alpha) + 1)

    ax.semilogy(
        idx,
        alpha,
        "o-",
        ms=3,
        lw=1.5,
        color=COLORS["Sigma_Y"],
        label=r"$\alpha_i$  ($\Sigma_Y$ vs $\Sigma_{signal}$)",
    )
    if beta is not None:
        ax.semilogy(
            idx,
            np.asarray(beta),
            "s--",
            ms=3,
            lw=1.2,
            color=COLORS["Sigma_noise"],
            label=r"$\beta_i$  ($\Sigma_{Y|\theta}$ vs $\Sigma_{noise}$)",
        )
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")

    if effective_rank is not None:
        ax.axvline(effective_rank + 0.5, color=COLORS["truth"], lw=1.0, ls="--")
        ax.text(
            effective_rank + 1,
            ax.get_ylim()[1] * 0.5,
            f"effective rank = {effective_rank}",
            fontsize=7,
            color=COLORS["truth"],
        )

    ax.set_xlabel("mode index")
    ax.set_ylabel("generalized eigenvalue")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_suboptimality(ms, inc, cons, eig_scale=None, ax=None, title=""):
    """Plot the incremental and conservative sub-optimality constants of Prop. 1 vs ``m``.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets.
    inc, cons : array_like, shape (M,)
        Incremental and conservative sub-optimality constants at each budget
        in `ms`.
    eig_scale : float, optional
        Order of magnitude of the EIG. Should be supplied whenever available:
        a sub-optimality constant is only meaningful relative to what it
        bounds. If it exceeds that quantity, the guarantee
        ``EIG >= max EIG - constant`` is vacuous, and the shaded region above
        the ``eig_scale`` line makes that visible.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if not given.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The parent figure of `ax`.
    """
    ax = ax or plt.subplots(figsize=(6, 3.4))[1]
    ax.plot(
        ms,
        inc,
        "o-",
        ms=3,
        lw=1.6,
        color=COLORS["incremental"],
        label=r"incremental  $\sum_{i\leq m}$",
    )
    ax.plot(
        ms,
        cons,
        "s--",
        ms=3,
        lw=1.6,
        color=COLORS["conservative"],
        label=r"conservative  $\sum_{i\leq p-m}$",
    )

    if eig_scale is not None:
        ax.axhline(
            eig_scale,
            color=COLORS["exact"],
            lw=1.2,
            ls=":",
            label=f"EIG scale ({eig_scale:.1f} nats)",
        )
        ax.fill_between(ms, eig_scale, max(max(inc), max(cons)), color="0.85", alpha=0.5, zorder=0)
        ax.text(
            ms[len(ms) // 2],
            eig_scale * 1.05,
            "vacuous guarantee",
            fontsize=7,
            style="italic",
            color="0.4",
        )

    ax.set_xlabel("number of sensors $m$")
    ax.set_ylabel("sub-optimality (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_gap_decomposition(alpha, beta, title=""):
    """Plot the cumulative contribution of each mode to the total gap, as a percentage.

    Parameters
    ----------
    alpha, beta : array_like, shape (p,)
        Generalized eigenvalues ``alpha_i``, ``beta_i`` (Prop. 1). The
        per-mode term is ``0.5 * (log(alpha_i) + log(beta_i))``.
    title : str, optional
        Axes title. The total gap (in nats) is appended to it.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    The curve reaches 100% at the effective rank. A sharp elbow means a
    concentrated gap; a straight line means a spread-out gap, and nothing to
    be gained by selecting modes.
    """
    terms = 0.5 * (np.log(np.asarray(alpha)) + np.log(np.asarray(beta)))
    total = terms.sum()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(
        np.arange(1, len(terms) + 1),
        100 * np.cumsum(terms) / total,
        "o-",
        ms=3,
        lw=1.6,
        color=COLORS["Sigma_signal"],
    )
    ax.axhline(90, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("number of modes")
    ax.set_ylabel("% of gap explained")
    ax.set_ylim(0, 105)
    if title:
        ax.set_title(f"{title}   (total gap = {total:.3f} nats)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_log_generalized_spectrum(alpha, beta, title=""):
    """Plot ``log(alpha_i)`` and ``log(beta_i)`` overlaid, linear scale.

    Parameters
    ----------
    alpha, beta : array_like, shape (p,)
        Generalized eigenvalues ``alpha_i``, ``beta_i`` (Prop. 1).
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Visual equivalent of :func:`plot_alpha_spectrum` plotted with
    ``semilogy`` -- both show the same quantity on different axes. This one
    expresses it in directly additive nats (Prop. 1: the gap is a sum of
    these log-values), without the effective-rank marker.
    """
    fig, ax = plt.subplots(figsize=(6, 3.4))

    i = np.arange(1, len(alpha) + 1)

    ax.plot(
        i,
        np.log(np.asarray(alpha)),
        "o-",
        ms=3,
        lw=1.5,
        color=COLORS["Sigma_Y"],
        label=r"$\log(\alpha_i)$  ($\Sigma_Y$ vs $\Sigma_{signal}$)",
    )
    ax.plot(
        i,
        np.log(np.asarray(beta)),
        "s--",
        ms=3,
        lw=1.2,
        color=COLORS["Sigma_noise"],
        label=r"$\log(\beta_i)$  ($\Sigma_{Y|\theta}$ vs $\Sigma_{noise}$)",
    )
    ax.axhline(0, color="0.6", lw=0.8, ls=":")

    ax.set_xlabel("mode index")
    ax.set_ylabel("generalized log-eigenvalue (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_suboptimality_vs_lambda(ms, inc_by_lambda, cons_by_lambda, title=""):
    r"""Plot incremental/conservative sub-optimality constants vs budget, one pair of curves
    per ``lambda``.

    Both families share a single panel so the incremental (rising) /
    conservative (falling) trade-off is visible directly.

    Parameters
    ----------
    ms : array_like, shape (M,)
        Sensor budgets (can be a dense range, not just the sensor budgets
        actually used -- the sum, eq. (22)/(23), is cheap to evaluate at any
        ``m``).
    inc_by_lambda, cons_by_lambda : dict[float, array_like]
        ``{lambda: [suboptimality(m, strategy) for m in ms]}``, each value of
        shape ``(M,)`` matching `ms` -- same keys in both dicts.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    (22) and (23) read the *same* spectral terms
    ``t_i = (log(alpha_i) + log(beta_i)) / 2`` two different ways -- a
    **partial sum** over the first ``m`` terms (incremental, solid) or the
    first ``d - m`` terms (conservative, dashed). One color per ``lambda``,
    consistent with :func:`plot_spectrum_vs_lambda`; line style carries the
    strategy, matching the convention used in ``viz.bounds``.

    Examples
    --------
    >>> ms = [1, 2, 3, 4]
    >>> inc_by_lambda = {0.0: [0.4, 0.3, 0.2, 0.1], 1.0: [0.5, 0.4, 0.3, 0.2]}
    >>> cons_by_lambda = {0.0: [0.1, 0.2, 0.3, 0.4], 1.0: [0.2, 0.3, 0.4, 0.5]}
    >>> fig = plot_suboptimality_vs_lambda(ms, inc_by_lambda, cons_by_lambda)
    """
    lams = sorted(inc_by_lambda)
    cmap = plt.get_cmap("viridis")
    colors = {lam: cmap(i / max(len(lams) - 1, 1)) for i, lam in enumerate(lams)}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for lam in lams:
        color = colors[lam]
        ax.plot(ms, inc_by_lambda[lam], "o-", ms=4, lw=1.8, color=color)
        ax.plot(ms, cons_by_lambda[lam], "s--", ms=4, lw=1.8, color=color)

    lambda_handles = [
        Line2D([0], [0], color=colors[lam], lw=1.8, label=rf"$\lambda={lam}$") for lam in lams
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="0.3",
            lw=1.8,
            ls="-",
            marker="o",
            ms=4,
            label=r"incremental $\sum_{i=1}^{m}$ (22)",
        ),
        Line2D(
            [0],
            [0],
            color="0.3",
            lw=1.8,
            ls="--",
            marker="s",
            ms=4,
            label=r"conservative $\sum_{i=1}^{d-m}$ (23)",
        ),
    ]
    legend_lambda = ax.legend(
        handles=lambda_handles, fontsize=7, loc="center left", title=r"$\lambda$"
    )
    ax.add_artist(legend_lambda)
    ax.legend(handles=style_handles, fontsize=7, loc="upper center")

    ax.set_xlabel("number of sensors $m$")
    ax.set_ylabel("sub-optimality constant (nats)")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_spectrum_vs_lambda(alpha_by_lambda, beta_by_lambda, title=""):
    r"""Plot ``log(alpha_i)``, ``log(beta_i)``, and their sum, one curve per ``lambda``.

    Three panels: growing non-linearity (increasing ``lambda``) shifts the
    spectrum upward (Prop. 1, the gap grows) -- directly visible here.

    Parameters
    ----------
    alpha_by_lambda, beta_by_lambda : dict[float, array_like]
        ``{lambda: alpha_i}`` / ``{lambda: beta_i}``, each value a 1-D array
        of generalized eigenvalues (length may differ across `lambda`
        values) -- same keys in both dicts.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> alpha_by_lambda = {0.0: [1.0, 1.0, 1.2], 1.0: [1.5, 1.3, 1.1]}
    >>> beta_by_lambda = {0.0: [1.0, 1.0, 1.1], 1.0: [1.4, 1.2, 1.05]}
    >>> fig = plot_spectrum_vs_lambda(alpha_by_lambda, beta_by_lambda)
    """
    lams = sorted(alpha_by_lambda)
    cmap = plt.get_cmap("viridis")
    colors = {lam: cmap(i / max(len(lams) - 1, 1)) for i, lam in enumerate(lams)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for lam in lams:
        alpha = np.asarray(alpha_by_lambda[lam])
        beta = np.asarray(beta_by_lambda[lam])
        idx = np.arange(1, len(alpha) + 1)
        label = rf"$\lambda={lam}$"
        axes[0].plot(idx, np.log(alpha), lw=1.5, color=colors[lam], label=label)
        axes[1].plot(idx, np.log(beta), lw=1.5, color=colors[lam], label=label)
        axes[2].plot(idx, np.log(alpha) + np.log(beta), lw=1.5, color=colors[lam], label=label)

    axes[0].set_ylabel(r"$\log(\alpha_i)$")
    axes[1].set_ylabel(r"$\log(\beta_i)$")
    axes[2].set_ylabel(r"$\log(\alpha_i) + \log(\beta_i)$")
    for ax in axes:
        ax.set_xlabel("mode index")
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
