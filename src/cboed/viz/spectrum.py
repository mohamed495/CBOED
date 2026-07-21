r"""Quasi-optimality -- Proposition 1.

The identity that structures everything:

.. math::
    \sum_i (\ln\alpha_i + \ln\beta_i)

The sub-optimality constants use this same spectrum but read it in two
different ways:
- the incremental one controls the modes not yet selected;
- the conservative one uses a sum over the first d-m modes in order to
  provide a bound independent of the choice of sensors.
"""

import matplotlib.pyplot as plt
import numpy as np

from cboed.viz.style import COLORS


def plot_alpha_spectrum(alpha, beta=None, effective_rank=None, ax=None, title=""):
    """Generalized spectrum, log scale.

    ``alpha_i, beta_i >= 1`` (Prop. 1): the line at 1 marks the floor. A
    mode at ``alpha_i = 1`` does **not** contribute to the gap -- it is
    Gaussian.

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
    """The two constants of Prop. 1 as a function of ``m``.

    Parameters
    ----------
    eig_scale : float or None
        Order of magnitude of the EIG. **Must be supplied**: a
        sub-optimality constant is only meaningful relative to what it
        bounds. If it exceeds that quantity, the guarantee ``EIG >= max EIG
        - constant`` is vacuous, and the figure must show it.
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
    """Cumulative contribution of each mode to the gap.

    The curve reaches 100% at the effective rank. A sharp elbow = concentrated
    gap; a straight line = spread-out gap, and nothing to be gained by
    selecting modes.
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
    """``log(alpha_i)`` and ``log(beta_i)`` overlaid, linear scale.

    Visual equivalent of :func:`plot_alpha_spectrum` in ``semilogy`` -- both
    plot the same quantity on different axes. This one expresses it in
    directly additive nats (Prop. 1: the gap is a sum of these log-values),
    without the effective-rank marker.
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


def plot_spectrum_vs_lambda(alpha_by_lambda, beta_by_lambda, title=""):
    r"""``log(alpha_i)``, ``log(beta_i)`` and their sum -- one curve per ``lambda``.

    Three panels: growing non-linearity (increasing ``lambda``) shifts the
    spectrum upward (Prop. 1, the gap grows) -- directly visible here.

    Parameters
    ----------
    alpha_by_lambda, beta_by_lambda : dict[float, array]
        ``{lambda: alpha_i}`` -- same keys in both dicts.
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
