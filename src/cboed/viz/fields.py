r"""Fields: prior samples, posterior samples, reconstruction.

The functions take already-computed arrays. No model calls.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cboed.viz.style import COLORS


def _mark_sensors_rug(ax, x, sensors, height_frac=0.045):
    """Short ticks anchored at the bottom of the axis -- one per sensor.

    Unlike a full-height ``axvline``, these never cross the plotted
    trajectories: they sit in a thin band at the current bottom of the axis,
    sized as a fraction of the y-range so the mark stays visible regardless
    of scale.
    """
    ymin, ymax = ax.get_ylim()
    tick_h = height_frac * (ymax - ymin)
    for j in np.asarray(sensors):
        ax.plot(
            [x[j], x[j]],
            [ymin, ymin + tick_h],
            color=COLORS["sensors"],
            lw=2.2,
            solid_capstyle="butt",
            zorder=4,
        )
    ax.set_ylim(ymin, ymax)


def plot_field_samples(
    x,
    samples,
    mean=None,
    std=None,
    truth=None,
    sensors=None,
    ax=None,
    color="prior",
    label="prior",
    n_show=20,
):
    """Trajectories + ``±2 sigma`` envelope.

    Parameters
    ----------
    x : (n,)
        Spatial grid.
    samples : (n_samples, n)
        Realizations. Only ``n_show`` are plotted; the envelope uses all of
        them.
    mean, std : (n,) or None
        If ``None``, computed from ``samples``.
    truth : (n,) or None
        ``theta_true``, overlaid.
    sensors : (m,) or None
        Indices of the selected sensors, marked on the axis.
    """
    ax = ax or plt.subplots(figsize=(7, 3.2))[1]
    samples = np.asarray(samples)
    mean = np.asarray(samples.mean(0) if mean is None else mean)
    std = np.asarray(samples.std(0) if std is None else std)
    c = COLORS[color]

    for s in samples[:n_show]:
        ax.plot(x, s, color=c, alpha=0.12, lw=0.8)
    ax.fill_between(
        x, mean - 2 * std, mean + 2 * std, color=c, alpha=0.22, label=rf"{label} $\pm 2\sigma$"
    )
    ax.plot(x, mean, color=c, lw=1.8, label=f"{label} mean")

    if truth is not None:
        ax.plot(x, truth, color=COLORS["truth"], lw=2.0, ls="--", label=r"$\theta_{\rm true}$")
    if sensors is not None:
        _mark_sensors_rug(ax, np.asarray(x), sensors)
        ax.plot([], [], color=COLORS["sensors"], lw=2.2, label="sensors")

    ax.set_xlabel("$x$")
    ax.legend(fontsize=7, ncol=2)
    return ax.figure


def plot_reconstruction(
    x,
    prior_samples,
    posterior_samples,
    truth,
    sensors=None,
    laplace_warning=False,
    n_show=30,
    qoi_span=None,
):
    """Prior, posterior, and ``theta_true`` overlaid on a single plot.

    Parameters
    ----------
    laplace_warning : bool
        If ``True``, annotate that the posterior is the Laplace
        approximation linearized at ``mu_prior`` -- exact only if the model
        is linear, and increasingly wrong the farther ``theta_true`` is from
        the linearization point.
    n_show : int
        Number of realizations plotted per cloud (prior, posterior).
    qoi_span : tuple[float, float] or None
        ``(x_min, x_max)`` of the goal-oriented region of interest, shaded in
        the background. Affects only the display: the plotted field remains
        the full field.
    """
    prior_samples = np.asarray(prior_samples)
    posterior_samples = np.asarray(posterior_samples)
    x = np.asarray(x)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    if qoi_span is not None:
        ax.axvspan(qoi_span[0], qoi_span[1], color=COLORS["sensors"], alpha=0.08, zorder=0)

    for s in prior_samples[:n_show]:
        ax.plot(x, s, color=COLORS["prior"], alpha=0.25, lw=0.7)
    for s in posterior_samples[:n_show]:
        ax.plot(x, s, color=COLORS["posterior"], alpha=0.35, lw=0.7)
    ax.plot(x, np.asarray(truth), color=COLORS["truth"], lw=2.2)

    handles = [
        Line2D([0], [0], color=COLORS["prior"], lw=1.5, label="prior (realizations)"),
        Line2D([0], [0], color=COLORS["posterior"], lw=1.5, label="posterior (realizations)"),
        Line2D([0], [0], color=COLORS["truth"], lw=2.0, label=r"$\theta_{\rm true}$"),
    ]
    if qoi_span is not None:
        from matplotlib.patches import Patch

        handles.append(
            Patch(facecolor=COLORS["sensors"], alpha=0.15, label="QoI region (goal-oriented)")
        )
    if sensors is not None:
        _mark_sensors_rug(ax, x, sensors)
        handles.append(Line2D([0], [0], color=COLORS["sensors"], lw=2.2, label="sensors"))

    if laplace_warning:
        ax.text(
            0.02,
            0.02,
            "Laplace (linearized at $\\mu_{prior}$)",
            transform=ax.transAxes,
            fontsize=7,
            style="italic",
            color="0.35",
        )

    ax.set_xlabel("$x$")
    ax.legend(handles=handles, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_contraction(x, prior_std, posterior_std, sensors=None):
    """``sigma_post / sigma_prior`` -- where the design actually informs.

    More legible than two overlaid envelopes: the contraction is local, and
    it is what distinguishes two designs of the same budget.
    """
    fig, ax = plt.subplots(figsize=(7, 2.8))
    x = np.asarray(x)
    ratio = np.asarray(posterior_std) / np.asarray(prior_std)
    ax.plot(x, ratio, color=COLORS["posterior"], lw=1.8)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    if sensors is not None:
        # Anchored directly on the curve rather than a separate vertical mark:
        # the dot already sits at the local contraction value, so it doubles
        # as "how much this sensor actually helped" -- no extra visual layer.
        j = np.asarray(sensors)
        ax.plot(
            x[j],
            ratio[j],
            marker="o",
            ls="none",
            color=COLORS["sensors"],
            ms=8,
            mec="white",
            mew=1.0,
            zorder=5,
            label="sensors",
        )
        ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\sigma_{post} / \sigma_{prior}$")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    return fig


def plot_contraction_spectrum(
    Gamma_prior,
    Gamma_post,
    title="",
):
    """Generalized contraction spectrum.

    Eigenvalues of:

        C = Gamma_prior^{-1/2}
            Gamma_post
            Gamma_prior^{-1/2}

    Values close to 1:
        no information gained.

    Values close to 0:
        strong uncertainty reduction.
    """

    Gamma_prior = np.asarray(Gamma_prior)
    Gamma_post = np.asarray(Gamma_post)

    eigvals, eigvecs = np.linalg.eigh(Gamma_prior)

    eigvals = np.maximum(
        eigvals,
        np.finfo(float).eps,
    )

    Gamma_prior_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    C = Gamma_prior_inv_sqrt @ Gamma_post @ Gamma_prior_inv_sqrt

    contraction = np.linalg.eigvalsh(C)
    contraction = np.sort(contraction)[::-1]

    fig, ax = plt.subplots(figsize=(6, 3.2))

    ax.semilogy(
        np.arange(1, len(contraction) + 1),
        contraction,
        "o-",
        ms=3,
        lw=1.5,
        color=COLORS["posterior"],
    )

    ax.axhline(
        1.0,
        color="0.6",
        ls=":",
        lw=0.8,
    )

    ax.set_xlabel("mode")
    ax.set_ylabel(
        r"$\lambda_i(\Gamma_{prior}^{-1/2}"
        r"\Gamma_{post}"
        r"\Gamma_{prior}^{-1/2})$"
    )

    if title:
        ax.set_title(title)

    fig.tight_layout()

    return fig
