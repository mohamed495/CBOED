r"""Plot fields: prior samples, posterior samples, reconstruction, contraction.

The functions take already-computed arrays. No model calls.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cboed.viz.style import COLORS


def _mark_sensors_rug(ax, x, sensors, height_frac=0.045):
    """Draw short tick marks at the bottom of the axis, one per sensor.

    Unlike a full-height ``axvline``, these never cross the plotted
    trajectories: they sit in a thin band at the current bottom of the axis,
    sized as a fraction of the y-range (`height_frac`) so the mark stays
    visible regardless of scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x : array_like, shape (n,)
        Spatial grid.
    sensors : array_like, shape (m,)
        Indices into `x` of the sensors to mark.
    height_frac : float, optional
        Tick height as a fraction of the current y-range.
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
    """Plot sample trajectories with a ``mean ± 2 sigma`` envelope.

    Parameters
    ----------
    x : array_like, shape (n,)
        Spatial grid.
    samples : array_like, shape (n_samples, n)
        Realizations. Only `n_show` of them are plotted individually; the
        ``mean``/``std`` envelope is computed from all of them.
    mean, std : array_like, shape (n,), optional
        Mean and standard deviation of the field. Computed from `samples` if
        not given.
    truth : array_like, shape (n,), optional
        ``theta_true``, overlaid as a dashed line.
    sensors : array_like, shape (m,), optional
        Indices into `x` of the selected sensors, marked on the axis as a
        rug of ticks (see :func:`_mark_sensors_rug`).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if not given.
    color : str, optional
        Key into ``COLORS`` used for the samples, mean, and envelope.
    label : str, optional
        Legend label prefix (e.g. ``"prior"`` or ``"posterior"``).
    n_show : int, optional
        Number of individual realizations plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The parent figure of `ax`.
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
    sensor_order=None,
    laplace_warning=False,
    n_show=30,
    qoi_span=None,
):
    """Plot prior, posterior, and ``theta_true`` realizations overlaid on one axes.

    Also draws the posterior mean (``posterior_samples.mean(axis=0)``) as a
    solid line in the posterior color, on top of the (thin, semi-transparent)
    individual posterior realizations, so the central estimate is visible at
    a glance alongside the spread and ``theta_true``.

    Parameters
    ----------
    x : array_like, shape (n,)
        Spatial grid.
    prior_samples : array_like, shape (n_samples, n)
        Prior realizations. Only `n_show` are plotted.
    posterior_samples : array_like, shape (n_samples, n)
        Posterior realizations. Only `n_show` are plotted individually; all
        of them are used to compute the posterior mean curve.
    truth : array_like, shape (n,)
        ``theta_true``, overlaid as a solid line.
    sensors : array_like, shape (m,), optional
        Indices into `x` of the selected sensors, marked on the axis as a
        rug of ticks (see :func:`_mark_sensors_rug`).
    sensor_order : array_like, shape (m,), optional
        Selection rank to display above each sensor. The order must correspond
        to `sensors`; for a greedy design this is ``1, 2, ..., m``.
    laplace_warning : bool, optional
        If ``True``, annotate that the posterior is the Laplace
        approximation linearized at ``mu_prior`` -- exact only if the model
        is linear, and increasingly wrong the farther ``theta_true`` is from
        the linearization point.
    n_show : int, optional
        Number of realizations plotted per cloud (prior, posterior).
    qoi_span : tuple[float, float], optional
        ``(x_min, x_max)`` of the goal-oriented region of interest, shaded in
        the background. Affects only the display: the plotted field remains
        the full field.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 50)
    >>> prior = np.random.randn(10, 50)
    >>> posterior = 0.3 * np.random.randn(10, 50)
    >>> truth = np.sin(2 * np.pi * x)
    >>> fig = plot_reconstruction(x, prior, posterior, truth)
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
    posterior_mean = posterior_samples.mean(axis=0)
    ax.plot(x, posterior_mean, color=COLORS["posterior"], lw=2.0, zorder=3)
    ax.plot(x, np.asarray(truth), color=COLORS["truth"], lw=2.2, zorder=4)

    if sensor_order is not None:
        if sensors is None:
            raise ValueError("sensor_order requires sensors")
        sensors = np.asarray(sensors)
        sensor_order = np.asarray(sensor_order)
        if sensor_order.shape != sensors.shape:
            raise ValueError("sensor_order and sensors must have the same shape")
        y_min, y_max = ax.get_ylim()
        label_offset = 0.035 * (y_max - y_min)
        label_base = y_min + 0.06 * (y_max - y_min)
        for label_index, (sensor, rank) in enumerate(zip(sensors, sensor_order, strict=True)):
            ax.text(
                x[sensor],
                label_base + (label_index % 3) * label_offset,
                str(rank),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=COLORS["sensors"],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.8},
                clip_on=True,
            )

    handles = [
        Line2D([0], [0], color=COLORS["prior"], lw=1.5, label="prior (realizations)"),
        Line2D([0], [0], color=COLORS["posterior"], lw=1.5, label="posterior (realizations)"),
        Line2D([0], [0], color=COLORS["posterior"], lw=2.0, label="posterior mean"),
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


def plot_energy_posterior_histograms(
    theta_samples_by_design,
    theta_true,
    prior_theta_samples=None,
    bins=60,
    show_density=True,
    overlay=True,
    x_max=None,
):
    """Plot posterior energy histograms for sensor designs.

    Parameters
    ----------
    theta_samples_by_design : dict[str, array_like]
        One-dimensional samples of ``||eta||**2``, keyed by design label.
    theta_true : float
        Energy of the latent field used to generate the common observation.
    prior_theta_samples : array_like, optional
        Prior energy samples, overlaid for reference in both panels.
    bins : int, optional
        Number of histogram bins.
    show_density : bool, optional
        Overlay a Gaussian-kernel density estimate on each histogram.
    overlay : bool, optional
        Overlay all designs on one axes when true. When false, use one panel
        per design.
    x_max : float, optional
        Upper display limit for the energy axis. Samples remain unchanged;
        this only zooms the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with one panel per design.

    Notes
    -----
    This figure visualizes ``theta | Y_m``. At ``lambda=0`` the posterior
    samples of ``eta | Y_m`` are exact Gaussian draws, while the energy
    transformation itself is nonlinear and generally non-Gaussian.
    """
    labels = list(theta_samples_by_design)
    if not labels:
        raise ValueError("theta_samples_by_design must not be empty")
    arrays = [np.asarray(theta_samples_by_design[label]).ravel() for label in labels]
    if any(not np.all(np.isfinite(samples)) for samples in arrays):
        raise ValueError("energy posterior samples must contain finite values")
    finite = arrays
    if prior_theta_samples is not None:
        prior_theta_samples = np.asarray(prior_theta_samples).ravel()
        if not np.all(np.isfinite(prior_theta_samples)):
            raise ValueError("prior energy samples must contain finite values")

    all_samples = np.concatenate(finite)
    if prior_theta_samples is not None:
        all_samples = np.concatenate([all_samples, prior_theta_samples])
    edges = np.histogram_bin_edges(all_samples, bins=bins)
    x_grid = np.linspace(0.0, max(float(all_samples.max()), float(theta_true)) * 1.05, 400)

    def kde(samples):
        bandwidth = 1.06 * np.std(samples) * samples.size ** (-1 / 5)
        bandwidth = max(float(bandwidth), np.finfo(float).eps)
        scaled = (x_grid[:, None] - samples[None, :]) / bandwidth
        return np.mean(np.exp(-0.5 * scaled**2), axis=1) / (bandwidth * np.sqrt(2 * np.pi))

    n_axes = 1 if overlay else len(labels)
    fig, axes = plt.subplots(1, n_axes, figsize=(5.8 * n_axes, 3.6), squeeze=False)
    axes = axes[0]
    posterior_colors = {
        "INC": COLORS["incremental"],
        "CONS": COLORS["conservative"],
    }
    for index, (label, samples) in enumerate(zip(labels, finite, strict=True)):
        ax = axes[0] if overlay else axes[index]
        color = posterior_colors.get(label, COLORS["posterior"])
        if prior_theta_samples is not None and index == 0:
            ax.hist(
                prior_theta_samples,
                bins=edges,
                density=True,
                color=COLORS["prior"],
                alpha=0.22,
                label="prior",
            )
        ax.hist(samples, bins=edges, density=True, color=color, alpha=0.32,
                label=f"{label} posterior")
        if show_density:
            ax.plot(x_grid, kde(samples), color=color, lw=2.0, label=f"{label} density")
        if not overlay:
            ax.axvline(theta_true, color=COLORS["truth"], ls="--", lw=1.8,
                       label=r"$\theta_{\rm true}$")
            ax.set_title(label)
        variance = float(np.var(samples, ddof=1)) if samples.size > 1 else float("nan")
        prior_variance = (
            float(np.var(prior_theta_samples, ddof=1))
            if prior_theta_samples is not None and prior_theta_samples.size > 1
            else float("nan")
        )
        ratio = variance / prior_variance if prior_variance > 0 else float("nan")
        if overlay:
            annotation = ax.texts[0] if ax.texts else None
            line = f"{label}: Var(post)/Var(prior) = {ratio:.3f}"
            if annotation is None:
                annotation = ax.text(
                    0.02,
                    0.97,
                    line,
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                    color=color,
                    linespacing=1.5,
                    bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.85, "pad": 3},
                )
            else:
                annotation.set_text(f"{annotation.get_text()}\n{line}")
                annotation.set_color("0.2")
                annotation.set_bbox(
                    {"facecolor": "white", "edgecolor": "0.35", "alpha": 0.85, "pad": 3}
                )
        else:
            ax.text(
                0.02,
                0.97,
                f"{label}: Var(post)/Var(prior) = {ratio:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color=color,
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.8, "pad": 2},
            )
    ax = axes[0]
    ax.axvline(theta_true, color=COLORS["truth"], ls="--", lw=1.8,
               label=r"$\theta_{\rm true}$")
    for ax in axes:
        ax.set_xlabel(r"$\theta = \|\eta\|^2$")
        ax.set_ylabel("density")
        if x_max is not None:
            if x_max <= 0:
                raise ValueError(f"x_max must be > 0, got {x_max}")
            ax.set_xlim(0.0, x_max)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_contraction(x, prior_std, posterior_std, sensors=None):
    r"""Plot the local contraction ratio ``sigma_post / sigma_prior``.

    Parameters
    ----------
    x : array_like, shape (n,)
        Spatial grid.
    prior_std, posterior_std : array_like, shape (n,)
        Prior and posterior standard deviation fields.
    sensors : array_like, shape (m,), optional
        Indices into `x` of the selected sensors, marked directly on the
        contraction curve.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    More legible than two overlaid envelopes: the contraction is local, and
    it is what distinguishes two designs of the same budget. Sensors are
    anchored directly on the curve rather than as a separate vertical mark:
    the dot already sits at the local contraction value, so it doubles as
    "how much this sensor actually helped" -- no extra visual layer.
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
    """Plot the generalized contraction spectrum, log scale.

    Computes and plots the eigenvalues of
    ``C = Gamma_prior^{-1/2} Gamma_post Gamma_prior^{-1/2}``, sorted in
    decreasing order.

    Parameters
    ----------
    Gamma_prior, Gamma_post : array_like, shape (q, q)
        Prior and posterior covariance matrices.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Values close to 1 mean no information was gained along that mode; values
    close to 0 mean a strong uncertainty reduction.
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
