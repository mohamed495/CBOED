r"""Plot diagnostic matrices and the moments that produce them.

The central figure is :func:`plot_diagnostics` -- the two pairs from Thm 2.1
and their differences.

At ``lambda = 0``: ``Sigma_signal = Sigma_Y`` and ``Sigma_noise =
Sigma_{Y|theta}``, so both difference panels are empty (Rem. 2.2). This is
the **validation figure**: if they are not empty in the linear case,
something is wrong.

At ``lambda != 0``, the structure that appears in the differences **is** the
gap.
"""

import matplotlib.pyplot as plt
import numpy as np

from cboed.viz.style import CMAP_DIFF, CMAP_PSD, symmetric_limits


def _imshow(ax, M, title, cmap=CMAP_PSD, vlim=None):
    """Draw a matrix with `imshow`, hidden ticks, and an attached colorbar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    M : array_like, shape (r, c)
        Matrix to display. Not required to be square (e.g. the Jacobian
        ``L(u)`` in :func:`plot_moments` is rectangular).
    title : str
        Axes title.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap passed to `imshow`. Defaults to ``CMAP_PSD``.
    vlim : tuple[float, float], optional
        ``(vmin, vmax)`` passed to `imshow`. Uses matplotlib's default
        autoscaling if not given.

    Returns
    -------
    im : matplotlib.image.AxesImage
    """
    M = np.asarray(M)
    kw = {} if vlim is None else {"vmin": vlim[0], "vmax": vlim[1]}
    im = ax.imshow(M, cmap=cmap, **kw)
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return im


def plot_diagnostics(diagnostics, title=""):
    """Plot the two matrix pairs from Thm 2.1 and their differences.

    Row 1: ``Sigma_Y``, ``Sigma_signal``, their difference -- the numerator.
    Row 2: ``Sigma_{Y|theta}``, ``Sigma_noise``, their difference -- the
    denominator.

    Parameters
    ----------
    diagnostics : object
        Must expose the attributes ``Sigma_Y``, ``Sigma_signal``,
        ``Sigma_Y_given_theta``, and ``Sigma_noise`` as arrays of matching
        shape ``(q, q)``.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Each pair shares its color scale: without that, two close matrices would
    look different simply because their extrema are.

    At ``lambda = 0``: ``Sigma_signal = Sigma_Y`` and ``Sigma_noise =
    Sigma_{Y|theta}``, so both difference panels are empty (Rem. 2.2). This
    is the **validation figure**: if they are not empty in the linear case,
    something is wrong. At ``lambda != 0``, the structure that appears in the
    differences **is** the gap.
    """
    d = diagnostics
    pairs = [
        (d.Sigma_Y, d.Sigma_signal, r"$\Sigma_Y$", r"$\Sigma_{signal}$"),
        (d.Sigma_Y_given_theta, d.Sigma_noise, r"$\Sigma_{Y|\theta}$", r"$\Sigma_{noise}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

    for row, (A, B, la, lb) in enumerate(pairs):
        A, B = np.asarray(A), np.asarray(B)
        vlim = (min(A.min(), B.min()), max(A.max(), B.max()))  # shared scale
        _imshow(axes[row, 0], A, la, vlim=vlim)
        _imshow(axes[row, 1], B, lb, vlim=vlim)
        diff = A - B
        rel = np.linalg.norm(diff) / np.linalg.norm(A)
        _imshow(
            axes[row, 2],
            diff,
            f"{la} $-$ {lb}   (rel. {rel:.1e})",
            cmap=CMAP_DIFF,
            vlim=symmetric_limits(diff),
        )

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_moments(L, H, I_eta, J_h=None, title=""):
    """Plot the matrix ingredients of Prop. 4.

    Parameters
    ----------
    L : array_like, shape (q, p)
        ``L(u)`` -- mean Jacobian.
    H : array_like, shape (q, q)
        ``H(u)`` -- covariance of the Jacobians. Zero if ``u`` is linear.
    I_eta : array_like, shape (q, q)
        ``I_eta = Gamma_prior^{-1}`` -- prior precision.
    J_h : array_like, shape (q, q), optional
        ``J(h)`` -- QoI term, the only difference between signal and noise.
        Omitted if not applicable.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    ``H(u)`` is the only matrix that distinguishes Prop. 4 from a
    linear-Gaussian computation: seeing it empty at ``lambda=0`` and full at
    ``lambda>0`` validates that branch.
    """
    mats = [
        (L, r"$L(u) = \mathbb{E}[\mathrm{Jac}\,u]^T$"),
        (H, r"$H(u)$ -- covariance of the jacobians"),
        (I_eta, r"$\mathcal{I}_\eta = \Gamma_{prior}^{-1}$"),
    ]
    if J_h is not None:
        mats.append((J_h, r"$\mathcal{J}(h)$"))

    fig, axes = plt.subplots(1, len(mats), figsize=(3.4 * len(mats), 3.4))
    for ax, (M, label) in zip(np.atleast_1d(axes), mats, strict=True):
        M = np.asarray(M)
        _imshow(ax, M, label, cmap=CMAP_DIFF, vlim=symmetric_limits(M))
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_matrix_comparison(mats, labels, reference=0, title=""):
    """Plot several estimates of the same matrix and their deviations from a reference.

    Row 1: each matrix, on a shared color scale. Row 2: each matrix's
    deviation from the reference (the reference panel itself is left blank),
    on a shared, zero-centered diverging color scale.

    Used to compare standard and goal-oriented in the linear case, or two
    routes to ``Sigma_signal`` (gradient §3.3 versus approximation §3.2).

    Parameters
    ----------
    mats : sequence of array_like
        The matrices to compare, all of the same shape ``(q, q)``.
    labels : sequence of str
        One label per matrix in `mats`.
    reference : int, optional
        Index into `mats`/`labels` of the matrix used as the reference for
        the deviations.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(mats)
    ref = np.asarray(mats[reference])
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.5), squeeze=False)

    vlim = (min(np.asarray(M).min() for M in mats), max(np.asarray(M).max() for M in mats))
    diffs = [np.asarray(M) - ref for M in mats]
    axes[1, reference].axis("off")
    dlim = (
        symmetric_limits(*[d for i, d in enumerate(diffs) if i != reference]) if n > 1 else (-1, 1)
    )

    for k, (M, label) in enumerate(zip(mats, labels, strict=True)):
        _imshow(axes[0, k], M, label, vlim=vlim)
        rel = np.linalg.norm(diffs[k]) / np.linalg.norm(ref)
        suffix = " (reference)" if k == reference else f"  (rel. {rel:.1e})"
        _imshow(axes[1, k], diffs[k], f"deviation{suffix}", cmap=CMAP_DIFF, vlim=dlim)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_spectrum_comparison(mats, labels, title=""):
    """Plot overlaid eigenvalue spectra of several matrices, log scale.

    Parameters
    ----------
    mats : sequence of array_like
        Matrices to compare, each of shape ``(q, q)``; each is symmetrized
        (``0.5 * (M + M.T)``) before its eigenvalues are computed.
    labels : sequence of str
        One label per matrix in `mats`.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Two matrices can look visually similar and have very different spectra
    -- it is the spectrum that drives the log-dets, and thus the bounds.
    """
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for M, label in zip(mats, labels, strict=True):
        M = 0.5 * (M + M.T)
        ev = np.linalg.eigvalsh(M)[::-1]
        ax.semilogy(np.arange(1, len(ev) + 1), np.clip(ev, 1e-16, None), lw=1.6, label=label)
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
