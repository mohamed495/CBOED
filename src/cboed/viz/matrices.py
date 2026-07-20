r"""Matrices diagnostiques et les moments qui les produisent.

La figure centrale est :func:`plot_diagnostics` -- les deux paires du théorème 2.1 et
leurs différences.

À ``lambda = 0`` : ``Sigma_signal = Sigma_Y`` et ``Sigma_noise = Sigma_{Y|theta}``,
donc les deux panneaux de différence sont vides (Rem. 2.2). C'est la **figure de
validation** : si elles ne sont pas vides en linéaire, quelque chose est faux.

À ``lambda != 0``, la structure qui apparaît dans les différences **est** le gap.
"""

import matplotlib.pyplot as plt
import numpy as np

from cboed.viz.style import CMAP_DIFF, CMAP_PSD, symmetric_limits


def _imshow(ax, M, title, cmap=CMAP_PSD, vlim=None):
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
    """Les deux paires du Thm 2.1 et leurs différences.

    Ligne 1 : ``Sigma_Y``, ``Sigma_signal``, leur différence -- le numérateur.
    Ligne 2 : ``Sigma_{Y|theta}``, ``Sigma_noise``, leur différence -- le dénominateur.

    Chaque paire partage son échelle de couleur : sans ça, deux matrices proches
    paraîtraient différentes parce que leurs extrema le sont.
    """
    d = diagnostics
    pairs = [
        (d.Sigma_Y, d.Sigma_signal, r"$\Sigma_Y$", r"$\Sigma_{signal}$"),
        (d.Sigma_Y_given_theta, d.Sigma_noise, r"$\Sigma_{Y|\theta}$", r"$\Sigma_{noise}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

    for row, (A, B, la, lb) in enumerate(pairs):
        A, B = np.asarray(A), np.asarray(B)
        vlim = (min(A.min(), B.min()), max(A.max(), B.max()))  # echelle partagee
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
    """Les ingrédients de Prop. 4.

    ``L(u)`` -- jacobienne moyenne, ``(q, p)``.
    ``H(u)`` -- covariance des jacobiennes, ``(q, q)``. **Nulle si u est linéaire.**
    ``I_eta`` -- précision du prior, ``(q, q)``.
    ``J(h)`` -- terme QoI, ``(q, q)``. Seule différence entre signal et noise.

    ``H(u)`` est la seule matrice qui distingue Prop. 4 d'un calcul linéaire-gaussien :
    la voir vide à ``lambda=0`` et pleine à ``lambda>0`` valide la branche.
    """
    mats = [
        (L, r"$L(u) = \mathbb{E}[\mathrm{Jac}\,u]^T$"),
        (H, r"$H(u)$ -- covariance des jacobiennes"),
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
    """Plusieurs estimations de la **même** matrice, et leurs écarts à une référence.

    Pour comparer standard et goal-oriented en linéaire, ou deux voies vers
    ``Sigma_signal`` (gradient §3.3 contre approximation §3.2).

    Parameters
    ----------
    reference : int
        Index de la matrice servant de référence pour les écarts.
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
        _imshow(axes[1, k], diffs[k], f"ecart{suffix}", cmap=CMAP_DIFF, vlim=dlim)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_spectrum_comparison(mats, labels, title=""):
    """Spectres superposés, échelle log.

    Deux matrices peuvent se ressembler visuellement et avoir des spectres très
    différents -- c'est le spectre qui pilote les log-dets, donc les bornes.
    """
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for M, label in zip(mats, labels, strict=True):
        M = 0.5 * (M + M.T)
        ev = np.linalg.eigvalsh(M)[::-1]
        ax.semilogy(np.arange(1, len(ev) + 1), np.clip(ev, 1e-16, None), lw=1.6, label=label)
    ax.set_xlabel("indice")
    ax.set_ylabel("valeur propre")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
