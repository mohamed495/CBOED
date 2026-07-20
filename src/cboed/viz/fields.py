r"""Champs : échantillons du prior, de la postérieure, reconstruction.

Les fonctions prennent des tableaux déjà calculés. Aucun appel au modèle.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cboed.viz.style import COLORS


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
    """Trajectoires + enveloppe ``±2 sigma``.

    Parameters
    ----------
    x : (n,)
        Grille spatiale.
    samples : (n_samples, n)
        Réalisations. Seules ``n_show`` sont tracées ; l'enveloppe utilise tout.
    mean, std : (n,) or None
        Si ``None``, calculés sur ``samples``.
    truth : (n,) or None
        ``theta_vrai``, superposé.
    sensors : (m,) or None
        Indices des capteurs retenus, marqués sur l'axe.
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
    ax.plot(x, mean, color=c, lw=1.8, label=f"{label} moyenne")

    if truth is not None:
        ax.plot(x, truth, color=COLORS["truth"], lw=2.0, ls="--", label=r"$\theta_{\rm vrai}$")
    if sensors is not None:
        ax.plot(
            np.asarray(x)[np.asarray(sensors)],
            np.full(len(sensors), ax.get_ylim()[0]),
            "|",
            color=COLORS["sensors"],
            ms=12,
            mew=2,
            label="capteurs",
        )

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
):
    """Prior, posterieur et ``theta_vrai`` superposes sur un seul graphique.

    Parameters
    ----------
    laplace_warning : bool
        Si ``True``, annote que la postérieure est l'approximation de Laplace
        linéarisée en ``mu_prior`` -- exacte seulement si le modèle est linéaire, et
        d'autant plus fausse que ``theta_vrai`` est loin du point de linéarisation.
    n_show : int
        Nombre de réalisations tracées par nuage (prior, posterieur).
    """
    prior_samples = np.asarray(prior_samples)
    posterior_samples = np.asarray(posterior_samples)
    x = np.asarray(x)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s in prior_samples[:n_show]:
        ax.plot(x, s, color=COLORS["prior"], alpha=0.25, lw=0.7)
    for s in posterior_samples[:n_show]:
        ax.plot(x, s, color=COLORS["posterior"], alpha=0.35, lw=0.7)
    ax.plot(x, np.asarray(truth), color=COLORS["truth"], lw=2.2)

    handles = [
        Line2D([0], [0], color=COLORS["prior"], lw=1.5, label="prior (realisations)"),
        Line2D([0], [0], color=COLORS["posterior"], lw=1.5, label="posterieur (realisations)"),
        Line2D([0], [0], color=COLORS["truth"], lw=2.0, label=r"$\theta_{\rm vrai}$"),
    ]
    if sensors is not None:
        for j in np.asarray(sensors):
            ax.axvline(x[j], color=COLORS["sensors"], lw=0.8, alpha=0.5)
        handles.append(Line2D([0], [0], color=COLORS["sensors"], lw=1.5, label="capteurs"))

    if laplace_warning:
        ax.text(
            0.02,
            0.02,
            "Laplace (linearise en $\\mu_{prior}$)",
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
    """``sigma_post / sigma_prior`` -- où le design informe réellement.

    Plus lisible que deux enveloppes superposées : la contraction est locale, et
    c'est elle qui distingue deux designs de même budget.
    """
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ratio = np.asarray(posterior_std) / np.asarray(prior_std)
    ax.plot(x, ratio, color=COLORS["posterior"], lw=1.8)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    if sensors is not None:
        for j in np.asarray(sensors):
            ax.axvline(np.asarray(x)[j], color=COLORS["sensors"], lw=0.8, alpha=0.5)
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
