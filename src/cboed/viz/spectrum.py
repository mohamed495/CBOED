r"""Quasi-optimalité -- Proposition 1.

L'identité qui structure tout :

.. math::
    \sum_i (\ln\alpha_i + \ln\beta_i)

Les constantes de sous-optimalité utilisent ce même spectre mais selon
deux lectures différentes :
- l'incrémental contrôle les modes non encore sélectionnés ;
- le conservatif utilise une somme sur les d-m premiers modes afin de
  fournir une borne indépendante du choix des capteurs.
"""

import matplotlib.pyplot as plt
import numpy as np

from cboed.viz.style import COLORS


def plot_alpha_spectrum(alpha, beta=None, effective_rank=None, ax=None, title=""):
    """Spectre généralisé, échelle log.

    ``alpha_i, beta_i >= 1`` (Prop. 1) : la ligne à 1 marque le plancher. Un mode à
    ``alpha_i = 1`` ne contribue **pas** au gap -- il est gaussien.

    Un spectre **concentré** (quelques modes >> 1, le reste à 1) veut dire que la
    non-gaussianité vit dans peu de directions.
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
            f"rang effectif = {effective_rank}",
            fontsize=7,
            color=COLORS["truth"],
        )

    ax.set_xlabel("indice du mode")
    ax.set_ylabel("valeur propre generalisee")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_suboptimality(ms, inc, cons, eig_scale=None, ax=None, title=""):
    """Les deux constantes de Prop. 1 en fonction de ``m``.

    Parameters
    ----------
    eig_scale : float or None
        Ordre de grandeur de l'EIG. **À fournir** : une constante de sous-optimalité
        n'a de sens que rapportée à ce qu'elle borne. Si elle la dépasse, la garantie
        ``EIG >= max EIG - constante`` est vide, et la figure doit le montrer.
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
        label=r"conservatif  $\sum_{i\leq p-m}$",
    )

    if eig_scale is not None:
        ax.axhline(
            eig_scale,
            color=COLORS["exact"],
            lw=1.2,
            ls=":",
            label=f"echelle de l'EIG ({eig_scale:.1f} nats)",
        )
        ax.fill_between(ms, eig_scale, max(max(inc), max(cons)), color="0.85", alpha=0.5, zorder=0)
        ax.text(
            ms[len(ms) // 2],
            eig_scale * 1.05,
            "garantie vide",
            fontsize=7,
            style="italic",
            color="0.4",
        )

    ax.set_xlabel("nombre de capteurs $m$")
    ax.set_ylabel("sous-optimalite (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_gap_decomposition(alpha, beta, title=""):
    """Contribution cumulée de chaque mode au gap.

    La courbe atteint 100 % au rang effectif. Un coude marqué = gap concentré ; une
    droite = gap étalé, et rien à gagner en sélectionnant des modes.
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
    ax.set_xlabel("nombre de modes")
    ax.set_ylabel("% du gap explique")
    ax.set_ylim(0, 105)
    if title:
        ax.set_title(f"{title}   (gap total = {total:.3f} nats)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_log_generalized_spectrum(alpha, beta, title=""):
    """``log(alpha_i)`` et ``log(beta_i)`` superposes, echelle lineaire.

    Equivalent visuel de :func:`plot_alpha_spectrum` en ``semilogy`` -- les deux
    tracent la meme quantite sur des axes differents. Celle-ci l'exprime en nats
    directement additifs (Prop. 1 : le gap est une somme de ces log-valeurs), sans
    le marqueur de rang effectif.
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

    ax.set_xlabel("indice du mode")
    ax.set_ylabel("log-valeur propre generalisee (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
