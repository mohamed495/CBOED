r"""Designs : positions des capteurs, comparaison des stratégies gloutonnes.

Le recouvrement entre deux designs n'est **pas** la bonne métrique : sur un champ
diffusif, des placements différents portent la même information. Ce qui compte est le
score, pas la coïncidence des indices.
"""

import matplotlib.pyplot as plt
import numpy as np

from cboed.viz.style import COLORS


def plot_sensor_positions(x, designs, m=None, ax=None, title=""):
    """Positions retenues par chaque stratégie, une ligne par design.

    Parameters
    ----------
    designs : dict[str, array]
        ``{"iEIG>= (19)": indices, "cEIG>= (20)": indices, ...}``. Ordre d'ajout
        respecté : la couleur code le rang.
    m : int or None
        Tronque à ``m`` capteurs.
    """
    ax = ax or plt.subplots(figsize=(8, 0.7 * len(designs) + 1.2))[1]
    x = np.asarray(x)

    for row, (_label, idx) in enumerate(designs.items()):
        idx = np.asarray(idx)[:m]
        ax.scatter(
            x[idx],
            np.full(len(idx), row),
            c=np.arange(len(idx)),
            cmap="viridis_r",
            s=60,
            marker="|",
            linewidths=2.5,
            zorder=3,
        )
        for rank, j in enumerate(idx):
            ax.annotate(
                str(rank + 1),
                (x[j], row),
                fontsize=5,
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color="0.4",
            )

    ax.set_yticks(range(len(designs)))
    ax.set_yticklabels(list(designs), fontsize=8)
    ax.set_xlabel("$x$")
    ax.set_ylim(-0.5, len(designs) - 0.5)
    ax.grid(axis="x", alpha=0.2)
    if title:
        ax.set_title(f"{title}   (le numero est l'ordre d'ajout)", fontsize=10)
    return ax.figure


def plot_greedy_comparison(ms, scores, title=""):
    """Score en fonction de ``m``, une courbe par stratégie gloutonne.

    Parameters
    ----------
    scores : dict[str, array]
        ``{"naif": ..., "batch": ..., "schur": ...}`` -- **tous évalués avec le même
        critère**. Comparer des scores issus de critères différents n'aurait pas de
        sens.

    Notes
    -----
    Le greedy naïf est l'**oracle** du greedy Schur : ils doivent rendre le même
    design. Un écart est un bug, pas un résultat.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, s in scores.items():
        ax.plot(ms, s, "o-", ms=3, lw=1.6, label=label)
    ax.set_xlabel("nombre de capteurs $m$")
    ax.set_ylabel("score (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_greedy_cost(ms, costs, title=""):
    """Cout de chaque strategie gloutonne en fonction de ``m``, echelle log.

    Parameters
    ----------
    costs : dict[str, array]
        Cout par strategie -- nombre d'evaluations du critere en boite noire
        pour ``naif``/``batch``, flops pour ``schur``. Unites heterogenes,
        d'ou l'echelle log : seul l'ordre de grandeur est comparable.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, c in costs.items():
        ax.semilogy(ms, c, "o-", ms=3, lw=1.6, label=label)
    ax.set_xlabel("nombre de capteurs $m$")
    ax.set_ylabel("cout (echelle log)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_design_on_field(x, field, designs, title=""):
    """Capteurs superposés au champ -- où l'information est-elle prélevée ?

    Parameters
    ----------
    field : (n,)
        Un champ de référence : ``std`` du prior, ``|E[u]|``, ou la diagonale de
        ``Sigma_Y - Sigma_signal`` (là où la non-gaussianité se loge).
    """
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.asarray(x)
    ax.plot(x, np.asarray(field), color="0.4", lw=1.5, zorder=1)
    ax.fill_between(x, 0, np.asarray(field), color="0.85", zorder=0)

    ymax = float(np.max(field))
    for k, (label, idx) in enumerate(designs.items()):
        idx = np.asarray(idx)
        ax.vlines(
            x[idx],
            0,
            ymax * (0.9 - 0.1 * k),
            lw=1.2,
            alpha=0.75,
            color=COLORS["incremental"] if k == 0 else COLORS["conservative"],
            label=label,
        )
    ax.set_xlabel("$x$")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
