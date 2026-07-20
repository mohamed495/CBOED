r"""Les bornes en fonction du budget.

⚠️ **Piège de direction.** Le même couple ``(A, B)`` donne des bornes **opposées**
selon la stratégie :

===============================  ==========================  ==================
                                 ``(signal, Y|theta)``       ``(Y, noise)``
===============================  ==========================  ==================
**Incrémental** (Cor. 1)         borne **INF** (15)          borne **SUP** (16)
**Conservatif** (Cor. 2)         borne **SUP** (18)          borne **INF** (17)
===============================  ==========================  ==================

Les légendes encodent donc les deux axes -- jamais ``lb``/``ub`` seuls.
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
    """Encadrements en fonction de ``m``.

    Parameters
    ----------
    ms : (M,)
        Budgets. Les bornes se calculent à tout ``m`` sans surcoût (le greedy
        télescope) : tracer une courbe continue, pas cinq points.
    inc_low, inc_up : (M,)
        Encadrement incrémental (Cor. 1).
    cons_low, cons_up : (M,) or None
        Encadrement conservatif (Cor. 2).
    mc : (R, K) or None
        ``R`` répétitions d'un estimateur MC aux budgets ``mc_ms`` -- boîtes.
    truth : (M,) or None
        EIG exacte, si calculable (cas linéaire-gaussien).
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
            label="conservatif (Cor. 2)",
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
        ax.plot(ms, truth, color=COLORS["exact"], lw=2.0, label="EIG exacte")

    ax.set_xlabel("nombre de capteurs $m$")
    ax.set_ylabel("gain d'information (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_two_strategies(ms, per_strategy, title=""):
    """Deux designs, quatre bornes chacun -- le protocole du papier §2.

    Parameters
    ----------
    per_strategy : dict[str, dict]
        ``{"iEIG>= (19)": {"inc_low":..., "inc_up":..., "cons_low":..., "cons_up":...}}``

    Notes
    -----
    Dualité §2 : maximiser (19) équivaut à maximiser la borne SUP conservative. Les
    deux designs **doivent** différer -- ce n'est pas une incohérence, c'est l'objet
    de la figure.
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
    """Largeur d'encadrement (``upper - lower``) par stratégie.

    Prop. 1 : la constante de sous-optimalité **croît** avec ``m`` en incrémental,
    **décroît** en conservatif. Les largeurs suivent, et leur croisement est la vraie
    transition entre les deux stratégies -- contrairement à ``crossover()``, qui vaut
    ``p//2 + 1`` quel que soit le spectre.
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
    ax.set_xlabel("nombre de capteurs $m$")
    ax.set_ylabel("largeur (nats)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_gap_vs_parameter(values, gaps, xlabel=r"$\lambda$", mc_floor=None, title=""):
    """Gap en fonction d'un paramètre du modèle.

    Parameters
    ----------
    mc_floor : float or None
        Plancher Monte-Carlo -- typiquement ``gap`` mesuré à ``lambda=0``, où le gap
        vaut zéro en théorie (Rem. 2.2). Tracé en pointillés : rien en dessous n'est
        un effet.

    Notes
    -----
    Le gap mesure la **non-gaussianité** de ``Y`` et ``Y|theta`` (Rem. 2.2 +
    Cramér-Rao), pas la non-linéarité. ``lambda`` pilote la seconde ; la première en
    est une conséquence, et le terme advectif ``lambda u d_x u`` étant quadratique en
    ``u``, l'amplitude du champ compte autant que ``lambda``.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(values, gaps, "o-", color=COLORS["Sigma_signal"], lw=1.8)
    if mc_floor is not None:
        ax.axhline(mc_floor, color="0.5", ls=":", lw=1.2, label="plancher MC")
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("gap (nats)")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


#: Palette categorielle validee (skill dataviz, references/palette.md) -- ordre
#: fixe, jamais recycle : slots 1-4 (bleu, vert, magenta, jaune), les seuls
#: quatre qui passent le controle CVD toutes paires confondues. Une couleur par
#: serie ; le trait plein partout (les 4 teintes suffisent a l'identite, pas
#: besoin d'un pointille en plus). Toutes les series partagent la meme position
#: ``x`` (pas de decalage) : quand deux series sont proches en valeur, leurs
#: boites se superposent -- c'est l'information (les bornes se resserrent), pas
#: un defaut d'affichage.
_SERIES_STYLE = {
    "inc_low": ("#2a78d6", "inc_lb"),
    "inc_up": ("#008300", "inc_ub"),
    "cons_low": ("#e87ba4", "cons_lb"),
    "cons_up": ("#eda100", "cons_ub"),
}


def _boxplot_series(ax, ms, values, color, label):
    """Ligne (mediane sur les repetitions) + boxplot a chaque ``m`` -- une serie.

    ``values`` : ``(n_repeats, n_budgets)``. A ``n_repeats = 1``, la mediane
    passe exactement par les points et chaque boite degenere en un trait --
    normal, pas une erreur.
    """
    values = np.atleast_2d(values)
    ax.plot(ms, np.median(values, axis=0), color=color, lw=2.0, zorder=2, solid_capstyle="round")
    width = (ms[1] - ms[0]) * 0.3 if len(ms) > 1 else 0.6
    line_props = dict(color=color, linewidth=1.4)
    ax.boxplot(
        values, positions=ms, widths=width,
        patch_artist=True, showfliers=False, manage_ticks=False,
        medianprops=line_props, whiskerprops=line_props, capprops=line_props,
        boxprops=dict(edgecolor=color, facecolor=color, alpha=0.12, linewidth=1.2),
        zorder=3,
    )
    return Line2D([0], [0], color=color, lw=2.0, label=label)


def plot_bounds_boxplot_vs_m(ms, inc_low, inc_up, cons_low=None, cons_up=None, ax=None, title=""):
    """Comme :func:`plot_bounds_vs_m`, mais un boxplot (sur les repetitions) a
    chaque ``m`` plutot qu'une bande continue -- 4 series superposees, une
    couleur chacune (cf. ``_SERIES_STYLE``), comme le prototype NumPy.

    Parameters
    ----------
    inc_low, inc_up : array ``(n_repeats, n_budgets)``
        Encadrement incremental (Cor. 1), une valeur par repetition et par budget.
    cons_low, cons_up : array ``(n_repeats, n_budgets)`` or None
        Encadrement conservatif (Cor. 2), meme forme.
    """
    ax = ax or plt.subplots(figsize=(6.5, 4.5))[1]
    ms = np.asarray(ms)

    series = {"inc_low": inc_low, "inc_up": inc_up}
    if cons_low is not None:
        series["cons_low"] = cons_low
        series["cons_up"] = cons_up

    handles = [
        _boxplot_series(ax, ms, arr, *_SERIES_STYLE[key])
        for key, arr in series.items()
    ]

    ax.set_xticks(ms)
    ax.set_xticklabels(ms)
    ax.set_xlabel("nombre de capteurs $m$")
    ax.set_ylabel("gain d'information (nats)")
    ax.legend(handles=handles, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_two_strategies_boxplot(ms, per_strategy, title=""):
    """Version boxplot de :func:`plot_two_strategies` -- meme mise en page.

    Parameters
    ----------
    per_strategy : dict[str, dict]
        ``{"iEIG>= (19)": {"inc_low": (n_repeats, n_budgets), ...}}`` -- memes
        cles que :func:`plot_two_strategies`, mais des arrays de repetitions.
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
