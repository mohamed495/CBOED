r"""Compléments de Schur et leurs mises à jour rank-1.

Équations (11)-(14) du théorème 2.1 : après avoir retenu le design ``W_m``, les
quatre matrices diagnostiques se conditionnent par

.. math::
    \Sigma(W_m) = \Sigma - \Sigma W_m (W_m^T \Sigma W_m)^{-1} W_m^T \Sigma

C'est **le** cœur computationnel de ``bounds/``. Le théorème n'est pas seulement
énoncé sous cette forme : c'est elle qui rend le greedy tractable, parce que les
compléments de Schur **se composent** -- conditionner par ``S U {j}`` revient à
conditionner par ``{j}`` la matrice déjà conditionnée par ``S``. D'où un update
rank-1 en ``O(p²)`` là où le recalcul direct coûte ``O(p² m + m³)``.

Ce module ne connaît ni le papier ni le BOED : de l'algèbre linéaire sur une SDP.
C'est le premier morceau de ce qui deviendra ``linalg/``.

Notes
-----
``W_m`` n'est jamais matérialisée. Pour une sélection canonique (colonnes = vecteurs
de base), ``W_m^T \Sigma W_m`` est la sous-matrice ``Sigma[ix_(design, design)]`` --
cf. ``core/selection.py``, qui est déjà ce ``W_m^T``.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def schur_complement(
    Sigma: Float[Array, "n_obs n_obs"],
    design: Int[Array, " n_sensors"] | None = None,
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma(W_m)`` recalculé depuis zéro -- équations (11)-(14).

    Coût ``O(p² m + m³)``. C'est l'**oracle** de :func:`schur_update` : chemin
    numérique indépendant, aucune récurrence.

    Parameters
    ----------
    Sigma : Float[Array, "n_obs n_obs"]
        Matrice SDP à conditionner (une des quatre diagnostiques).
    design : Int[Array, " n_sensors"] | None
        Indices retenus. ``None`` ou vide : aucun conditionnement, renvoie ``Sigma``.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        Complément de Schur, symétrisé.

    Notes
    -----
    Les lignes et colonnes de ``design`` sont **exactement nulles** en sortie
    (à l'arrondi près) : conditionner par une observation retire toute
    l'information qu'elle portait. Voir :func:`schur_gain_diagonal`.
    """
    if design is None or design.shape[0] == 0:
        return Sigma

    Sigma_S = Sigma[:, design]  # (p, m)
    Sigma_SS = Sigma[jnp.ix_(design, design)]  # (m, m)
    chol = jsp.linalg.cho_factor(Sigma_SS, lower=True)
    out = Sigma - Sigma_S @ jsp.linalg.cho_solve(chol, Sigma_S.T)
    return 0.5 * (out + out.T)


@jaxtyped(typechecker=beartype)
def schur_update(
    Sigma_cond: Float[Array, "n_obs n_obs"],
    j: int,
) -> Float[Array, "n_obs n_obs"]:
    r"""Ajoute le capteur ``j`` à un complément de Schur -- update **rank-1**.

    .. math::
        \Sigma(S \cup \{j\}) = \Sigma(S)
        - \frac{\Sigma(S)_{:,j}\, \Sigma(S)_{j,:}}{\Sigma(S)_{j,j}}

    Coût ``O(p²)`` : c'est ce qui fait passer le greedy de
    ``O(n_sensors * p² m)`` à ``O(n_sensors * p²)``.

    Parameters
    ----------
    Sigma_cond : Float[Array, "n_obs n_obs"]
        Complément déjà conditionné par ``S`` (ou ``Sigma`` brute si ``S`` vide).
    j : int
        Indice du capteur ajouté. **Ne doit pas être déjà dans ``S``** : le pivot
        ``Sigma_cond[j, j]`` y est numériquement nul et la division explose.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        ``Sigma(S U {j})``, symétrisé.

    Notes
    -----
    La symétrisation à chaque pas n'est pas cosmétique : sur une chaîne de
    ``n_sensors`` updates, l'asymétrie d'arrondi s'accumule et finit par faire
    échouer ``cho_factor`` en aval.
    """
    col = Sigma_cond[:, j]
    pivot = Sigma_cond[j, j]
    out = Sigma_cond - jnp.outer(col, col) / pivot
    return 0.5 * (out + out.T)


@jaxtyped(typechecker=beartype)
def schur_gain_diagonal(
    Sigma_num_cond: Float[Array, "n_obs n_obs"],
    Sigma_den_cond: Float[Array, "n_obs n_obs"],
    selected: Int[Array, " n_sensors"] | None = None,
) -> Float[Array, " n_obs"]:
    r"""Gain marginal de **chaque** candidat, d'un coup.

    Pour ``W_new = e_j``, le terme incrémental du théorème 2.1 se réduit à un
    rapport de deux entrées diagonales :

    .. math::
        \mathrm{gain}(j) = \tfrac12 \ln
        \frac{\Sigma_{\rm num}(W_m)_{j,j}}{\Sigma_{\rm den}(W_m)_{j,j}}

    C'est **toute** l'astuce : ``O(1)`` par candidat, ``O(p)`` pour la passe
    complète, aucune factorisation, aucun appel au modèle direct.

    Parameters
    ----------
    Sigma_num_cond, Sigma_den_cond : Float[Array, "n_obs n_obs"]
        Numérateur et dénominateur **déjà conditionnés** par le design courant.
        Incrémental : ``(Sigma_signal, Sigma_Y_given_theta)``.
        Conservatif : ``(Sigma_Y, Sigma_noise)``.
    selected : Int[Array, " n_sensors"] | None
        Capteurs déjà retenus. Masqués à ``-inf``.

    Returns
    -------
    Float[Array, " n_obs"]
        Gain par candidat ; ``-inf`` sur les indices déjà retenus.

    Notes
    -----
    ⚠️ Le masquage est **obligatoire**, pas défensif. Pour ``j`` déjà retenu, les
    deux diagonales valent ~1e-16 : le rapport est un quotient de bruit d'arrondi,
    fini et arbitraire. Sans masque le greedy resélectionne, en silence, un capteur
    déjà pris.
    """
    num = jnp.diagonal(Sigma_num_cond)
    den = jnp.diagonal(Sigma_den_cond)
    gain = 0.5 * (jnp.log(num) - jnp.log(den))

    if selected is not None and selected.shape[0] > 0:
        gain = gain.at[selected].set(-jnp.inf)
    return gain


@jaxtyped(typechecker=beartype)
def log_ratio(
    Sigma_num: Float[Array, "n_obs n_obs"],
    Sigma_den: Float[Array, "n_obs n_obs"],
    design: Int[Array, " n_sensors"] | None = None,
) -> Float[Array, ""]:
    r"""``½ ln |W^T A W| / |W^T B W|`` -- quotient de Rayleigh généralisé, à plat.

    Sous-matrices et ``slogdet``, aucun complément de Schur : chemin indépendant
    de :func:`schur_gain_diagonal` et de ``greedy_schur``, donc leur oracle.

    ``design=None`` -> design complet ``I_p``, utilisé par les bornes
    conservatives pour leur terme de référence.
    """
    if design is None:
        num, den = Sigma_num, Sigma_den
    else:
        ix = jnp.ix_(design, design)
        num, den = Sigma_num[ix], Sigma_den[ix]
    _, ln = jnp.linalg.slogdet(num)
    _, ld = jnp.linalg.slogdet(den)
    return 0.5 * (ln - ld)
