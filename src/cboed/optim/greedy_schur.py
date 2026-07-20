r"""Greedy par compléments de Schur -- ``O(n_sensors * p²)``.

Sélection gloutonne du design maximisant un quotient de Rayleigh généralisé

.. math::
    \tfrac12 \ln \frac{|W_m^T A\, W_m|}{|W_m^T B\, W_m|}

où ``(A, B)`` est un couple de matrices SDP en espace observation. Le module ne
sait pas d'où elles viennent : ``(Sigma_signal, Sigma_Y_given_theta)`` donne la
borne inférieure incrémentale (19), ``(Sigma_Y, Sigma_noise)`` la borne inférieure
conservative (20).

Pourquoi c'est tractable
------------------------
Le greedy générique (``optim/greedy.py``) évalue le critère en boîte noire :
``n_candidates * n_sensors`` appels, chacun refactorisant ``Gamma_post^{-1}``. Ici :

* le gain marginal d'un candidat est un **rapport de deux entrées diagonales** des
  matrices conditionnées -- ``O(1)`` par candidat ;
* l'ajout d'un capteur est un **update rank-1** des deux matrices -- ``O(p²)``.

Aucune factorisation, aucun appel au modèle direct : tout le coût du modèle est déjà
payé, une fois, dans la construction de ``A`` et ``B``.

⚠️ ``optim/greedy.py`` reste l'**oracle** de ce module. Ne pas le supprimer.
"""

import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped

from cboed.bounds.schur import schur_gain_diagonal, schur_update
from cboed.optim.base import Result


@jaxtyped(typechecker=beartype)
def greedy_schur(
    Sigma_num: Float[Array, "n_obs n_obs"],
    Sigma_den: Float[Array, "n_obs n_obs"],
    n_sensors: int,
) -> Result:
    r"""Design glouton maximisant ``½ ln |W^T A W| / |W^T B W|``.

    Parameters
    ----------
    Sigma_num, Sigma_den : Float[Array, "n_obs n_obs"]
        Numérateur ``A`` et dénominateur ``B``, SDP, **non conditionnées**.
        Incrémental : ``(Sigma_signal, Sigma_Y_given_theta)``.
        Conservatif : ``(Sigma_Y, Sigma_noise)``.
    n_sensors : int
        Budget. Doit valoir au plus ``n_obs``.

    Returns
    -------
    Result
        ``design`` : indices dans l'ordre d'ajout.
        ``scores`` : valeur **cumulée** du quotient après chaque ajout -- cf. Notes.

    Notes
    -----
    **Les scores télescopent.** Le gain marginal vaut
    ``log_ratio(S U {j}) - log_ratio(S)``, et ``log_ratio(∅) = 0`` ; la somme
    cumulée des gains est donc **exactement** ``log_ratio(S_k)`` à chaque étape.
    C'est ce qui rend ``scores`` comparable à celui de ``GreedyOptimizer``, dont le
    contrat est « score du critère après chaque ajout », et non un gain marginal.

    Ce n'est pas un détail d'implémentation : le télescopage **est** la
    décomposition incrémentale du théorème 2.1, et la borne inférieure du
    corollaire 1 se lit directement dans ``scores[-1]``.

    ⚠️ Pas de ``jit`` autour de la boucle : ``argmax`` rend un indice qui pilote un
    ``schur_update``, donc une valeur statique. Le greedy discret n'est de toute
    façon pas différentiable -- le design paramètre l'opérateur, il n'entre jamais
    dans ``jax.grad``.
    """
    p = Sigma_num.shape[0]
    if not 0 < n_sensors <= p:
        raise ValueError(f"n_sensors must be in (0, {p}], got {n_sensors}")

    num_cond, den_cond = Sigma_num, Sigma_den
    selected: list[int] = []
    scores: list[float] = []
    cumulative = 0.0

    for _ in range(n_sensors):
        gain = schur_gain_diagonal(num_cond, den_cond, jnp.asarray(selected, dtype=int))
        j = int(jnp.argmax(gain))

        cumulative += float(gain[j])
        selected.append(j)
        scores.append(cumulative)

        num_cond = schur_update(num_cond, j)
        den_cond = schur_update(den_cond, j)

    return Result(design=jnp.asarray(selected, dtype=int), scores=scores)


@jaxtyped(typechecker=beartype)
def log_ratio(
    Sigma_num: Float[Array, "n_obs n_obs"],
    Sigma_den: Float[Array, "n_obs n_obs"],
    design: Array,
) -> Float[Array, ""]:
    r"""``½ ln |W^T A W| / |W^T B W|`` évalué à plat.

    Chemin indépendant de :func:`greedy_schur` : sous-matrices et ``slogdet``,
    aucun complément de Schur. C'est l'oracle des ``scores``.
    """
    ix = jnp.ix_(design, design)
    _, ln = jnp.linalg.slogdet(Sigma_num[ix])
    _, ld = jnp.linalg.slogdet(Sigma_den[ix])
    return 0.5 * (ln - ld)
