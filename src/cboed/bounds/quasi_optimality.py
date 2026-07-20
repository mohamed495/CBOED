r"""Quasi-optimalité -- Proposition 1.

Le gap n'est pas un scalaire opaque : c'est une **somme de log-valeurs propres
généralisées**, et sa répartition décide laquelle des deux stratégies est utilisable.

Prop. 1 pose les deux problèmes aux valeurs propres généralisées

.. math::
    \Sigma_Y u_i = \alpha_i \Sigma_{\rm signal} u_i, \qquad
    \Sigma_{Y|\theta} v_i = \beta_i \Sigma_{\rm noise} v_i

avec ``alpha_i, beta_i >= 1`` (car ``Sigma_Y ⪰ Sigma_signal`` et
``Sigma_{Y|theta} ⪰ Sigma_noise``), et borne la sous-optimalité des designs gloutons :

.. math::
    \mathrm{EIG}(W^{\rm inc}_m) &\ge \max_W \mathrm{EIG}(W)
        - \sum_{i=1}^{m} \tfrac{\ln\alpha_i + \ln\beta_i}{2} \\
    \mathrm{EIG}(W^{\rm cons}_m) &\ge \max_W \mathrm{EIG}(W)
        - \sum_{i=1}^{d-m} \tfrac{\ln\alpha_i + \ln\beta_i}{2}

L'identité qui relie tout
-------------------------
En sommant toutes les valeurs propres,

.. math::
    \sum_i (\ln\alpha_i + \ln\beta_i) = 2\,\mathrm{gap}(I_p).

Le gap au design complet est donc la somme des contributions spectrales

.. math::
    t_i = \frac{\ln\alpha_i+\ln\beta_i}{2}.

Les constantes de la Proposition 1 sont simplement des sommes partielles de ces
contributions :

* incrémental : ``\sum_{i=1}^{m} t_i`` ;
* conservatif : ``\sum_{i=1}^{d-m} t_i``.

Les deux utilisent les mêmes contributions spectrales, mais avec un nombre de termes
différent. La constante incrémentale est donc croissante avec le budget ``m``, tandis
que la constante conservative est décroissante.

Cas standard
------------
``Sigma_{Y|theta} = Sigma_noise = Sigma_obs`` **exactement**, donc ``beta_i = 1`` pour
tout ``i`` et ``ln beta_i = 0`` : la sous-optimalité ne dépend que de ``alpha``. Le
cadre standard isole ``gap_G``, y compris spectralement.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped

from cboed.bounds.base import DiagnosticMatrices


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QuasiOptimality:
    """Le spectre du gap et ce qu'il coûte."""

    alpha: Float[Array, " n_obs"]
    """Valeurs propres de ``(Sigma_Y, Sigma_signal)``, **décroissantes**. ``>= 1``."""

    beta: Float[Array, " n_obs"]
    """Valeurs propres de ``(Sigma_Y_given_theta, Sigma_noise)``, décroissantes. ``>= 1``.

    Identiquement 1 en cadre standard : les deux matrices y valent ``Sigma_obs``.
    """

    def suboptimality(self, n_sensors: int, strategy: str = "incremental") -> float:
        """Borne sur ``max_W EIG(W) - EIG(W_greedy)`` -- éq. (22)/(23).

        Parameters
        ----------
        n_sensors : int
            Budget ``m``.
        strategy : {"incremental", "conservative"}
            Incrémental : somme des ``m`` **premières** (les plus grandes).
            Conservatif : somme des ``d - m`` **dernières**.

        Notes
        -----
        La constante **croît avec m** en incrémental et **décroît** en conservatif.
        Petit budget -> incrémental ; grand budget -> conservatif. Complémentaires,
        pas concurrentes.
        """
        terms = 0.5 * (jnp.log(self.alpha) + jnp.log(self.beta))
        if strategy == "incremental":
            return float(jnp.sum(terms[:n_sensors]))
        if strategy == "conservative":
            n_tail = self.alpha.shape[0] - n_sensors
            return float(jnp.sum(terms[:n_tail]))
        raise ValueError(f"strategy must be incremental|conservative, got {strategy}")

    def crossover(self) -> int:
        """Premier budget où la borne conservative devient plus serrée.

        Les constantes des équations (22) et (23) sont monotones en sens opposés.
        Cette méthode renvoie le premier ``m`` pour lequel la borne conservative est
        plus petite que la borne incrémentale ; si cela n'arrive pas, elle renvoie
        ``p``.
        """
        p = self.alpha.shape[0]
        gaps = [
            (m, self.suboptimality(m, "incremental") - self.suboptimality(m, "conservative"))
            for m in range(1, p)
        ]
        crossings = [m for m, d in gaps if d > 0]
        return crossings[0] if crossings else p

    @property
    def total_gap(self) -> float:
        """``gap(I_p) = ½ sum (ln alpha_i + ln beta_i)``.

        Oracle : doit égaler ``incremental_bounds(diagnostics, None).gap``, calculé
        par des ``slogdet`` qui ne diagonalisent rien.
        """
        return float(0.5 * jnp.sum(jnp.log(self.alpha) + jnp.log(self.beta)))

    @property
    def effective_rank(self) -> int:
        """Nombre minimal de contributions spectrales expliquant 90 % du gap total.

        Il s'agit d'un indicateur de concentration spectrale : une faible valeur
        signifie que le gap est dominé par un petit nombre de modes.
        """
        terms = 0.5 * (jnp.log(self.alpha) + jnp.log(self.beta))
        total = jnp.sum(terms)
        if total <= 0:
            return 0
        return int(jnp.searchsorted(jnp.cumsum(terms) / total, 0.9) + 1)


@jax.jit
@jaxtyped(typechecker=beartype)
def generalized_eigenvalues(
    A: Float[Array, "n_obs n_obs"],
    B: Float[Array, "n_obs n_obs"],
) -> Float[Array, " n_obs"]:
    r"""Valeurs propres de ``A u = alpha B u``, décroissantes. ``B`` SDP.

    Par Cholesky de ``B`` puis ``eigvalsh`` de ``L^{-1} A L^{-T}`` : JAX n'a pas de
    ``eigh`` généralisé, et former ``B^{-1}A`` détruirait la symétrie **et** le
    conditionnement.
    """
    L = jsp.linalg.cho_factor(B, lower=True)[0]
    L = jnp.tril(L)
    X = jsp.linalg.solve_triangular(L, A, lower=True)
    C = jsp.linalg.solve_triangular(L, X.T, lower=True).T
    return jnp.flip(jnp.linalg.eigvalsh(0.5 * (C + C.T)))


@jax.jit
@jaxtyped(typechecker=beartype)
def quasi_optimality(diagnostics: DiagnosticMatrices) -> QuasiOptimality:
    """Le spectre du gap -- Prop. 1.

    Notes
    -----
    ⚠️ ``eigvalsh`` sur des ``p x p`` denses : c'est un **diagnostic**, pas un chemin
    de production. Les bornes et le greedy n'en ont jamais besoin -- eux passent par
    Cholesky et compléments de Schur.
    """
    return QuasiOptimality(
        alpha=generalized_eigenvalues(diagnostics.Sigma_Y, diagnostics.Sigma_signal),
        beta=generalized_eigenvalues(diagnostics.Sigma_Y_given_theta, diagnostics.Sigma_noise),
    )
