"""Contrat de vraisemblance."""

from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from cboed.core.linear_operator import LinearizedOperator


class Likelihood(ABC):
    r"""``p(y | theta, design)``. Porte l'opérateur d'observation et le bruit.

    Le ``design`` entre **ici et nulle part ailleurs** : il sélectionne ce qui
    est observé, sans toucher au prior ni à la dynamique directe.

    Notes
    -----
    **Espace des observations.** Quand un ``design`` est fourni, ``y`` a
    ``n_sensors`` composantes (``m``) ; quand il vaut ``None``, on observe tout
    et ``m = p``. Dans les deux cas ``y`` vit en ``n_sensors`` -- aucun ``y`` de
    ce module n'a la dimension ``n_obs``. Seule ``Sigma_obs`` (``p x p``), non
    restreinte, la porte légitimement.
    """

    @abstractmethod
    def log_likelihood(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """``log p(y | theta, design)``."""
        ...

    @abstractmethod
    def jacobian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """``d(mean)/dtheta`` en ``(theta, design)``, matrix-free. Indépendant de y."""
        ...

    @abstractmethod
    def grad_log_likelihood(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_param"]:
        """``J^T Sigma_obs^{-1} (y - M(theta))``, en espace paramètre."""
        ...

    @abstractmethod
    def hessian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """Gauss-Newton ``-J^T Sigma_obs^{-1} J``, matrix-free et symétrique."""
        ...

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_sensors"]:
        """``y ~ p(. | theta, design)``."""
        ...

    # -- oracles denses : matérialisés depuis les opérateurs ---------------

    @partial(jax.jit, static_argnums=(0,))
    def jacobian(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_sensors n_param"]:
        """``J`` dense, ``(m, d)``.

        Le ``.T`` compte : ``vmap`` empile les images en **lignes**, on veut les
        colonnes.
        """
        op = self.jacobian_operator(theta, design)
        return jax.vmap(op.matvec)(jnp.eye(op.shape[1])).T

    @partial(jax.jit, static_argnums=(0,))
    def hessian(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""Gauss-Newton dense, ``(d, d)``. Oracle -- interdite en haute dim.

        Concrète : matérialise :meth:`hessian_operator`, un seul chemin
        numérique, aucune sous-classe ne la réimplémente de travers.

        **Ce n'est pas la vraie Hessienne** :

        .. math::
            \nabla^2 \log p = -J^T \Sigma^{-1} J
            + [\text{terme en } \partial^2 u/\partial\theta^2 -- \text{IGNORÉ}]

        À ``lambda=0`` le terme omis est nul et Gauss-Newton est exact. À
        ``lambda>0`` il diffère de l'autodiff : ce n'est **pas un bug**, c'est
        l'approximation de Laplace, et l'écart *est* la non-linéarité que les
        bornes quantifient. Ne pas écrire de test contre
        ``jax.hessian(log_likelihood)`` à ``lambda>0`` -- il échouera à raison.
        """
        op = self.hessian_operator(theta, design)
        H = jax.vmap(op.matvec)(jnp.eye(op.shape[1])).T
        return 0.5 * (H + H.T)
