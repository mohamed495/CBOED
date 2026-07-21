# estimators/base.py
from abc import ABC, abstractmethod

import jax
from jaxtyping import Array, Float, Int


def chunked_vmap(f, *xs, chunk_size: int | None = None):
    """``vmap(f)(*xs)``, ou ``lax.map`` par lots de ``chunk_size`` si fourni.

    Les estimateurs NMC vectorisent une boucle externe (``n_outer``) dont
    chaque élément lance déjà une boucle interne vectorisée (``n_inner``) --
    XLA fusionne les deux en un batch effectif ``n_outer x n_inner``, et à
    l'échelle réelle (champ complet, gros ``n_outer``/``n_inner``) ce batch
    dépasse largement la mémoire GPU disponible. ``chunk_size`` borne la
    mémoire de crête à ``chunk_size x n_inner`` en traitant l'axe externe par
    lots séquentiels (``lax.map``) plutôt qu'en un seul ``vmap`` -- plus lent,
    mais indispensable dès que ``n_outer x n_inner`` ne tient plus en mémoire.
    ``None`` (défaut) : un seul ``vmap``, comme avant -- pas de surcoût aux
    petites échelles (tests, dev).
    """
    if chunk_size is None:
        return jax.vmap(f)(*xs)
    if len(xs) == 1:
        return jax.lax.map(f, xs[0], batch_size=chunk_size)
    return jax.lax.map(lambda args: f(*args), xs, batch_size=chunk_size)


class EIGEstimator(ABC):
    """Estimateur d'EIG. Les sous-classes décident *comment* l'approximer."""

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def inference(self):
        return self._hyperparameters["inference"]

    @abstractmethod
    def estimate(
        self,
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]: ...
