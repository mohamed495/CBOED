"""Contrat d'inférence : prior + vraisemblance -> postérieure sur theta."""

from abc import ABC, abstractmethod

from jax import Array
from jaxtyping import Float, Int


class InferenceModel(ABC):
    r"""Prior + vraisemblance -> postérieure sur ``theta``.

    Décide **comment on obtient la postérieure** (forme fermée, linéarisation,
    propagation vers une QoI...). *Comment on estime l'EIG* relève de
    ``estimators/`` -- ce n'est pas ce contrat.

    Le contrat est en **actions** : ``Gamma_post`` (d x d) n'est pas
    matérialisable en haute dimension. Tout ce qui figure ici est
    **``y``-indépendant** -- c'est exactement ce que consomment les critères, et
    c'est pourquoi l'EIG se calcule avant toute observation.

    Notes
    -----
    ``theta`` est le **point de linéarisation**. Ignoré dans le cas
    linéaire-gaussien (jacobienne constante), il reste dans la signature : c'est
    l'interface qui survit au non-linéaire.
    """

    @abstractmethod
    def log_det_posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """``log det Gamma_post^{-1}`` au point ``theta``.

        Via Cholesky (``2 sum log diag L``) -- sans inverser, sans diagonaliser.
        """
        ...

    @abstractmethod
    def log_det_prior_precision(self) -> Float[Array, ""]:
        """``log det Gamma_prior^{-1}``.

        Aucun argument : ne dépend ni de ``theta`` ni du ``design``.
        """
        ...

    @abstractmethod
    def posterior_cov_matmul(
        self,
        B: Float[Array, "n_param k"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param k"]:
        """``Gamma_post @ B``, sans matérialiser ``Gamma_post``.

        Primitive unique : la moyenne postérieure (``B = grad``), la
        propagation QoI (``B = H^T``), la trace A-optimale et l'oracle dense
        (``B = I``) en dérivent tous.
        """
        ...
