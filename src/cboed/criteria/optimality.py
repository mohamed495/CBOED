import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from cboed.criteria.base import Criterion


class EIG(Criterion):
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        return 0.5 * (
            self.inference.log_det_posterior_precision(theta, design)
            - self.inference.log_det_prior_precision()
        )


class DOptimal(Criterion):
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        return self.inference.log_det_posterior_precision(theta, design)


class AOptimal(Criterion):
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        eigs = self._posterior_precision_eigvals(theta, design)
        return -jnp.sum(1.0 / eigs)
