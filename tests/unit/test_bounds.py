import jax.numpy as jnp

from cboed.bounds.base import DiagnosticMatrices
from cboed.bounds.bounds import conservative_bounds, incremental_bounds


def test_conservative_full_design_preserves_certified_full_interval():
    diagnostics = DiagnosticMatrices(
        Sigma_Y=jnp.diag(jnp.array([4.0, 3.0])),
        Sigma_Y_given_theta=jnp.eye(2),
        Sigma_signal=jnp.diag(jnp.array([2.0, 1.5])),
        Sigma_noise=jnp.eye(2),
        certified=True,
    )

    incremental = incremental_bounds(diagnostics)
    conservative = conservative_bounds(diagnostics)

    assert jnp.allclose(conservative.lower, incremental.lower)
    assert jnp.allclose(conservative.upper, incremental.upper)
