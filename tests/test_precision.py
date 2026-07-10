# tests/unit/test_precision.py
import jax.numpy as jnp


def test_x64_is_enabled() -> None:
    assert jnp.zeros(1).dtype == jnp.float64
