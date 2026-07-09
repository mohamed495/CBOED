from cboed.priors import kernel
from cboed.priors.gaussian_priors import GaussianProcessPrior
import pytest 
import jax.numpy as jnp

def test_Gaussian():
    x1 = jnp.ones(3)
    x2 = jnp.zeros(3)

    gaussian = kernel.Gaussian(length_scale=1.0, sigma=1.0)

    expected = jnp.exp(-0.5) * jnp.ones((3, 3))

    assert jnp.allclose(gaussian(x1=x1, x2=x2), expected)


def test_matern12():
    x1 = jnp.ones(3)
    x2 = jnp.zeros(3)

    matern12 = kernel.Matern12(length_scale=1.0, sigma=1.0)
    expected = jnp.exp(-1.) * jnp.ones((3,3))
    print(f"expected = {expected}")
    print()
    assert jnp.allclose(expected, matern12(x1=x1, x2=x2))

def test_matern32():
    x1 = jnp.ones(3)
    x2 = jnp.zeros(3)

    matern32 = kernel.Matern32(length_scale=1.0, sigma=1.0)

    expected = (1 + jnp.sqrt(3)) * jnp.exp(-jnp.sqrt(3)) * jnp.ones((3,3))

    output = matern32(x1, x2)

    assert jnp.allclose(output, expected)

def test_matern52():
    x1 = jnp.ones(3)
    x2 = jnp.zeros(3)

    matern52 = kernel.Matern52(length_scale=1.0, sigma=1.0)

    d = jnp.sqrt(5.)                     # = sqrt(5) * r / ell, with r = 1, ell = 1
    expected = (1 + d + d**2 / 3) * jnp.exp(-d) * jnp.ones((3, 3))
    output = matern52(x1, x2)
    assert jnp.allclose(output, expected)

def test_RationalQuadratique():
    x1 = jnp.ones(3)
    x2 = jnp.zeros(3)

    RQ = kernel.RationalQuadratic(
        length_scale=1.,
        sigma=1.,
        alpha=1.
    )

    expected = (1 + 1./2) ** (-1) * jnp.ones((3, 3))
    output = RQ(x1=x1, x2=x2)

    assert jnp.allclose(output, expected)


def test_Periodic() : 
    x1 = jnp.ones(3)
    x2 = jnp.zeros(3)

    periodic = kernel.Periodic(
        length_scale=1.,
        sigma=1.,
        period=1.
    )

    expected = jnp.ones((3, 3))
    output = periodic(x1=x1, x2=x2)

    assert jnp.allclose(output, expected)

