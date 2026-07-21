from typing import NamedTuple

import jax.numpy as jnp
import pytest  # type: ignore

from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.core.burgers import Burgers


class Setup(NamedTuple):
    diffusion_model: AdvectionDiffusion
    burgers_model: Burgers
    burgers_diffusion_model: Burgers


@pytest.fixture
def setup() -> Setup:
    diffusion_model = AdvectionDiffusion(
        diffusivity=1.0,
        velocity=0.0,
        T=1.0,
        domain=[0, 1],
        nt=5,
        n=4,
    )

    burgers_model = Burgers(diffusivity=1.0, lambda_=1.0, T=1.0, domain=[0, 1], nt=5, n=4)

    burgers_diffusion_model = Burgers(diffusivity=1.0, lambda_=0.0, T=1.0, domain=[0, 1], nt=5, n=4)
    return Setup(
        burgers_diffusion_model=burgers_diffusion_model,
        burgers_model=burgers_model,
        diffusion_model=diffusion_model,
    )


def test_properties(setup: Setup):
    model = setup.burgers_model

    assert model.dt == 1.0 / 5
    assert model.n == 4
    assert model.lambda_ == 1.0
    assert model.diffusivity == 1.0
    assert model.T == 1.0


def test_burgers_lambda_zero_is_linear(setup: Setup):
    """At λ=0, Burgers is linear: Jacobian independent of theta."""
    model_burgers = setup.burgers_diffusion_model
    J1 = model_burgers.jacobian(jnp.zeros(model_burgers.n))
    J2 = model_burgers.jacobian(jnp.ones(model_burgers.n) * 5.0)
    assert jnp.allclose(J1, J2, atol=1e-10)  # constant Jacobian


def test_burgers_nonlinear_jacobian_varies(setup: Setup):
    """At λ>0, the Jacobian depends on theta."""
    model = setup.burgers_model
    J1 = model.jacobian(jnp.zeros(model.n))
    J2 = model.jacobian(jnp.ones(model.n) * 5.0)
    assert not jnp.allclose(J1, J2)  # Jacobian varies


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_burgers_adjoint_holds(lam):
    model = Burgers(diffusivity=1.0, lambda_=lam, T=1.0, domain=[0, 1], nt=5, n=4)
    theta = jnp.ones(model.n)
    J = model.jacobian_operator(theta)
    v = jnp.arange(1.0, model.n + 1)
    w = jnp.arange(2.0, model.n + 2)
    assert jnp.allclose(jnp.dot(J.matvec(v), w), jnp.dot(v, J.rmatvec(w)))


def test_burgers_lambda_zero_matches_analytic_diffusion(setup):
    """λ=0: exact value against (A⁻¹B)^nt."""
    model = setup.burgers_diffusion_model
    r = model.diffusivity * model.dt / (2 * model.dx**2)
    L = -2 * jnp.eye(4) + jnp.eye(4, k=1) + jnp.eye(4, k=-1)
    A = jnp.eye(4) - r * L
    B = jnp.eye(4) + r * L
    M = jnp.linalg.inv(A) @ B

    theta = jnp.ones(4)
    expected = jnp.linalg.matrix_power(M, model.nt) @ theta
    assert jnp.allclose(model(theta), expected, atol=1e-10)


def test_burgers_selects_rows(setup):
    model = setup.burgers_model
    theta = jnp.ones(model.n)
    full = model.jacobian(theta)
    design = jnp.array([0, 2])
    selected = model.jacobian(theta, design)
    assert selected.shape == (2, model.n_parameters)
    assert jnp.allclose(selected, full[design])
