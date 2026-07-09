from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.core.linear_operator import LinearizedOperator
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

def test_properties():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,
    )
    
    assert model.dt == 1./5
    assert model.n ==4
    assert model.velocity == 2.0
    assert model.diffusivity == 0.0
    assert model.T == 1

def test_jacobian_operator():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,
    )

    theta = jnp.arange(1., model.n + 1)
    direction = jnp.ones(model.n)

    J = model.jacobian_operator(theta)

    eps = 1e-3

    fd = (
        model(theta + eps * direction)
        - model(theta)
    ) / eps

    assert jnp.allclose(
        J.matvec(direction),
        fd,
        rtol=1e-3,
        atol=1e-5,
    )

def test_jacobian_returns_linearized_operator():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=11,
        n=11,
    )

    theta = jnp.zeros(model.n)

    J = model.jacobian_operator(theta)

    assert isinstance(J, LinearizedOperator)
    assert J.shape == (model.n, model.n)

def test_jacobian_is_linearize():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=11,
        n=11,
    )

    theta = jnp.zeros(model.n)
    v = jnp.arange(model.n, dtype=float)

    J1 = model.jacobian_operator(theta)
    J2 = model.linearize(theta)

    assert jnp.allclose(J1.matvec(v), J2.matvec(v))
    assert jnp.allclose(J1.rmatvec(v), J2.rmatvec(v))
    assert J1.shape == J2.shape

def test_advection_adjoint():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,
    )

    theta = jnp.ones(model.n)
    J = model.jacobian_operator(theta)

    v = jnp.arange(1., model.n + 1)
    w = jnp.arange(2., model.n + 2)

    lhs = jnp.dot(J.matvec(v), w)
    rhs = jnp.dot(v, J.rmatvec(w))

    assert jnp.allclose(lhs, rhs)

def test_jacobian_matrix():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=4.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,
    )

    A = (jnp.diag(jnp.ones(model.n)) +
         jnp.diag(jnp.ones(model.n-1), 1) -
         jnp.diag(jnp.ones(model.n-1), -1))

    B = (jnp.diag(jnp.ones(model.n)) +
         jnp.diag(jnp.ones(model.n-1), -1) -
         jnp.diag(jnp.ones(model.n-1), 1))

    M = jnp.linalg.inv(A) @ B

    theta = jnp.ones(model.n)

    expected = jnp.linalg.matrix_power(M, model.nt)
    computed = model.jacobian(theta=theta)

    assert jnp.allclose(expected, computed)

def test_solve():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=4.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,          # noeuds INTERIEURS -> U0 de taille n+2 = 6
    )

    # bords nuls (Dirichlet homogene) et flottants
    U0 = jnp.arange(model.n + 2, dtype=float).at[0].set(0.0).at[-1].set(0.0)

    # r = 0, c = 1  (dx = 1/(n+1) = 0.2)
    A = (jnp.diag(jnp.ones(model.n))
         + jnp.diag(jnp.ones(model.n - 1), 1)
         - jnp.diag(jnp.ones(model.n - 1), -1))

    B = (jnp.diag(jnp.ones(model.n))
         + jnp.diag(jnp.ones(model.n - 1), -1)
         - jnp.diag(jnp.ones(model.n - 1), 1))

    M = jnp.linalg.inv(A) @ B

    expected_int = jnp.linalg.matrix_power(M, model.nt) @ U0[1:-1]
    expected = U0.at[1:-1].set(expected_int)

    computed = model.solve(U0=U0)

    assert jnp.allclose(computed, expected)

def test_call():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=4.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,          # noeuds INTERIEURS -> U0 de taille n+2 = 6
    )

    # r = 0, c = 1  (dx = 1/(n+1) = 0.2)
    A = (jnp.diag(jnp.ones(model.n))
         + jnp.diag(jnp.ones(model.n - 1), 1)
         - jnp.diag(jnp.ones(model.n - 1), -1))

    B = (jnp.diag(jnp.ones(model.n))
         + jnp.diag(jnp.ones(model.n - 1), -1)
         - jnp.diag(jnp.ones(model.n - 1), 1))

    M = jnp.linalg.inv(A) @ B
    theta = jnp.ones(model.n)
    expected = jnp.linalg.matrix_power(M, model.nt) @ theta
    computed = model(theta=theta)

    assert jnp.allclose(computed, expected)
   