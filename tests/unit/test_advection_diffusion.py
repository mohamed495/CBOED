import jax.numpy as jnp

from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.core.linear_operator import LinearizedOperator


def test_properties():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,
    )

    assert model.dt == 1.0 / 5
    assert model.n == 4
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

    theta = jnp.arange(1.0, model.n + 1)
    direction = jnp.ones(model.n)

    J = model.jacobian_operator(theta)

    eps = 1e-3

    fd = (model(theta + eps * direction) - model(theta)) / eps

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

    v = jnp.arange(1.0, model.n + 1)
    w = jnp.arange(2.0, model.n + 2)

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

    A = (
        jnp.diag(jnp.ones(model.n))
        + jnp.diag(jnp.ones(model.n - 1), 1)
        - jnp.diag(jnp.ones(model.n - 1), -1)
    )

    B = (
        jnp.diag(jnp.ones(model.n))
        + jnp.diag(jnp.ones(model.n - 1), -1)
        - jnp.diag(jnp.ones(model.n - 1), 1)
    )

    M = jnp.linalg.inv(A) @ B

    theta = jnp.ones(model.n)

    expected = jnp.linalg.matrix_power(M, model.nt)
    computed = model.jacobian(theta=theta)

    assert jnp.allclose(expected, computed)


def test_jacobian_operator_is_exact_for_linear():
    """Modèle linéaire : J(d) = G(d) - G(0), sans erreur de troncature."""
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1.0, domain=[0, 1], nt=5, n=4)
    d = jnp.ones(model.n)
    J = model.jacobian_operator(jnp.zeros(model.n))
    # linéaire : G(d) = G(0) + J·d, et G(0)=0 (bords nuls) → J·d = G(d)
    assert jnp.allclose(J.matvec(d), model(d), atol=1e-12)


def test_solve():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=4.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,  # noeuds INTERIEURS -> U0 de taille n+2 = 6
    )

    # bords nuls (Dirichlet homogene) et flottants
    U0 = jnp.arange(model.n + 2, dtype=float).at[0].set(0.0).at[-1].set(0.0)

    # r = 0, c = 1  (dx = 1/(n+1) = 0.2)
    A = (
        jnp.diag(jnp.ones(model.n))
        + jnp.diag(jnp.ones(model.n - 1), 1)
        - jnp.diag(jnp.ones(model.n - 1), -1)
    )

    B = (
        jnp.diag(jnp.ones(model.n))
        + jnp.diag(jnp.ones(model.n - 1), -1)
        - jnp.diag(jnp.ones(model.n - 1), 1)
    )

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
        n=4,  # noeuds INTERIEURS -> U0 de taille n+2 = 6
    )

    # r = 0, c = 1  (dx = 1/(n+1) = 0.2)
    A = (
        jnp.diag(jnp.ones(model.n))
        + jnp.diag(jnp.ones(model.n - 1), 1)
        - jnp.diag(jnp.ones(model.n - 1), -1)
    )

    B = (
        jnp.diag(jnp.ones(model.n))
        + jnp.diag(jnp.ones(model.n - 1), -1)
        - jnp.diag(jnp.ones(model.n - 1), 1)
    )

    M = jnp.linalg.inv(A) @ B
    theta = jnp.ones(model.n)
    expected = jnp.linalg.matrix_power(M, model.nt) @ theta
    computed = model(theta=theta)

    assert jnp.allclose(computed, expected)


def test_jacobian_selects_rows():
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,
    )
    theta = jnp.ones(model.n)
    full = model.jacobian(theta)  # (n, n_param)
    design = jnp.array([0, 2])
    selected = model.jacobian(theta, design)  # (2, n_param)
    assert selected.shape == (2, model.n_parameters)
    assert jnp.allclose(selected, full[design])


def test_selection_extracts_correct_rows():
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1, domain=[0, 1], nt=5, n=4)
    theta = jnp.ones(model.n)
    full = model.jacobian(theta)
    for design in [jnp.array([0]), jnp.array([1, 3]), jnp.array([0, 1, 2, 3])]:
        sel = model.jacobian(theta, design)
        assert sel.shape == (len(design), model.n_parameters)
        assert jnp.allclose(sel, full[design])


def test_selection_full_equals_no_selection():
    """design = tous les indices ⟺ pas de sélection."""
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1, domain=[0, 1], nt=5, n=4)
    theta = jnp.ones(model.n)
    full = model.jacobian(theta)
    all_idx = model.jacobian(theta, jnp.arange(model.n))
    assert jnp.allclose(full, all_idx)


def test_selection_order_matters():
    """design=[2,0] sélectionne dans cet ordre, pas trié."""
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1, domain=[0, 1], nt=5, n=4)
    theta = jnp.ones(model.n)
    full = model.jacobian(theta)
    sel = model.jacobian(theta, jnp.array([2, 0]))
    assert jnp.allclose(sel[0], full[2])
    assert jnp.allclose(sel[1], full[0])


def test_selected_operator_adjoint():
    """⟨H∘G v, w⟩ = ⟨v, (H∘G)ᵀ w⟩ avec sélection."""
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1, domain=[0, 1], nt=5, n=4)
    theta = jnp.ones(model.n)
    design = jnp.array([0, 2])
    op = model.jacobian_operator(theta, design)

    v = jnp.arange(1.0, model.n + 1)  # espace param (n_param)
    w = jnp.arange(1.0, len(design) + 1)  # espace obs (n_obs)

    lhs = jnp.dot(op.matvec(v), w)
    rhs = jnp.dot(v, op.rmatvec(w))
    assert jnp.allclose(lhs, rhs)


def test_operator_matches_dense_with_selection():
    """jacobian_operator(design) et jacobian(design) cohérents."""
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1, domain=[0, 1], nt=5, n=4)
    theta = jnp.ones(model.n)
    design = jnp.array([1, 3])
    op = model.jacobian_operator(theta, design)
    dense = model.jacobian(theta, design)

    v = jnp.arange(1.0, model.n + 1)
    assert jnp.allclose(op.matvec(v), dense @ v)


def test_forward_applies_design():
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1, domain=[0, 1], nt=5, n=4)
    theta = jnp.ones(model.n)

    y_full = model(theta)  # ℝᵖ, p=4
    assert y_full.shape == (4,)

    design = jnp.array([0, 2])
    y_obs = model(theta, design)  # ℝᵐ, m=2
    assert y_obs.shape == (2,)
    assert jnp.allclose(y_obs, y_full[design])  # Yₘ = Wₘᵀ Y
