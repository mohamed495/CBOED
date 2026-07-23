"""Prop. 4 -- linear case and assembly.

Two levels, now that moments are separated from assembly:

* **assembly** -- `L` and `H` set by hand, no Jacobian involved. Two paths
  (`assemble`, `assemble_misfit`) toward the same matrix: each an oracle for the other.
* **end to end** -- `u` linear, `H(u) = 0`, everything has a closed form.

The nonlinear case, where `H(u) != 0`, is in `test_nonlinear_diagnostics.py`.
"""

import inspect

import jax.numpy as jnp
import jax.random as jr
import pytest

from cboed.bounds.diagnostics.gradient_based import (
    assemble,
    assemble_misfit,
    expected_jacobian_moments,
    fisher_information_prior,
    fisher_information_prior_mc,
    gradient_diagnostics,
    gradient_diagnostics_standard,
    psd_sqrt,
)
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Matern32

Q, P, D = 6, 6, 3


@pytest.fixture
def prior_eta():
    gp = GaussianProcess(Matern32(length_scale=0.3, sigma=1.0), jnp.zeros(Q))
    return GaussianPrior(prior=gp)


@pytest.fixture
def linear_setup():
    """`u` linear (constant Jacobian) and `h` = selection of the first D."""
    A = jr.normal(jr.key(3), (P, Q))
    B = jnp.eye(D, Q)
    return (lambda eta: A @ eta), (lambda eta: B @ eta), A, B


@pytest.fixture
def moments():
    """Synthetic `L` and `H` -- neither Jacobian nor forward model.

    `H` is a nontrivial SPD matrix: `H = 0` (the linear case) would not
    exercise anything about the assembly.
    """
    L = jr.normal(jr.key(7), (Q, P))
    X = jr.normal(jr.key(8), (Q, Q))
    return L, X @ X.T


# =============================================================================
# Assembly -- without Jacobians
# =============================================================================


def test_misfit_matches_direct(prior_eta, moments):
    """`assemble` and `assemble_misfit`: two paths, one matrix.

    The identity: `(H + Sigma^{-1})^{-1} = Ssq (I + Ssq H Ssq)^{-1} Ssq`.

    The NumPy prototype did `solve(A_mis, l)` while `A_mis` **was already the
    inverse** -- a 248% relative error, with a result that stayed SPD and thus
    silent. This test would have caught it in three lines.
    """
    L, H = moments
    direct = assemble(L, H + fisher_information_prior(prior_eta), jnp.eye(P))
    misfit = assemble_misfit(L, H, prior_eta.Sigma(), jnp.eye(P))
    rel = jnp.linalg.norm(direct - misfit) / jnp.linalg.norm(direct)
    print(f"\nassemble vs assemble_misfit: relative error = {rel:.3e}")
    assert rel < 1e-8


def test_misfit_matches_direct_with_extra(prior_eta, moments):
    """Same with `J(h)`: the identity only concerns `Sigma_eta^{-1}`."""
    L, H = moments
    J_h = jnp.diag(jnp.arange(1.0, Q + 1.0))
    direct = assemble(L, H + fisher_information_prior(prior_eta) + J_h, jnp.eye(P))
    misfit = assemble_misfit(L, H, prior_eta.Sigma(), jnp.eye(P), extra=J_h)
    assert jnp.allclose(direct, misfit, rtol=1e-8)


def test_misfit_never_forms_prior_precision():
    """Only uses `Sigma_eta^{1/2}` -- what survives when the precision no longer does."""
    src = inspect.getsource(assemble_misfit)
    assert "inv(" not in src
    assert "prior_precision" not in src


def test_assemble_is_symmetric(moments):
    L, H = moments
    out = assemble(L, H + jnp.eye(Q), jnp.eye(P))
    assert jnp.allclose(out, out.T, atol=1e-12)


def test_assemble_reduces_to_Sigma_obs_when_L_vanishes(moments):
    """`L = 0` -> no information -> `Sigma_signal = Sigma_obs`."""
    _, H = moments
    Sigma_obs = jnp.diag(jnp.arange(1.0, P + 1.0))
    assert jnp.allclose(assemble(jnp.zeros((Q, P)), H + jnp.eye(Q), Sigma_obs), Sigma_obs)


def test_psd_sqrt_handles_singular():
    """`Sigma_xi = 0` is a nominal case: `cholesky` would return `nan` there."""
    assert jnp.allclose(psd_sqrt(jnp.zeros((4, 4))), 0.0)


def test_psd_sqrt_squares_back(prior_eta):
    R = psd_sqrt(prior_eta.Sigma())
    assert jnp.allclose(R @ R, prior_eta.Sigma(), atol=1e-8)


# =============================================================================
# Prior Fisher information
# =============================================================================


def test_fisher_exact_matches_monte_carlo(prior_eta):
    """Two paths toward `I_eta` -- and proof that the prior is indeed Gaussian."""
    exact = fisher_information_prior(prior_eta)
    mc = fisher_information_prior_mc(prior_eta, jr.key(0), 200_000)
    rel = jnp.linalg.norm(exact - mc) / jnp.linalg.norm(exact)
    print(f"I_eta exact vs MC: relative error = {rel:.3e}")
    assert rel < 0.05


def test_fisher_exact_is_prior_precision(prior_eta):
    """`I_eta == Gamma_eta^{-1}`: oracle via `Sigma() @ I_eta == I`."""
    assert jnp.allclose(
        prior_eta.Sigma() @ fisher_information_prior(prior_eta), jnp.eye(Q), atol=1e-8
    )


# =============================================================================
# Moments -- linear case
# =============================================================================


def test_H_is_zero_for_constant_jacobian(prior_eta, linear_setup):
    """`H(u) = 0` **exactly** -- what both passes guarantee."""
    u, _, _, _ = linear_setup
    etas = prior_eta.sample(jr.key(0), 64)
    _, H = expected_jacobian_moments(u, etas, jnp.eye(P))
    assert jnp.allclose(H, 0.0, atol=1e-10)


def test_L_is_the_transposed_jacobian(prior_eta, linear_setup):
    """`L(u) = E[Jac]^T = A^T`. The `.T` matters."""
    u, _, A, _ = linear_setup
    etas = prior_eta.sample(jr.key(0), 64)
    L, _ = expected_jacobian_moments(u, etas, jnp.eye(P))
    assert jnp.allclose(L, A.T, atol=1e-10)


# =============================================================================
# End to end -- linear case
# =============================================================================


def test_lg_collapse(prior_eta, linear_setup):
    """`H(u) = 0` -> `Sigma_signal = Sigma_obs + A Sigma_eta A^T = Sigma_Y`.

    The collapse from Rem. 2.2 -- and what the "computational" formula for
    `H(u)` would destroy through cancellation.
    """
    u, _, A, _ = linear_setup
    Sigma_obs = jnp.diag(jnp.arange(1.0, P + 1.0)) * 0.01
    Sigma_signal, _ = gradient_diagnostics_standard(u, prior_eta, Sigma_obs, jr.key(1), 64)
    assert jnp.allclose(Sigma_signal, Sigma_obs + A @ prior_eta.Sigma() @ A.T, atol=1e-8)


def test_standard_noise_is_exactly_Sigma_obs(prior_eta, linear_setup):
    """Prop. 2: posited, not approximated. `array_equal`, not `allclose`."""
    u, _, _, _ = linear_setup
    Sigma_obs = jnp.eye(P) * 0.01
    _, Sigma_noise = gradient_diagnostics_standard(u, prior_eta, Sigma_obs, jr.key(0), 64)
    assert jnp.array_equal(Sigma_noise, Sigma_obs)


def test_signal_independent_of_Sigma_xi(prior_eta, linear_setup):
    """`Sigma_signal` does not contain `J(h)`: `xi` must not move it."""
    u, h, _, _ = linear_setup
    Sigma_obs = jnp.eye(P) * 0.01
    kw = {"key": jr.key(2), "n_samples": 64}
    s1, _ = gradient_diagnostics(u, h, prior_eta, Sigma_obs, jnp.eye(D) * 1e-2, **kw)
    s2, _ = gradient_diagnostics(u, h, prior_eta, Sigma_obs, jnp.eye(D) * 1e2, **kw)
    assert jnp.allclose(s1, s2, atol=1e-10)


def test_noise_preceq_signal(prior_eta, linear_setup):
    """`J(h)` PSD -> `(H+I+J)^{-1} ⪯ (H+I)^{-1}` -> `Sigma_noise ⪯ Sigma_signal`.

    The gap **is** `J(h)`: that's `gap_h`.
    """
    u, h, _, _ = linear_setup
    Sigma_signal, Sigma_noise = gradient_diagnostics(
        u, h, prior_eta, jnp.eye(P) * 0.01, jnp.eye(D) * 1e-2, jr.key(4), 64
    )
    assert jnp.min(jnp.linalg.eigvalsh(Sigma_signal - Sigma_noise)) > -1e-8


def test_diagnostics_are_spd(prior_eta, linear_setup):
    u, h, _, _ = linear_setup
    for M in gradient_diagnostics(u, h, prior_eta, jnp.eye(P) * 0.01, jnp.eye(D), jr.key(6), 64):
        assert jnp.min(jnp.linalg.eigvalsh(M)) > 0


@pytest.mark.parametrize("scale", [1e0, 1e-2, 1e-4, 1e-6])
def test_noise_tends_to_Sigma_obs_as_Sigma_xi_vanishes(prior_eta, linear_setup, scale):
    """The `Sigma_xi -> 0` limit is singular: the gap decreases as `O(Sigma_xi)`.

    This is why `gradient_diagnostics_standard` exists.
    """
    u, h, _, _ = linear_setup
    Sigma_obs = jnp.eye(P) * 0.01
    _, Sigma_noise = gradient_diagnostics(
        u, h, prior_eta, Sigma_obs, jnp.eye(D) * scale, jr.key(5), 64
    )
    gap = jnp.linalg.norm(Sigma_noise - Sigma_obs) / jnp.linalg.norm(Sigma_obs)
    print(f"Sigma_xi = {scale:.0e} -> relative gap = {gap:.3e}")
    assert jnp.all(jnp.isfinite(Sigma_noise))
    assert jnp.min(jnp.linalg.eigvalsh(Sigma_noise - Sigma_obs)) > -1e-8
