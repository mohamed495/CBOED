r"""Prop. 4 and §3.1 in the **nonlinear** case.

All other tests use `u(theta) = A theta`. There, the Jacobian is constant,
so `H(u) = 0` **exactly**, and the one quantity that distinguishes Prop. 4 from a
linear-Gaussian computation is never exercised. Same story for `sample_based`:
`Cov(u(theta))` there equals `A Sigma A^T`, a form that any centering bug would
also reproduce.

This file uses `u(theta) = theta**2`, the only nonlinear case where **every**
quantity has a closed form:

    Jac u(theta) = 2 diag(theta)
    L(u)  = E[Jac]^T = 2 diag(m)
    H(u)  = E[(Jac - E[Jac])^T S^{-1} (Jac - E[Jac])]
          = 4 E[diag(delta) S^{-1} diag(delta)]        delta = theta - m
          = 4 (Sigma_theta ⊙ S^{-1})                   Hadamard product
    Cov(u(theta))_ij = 2 Sigma_ij^2 + 4 m_i m_j Sigma_ij      (Isserlis)

Both identities are exact for Gaussian `theta` -- no Monte Carlo in the
oracles. Verified to 8e-4 (H) and 2.5e-3 (Isserlis) over 4e6 draws.

Warning: `m != 0` is **mandatory**: `L(u) = 2 diag(m)` would otherwise vanish, and
`Sigma_signal = Sigma_obs` degenerates. The production benchmark uses `mu = zeros`;
here it is `mu = ones`, and that is a deliberate choice, not an oversight.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest  # type: ignore

from cboed.bounds.diagnostics.gradient_based import (
    expected_jacobian_moments,
    fisher_information_prior,
    gradient_diagnostics_standard,
)
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Matern32

D = 5
N_SAMPLES = 200_000


def u_square(theta):
    """`u(theta) = theta**2`. Jac = 2 diag(theta) -- not constant."""
    return theta**2


@pytest.fixture(scope="module")
def setup():
    """Gaussian prior with **nonzero** mean and anisotropic Sigma_obs.

    `Sigma_obs = I` would mask any transposition error in `H(u)`: the Hadamard
    product with the identity is diagonal.
    """
    gp = GaussianProcess(
        kernel=Matern32(length_scale=0.4, sigma=1.0),
        mu=jnp.ones(D),
        domain=(0.0, 1.0),
    )
    prior = GaussianPrior(prior=gp)
    Sigma_obs = jnp.diag(jnp.arange(1.0, D + 1.0)) * 0.5
    return prior, Sigma_obs


@pytest.mark.slow
def test_H_matches_hadamard_formula(setup):
    """`H(u) = 4 (Sigma_theta ⊙ Sigma_obs^{-1})` -- the nonlinear branch.

    This is THE missing test: `H(u)` is the one quantity that distinguishes
    Prop. 4 from an LG computation, and it is zero everywhere else in the
    suite. A wrong factor, sign, or transposition would go unnoticed here --
    `Sigma_signal` would stay SPD and plausible.
    """
    prior, Sigma_obs = setup
    thetas = prior.sample(jr.key(0), N_SAMPLES)
    _, H = expected_jacobian_moments(u_square, thetas, Sigma_obs)

    expected = 4.0 * prior.Sigma() * jnp.linalg.inv(Sigma_obs)  # Hadamard
    rel = jnp.linalg.norm(H - expected) / jnp.linalg.norm(expected)
    print(f"\nH(u): relative error = {rel:.3e}")
    assert rel < 0.02


@pytest.mark.slow
def test_L_matches_analytic(setup):
    """`L(u) = E[Jac]^T = 2 diag(m)`. Exact from a single sample if m is known."""
    prior, Sigma_obs = setup
    thetas = prior.sample(jr.key(0), N_SAMPLES)
    L, _ = expected_jacobian_moments(u_square, thetas, Sigma_obs)

    expected = 2.0 * jnp.diag(prior.mu)
    rel = jnp.linalg.norm(L - expected) / jnp.linalg.norm(expected)
    print(f"L(u): relative error = {rel:.3e}")
    assert rel < 0.02


@pytest.mark.slow
def test_H_is_zero_for_linear_u(setup):
    """The control: `H(u) = 0` **exactly** when the Jacobian is constant.

    This is what the two-pass computation guarantees. The "computational"
    formula `E[J^T S^{-1} J] - Jbar^T S^{-1} Jbar` would here return rounding
    noise instead of zero -- two large, equal numbers being subtracted.
    """
    prior, Sigma_obs = setup
    A = jr.normal(jr.key(1), (D, D))
    thetas = prior.sample(jr.key(0), 1_000)
    _, H = expected_jacobian_moments(lambda th: A @ th, thetas, Sigma_obs)
    assert jnp.allclose(H, 0.0, atol=1e-10)


@pytest.mark.slow
def test_Sigma_Y_matches_isserlis(setup):
    """§3.1 in the nonlinear case: `Cov(theta**2)_ij = 2 Sigma_ij^2 + 4 m_i m_j Sigma_ij`.

    The paired estimator (26) has never been tested on anything but a linear
    model, where `Cov(u) = A Sigma A^T` -- a form that any centering bug would
    also reproduce.
    """
    prior, Sigma_obs = setup
    Sigma_Y = sample_Sigma_Y(u_square, prior, Sigma_obs, jr.key(2), N_SAMPLES)

    S, m = prior.Sigma(), prior.mu
    expected = Sigma_obs + 2.0 * S**2 + 4.0 * jnp.outer(m, m) * S
    rel = jnp.linalg.norm(Sigma_Y - expected) / jnp.linalg.norm(expected)
    print(f"Sigma_Y: relative error = {rel:.3e}")
    assert rel < 0.05


@pytest.mark.slow
def test_Sigma_signal_matches_analytic(setup):
    """The full assembly against its closed form -- nonlinear.

    Sigma_signal = Sigma_obs + L^T (H + I_theta)^{-1} L
                 = Sigma_obs + 4 diag(m) (4(Sigma ⊙ S^{-1}) + Sigma^{-1})^{-1} diag(m)
    """
    prior, Sigma_obs = setup
    Sigma_signal, _ = gradient_diagnostics_standard(
        u_square, prior, Sigma_obs, jr.key(3), N_SAMPLES
    )

    S, m = prior.Sigma(), prior.mu
    H = 4.0 * S * jnp.linalg.inv(Sigma_obs)
    I_theta = fisher_information_prior(prior)
    L = 2.0 * jnp.diag(m)
    expected = Sigma_obs + L.T @ jnp.linalg.solve(H + I_theta, L)

    rel = jnp.linalg.norm(Sigma_signal - expected) / jnp.linalg.norm(expected)
    print(f"Sigma_signal: relative error = {rel:.3e}")
    assert rel < 0.03


@pytest.mark.slow
def test_signal_preceq_Sigma_Y_nonlinear(setup):
    """Prop. 1 in the **nonlinear** case: `Sigma_Y ⪰ Sigma_signal`, hence `alpha_i >= 1`.

    Via Cramer-Rao: `Sigma_signal^{-1} ⪰ I_Y` gives `Sigma_signal ⪯ I_Y^{-1} ⪯ Cov(Y)`.
    Tested here with **both closed forms**, without any MC error.
    """
    prior, Sigma_obs = setup
    S, m = prior.Sigma(), prior.mu

    Sigma_Y = Sigma_obs + 2.0 * S**2 + 4.0 * jnp.outer(m, m) * S
    H = 4.0 * S * jnp.linalg.inv(Sigma_obs)
    L = 2.0 * jnp.diag(m)
    Sigma_signal = Sigma_obs + L.T @ jnp.linalg.solve(H + fisher_information_prior(prior), L)

    lo = float(jnp.min(jnp.linalg.eigvalsh(Sigma_Y - Sigma_signal)))
    print(f"lambda_min(Sigma_Y - Sigma_signal) = {lo:.3e}")
    assert lo > -1e-8, "Loewner order violated in the nonlinear case"
