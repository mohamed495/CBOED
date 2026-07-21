"""§3.2 -- diagnostics via affine+network denoiser (`denoisers.py`, `approximation_based.py`).

Same protocol as `test_approximation_diagnostics.py` (pure affine denoiser,
`linear_approximation_based.py`), extended to the composed denoiser
`affine + network` (`ResidualDenoiser`). Two sets of toy models:

- `linear_model`: `u` is linear -- the network has nothing to learn (target
  residual is zero). Comparisons are on the raw residual `R`, never on the
  assembled `Sigma_signal`/`Sigma_noise`: `test_approximation_diagnostics.py`
  shows that this inversion is nearly singular at `lambda = 0`, which amplifies
  estimation noise far beyond what these tests intend to check.
- `nonlinear_model`: `u` is nonlinear -- here the network has a reason to
  exist: `R_g` (affine + network) must be strictly smaller than `R_f` (affine
  alone), otherwise `ResidualDenoiser` adds nothing.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from cboed.bounds.diagnostics.approximation_based import denoiser_residual
from cboed.bounds.diagnostics.denoisers import AffineDenoiser, ResidualDenoiser
from cboed.bounds.diagnostics.linear_approximation_based import (
    affine_denoiser as affine_denoiser_standalone,
)
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Matern32

Q, P = 6, 6
N_SAMPLES = 200_000
NET_STEPS = 300  # small training budget: verification, not fine convergence


@pytest.fixture
def prior():
    gp = GaussianProcess(Matern32(length_scale=0.3, sigma=1.0), jnp.zeros(Q))
    return GaussianPrior(prior=gp)


@pytest.fixture
def linear_model():
    """`u(theta) = A theta` -- linear, the network's target residual is zero."""
    A = jr.normal(jr.key(3), (P, Q))
    return lambda theta: A @ theta, A


@pytest.fixture
def nonlinear_model():
    """`u(theta) = A theta + c * sin(B theta)` -- nonlinear, nonzero residual.

    The `sin` term isn't affine: `E[u|Y]` deviates from the affine, and it is
    this deviation the network must capture.
    """
    k_A, k_B = jr.split(jr.key(7))
    A = jr.normal(k_A, (P, Q))
    B = jr.normal(k_B, (P, Q)) * 0.5
    c = 0.3
    return lambda theta: A @ theta + c * jnp.sin(B @ theta)


def _paired(u, prior, Sigma_obs, key, n):
    """`(u(eta), Y = u(eta) + eps)` -- the same pairs as sample_Sigma_Y."""
    k_eta, k_eps = jr.split(key)
    eta = prior.sample(k_eta, n)
    u_vals = jax.vmap(u)(eta)
    L = jnp.linalg.cholesky(Sigma_obs)
    Y = u_vals + jr.normal(k_eps, u_vals.shape) @ L.T
    return u_vals, Y, eta


# ─────────────────────────────────────────────────────────
# AffineDenoiser (equinox) == affine_denoiser (standalone closed form)
# ─────────────────────────────────────────────────────────


def test_affine_denoiser_module_matches_standalone_function(prior, linear_model):
    """The two affine implementations (linear_approximation_based vs denoisers)
    must produce the same `A`, `b` -- same formula, two code paths.
    """
    u, _ = linear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    u_vals, Y, _ = _paired(u, prior, Sigma_obs, jr.key(1), N_SAMPLES)

    A_std, b_std = affine_denoiser_standalone(u_vals, Y)
    fitted = AffineDenoiser.fit(u_vals, Y)

    assert jnp.allclose(fitted.A, A_std, atol=1e-6)
    assert jnp.allclose(fitted.b, b_std, atol=1e-6)


# ─────────────────────────────────────────────────────────
# Linear case: the network has nothing to learn
# ─────────────────────────────────────────────────────────


def test_residual_denoiser_matches_affine_residual_when_linear(prior, linear_model):
    """At `lambda = 0`, the network has nothing to learn: the residual `R`
    (affine+net) must stay close to that of the affine alone.

    Comparison on the **raw** `R` (before the Prop. 3 inversion), not on the
    assembled `Sigma_signal`: the latter is nearly singular at this exact point
    (the affine is nearly exact, cf. test_approximation_diagnostics.py), and an
    infinitesimal difference on `R` gets amplified uncontrollably there -- that
    is not the property this test intends to check.
    """
    u, _ = linear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    u_vals, Y, _ = _paired(u, prior, Sigma_obs, jr.key(2), N_SAMPLES)

    A_aff, b_aff = affine_denoiser_standalone(u_vals, Y)
    R_affine = denoiser_residual_standalone(u_vals, Y, A_aff, b_aff)

    denoiser = ResidualDenoiser.fit(u_vals, Y, jr.key(0), steps=NET_STEPS)
    R_with_net = denoiser_residual(denoiser, u_vals, Y)

    rel = jnp.linalg.norm(R_with_net - R_affine) / jnp.linalg.norm(R_affine)
    print(f"\n||R(affine+net) - R(affine)|| / ||.|| = {rel:.3e}")
    assert rel < 0.1


def test_noise_preceq_signal_with_residual_denoiser(prior, linear_model):
    """`g` sees theta in addition to Y: `R_g <= R_f`.

    Tested on the raw residuals (stable), not on the assembled
    `Sigma_signal`/`Sigma_noise` -- same reason as above: the Prop. 3 inversion
    is nearly singular at `lambda = 0` and amplifies the estimation noise on `R`
    to the point of breaking the expected order on the assembled matrix (cf.
    test_approximation_diagnostics.py::test_noise_preceq_signal). The order on
    `R` itself, upstream of the inversion, is the property that can be checked.
    """
    u, _ = linear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    u_vals, Y, eta = _paired(u, prior, Sigma_obs, jr.key(4), N_SAMPLES)

    denoiser_f = ResidualDenoiser.fit(u_vals, Y, jr.key(2), steps=NET_STEPS)
    features_g = jnp.concatenate([Y, eta], axis=1)
    denoiser_g = ResidualDenoiser.fit(u_vals, features_g, jr.key(3), steps=NET_STEPS)

    R_f = denoiser_residual(denoiser_f, u_vals, Y)
    R_g = denoiser_residual(denoiser_g, u_vals, features_g)
    assert jnp.min(jnp.linalg.eigvalsh(R_f - R_g)) > -1e-3


# ─────────────────────────────────────────────────────────
# Nonlinear case: the network must reduce the residual below the affine alone
# ─────────────────────────────────────────────────────────


def test_residual_denoiser_beats_affine_when_nonlinear(prior, nonlinear_model):
    """The network's reason to exist: on a nonlinear `u`, `affine + network`
    must leave a smaller residual than the affine alone -- otherwise the
    network learns nothing useful.
    """
    u = nonlinear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    u_vals, Y, _ = _paired(u, prior, Sigma_obs, jr.key(5), N_SAMPLES)

    A_aff, b_aff = affine_denoiser_standalone(u_vals, Y)
    R_affine = denoiser_residual_standalone(u_vals, Y, A_aff, b_aff)

    denoiser = ResidualDenoiser.fit(u_vals, Y, jr.key(4), steps=2000, lr=3e-3)
    R_residual = denoiser_residual(denoiser, u_vals, Y)

    trace_affine = jnp.trace(R_affine)
    trace_residual = jnp.trace(R_residual)
    print(f"\ntrace(R_affine) = {trace_affine:.4e}, trace(R_affine+net) = {trace_residual:.4e}")
    assert trace_residual < trace_affine


def denoiser_residual_standalone(u_samples, features, A, b):
    """Residual of the standalone affine -- same formula as `denoiser_residual`
    in `approximation_based.py`, without a `Denoiser` object.
    """
    resid = u_samples - (features @ A.T + b)
    out = resid.T @ resid / resid.shape[0]
    return 0.5 * (out + out.T)
