"""§3.2 -- diagnostiques par approximation (débruiteur).

L'oracle inter-modules : à `lambda = 0`, `u` est linéaire, `Y` gaussien, donc
`E[u|Y]` est **exactement affine**. Le débruiteur affine est alors exact, et
`Sigma^{(F)}_signal` (§3.2) égale `Sigma_signal` (§3.3, gradient). Deux voies sans
code commun, reliées par le fait que les deux estiment la même matrice.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from cboed.bounds.diagnostics.linear_approximation_based import (
    affine_denoiser,
    approximation_noise,
    approximation_signal,
    denoiser_residual,
)
from cboed.bounds.diagnostics.gradient_based import gradient_diagnostics_standard
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Matern32

Q, P = 6, 6
N_SAMPLES = 200_000


@pytest.fixture
def prior():
    gp = GaussianProcess(Matern32(length_scale=0.3, sigma=1.0), jnp.zeros(Q))
    return GaussianPrior(prior=gp)


@pytest.fixture
def linear_model():
    """`u(theta) = A theta` -- lineaire, donc `E[u|Y]` affine et le debruiteur exact."""
    A = jr.normal(jr.key(3), (P, Q))
    return lambda theta: A @ theta, A


def _paired(u, prior, Sigma_obs, key, n):
    """`(u(eta), Y = u(eta) + eps)` -- les memes paires que sample_Sigma_Y."""
    k_eta, k_eps = jr.split(key)
    eta = prior.sample(k_eta, n)
    u_vals = jax.vmap(u)(eta)
    L = jnp.linalg.cholesky(Sigma_obs)
    Y = u_vals + jr.normal(k_eps, u_vals.shape) @ L.T
    return u_vals, Y, eta


def test_affine_denoiser_recovers_linear_map(prior, linear_model):
    """A `Sigma_obs -> 0`, le debruiteur doit retrouver `f(Y) = Y` (le bruit disparait).

    Sans bruit, `Y = u(eta)` exactement, donc `E[u|Y] = Y` : `A -> I`, `b -> 0`.
    """
    u, _ = linear_model
    Sigma_obs = 1e-6 * jnp.eye(P)
    u_vals, Y, _ = _paired(u, prior, Sigma_obs, jr.key(0), 20_000)
    A_f, b_f = affine_denoiser(u_vals, Y)
    assert jnp.allclose(A_f, jnp.eye(P), atol=1e-2)
    assert jnp.allclose(b_f, 0.0, atol=1e-2)


def test_residual_below_Sigma_obs(prior, linear_model):
    """Prop. 3 : `R_f < Sigma_obs` -- condition d'existence de `Sigma^{(F)}_signal`."""
    u, _ = linear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    u_vals, Y, _ = _paired(u, prior, Sigma_obs, jr.key(1), N_SAMPLES)
    A_f, b_f = affine_denoiser(u_vals, Y)
    R_f = denoiser_residual(u_vals, Y, A_f, b_f)
    assert jnp.max(jnp.linalg.eigvalsh(R_f - Sigma_obs)) < 0


def test_signal_matches_gradient_at_lambda_zero(prior, linear_model):
    """⭐ §3.2 == §3.3 en lineaire. Deux voies, meme matrice.

    A `lambda = 0`, `E[u|Y]` est affine (modele lineaire, tout gaussien), donc le
    debruiteur affine est exact et `Sigma^{(F)}_signal` egale `Sigma_signal` gradient.
    """
    u, _ = linear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    key = jr.key(2)

    Sigma_signal_grad, _ = gradient_diagnostics_standard(
        u, prior, Sigma_obs, key, 64  # gradient exact des 1 echantillon (Jac constante)
    )
    u_vals, Y, _ = _paired(u, prior, Sigma_obs, key, N_SAMPLES)
    Sigma_signal_approx = approximation_signal(u_vals, Y, Sigma_obs)

    rel = jnp.linalg.norm(Sigma_signal_approx - Sigma_signal_grad) / jnp.linalg.norm(
        Sigma_signal_grad
    )
    print(f"\n||approx - gradient|| / ||.|| = {rel:.3e}")
    assert rel < 0.05


def test_signal_matches_sample_Sigma_Y_at_lambda_zero(prior, linear_model):
    """Coherence avec §3.1 : en lineaire `Sigma_signal = Sigma_Y` (Rem. 2.2)."""
    u, A = linear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    key = jr.key(4)

    u_vals, Y, _ = _paired(u, prior, Sigma_obs, key, N_SAMPLES)
    Sigma_signal = approximation_signal(u_vals, Y, Sigma_obs)
    Sigma_Y_exact = Sigma_obs + A @ prior.Sigma() @ A.T

    rel = jnp.linalg.norm(Sigma_signal - Sigma_Y_exact) / jnp.linalg.norm(Sigma_Y_exact)
    print(f"||Sigma_signal - Sigma_Y|| / ||.|| = {rel:.3e}")
    assert rel < 0.05


def test_noise_preceq_signal(prior, linear_model):
    """`g` voit theta en plus de Y, donc debruite mieux : `R_g <= R_f`, `Sigma_noise <= Sigma_signal`."""
    u, _ = linear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    u_vals, Y, eta = _paired(u, prior, Sigma_obs, jr.key(5), N_SAMPLES)

    Sigma_signal = approximation_signal(u_vals, Y, Sigma_obs)
    Sigma_noise = approximation_noise(u_vals, Y, eta, Sigma_obs)
    assert jnp.min(jnp.linalg.eigvalsh(Sigma_signal - Sigma_noise)) > -1e-6
