"""§3.2 -- diagnostiques par débruiteur affine+réseau (`denoisers.py`, `approximation_based.py`).

Même protocole que `test_approximation_diagnostics.py` (débruiteur affine pur,
`linear_approximation_based.py`), étendu au débruiteur composé
`affine + reseau` (`ResidualDenoiser`). Deux jeux de modèles jouets :

- `linear_model` : `u` linéaire -- le réseau n'a rien à apprendre (résidu
  cible nul). Comparaisons sur le résidu `R` brut, jamais sur `Sigma_signal`/
  `Sigma_noise` assemblées : `test_approximation_diagnostics.py` montre que
  cette inversion est quasi singulière à `lambda = 0`, ce qui amplifie le
  bruit d'estimation bien au-delà de ce que ces tests veulent vérifier.
- `nonlinear_model` : `u` non linéaire -- ici le réseau a une raison d'être :
  `R_g` (affine + réseau) doit être strictement plus petit que `R_f` (affine
  seul), sinon `ResidualDenoiser` n'apporte rien.
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
NET_STEPS = 300  # petit budget d'entrainement : verification, pas convergence fine


@pytest.fixture
def prior():
    gp = GaussianProcess(Matern32(length_scale=0.3, sigma=1.0), jnp.zeros(Q))
    return GaussianPrior(prior=gp)


@pytest.fixture
def linear_model():
    """`u(theta) = A theta` -- lineaire, le residu cible du reseau est nul."""
    A = jr.normal(jr.key(3), (P, Q))
    return lambda theta: A @ theta, A


@pytest.fixture
def nonlinear_model():
    """`u(theta) = A theta + c * sin(B theta)` -- non lineaire, residu non nul.

    Le terme `sin` n'est pas affine : `E[u|Y]` s'ecarte de l'affine, et c'est
    cet ecart que le reseau doit capter.
    """
    k_A, k_B = jr.split(jr.key(7))
    A = jr.normal(k_A, (P, Q))
    B = jr.normal(k_B, (P, Q)) * 0.5
    c = 0.3
    return lambda theta: A @ theta + c * jnp.sin(B @ theta)


def _paired(u, prior, Sigma_obs, key, n):
    """`(u(eta), Y = u(eta) + eps)` -- les memes paires que sample_Sigma_Y."""
    k_eta, k_eps = jr.split(key)
    eta = prior.sample(k_eta, n)
    u_vals = jax.vmap(u)(eta)
    L = jnp.linalg.cholesky(Sigma_obs)
    Y = u_vals + jr.normal(k_eps, u_vals.shape) @ L.T
    return u_vals, Y, eta


# ─────────────────────────────────────────────────────────
# AffineDenoiser (equinox) == affine_denoiser (forme fermee standalone)
# ─────────────────────────────────────────────────────────


def test_affine_denoiser_module_matches_standalone_function(prior, linear_model):
    """Les deux implementations affines (linear_approximation_based vs denoisers)
    doivent rendre le meme `A`, `b` -- meme formule, deux code paths.
    """
    u, _ = linear_model
    Sigma_obs = 0.01 * jnp.eye(P)
    u_vals, Y, _ = _paired(u, prior, Sigma_obs, jr.key(1), N_SAMPLES)

    A_std, b_std = affine_denoiser_standalone(u_vals, Y)
    fitted = AffineDenoiser.fit(u_vals, Y)

    assert jnp.allclose(fitted.A, A_std, atol=1e-6)
    assert jnp.allclose(fitted.b, b_std, atol=1e-6)


# ─────────────────────────────────────────────────────────
# Cas lineaire : le reseau n'a rien a apprendre
# ─────────────────────────────────────────────────────────


def test_residual_denoiser_matches_affine_residual_when_linear(prior, linear_model):
    """A `lambda = 0`, le reseau n'a rien a apprendre : le residu `R` (affine+net)
    doit rester proche de celui de l'affine seul.

    Comparaison sur `R` **brut** (avant l'inversion de Prop. 3), pas sur
    `Sigma_signal` assemblee : celle-ci est quasi singuliere a ce point precis
    (l'affine est quasi exacte, cf. test_approximation_diagnostics.py), et une
    difference infime sur `R` y est amplifiee de facon incontrolable -- ce
    n'est pas la propriete que ce test veut verifier.
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
    """`g` voit theta en plus de Y : `R_g <= R_f`.

    Teste sur les residus bruts (stable), pas sur `Sigma_signal`/`Sigma_noise`
    assemblees -- meme raison que ci-dessus : l'inversion de Prop. 3 est
    quasi singuliere a `lambda = 0` et amplifie le bruit d'estimation de `R`
    au point de casser l'ordre attendu sur la matrice assemblee (cf.
    test_approximation_diagnostics.py::test_noise_preceq_signal). L'ordre sur
    `R` lui-meme, en amont de l'inversion, est la propriete verifiable.
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
# Cas non lineaire : le reseau doit reduire le residu sous l'affine seul
# ─────────────────────────────────────────────────────────


def test_residual_denoiser_beats_affine_when_nonlinear(prior, nonlinear_model):
    """La raison d'etre du reseau : sur un `u` non lineaire, `affine + reseau`
    doit laisser un residu plus petit que l'affine seul -- sinon le reseau
    n'apprend rien d'utile.
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
    """Residu de l'affine standalone -- meme formule que `denoiser_residual`
    de `approximation_based.py`, sans objet `Denoiser`.
    """
    resid = u_samples - (features @ A.T + b)
    out = resid.T @ resid / resid.shape[0]
    return 0.5 * (out + out.T)
