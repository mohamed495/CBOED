"""§3.1 : les diagnostiques par échantillonnage.

Oracles indépendants, tous à `lambda=0` où le modèle est linéaire :

* `Sigma_Y`        contre `Sigma_obs + A Sigma_eta A^T`  (24)
* `Sigma_{Y|theta}` contre `Sigma_obs + A (Sigma_eta^{-1} + B^T Sigma_xi^{-1} B)^{-1} A^T`  (24)

Aucun code commun : d'un côté du Monte-Carlo sur le modèle direct, de l'autre de
l'algèbre linéaire sur la jacobienne.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest  # type: ignore

from cboed.bounds.diagnostics.sample_based import (
    sample_diagnostics_standard,
    sample_Sigma_Y,
    sample_Sigma_Y_given_theta,
)
from cboed.core.burgers import Burgers
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Gaussian

N = 50
N_SAMPLES = 20_000


@pytest.fixture
def bench():
    model = Burgers(diffusivity=0.01, lambda_=0.0, T=1.0, domain=[0, 1], nt=5, n=N)
    gp = GaussianProcess(kernel=Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(N))
    prior = GaussianPrior(prior=gp)
    Sigma_obs = 0.001 * jnp.eye(N)
    return model, prior, Sigma_obs


def test_Sigma_Y_matches_linear_closed_form(bench):
    """(26) contre `Sigma_obs + A Sigma_eta A^T`. Erreur MC en O(1/sqrt(N))."""
    model, prior, Sigma_obs = bench

    def u(eta):
        return model(eta, None)

    sampled = sample_Sigma_Y(u, prior, Sigma_obs, jr.key(0), N_SAMPLES)
    A = model.jacobian(prior.mu, None)
    exact = Sigma_obs + A @ prior.Sigma() @ A.T

    rel = jnp.linalg.norm(sampled - exact) / jnp.linalg.norm(exact)
    print(f"||Sigma_Y_MC - Sigma_Y_exact|| / ||.|| = {rel:.3e}")
    assert rel < 0.02


@pytest.mark.parametrize("sigma_xi", [1e-1, 1e-2])
def test_Sigma_Y_given_theta_matches_linear_closed_form(bench, sigma_xi):
    """(27) via Rem. 3.1 contre la forme fermée (24) -- oracle indépendant."""
    model, prior, Sigma_obs = bench
    B = jnp.eye(N)
    Sigma_xi = sigma_xi * jnp.eye(N)

    def u(eta):
        return model(eta, None)

    sampled = sample_Sigma_Y_given_theta(u, prior, B, Sigma_obs, Sigma_xi, jr.key(1), N_SAMPLES)
    A = model.jacobian(prior.mu, None)
    M = prior.prior_precision_matmul(jnp.eye(N)) + jnp.linalg.inv(Sigma_xi)
    exact = Sigma_obs + A @ jnp.linalg.solve(M, A.T)

    rel = jnp.linalg.norm(sampled - exact) / jnp.linalg.norm(exact)
    print(f"Sigma_xi={sigma_xi:.0e} -> erreur relative = {rel:.3e}")
    assert rel < 0.05


def test_law_of_total_variance(bench):
    """`Sigma_Y ⪰ Sigma_{Y|theta}` : conditionner ne peut qu'enlever de la variance.

    `Cov(u) = E[Cov(u|theta)] + Cov(E[u|theta])`, second terme PSD.
    """
    model, prior, Sigma_obs = bench

    def u(eta):
        return model(eta, None)

    Sigma_Y = sample_Sigma_Y(u, prior, Sigma_obs, jr.key(2), N_SAMPLES)
    Sigma_Y_given_theta = sample_Sigma_Y_given_theta(
        u, prior, jnp.eye(N), Sigma_obs, 1e-1 * jnp.eye(N), jr.key(3), N_SAMPLES
    )
    gap = Sigma_Y - Sigma_Y_given_theta
    assert jnp.min(jnp.linalg.eigvalsh(gap)) > -1e-3


def test_zero_Sigma_xi_degenerates_to_Sigma_obs(bench):
    """⭐ `Sigma_xi = 0` exactement : Rem. 3.1 y est **régulière**.

    Là où Prop. 4 a une limite singulière (`J(h) = Sigma_xi^{-1}` explose), la voie
    échantillon accepte `Sigma_xi = 0` : `eta|theta` dégénère en Dirac, donc
    `eta' = eta`, donc les différences appariées sont **exactement nulles** et
    `Sigma_{Y|theta} = Sigma_obs`. C'est le cas standard, retrouvé sans détour.
    """
    model, prior, Sigma_obs = bench

    def u(eta):
        return model(eta, None)

    sampled = sample_Sigma_Y_given_theta(
        u, prior, jnp.eye(N), Sigma_obs, jnp.zeros((N, N)), jr.key(4), 1_000
    )
    assert jnp.allclose(sampled, Sigma_obs, atol=1e-10)


def test_standard_entry_point_poses_Sigma_obs(bench):
    """Symétrique de `gradient_diagnostics_standard` : posé, pas échantillonné."""
    model, prior, Sigma_obs = bench

    def u(eta):
        return model(eta, None)

    _, Sigma_Y_given_theta = sample_diagnostics_standard(u, prior, Sigma_obs, jr.key(5), 1_000)
    assert jnp.array_equal(Sigma_Y_given_theta, Sigma_obs)


def test_paired_estimator_matches_empirical_covariance(bench):
    """L'estimateur apparié (26) contre `jnp.cov` -- deux estimateurs du même objet.

    L'apparié évite de retrancher une moyenne empirique, donc la cancellation qui
    guette `E[XX^T] - mean mean^T`. Les deux doivent converger au même endroit.
    """
    import jax

    model, prior, Sigma_obs = bench

    def u(eta):
        return model(eta, None)

    paired = sample_Sigma_Y(u, prior, Sigma_obs, jr.key(6), N_SAMPLES)
    U = jax.vmap(u)(prior.sample(jr.key(7), N_SAMPLES))
    empirical = Sigma_obs + jnp.cov(U, rowvar=False)

    rel = jnp.linalg.norm(paired - empirical) / jnp.linalg.norm(empirical)
    assert rel < 0.02
