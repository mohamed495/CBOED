"""Prop. 4 -- cas linéaire et assemblage.

Deux niveaux, depuis que les moments sont séparés de l'assemblage :

* **assemblage** -- `L` et `H` posés à la main, aucune jacobienne. Deux chemins
  (`assemble`, `assemble_misfit`) vers la même matrice : oracles l'un de l'autre.
* **bout en bout** -- `u` linéaire, `H(u) = 0`, tout a une forme fermée.

Le cas non linéaire, où `H(u) != 0`, est dans `test_nonlinear_diagnostics.py`.
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
    """`u` lineaire (jacobienne constante) et `h` = selection des D premieres."""
    A = jr.normal(jr.key(3), (P, Q))
    B = jnp.eye(D, Q)
    return (lambda eta: A @ eta), (lambda eta: B @ eta), A, B


@pytest.fixture
def moments():
    """`L` et `H` synthetiques -- ni jacobienne ni modele direct.

    `H` SDP non triviale : `H = 0` (le cas lineaire) ne testerait rien de
    l'assemblage.
    """
    L = jr.normal(jr.key(7), (Q, P))
    X = jr.normal(jr.key(8), (Q, Q))
    return L, X @ X.T


# =============================================================================
# Assemblage -- sans jacobiennes
# =============================================================================


def test_misfit_matches_direct(prior_eta, moments):
    """⭐ `assemble` et `assemble_misfit` : deux chemins, une seule matrice.

    L'identite : `(H + Sigma^{-1})^{-1} = Ssq (I + Ssq H Ssq)^{-1} Ssq`.

    Le prototype NumPy faisait `solve(A_mis, l)` alors que `A_mis` **est deja
    l'inverse** -- 248 % d'erreur relative, resultat toujours SDP, donc silencieux.
    Ce test l'aurait attrape en trois lignes.
    """
    L, H = moments
    direct = assemble(L, H + fisher_information_prior(prior_eta), jnp.eye(P))
    misfit = assemble_misfit(L, H, prior_eta.Sigma(), jnp.eye(P))
    rel = jnp.linalg.norm(direct - misfit) / jnp.linalg.norm(direct)
    print(f"\nassemble vs assemble_misfit : erreur relative = {rel:.3e}")
    assert rel < 1e-8


def test_misfit_matches_direct_with_extra(prior_eta, moments):
    """Idem avec `J(h)` : l'identite ne porte que sur `Sigma_eta^{-1}`."""
    L, H = moments
    J_h = jnp.diag(jnp.arange(1.0, Q + 1.0))
    direct = assemble(L, H + fisher_information_prior(prior_eta) + J_h, jnp.eye(P))
    misfit = assemble_misfit(L, H, prior_eta.Sigma(), jnp.eye(P), extra=J_h)
    assert jnp.allclose(direct, misfit, rtol=1e-8)


def test_misfit_never_forms_prior_precision():
    """N'utilise que `Sigma_eta^{1/2}` -- ce qui survit quand la precision ne l'est plus."""
    src = inspect.getsource(assemble_misfit)
    assert "inv(" not in src
    assert "prior_precision" not in src


def test_assemble_is_symmetric(moments):
    L, H = moments
    out = assemble(L, H + jnp.eye(Q), jnp.eye(P))
    assert jnp.allclose(out, out.T, atol=1e-12)


def test_assemble_reduces_to_Sigma_obs_when_L_vanishes(moments):
    """`L = 0` -> aucune information -> `Sigma_signal = Sigma_obs`."""
    _, H = moments
    Sigma_obs = jnp.diag(jnp.arange(1.0, P + 1.0))
    assert jnp.allclose(assemble(jnp.zeros((Q, P)), H + jnp.eye(Q), Sigma_obs), Sigma_obs)


def test_psd_sqrt_handles_singular():
    """`Sigma_xi = 0` est un cas nominal : `cholesky` y rendrait des `nan`."""
    assert jnp.allclose(psd_sqrt(jnp.zeros((4, 4))), 0.0)


def test_psd_sqrt_squares_back(prior_eta):
    R = psd_sqrt(prior_eta.Sigma())
    assert jnp.allclose(R @ R, prior_eta.Sigma(), atol=1e-8)


# =============================================================================
# Information de Fisher du prior
# =============================================================================


def test_fisher_exact_matches_monte_carlo(prior_eta):
    """Deux chemins vers `I_eta` -- et preuve que le prior est bien gaussien."""
    exact = fisher_information_prior(prior_eta)
    mc = fisher_information_prior_mc(prior_eta, jr.key(0), 200_000)
    rel = jnp.linalg.norm(exact - mc) / jnp.linalg.norm(exact)
    print(f"I_eta exact vs MC : erreur relative = {rel:.3e}")
    assert rel < 0.05


def test_fisher_exact_is_prior_precision(prior_eta):
    """`I_eta == Gamma_eta^{-1}` : oracle via `Sigma() @ I_eta == I`."""
    assert jnp.allclose(
        prior_eta.Sigma() @ fisher_information_prior(prior_eta), jnp.eye(Q), atol=1e-8
    )


# =============================================================================
# Moments -- cas lineaire
# =============================================================================


def test_H_is_zero_for_constant_jacobian(prior_eta, linear_setup):
    """`H(u) = 0` **exactement** -- ce que garantissent les deux passes."""
    u, _, _, _ = linear_setup
    etas = prior_eta.sample(jr.key(0), 64)
    _, H = expected_jacobian_moments(u, etas, jnp.eye(P))
    assert jnp.allclose(H, 0.0, atol=1e-10)


def test_L_is_the_transposed_jacobian(prior_eta, linear_setup):
    """`L(u) = E[Jac]^T = A^T`. Le `.T` compte."""
    u, _, A, _ = linear_setup
    etas = prior_eta.sample(jr.key(0), 64)
    L, _ = expected_jacobian_moments(u, etas, jnp.eye(P))
    assert jnp.allclose(L, A.T, atol=1e-10)


# =============================================================================
# Bout en bout -- cas lineaire
# =============================================================================


def test_lg_collapse(prior_eta, linear_setup):
    """`H(u) = 0` -> `Sigma_signal = Sigma_obs + A Sigma_eta A^T = Sigma_Y`.

    L'effondrement du Rem. 2.2 -- et ce que la formule « de calcul » pour `H(u)`
    detruirait par cancellation.
    """
    u, _, A, _ = linear_setup
    Sigma_obs = jnp.diag(jnp.arange(1.0, P + 1.0)) * 0.01
    Sigma_signal, _ = gradient_diagnostics_standard(u, prior_eta, Sigma_obs, jr.key(1), 64)
    assert jnp.allclose(Sigma_signal, Sigma_obs + A @ prior_eta.Sigma() @ A.T, atol=1e-8)


def test_standard_noise_is_exactly_Sigma_obs(prior_eta, linear_setup):
    """Prop. 2 : pose, pas approche. `array_equal`, pas `allclose`."""
    u, _, _, _ = linear_setup
    Sigma_obs = jnp.eye(P) * 0.01
    _, Sigma_noise = gradient_diagnostics_standard(u, prior_eta, Sigma_obs, jr.key(0), 64)
    assert jnp.array_equal(Sigma_noise, Sigma_obs)


def test_signal_independent_of_Sigma_xi(prior_eta, linear_setup):
    """`Sigma_signal` ne contient pas `J(h)` : `xi` ne doit pas la bouger."""
    u, h, _, _ = linear_setup
    Sigma_obs = jnp.eye(P) * 0.01
    kw = {"key": jr.key(2), "n_samples": 64}
    s1, _ = gradient_diagnostics(u, h, prior_eta, Sigma_obs, jnp.eye(D) * 1e-2, **kw)
    s2, _ = gradient_diagnostics(u, h, prior_eta, Sigma_obs, jnp.eye(D) * 1e2, **kw)
    assert jnp.allclose(s1, s2, atol=1e-10)


def test_noise_preceq_signal(prior_eta, linear_setup):
    """`J(h)` PSD -> `(H+I+J)^{-1} ⪯ (H+I)^{-1}` -> `Sigma_noise ⪯ Sigma_signal`.

    L'ecart **est** `J(h)` : c'est `gap_h`.
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
    """La limite `Sigma_xi -> 0` est singuliere : l'ecart decroit en `O(Sigma_xi)`.

    C'est pourquoi `gradient_diagnostics_standard` existe.
    """
    u, h, _, _ = linear_setup
    Sigma_obs = jnp.eye(P) * 0.01
    _, Sigma_noise = gradient_diagnostics(
        u, h, prior_eta, Sigma_obs, jnp.eye(D) * scale, jr.key(5), 64
    )
    gap = jnp.linalg.norm(Sigma_noise - Sigma_obs) / jnp.linalg.norm(Sigma_obs)
    print(f"Sigma_xi = {scale:.0e} -> ecart relatif = {gap:.3e}")
    assert jnp.all(jnp.isfinite(Sigma_noise))
    assert jnp.min(jnp.linalg.eigvalsh(Sigma_noise - Sigma_obs)) > -1e-8
