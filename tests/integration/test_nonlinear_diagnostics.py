r"""Prop. 4 et §3.1 dans le cas **non linéaire**.

Tous les autres tests utilisent `u(theta) = A theta`. La jacobienne y est constante,
donc `H(u) = 0` **exactement**, et la seule quantité qui distingue Prop. 4 d'un calcul
linéaire-gaussien n'est jamais exécutée. Idem pour `sample_based` : `Cov(u(theta))` y
vaut `A Sigma A^T`, ce que n'importe quelle erreur de centrage donnerait aussi.

Ce fichier prend `u(theta) = theta**2`, le seul cas non linéaire dont **toutes** les
quantités ont une forme fermée :

    Jac u(theta) = 2 diag(theta)
    L(u)  = E[Jac]^T = 2 diag(m)
    H(u)  = E[(Jac - E[Jac])^T S^{-1} (Jac - E[Jac])]
          = 4 E[diag(delta) S^{-1} diag(delta)]        delta = theta - m
          = 4 (Sigma_theta ⊙ S^{-1})                   produit de Hadamard
    Cov(u(theta))_ij = 2 Sigma_ij^2 + 4 m_i m_j Sigma_ij      (Isserlis)

Les deux identités sont exactes pour `theta` gaussien -- aucun Monte-Carlo dans les
oracles. Vérifiées à 8e-4 (H) et 2.5e-3 (Isserlis) sur 4e6 tirages.

⚠️ `m != 0` est **obligatoire** : `L(u) = 2 diag(m)` s'annule sinon, et
`Sigma_signal = Sigma_obs` degenere. Le banc de production utilise `mu = zeros` ; ici
c'est `mu = ones`, et c'est un choix, pas un oubli.
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
    """`u(theta) = theta**2`. Jac = 2 diag(theta) -- non constante."""
    return theta**2


@pytest.fixture(scope="module")
def setup():
    """Prior gaussien a moyenne **non nulle** et Sigma_obs anisotrope.

    `Sigma_obs = I` masquerait toute erreur de transposition dans `H(u)` : le produit
    de Hadamard avec l'identite est diagonal.
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
    """⭐ `H(u) = 4 (Sigma_theta ⊙ Sigma_obs^{-1})` -- la branche non lineaire.

    C'est LE test manquant : `H(u)` est la seule quantite qui distingue Prop. 4 d'un
    calcul LG, et elle vaut zero partout ailleurs dans la suite. Un facteur, un signe
    ou une transposition y passeraient inapercus -- `Sigma_signal` resterait SDP et
    plausible.
    """
    prior, Sigma_obs = setup
    thetas = prior.sample(jr.key(0), N_SAMPLES)
    _, H = expected_jacobian_moments(u_square, thetas, Sigma_obs)

    expected = 4.0 * prior.Sigma() * jnp.linalg.inv(Sigma_obs)  # Hadamard
    rel = jnp.linalg.norm(H - expected) / jnp.linalg.norm(expected)
    print(f"\nH(u) : erreur relative = {rel:.3e}")
    assert rel < 0.02


@pytest.mark.slow
def test_L_matches_analytic(setup):
    """`L(u) = E[Jac]^T = 2 diag(m)`. Exact des un echantillon si m est connu."""
    prior, Sigma_obs = setup
    thetas = prior.sample(jr.key(0), N_SAMPLES)
    L, _ = expected_jacobian_moments(u_square, thetas, Sigma_obs)

    expected = 2.0 * jnp.diag(prior.mu)
    rel = jnp.linalg.norm(L - expected) / jnp.linalg.norm(expected)
    print(f"L(u) : erreur relative = {rel:.3e}")
    assert rel < 0.02


@pytest.mark.slow
def test_H_is_zero_for_linear_u(setup):
    """Le controle : `H(u) = 0` **exactement** quand la jacobienne est constante.

    C'est ce que garantit le calcul en deux passes. La formule « de calcul »
    `E[J^T S^{-1} J] - Jbar^T S^{-1} Jbar` rendrait ici du bruit d'arrondi au lieu
    d'un zero -- deux grands nombres egaux qui se soustraient.
    """
    prior, Sigma_obs = setup
    A = jr.normal(jr.key(1), (D, D))
    thetas = prior.sample(jr.key(0), 1_000)
    _, H = expected_jacobian_moments(lambda th: A @ th, thetas, Sigma_obs)
    assert jnp.allclose(H, 0.0, atol=1e-10)


@pytest.mark.slow
def test_Sigma_Y_matches_isserlis(setup):
    """⭐ §3.1 en non lineaire : `Cov(theta**2)_ij = 2 Sigma_ij^2 + 4 m_i m_j Sigma_ij`.

    L'estimateur apparie (26) n'a jamais ete teste sur autre chose qu'un modele
    lineaire, ou `Cov(u) = A Sigma A^T` -- une forme que n'importe quelle erreur de
    centrage reproduirait.
    """
    prior, Sigma_obs = setup
    Sigma_Y = sample_Sigma_Y(u_square, prior, Sigma_obs, jr.key(2), N_SAMPLES)

    S, m = prior.Sigma(), prior.mu
    expected = Sigma_obs + 2.0 * S**2 + 4.0 * jnp.outer(m, m) * S
    rel = jnp.linalg.norm(Sigma_Y - expected) / jnp.linalg.norm(expected)
    print(f"Sigma_Y : erreur relative = {rel:.3e}")
    assert rel < 0.05


@pytest.mark.slow
def test_Sigma_signal_matches_analytic(setup):
    """L'assemblage complet contre sa forme fermee -- non lineaire.

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
    print(f"Sigma_signal : erreur relative = {rel:.3e}")
    assert rel < 0.03


@pytest.mark.slow
def test_signal_preceq_Sigma_Y_nonlinear(setup):
    """Prop. 1 en **non lineaire** : `Sigma_Y ⪰ Sigma_signal`, donc `alpha_i >= 1`.

    Via Cramer-Rao : `Sigma_signal^{-1} ⪰ I_Y` donne `Sigma_signal ⪯ I_Y^{-1} ⪯ Cov(Y)`.
    Teste ici avec les **deux formes fermees**, sans erreur MC.
    """
    prior, Sigma_obs = setup
    S, m = prior.Sigma(), prior.mu

    Sigma_Y = Sigma_obs + 2.0 * S**2 + 4.0 * jnp.outer(m, m) * S
    H = 4.0 * S * jnp.linalg.inv(Sigma_obs)
    L = 2.0 * jnp.diag(m)
    Sigma_signal = Sigma_obs + L.T @ jnp.linalg.solve(H + fisher_information_prior(prior), L)

    lo = float(jnp.min(jnp.linalg.eigvalsh(Sigma_Y - Sigma_signal)))
    print(f"lambda_min(Sigma_Y - Sigma_signal) = {lo:.3e}")
    assert lo > -1e-8, "ordre de Loewner viole en non lineaire"
