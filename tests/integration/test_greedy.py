from typing import NamedTuple

import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.criteria.optimality import EIG
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.optim.greedy import GreedyOptimizer
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess


class Setup(NamedTuple):
    model: AdvectionDiffusion
    prior: GaussianProcess
    likelihood: GaussianLikelihood
    inference: LinearModel


@pytest.fixture
def setup() -> Setup:
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1.0,
        domain=[0, 1],
        nt=5,
        n=4,
    )
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    likelihood = GaussianLikelihood(model=model, prior=prior, Sigma_obs=jnp.eye(model.n))
    gaussian_prior = GaussianPrior(prior=prior)
    inference = LinearModel(prior=gaussian_prior, likelihood=likelihood)
    return Setup(model, prior, likelihood, inference)


def test_greedy_selects_monotone_improving(setup):
    """Le greedy produit des scores croissants."""

    eig = EIG(inference=setup.inference)
    opt = GreedyOptimizer(criterion=eig)
    result = opt.run(theta=setup.prior.mu, n_sensors=3, n_candidates=4)

    assert len(result.design) == 3
    assert len(set(result.design.tolist())) == 3  # pas de doublon
    # scores croissants (monotonie de l'EIG)
    assert all(a <= b for a, b in zip(result.scores, result.scores[1:], strict=False))


def test_greedy_first_pick_is_argmax(setup):
    """Le premier capteur est celui de meilleur EIG individuel."""

    eig = EIG(inference=setup.inference)
    theta = setup.prior.mu

    # calcul brut du meilleur capteur seul
    singles = [eig.evaluate(theta, jnp.array([i])) for i in range(4)]
    best_single = int(jnp.argmax(jnp.array(singles)))

    opt = GreedyOptimizer(criterion=eig)
    result = opt.run(theta, n_sensors=1, n_candidates=4)
    assert result.design[0] == best_single


def test_greedy_close_to_exhaustive(setup):
    """Sur petit n, greedy vs meilleur design exhaustif."""
    from itertools import combinations

    eig = EIG(inference=setup.inference)
    theta = setup.prior.mu

    # exhaustif : meilleur design de taille 2 parmi 4
    best_exhaustive = max(
        eig.evaluate(theta, jnp.array(list(c))) for c in combinations(range(4), 2)
    )
    opt = GreedyOptimizer(criterion=eig)
    greedy = eig.evaluate(theta, opt.run(theta, 2, 4).design)

    # greedy ≥ (1 - 1/e) x optimal (borne de Nemhauser, si sous-modulaire)
    assert greedy >= 0.63 * best_exhaustive
