import jax.numpy as jnp
import jax.random as jr
import pytest

from cboed.benchmarks import (
    LAMBDAS,
    SIGMA_OBS_MATRIX,
    cfl,
    forward,
    make_model,
    make_prior,
    peclet,
)
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y

N_SAMPLES_REFERENCE = 20_000


@pytest.fixture(scope="module")
def draws():
    thetas = make_prior().sample(jr.key(0), 1_000)
    return thetas, float(jnp.max(jnp.abs(thetas)))


def test_prior_is_well_conditioned():
    """`Matern32(0.2)` sur 200 points : la Gram tient-elle ?

    `cho_factor` rend des `nan` **sans lever** sur une Gram trop mal conditionnee.
    Le symptome se voit a `-inf` (XLA initialise `max` a `-inf`, et `nan > -inf` est
    `False`), pas a `nan`.
    """
    prior = make_prior()
    cond = float(jnp.linalg.cond(prior.prior.Sigma))
    print(f"\ncond(Sigma_theta) = {cond:.2e}")
    assert jnp.all(jnp.isfinite(prior.sample(jr.key(0), 10)))
    assert cond < 1e12


@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_bench_is_resolved(draws, lambda_):
    """`Pe <= 2` et `CFL <= 1`. Garde-fou : sans lui, le gap mesure la divergence."""
    _, u_max = draws
    pe, c = peclet(u_max), cfl(u_max, lambda_)
    print(f"lambda={lambda_:>5.2f}  Pe={pe:>5.2f}  CFL={c:>5.2f}  max|theta|={u_max:.2f}")
    assert pe <= 2.0, f"Pe={pe:.2f} : sous-resolu, monter n ou nu"
    assert c <= 1.0, f"CFL={c:.2f} : monter nt"


def test_lg_reference_at_lambda_zero():
    """⭐ La reference exacte : a `lambda=0`, aucune quantite n'a besoin de Monte-Carlo.

        J = jacobienne (constante)
        Sigma_Y = Sigma_signal = Sigma_obs + J Sigma_theta J^T

    C'est contre elle que se mesurent l'echantillonnage actuel, un futur Halton, et
    le surrogate (§3.2). Sans elle, un changement d'echantillonneur est invisible.

    Le chiffre imprime est **l'erreur du MC actuel a N donne** : c'est l'etalon.
    """
    prior = make_prior()
    model = make_model(0.0)

    J = model.jacobian(prior.mu, None)
    exact = SIGMA_OBS_MATRIX + J @ prior.Sigma() @ J.T
    sampled = sample_Sigma_Y(forward(0.0), prior, SIGMA_OBS_MATRIX, jr.key(0), N_SAMPLES_REFERENCE)

    rel = float(jnp.linalg.norm(sampled - exact) / jnp.linalg.norm(exact))
    print(f"\nN={N_SAMPLES_REFERENCE:,} -> erreur relative sur Sigma_Y = {rel:.3e}")
    assert rel < 0.05
