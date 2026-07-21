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
    """`Matern32(0.2)` on 200 points: does the Gram matrix hold up?

    `cho_factor` returns `nan` **without raising** on a poorly conditioned Gram
    matrix. The symptom shows up as `-inf` (XLA initializes `max` to `-inf`, and
    `nan > -inf` is `False`), not as `nan`.
    """
    prior = make_prior()
    cond = float(jnp.linalg.cond(prior.prior.Sigma))
    print(f"\ncond(Sigma_theta) = {cond:.2e}")
    assert jnp.all(jnp.isfinite(prior.sample(jr.key(0), 10)))
    assert cond < 1e12


@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_bench_is_resolved(draws, lambda_):
    """`Pe <= 2` and `CFL <= 1`. Safety net: without it, the gap measures divergence."""
    _, u_max = draws
    pe, c = peclet(u_max), cfl(u_max, lambda_)
    print(f"lambda={lambda_:>5.2f}  Pe={pe:>5.2f}  CFL={c:>5.2f}  max|theta|={u_max:.2f}")
    assert pe <= 2.0, f"Pe={pe:.2f}: under-resolved, increase n or nu"
    assert c <= 1.0, f"CFL={c:.2f}: increase nt"


def test_lg_reference_at_lambda_zero():
    """⭐ The exact reference: at `lambda=0`, no quantity needs Monte Carlo.

        J = Jacobian (constant)
        Sigma_Y = Sigma_signal = Sigma_obs + J Sigma_theta J^T

    This is what current sampling, a future Halton sequence, and the surrogate
    (§3.2) are all measured against. Without it, a change of sampler is invisible.

    The printed number is **the error of the current MC estimator at a given N**:
    that's the yardstick.
    """
    prior = make_prior()
    model = make_model(0.0)

    J = model.jacobian(prior.mu, None)
    exact = SIGMA_OBS_MATRIX + J @ prior.Sigma() @ J.T
    sampled = sample_Sigma_Y(forward(0.0), prior, SIGMA_OBS_MATRIX, jr.key(0), N_SAMPLES_REFERENCE)

    rel = float(jnp.linalg.norm(sampled - exact) / jnp.linalg.norm(exact))
    print(f"\nN={N_SAMPLES_REFERENCE:,} -> relative error on Sigma_Y = {rel:.3e}")
    assert rel < 0.05
