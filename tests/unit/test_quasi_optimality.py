"""Prop. 1: the spectrum of the gap.

The oracle is free and exact: `½ sum(ln alpha + ln beta) = gap(I_p)`. The left-hand
side goes through `eigvalsh` on a generalized problem; the right-hand side through
`slogdet` calls that diagonalize nothing. No shared code.
"""

from itertools import pairwise

import jax.numpy as jnp
import jax.random as jr
import pytest

from cboed.bounds.base import DiagnosticMatrices
from cboed.bounds.bounds import incremental_bounds
from cboed.bounds.quasi_optimality import generalized_eigenvalues, quasi_optimality

P = 8


@pytest.fixture
def diagnostics():
    """Synthetic standard setting: `Sigma_{Y|theta} = Sigma_noise = Sigma_obs`.

    `Sigma_signal ⪯ Sigma_Y` built explicitly, in accordance with Prop. 1
    (`alpha >= 1`).
    """
    X = jr.normal(jr.key(0), (P, P))
    Sigma_obs = 0.01 * jnp.eye(P)
    Sigma_signal = Sigma_obs + 0.1 * (X @ X.T + jnp.eye(P))
    Sigma_Y = Sigma_signal + 0.5 * jnp.diag(jnp.arange(1.0, P + 1.0))
    return DiagnosticMatrices(
        Sigma_Y=Sigma_Y,
        Sigma_Y_given_theta=Sigma_obs,
        Sigma_signal=Sigma_signal,
        Sigma_noise=Sigma_obs,
        certified=True,
    )


def test_generalized_eigenvalues_match_definition(diagnostics):
    """`A u = alpha B u`: verification via `det(A - alpha B) ~ 0`."""
    alpha = generalized_eigenvalues(diagnostics.Sigma_Y, diagnostics.Sigma_signal)
    for a in alpha:
        _, ld = jnp.linalg.slogdet(diagnostics.Sigma_Y - a * diagnostics.Sigma_signal)
        assert ld < -10, f"alpha={a} does not zero out det(A - alpha B)"


def test_eigenvalues_are_decreasing(diagnostics):
    assert jnp.all(jnp.diff(quasi_optimality(diagnostics).alpha) <= 1e-10)


def test_alpha_at_least_one(diagnostics):
    """Prop. 1: `Sigma_Y ⪰ Sigma_signal` therefore `alpha_i >= 1`."""
    assert jnp.min(quasi_optimality(diagnostics).alpha) >= 1.0 - 1e-8


def test_beta_is_one_in_standard_setting(diagnostics):
    """`Sigma_{Y|theta} = Sigma_noise = Sigma_obs` -> beta identically 1.

    The standard setting isolates `gap_G` **including spectrally**: `ln beta = 0`,
    so suboptimality depends only on `alpha`.
    """
    assert jnp.allclose(quasi_optimality(diagnostics).beta, 1.0, atol=1e-8)


def test_spectrum_sums_to_gap(diagnostics):
    """`½ sum(ln alpha + ln beta) == gap(I_p)`.

    Two paths: a generalized `eigvalsh` on one side, `slogdet` of submatrices on
    the other. The gap **is** the sum of the generalized log-eigenvalues, and the
    suboptimality constants are partial sums of it.
    """
    q = quasi_optimality(diagnostics)
    gap = float(incremental_bounds(diagnostics, None).gap)
    assert jnp.allclose(q.total_gap, gap, atol=1e-8)


def test_suboptimality_is_monotone(diagnostics):
    """The constant **increases** with m incrementally, **decreases** conservatively.

    Both strategies read the same spectrum from opposite ends.
    """
    q = quasi_optimality(diagnostics)
    inc = [q.suboptimality(m, "incremental") for m in range(1, P)]
    cons = [q.suboptimality(m, "conservative") for m in range(1, P)]
    assert all(b >= a - 1e-10 for a, b in pairwise(inc))
    assert all(b <= a + 1e-10 for a, b in pairwise(cons))


def test_suboptimality_at_full_budget_equals_gap(diagnostics):
    """At m = p, incremental has paid the entire gap. At m = 0, so has conservative."""
    q = quasi_optimality(diagnostics)
    assert jnp.allclose(q.suboptimality(P, "incremental"), q.total_gap, atol=1e-8)
    assert jnp.allclose(q.suboptimality(0, "conservative"), q.total_gap, atol=1e-8)


def test_unknown_strategy_rejected(diagnostics):
    with pytest.raises(ValueError, match="incremental"):
        quasi_optimality(diagnostics).suboptimality(3, "greedy")


def test_crossover_is_half_the_dimension(diagnostics):
    """`crossover` is **trivial**: `p//2 + 1`, regardless of the spectrum.

    `inc(m)` sums the first `m` terms, `cons(m)` the first `p-m` terms, of the
    **same** positive sequence (`alpha, beta >= 1` so `ln >= 0`). So
    `inc(m) > cons(m)` reduces to `m > p-m`, i.e. `m > p/2`. The first integer
    satisfying this is `p//2 + 1` -- at `m = p/2` the two sums are identical.

    This test documents that the function learns nothing from the spectrum.
    """
    assert quasi_optimality(diagnostics).crossover() == P // 2 + 1


def test_effective_rank_detects_concentration():
    """A gap concentrated on a single mode -> `effective_rank = 1`."""
    Sigma_obs = 0.01 * jnp.eye(P)
    Sigma_signal = Sigma_obs + jnp.eye(P)
    Sigma_Y = Sigma_signal + jnp.diag(jnp.zeros(P).at[0].set(100.0))
    d = DiagnosticMatrices(
        Sigma_Y=Sigma_Y,
        Sigma_Y_given_theta=Sigma_obs,
        Sigma_signal=Sigma_signal,
        Sigma_noise=Sigma_obs,
        certified=True,
    )
    assert quasi_optimality(d).effective_rank == 1


def test_effective_rank_detects_spread():
    """A gap spread over all modes -> `effective_rank` close to p."""
    Sigma_obs = 0.01 * jnp.eye(P)
    Sigma_signal = Sigma_obs + jnp.eye(P)
    Sigma_Y = Sigma_signal + 2.0 * jnp.eye(P)  # all alpha equal
    d = DiagnosticMatrices(
        Sigma_Y=Sigma_Y,
        Sigma_Y_given_theta=Sigma_obs,
        Sigma_signal=Sigma_signal,
        Sigma_noise=Sigma_obs,
        certified=True,
    )
    assert quasi_optimality(d).effective_rank >= P - 1
