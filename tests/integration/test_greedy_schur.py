"""Greedy Schur against brute force.

Two independent oracles:
* the naive greedy, which rescans all candidates and recomputes `slogdet` from scratch;
* `log_ratio`, which evaluates the ratio without ever touching a Schur complement.

Neither one takes the code path under test.
"""

from itertools import pairwise

import jax.numpy as jnp
import jax.random as jr
import pytest  # type: ignore

from cboed.optim.greedy_schur import greedy_schur, log_ratio


@pytest.fixture
def pair():
    """Two distinct, nontrivial SPD matrices (A != B, neither equal to I)."""
    k1, k2 = jr.split(jr.key(1))
    p = 8
    X = jr.normal(k1, (p, p))
    Y = jr.normal(k2, (p, p))
    A = X @ X.T + jnp.diag(jnp.arange(1.0, p + 1.0))
    B = 0.5 * (Y @ Y.T) + jnp.eye(p)
    return A, B


def _greedy_bruteforce(A, B, n_sensors):
    """Oracle: rescans everything, recomputes `slogdet` from scratch, no Schur."""
    selected: list[int] = []
    for _ in range(n_sensors):
        best_j, best = None, -jnp.inf
        for j in range(A.shape[0]):
            if j in selected:
                continue
            score = log_ratio(A, B, jnp.asarray([*selected, j], dtype=int))
            if score > best:
                best, best_j = score, j
        selected.append(best_j)
    return selected


@pytest.mark.parametrize("n_sensors", [1, 3, 5])
def test_matches_bruteforce_greedy(pair, n_sensors):
    """THE test: same design as the flat greedy."""
    A, B = pair
    result = greedy_schur(A, B, n_sensors)
    assert list(result.design) == _greedy_bruteforce(A, B, n_sensors)


def test_scores_telescope(pair):
    """scores[k] == log_ratio(design[:k+1]) -- the sum of gains telescopes."""
    A, B = pair
    result = greedy_schur(A, B, 4)
    for k in range(4):
        expected = log_ratio(A, B, result.design[: k + 1])
        assert jnp.allclose(result.scores[k], expected, atol=1e-8)


def test_no_sensor_selected_twice(pair):
    """Without a mask, a ~1e-16 diagonal entry over ~1e-16 would give a finite gain."""
    A, B = pair
    design = greedy_schur(A, B, 6).design
    assert len(jnp.unique(design)) == 6


def test_full_budget_is_all_sensors(pair):
    A, B = pair
    design = greedy_schur(A, B, A.shape[0]).design
    assert set(int(j) for j in design) == set(range(A.shape[0]))


@pytest.mark.parametrize("n_sensors", [0, -1, 99])
def test_invalid_budget_rejected(pair, n_sensors):
    A, B = pair
    with pytest.raises(ValueError):
        greedy_schur(A, B, n_sensors)


def test_scores_are_monotone(pair):
    A, B = pair
    scores = greedy_schur(A, B, 5).scores
    assert all(b >= a for a, b in pairwise(scores))
