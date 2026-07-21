"""Schur: the rank-1 update against the direct recomputation.

`schur_complement` and `schur_update` are two independent paths to the same
matrix -- one in a single pass, the other by recurrence. This is the oracle
required by the golden rule: neither one reuses the other's code.
"""

import jax.numpy as jnp
import pytest  # type: ignore

from cboed.bounds.schur import schur_complement, schur_gain_diagonal, schur_update


@pytest.fixture
def spd():
    """Nontrivial SPD matrix, deliberately ill-conditioned (Sigma = I would hide everything)."""
    import jax.random as jr

    key = jr.key(0)
    p = 9
    A = jr.normal(key, (p, p))
    return A @ A.T + jnp.diag(jnp.arange(1.0, p + 1.0))


def test_empty_design_is_identity_operation(spd):
    assert jnp.allclose(schur_complement(spd, None), spd)
    assert jnp.allclose(schur_complement(spd, jnp.array([], dtype=int)), spd)


def test_selected_rows_and_columns_vanish(spd):
    """Conditionner par j retire toute l'information portée par j."""
    design = jnp.array([2, 5])
    cond = schur_complement(spd, design)
    assert jnp.allclose(cond[design, :], 0.0, atol=1e-8)
    assert jnp.allclose(cond[:, design], 0.0, atol=1e-8)


def test_schur_complement_is_psd(spd):
    """The complement remains PSD -- condition of Theorem 2.1."""
    cond = schur_complement(spd, jnp.array([1, 4, 7]))
    assert jnp.min(jnp.linalg.eigvalsh(cond)) > -1e-8


@pytest.mark.parametrize("design", [[3], [0, 6], [1, 4, 7], [0, 2, 5, 8]])
def test_rank1_chain_matches_direct(spd, design):
    """THE test of the module: the rank-1 recurrence == the direct recomputation."""
    cond = spd
    for j in design:
        cond = schur_update(cond, j)
    assert jnp.allclose(cond, schur_complement(spd, jnp.array(design)), atol=1e-8)


def test_order_does_not_matter(spd):
    """The complement depends only on the set, not on the order of addition."""
    a = schur_complement(spd, jnp.array([1, 4, 7]))
    b = schur_complement(spd, jnp.array([7, 1, 4]))
    assert jnp.allclose(a, b, atol=1e-8)


def test_gain_masks_selected(spd):
    """Without masking, a ~1e-16 diagonal over ~1e-16 yields a finite but wrong gain."""
    selected = jnp.array([2, 5])
    num = schur_complement(spd, selected)
    den = schur_complement(spd + jnp.eye(spd.shape[0]), selected)
    gain = schur_gain_diagonal(num, den, selected)
    assert jnp.all(jnp.isneginf(gain[selected]))
    free = jnp.setdiff1d(jnp.arange(spd.shape[0]), selected)
    assert jnp.all(jnp.isfinite(gain[free]))


def test_gain_matches_logdet_ratio(spd):
    """Independent oracle: the diagonal gain == the increase in the log-det ratio.

    Verifies that `½ln(num[j,j]/den[j,j])` is indeed the incremental term of
    the theorem, and not merely a formula that resembles it.
    """
    num, den = spd, spd + jnp.eye(spd.shape[0])
    selected = jnp.array([1, 6])
    j = 3

    def log_ratio(design):
        _, ln = jnp.linalg.slogdet(num[jnp.ix_(design, design)])
        _, ld = jnp.linalg.slogdet(den[jnp.ix_(design, design)])
        return 0.5 * (ln - ld)

    expected = log_ratio(jnp.append(selected, j)) - log_ratio(selected)
    gain = schur_gain_diagonal(
        schur_complement(num, selected), schur_complement(den, selected), selected
    )
    assert jnp.allclose(gain[j], expected, atol=1e-8)
