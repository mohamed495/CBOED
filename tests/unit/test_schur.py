"""Schur : l'update rank-1 contre le recalcul direct.

`schur_complement` et `schur_update` sont deux chemins indépendants vers la même
matrice -- l'un en une passe, l'autre par récurrence. C'est l'oracle exigé par la
règle d'or : aucun des deux ne réutilise le code de l'autre.
"""

import jax.numpy as jnp
import pytest  # type: ignore

from cboed.bounds.schur import schur_complement, schur_gain_diagonal, schur_update


@pytest.fixture
def spd():
    """SDP non triviale, mal conditionnée à dessein (Sigma = I masque tout)."""
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
    """Le complément reste PSD -- condition du théorème 2.1."""
    cond = schur_complement(spd, jnp.array([1, 4, 7]))
    assert jnp.min(jnp.linalg.eigvalsh(cond)) > -1e-8


@pytest.mark.parametrize("design", [[3], [0, 6], [1, 4, 7], [0, 2, 5, 8]])
def test_rank1_chain_matches_direct(spd, design):
    """LE test du module : la récurrence rank-1 == le recalcul direct."""
    cond = spd
    for j in design:
        cond = schur_update(cond, j)
    assert jnp.allclose(cond, schur_complement(spd, jnp.array(design)), atol=1e-8)


def test_order_does_not_matter(spd):
    """Le complément ne dépend que de l'ensemble, pas de l'ordre d'ajout."""
    a = schur_complement(spd, jnp.array([1, 4, 7]))
    b = schur_complement(spd, jnp.array([7, 1, 4]))
    assert jnp.allclose(a, b, atol=1e-8)


def test_gain_masks_selected(spd):
    """Sans masque, une diagonale ~1e-16 sur ~1e-16 rend un gain fini et faux."""
    selected = jnp.array([2, 5])
    num = schur_complement(spd, selected)
    den = schur_complement(spd + jnp.eye(spd.shape[0]), selected)
    gain = schur_gain_diagonal(num, den, selected)
    assert jnp.all(jnp.isneginf(gain[selected]))
    free = jnp.setdiff1d(jnp.arange(spd.shape[0]), selected)
    assert jnp.all(jnp.isfinite(gain[free]))


def test_gain_matches_logdet_ratio(spd):
    """Oracle indépendant : le gain diagonal == l'accroissement du log-det ratio.

    Vérifie que `½ln(num[j,j]/den[j,j])` est bien le terme incrémental du
    théorème, et pas seulement une formule qui y ressemble.
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
