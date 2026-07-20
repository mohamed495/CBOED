"""Prop. 1 : le spectre du gap.

L'oracle est gratuit et exact : `½ sum(ln alpha + ln beta) = gap(I_p)`. Le membre de
gauche passe par `eigvalsh` sur un problème généralisé ; le droit par des `slogdet` qui
ne diagonalisent rien. Aucun code commun.
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
    """Cadre standard synthetique : `Sigma_{Y|theta} = Sigma_noise = Sigma_obs`.

    `Sigma_signal ⪯ Sigma_Y` construit explicitement, conformement a Prop. 1
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
    """`A u = alpha B u` : verification par `det(A - alpha B) ~ 0`."""
    alpha = generalized_eigenvalues(diagnostics.Sigma_Y, diagnostics.Sigma_signal)
    for a in alpha:
        _, ld = jnp.linalg.slogdet(diagnostics.Sigma_Y - a * diagnostics.Sigma_signal)
        assert ld < -10, f"alpha={a} n'annule pas det(A - alpha B)"


def test_eigenvalues_are_decreasing(diagnostics):
    assert jnp.all(jnp.diff(quasi_optimality(diagnostics).alpha) <= 1e-10)


def test_alpha_at_least_one(diagnostics):
    """Prop. 1 : `Sigma_Y ⪰ Sigma_signal` donc `alpha_i >= 1`."""
    assert jnp.min(quasi_optimality(diagnostics).alpha) >= 1.0 - 1e-8


def test_beta_is_one_in_standard_setting(diagnostics):
    """`Sigma_{Y|theta} = Sigma_noise = Sigma_obs` -> beta identiquement 1.

    Le cadre standard isole `gap_G` **y compris spectralement** : `ln beta = 0`, donc
    la sous-optimalite ne depend que de `alpha`.
    """
    assert jnp.allclose(quasi_optimality(diagnostics).beta, 1.0, atol=1e-8)


def test_spectrum_sums_to_gap(diagnostics):
    """⭐ `½ sum(ln alpha + ln beta) == gap(I_p)`.

    Deux chemins : `eigvalsh` generalise d'un cote, `slogdet` de sous-matrices de
    l'autre. Le gap **est** la somme des log-valeurs propres generalisees, et les
    constantes de sous-optimalite en sont des sommes partielles.
    """
    q = quasi_optimality(diagnostics)
    gap = float(incremental_bounds(diagnostics, None).gap)
    assert jnp.allclose(q.total_gap, gap, atol=1e-8)


def test_suboptimality_is_monotone(diagnostics):
    """La constante **croit** avec m en incremental, **decroit** en conservatif.

    Les deux strategies lisent le meme spectre par les deux bouts.
    """
    q = quasi_optimality(diagnostics)
    inc = [q.suboptimality(m, "incremental") for m in range(1, P)]
    cons = [q.suboptimality(m, "conservative") for m in range(1, P)]
    assert all(b >= a - 1e-10 for a, b in pairwise(inc))
    assert all(b <= a + 1e-10 for a, b in pairwise(cons))


def test_suboptimality_at_full_budget_equals_gap(diagnostics):
    """A m = p, l'incremental a paye tout le gap. A m = 0, le conservatif aussi."""
    q = quasi_optimality(diagnostics)
    assert jnp.allclose(q.suboptimality(P, "incremental"), q.total_gap, atol=1e-8)
    assert jnp.allclose(q.suboptimality(0, "conservative"), q.total_gap, atol=1e-8)


def test_unknown_strategy_rejected(diagnostics):
    with pytest.raises(ValueError, match="incremental"):
        quasi_optimality(diagnostics).suboptimality(3, "greedy")


def test_crossover_is_half_the_dimension(diagnostics):
    """`crossover` est **trivial** : `p//2 + 1`, quel que soit le spectre.

    `inc(m)` somme les `m` premiers termes, `cons(m)` les `p-m` premiers, de la
    **meme** suite positive (`alpha, beta >= 1` donc `ln >= 0`). Donc
    `inc(m) > cons(m)` revient a `m > p-m`, soit `m > p/2`. Le premier entier qui
    satisfait ca est `p//2 + 1` -- a `m = p/2` les deux sommes sont identiques.

    Ce test documente que la fonction n'apprend rien du spectre.
    """
    assert quasi_optimality(diagnostics).crossover() == P // 2 + 1


def test_effective_rank_detects_concentration():
    """Un gap concentre sur un seul mode -> `effective_rank = 1`."""
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
    """Un gap etale sur tous les modes -> `effective_rank` proche de p."""
    Sigma_obs = 0.01 * jnp.eye(P)
    Sigma_signal = Sigma_obs + jnp.eye(P)
    Sigma_Y = Sigma_signal + 2.0 * jnp.eye(P)  # tous les alpha egaux
    d = DiagnosticMatrices(
        Sigma_Y=Sigma_Y,
        Sigma_Y_given_theta=Sigma_obs,
        Sigma_signal=Sigma_signal,
        Sigma_noise=Sigma_obs,
        certified=True,
    )
    assert quasi_optimality(d).effective_rank >= P - 1
