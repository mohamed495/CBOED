r"""Quasi-optimality -- Proposition 1.

The gap is not an opaque scalar: it is a **sum of generalized log-eigenvalues**,
and its distribution decides which of the two strategies is usable.

Prop. 1 poses the two generalized eigenvalue problems

.. math::
    \Sigma_Y u_i = \alpha_i \Sigma_{\rm signal} u_i, \qquad
    \Sigma_{Y|\theta} v_i = \beta_i \Sigma_{\rm noise} v_i

with ``alpha_i, beta_i >= 1`` (since ``Sigma_Y ⪰ Sigma_signal`` and
``Sigma_{Y|theta} ⪰ Sigma_noise``), and bounds the suboptimality of the greedy
designs:

.. math::
    \mathrm{EIG}(W^{\rm inc}_m) &\ge \max_W \mathrm{EIG}(W)
        - \sum_{i=1}^{m} \tfrac{\ln\alpha_i + \ln\beta_i}{2} \\
    \mathrm{EIG}(W^{\rm cons}_m) &\ge \max_W \mathrm{EIG}(W)
        - \sum_{i=1}^{d-m} \tfrac{\ln\alpha_i + \ln\beta_i}{2}

The identity that ties everything together
--------------------------------------------
Summing all the eigenvalues,

.. math::
    \sum_i (\ln\alpha_i + \ln\beta_i) = 2\,\mathrm{gap}(I_p).

The gap at the full design is therefore the sum of the spectral
contributions

.. math::
    t_i = \frac{\ln\alpha_i+\ln\beta_i}{2}.

The constants of Proposition 1 are simply partial sums of these
contributions:

* incremental: ``\sum_{i=1}^{m} t_i``;
* conservative: ``\sum_{i=1}^{d-m} t_i``.

Both use the same spectral contributions, but with a different number of
terms. The incremental constant is therefore increasing with the budget
``m``, while the conservative constant is decreasing.

Standard case
-------------
``Sigma_{Y|theta} = Sigma_noise = Sigma_obs`` **exactly**, so ``beta_i = 1``
for all ``i`` and ``ln beta_i = 0``: the suboptimality then depends only on
``alpha``. The standard setting isolates ``gap_G``, spectrally as well.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped

from cboed.bounds.base import DiagnosticMatrices


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QuasiOptimality:
    """The spectrum of the gap and what it costs."""

    alpha: Float[Array, " n_obs"]
    """Eigenvalues of ``(Sigma_Y, Sigma_signal)``, **decreasing**. ``>= 1``."""

    beta: Float[Array, " n_obs"]
    """Eigenvalues of ``(Sigma_Y_given_theta, Sigma_noise)``, decreasing. ``>= 1``.

    Identically 1 in the standard setting: both matrices equal ``Sigma_obs``
    there.
    """

    def suboptimality(self, n_sensors: int, strategy: str = "incremental") -> float:
        """Bound on ``max_W EIG(W) - EIG(W_greedy)`` -- eq. (22)/(23).

        Parameters
        ----------
        n_sensors : int
            Budget ``m``.
        strategy : {"incremental", "conservative"}
            Incremental: sum of the **first** ``m`` (the largest).
            Conservative: sum of the **last** ``d - m``.

        Notes
        -----
        The constant **grows with m** in the incremental strategy and
        **shrinks** in the conservative one. Small budget -> incremental;
        large budget -> conservative. Complementary, not competing.
        """
        terms = 0.5 * (jnp.log(self.alpha) + jnp.log(self.beta))
        if strategy == "incremental":
            return float(jnp.sum(terms[:n_sensors]))
        if strategy == "conservative":
            n_tail = self.alpha.shape[0] - n_sensors
            return float(jnp.sum(terms[:n_tail]))
        raise ValueError(f"strategy must be incremental|conservative, got {strategy}")

    def crossover(self) -> int:
        """First budget at which the conservative bound becomes tighter.

        The constants in equations (22) and (23) are monotone in opposite
        directions. This method returns the first ``m`` for which the
        conservative bound is smaller than the incremental bound; if that
        never happens, it returns ``p``.
        """
        p = self.alpha.shape[0]
        gaps = [
            (m, self.suboptimality(m, "incremental") - self.suboptimality(m, "conservative"))
            for m in range(1, p)
        ]
        crossings = [m for m, d in gaps if d > 0]
        return crossings[0] if crossings else p

    @property
    def total_gap(self) -> float:
        """``gap(I_p) = ½ sum (ln alpha_i + ln beta_i)``.

        Oracle: must equal ``incremental_bounds(diagnostics, None).gap``,
        computed via ``slogdet`` calls that diagonalize nothing.
        """
        return float(0.5 * jnp.sum(jnp.log(self.alpha) + jnp.log(self.beta)))

    @property
    def effective_rank(self) -> int:
        """Minimal number of spectral contributions explaining 90% of the total gap.

        This is a spectral concentration indicator: a low value means the
        gap is dominated by a small number of modes.
        """
        terms = 0.5 * (jnp.log(self.alpha) + jnp.log(self.beta))
        total = jnp.sum(terms)
        if total <= 0:
            return 0
        return int(jnp.searchsorted(jnp.cumsum(terms) / total, 0.9) + 1)


@jax.jit
@jaxtyped(typechecker=beartype)
def generalized_eigenvalues(
    A: Float[Array, "n_obs n_obs"],
    B: Float[Array, "n_obs n_obs"],
) -> Float[Array, " n_obs"]:
    r"""Eigenvalues of ``A u = alpha B u``, decreasing. ``B`` SDP.

    Via Cholesky of ``B`` then ``eigvalsh`` of ``L^{-1} A L^{-T}``: JAX has no
    generalized ``eigh``, and forming ``B^{-1}A`` would destroy both the
    symmetry **and** the conditioning.
    """
    L = jsp.linalg.cho_factor(B, lower=True)[0]
    L = jnp.tril(L)
    X = jsp.linalg.solve_triangular(L, A, lower=True)
    C = jsp.linalg.solve_triangular(L, X.T, lower=True).T
    return jnp.flip(jnp.linalg.eigvalsh(0.5 * (C + C.T)))


@jax.jit
@jaxtyped(typechecker=beartype)
def quasi_optimality(diagnostics: DiagnosticMatrices) -> QuasiOptimality:
    """The spectrum of the gap -- Prop. 1.

    Notes
    -----
    ⚠️ ``eigvalsh`` on dense ``p x p`` matrices: this is a **diagnostic**, not
    a production path. The bounds and the greedy algorithm never need it --
    they go through Cholesky and Schur complements instead.
    """
    return QuasiOptimality(
        alpha=generalized_eigenvalues(diagnostics.Sigma_Y, diagnostics.Sigma_signal),
        beta=generalized_eigenvalues(diagnostics.Sigma_Y_given_theta, diagnostics.Sigma_noise),
    )
