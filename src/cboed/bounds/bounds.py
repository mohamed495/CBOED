r"""The two bound families -- Corollaries 1 and 2.

⚠️ **The direction trap.** The *same* pair ``(A, B)`` gives **opposite**
bounds depending on the strategy:

===============================  ==========================  ==================
                                 ``(signal, Y|theta)``       ``(Y, noise)``
===============================  ==========================  ==================
**Incremental** (Cor. 1)         **LOWER** bound (15)        **UPPER** bound (16)
**Conservative** (Cor. 2)        **UPPER** bound (18)        **LOWER** bound (17)
===============================  ==========================  ==================

Hence names that encode both axes. Never ``BS``/``BI``: the prototype uses
them, and the goal-oriented notebook already writes "upper bound (LB)" then
"Upper bound (UB)" two cells apart.

**Regimes** (Prop. 1): the suboptimality constant **grows with ``m``** in the
incremental strategy, **shrinks** in the conservative one. Small budget ->
incremental; large budget -> conservative. Complementary, not competing.

**Duality** (§2 of the paper): maximizing the incremental lower bound ≡
maximizing the conservative upper bound. And conversely.
"""

from dataclasses import dataclass, field

import jax
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped

from cboed.bounds.base import DiagnosticMatrices
from cboed.bounds.schur import log_ratio


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoundResult:
    """Certified enclosure of ``EIG(design)``."""

    lower: Float[Array, ""]
    upper: Float[Array, ""]
    certified: bool = field(metadata=dict(static=True))
    """Is the Loewner order of Thm 2.1 guaranteed? (inherited from the diagnostic)"""

    @property
    def gap(self) -> Float[Array, ""]:
        """``upper - lower``.

        Measures the **non-Gaussianity** of ``Y`` and ``Y|theta``, not the
        non-linearity of ``u``: Rem. 2.2 states that the bounds are tight iff
        ``Sigma_signal = Sigma_Y`` and ``Sigma_noise = Sigma_{Y|theta}``, which
        by Cramér-Rao forces ``Y`` Gaussian. ``lambda`` drives the
        non-linearity; the non-Gaussianity is a consequence of it, not its
        definition.
        """
        return self.upper - self.lower

    def is_tight(self, tolerance: float) -> bool:
        """``gap < tolerance``. Not to be confused with :attr:`certified`."""
        return bool(self.gap < tolerance)


@jax.jit
@jaxtyped(typechecker=beartype)
def incremental_bounds(
    diagnostics: DiagnosticMatrices,
    design: Int[Array, " n_sensors"] | None = None,
) -> BoundResult:
    r"""Corollary 1 -- equations (15)-(16).

    .. math::
        \tfrac12 \ln \frac{|W^T \Sigma_{\rm signal} W|}{|W^T \Sigma_{Y|\theta} W|}
        \;\le\; \mathrm{EIG}(W) \;\le\;
        \tfrac12 \ln \frac{|W^T \Sigma_Y W|}{|W^T \Sigma_{\rm noise} W|}

    Compares the information drawn from ``Y_m = W^T Y`` to that of **no**
    observation at all (``EIG(∅) = 0``). Fully computable, no unknown term.
    """
    return BoundResult(
        lower=log_ratio(diagnostics.Sigma_signal, diagnostics.Sigma_Y_given_theta, design),
        upper=log_ratio(diagnostics.Sigma_Y, diagnostics.Sigma_noise, design),
        certified=diagnostics.certified,
    )


@jax.jit
@jaxtyped(typechecker=beartype)
def conservative_bounds(
    diagnostics: DiagnosticMatrices,
    design: Int[Array, " n_sensors"] | None = None,
    eig_full: Float[Array, ""] | None = None,
) -> BoundResult:
    r"""Corollary 2 -- equations (17)-(18).

    Compares the information drawn from ``Y_m`` to that of the **full**
    dataset ``EIG(I_p)``, hence "conservative": the strategy tries to retain
    as much of the available information as possible.

    Parameters
    ----------
    diagnostics : DiagnosticMatrices
    design : Int[Array, " n_sensors"] | None
    eig_full : Float[Array, ""] | None
        ``EIG(I_p)``, **unknown** in practice. ``None`` (default) -> it is
        bounded by Corollary 1 applied to ``W = I_p``, which keeps the bounds
        fully computable **and** certified. See Notes.

    Notes
    -----
    ⚠️ **Why not estimate ``eig_full`` by Monte Carlo.** The NumPy prototype
    injects an MC-estimated ``eig_offset`` into (17)-(18) -- which
    **decertifies** the bound: a guaranteed enclosure plus a noisy estimate is
    no longer an enclosure.

    No need to estimate it. Cor. 1 at ``W = I_p`` bounds ``EIG(I_p)`` for
    free:

    .. math::
        \tfrac12 \ln \tfrac{|\Sigma_{\rm signal}|}{|\Sigma_{Y|\theta}|}
        \;\le\; \mathrm{EIG}(I_p) \;\le\;
        \tfrac12 \ln \tfrac{|\Sigma_Y|}{|\Sigma_{\rm noise}|}

    Substituting the LOWER bound into (17) and the UPPER bound into (18)
    keeps the conservative bounds valid. Looser than if ``EIG(I_p)`` were
    known -- but certified, which is the point of the module.

    ``eig_full`` is still accepted for comparison against the prototype's
    approach.
    """
    delta_lower = log_ratio(diagnostics.Sigma_Y, diagnostics.Sigma_noise, design) - log_ratio(
        diagnostics.Sigma_Y, diagnostics.Sigma_noise, None
    )

    delta_upper = log_ratio(
        diagnostics.Sigma_signal, diagnostics.Sigma_Y_given_theta, design
    ) - log_ratio(diagnostics.Sigma_signal, diagnostics.Sigma_Y_given_theta, None)

    if eig_full is None:
        full = incremental_bounds(diagnostics, None)
        base_lower, base_upper = full.lower, full.upper
    else:
        base_lower = base_upper = eig_full

    return BoundResult(
        lower=base_lower + delta_lower,
        upper=base_upper + delta_upper,
        certified=diagnostics.certified,
    )
