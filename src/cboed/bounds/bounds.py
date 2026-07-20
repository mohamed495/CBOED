r"""Les deux familles de bornes -- corollaires 1 et 2.

⚠️ **Le piège de direction.** Le *même* couple ``(A, B)`` donne des bornes
**opposées** selon la stratégie :

===============================  ==========================  ==================
                                 ``(signal, Y|theta)``       ``(Y, noise)``
===============================  ==========================  ==================
**Incrémental** (Cor. 1)         borne **INF** (15)          borne **SUP** (16)
**Conservatif** (Cor. 2)         borne **SUP** (18)          borne **INF** (17)
===============================  ==========================  ==================

D'où des noms qui encodent les deux axes. Jamais ``BS``/``BI`` : le prototype les
utilise et le notebook goal-oriented écrit déjà « upper bound (LB) » puis
« Upper bound (UB) » à deux cellules d'écart.

**Régimes** (Prop. 1) : la constante de sous-optimalité **croît avec ``m``** en
incrémental, **décroît** en conservatif. Petit budget -> incrémental ; grand budget
-> conservatif. Complémentaires, pas concurrentes.

**Dualité** (§2 du papier) : maximiser la borne inf incrémentale ≡ maximiser la
borne sup conservative. Et inversement.
"""

from dataclasses import dataclass

from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped

from cboed.bounds.base import DiagnosticMatrices
from cboed.bounds.schur import log_ratio


@dataclass(frozen=True)
class BoundResult:
    """Encadrement certifié de ``EIG(design)``."""

    lower: Float[Array, ""]
    upper: Float[Array, ""]
    certified: bool
    """L'ordre de Loewner du Thm 2.1 est-il garanti ? (hérité du diagnostic)"""

    @property
    def gap(self) -> Float[Array, ""]:
        """``upper - lower``.

        Mesure la **non-gaussianité** de ``Y`` et de ``Y|theta``, pas la
        non-linéarité de ``u`` : Rem. 2.2 dit que les bornes sont serrées ssi
        ``Sigma_signal = Sigma_Y`` et ``Sigma_noise = Sigma_{Y|theta}``, ce qui par
        Cramér-Rao force ``Y`` gaussien. ``lambda`` pilote la non-linéarité ; la
        non-gaussianité en est la conséquence, pas la définition.
        """
        return self.upper - self.lower

    def is_tight(self, tolerance: float) -> bool:
        """``gap < tolerance``. À ne pas confondre avec :attr:`certified`."""
        return bool(self.gap < tolerance)


@jaxtyped(typechecker=beartype)
def incremental_bounds(
    diagnostics: DiagnosticMatrices,
    design: Int[Array, " n_sensors"] | None = None,
) -> BoundResult:
    r"""Corollaire 1 -- équations (15)-(16).

    .. math::
        \tfrac12 \ln \frac{|W^T \Sigma_{\rm signal} W|}{|W^T \Sigma_{Y|\theta} W|}
        \;\le\; \mathrm{EIG}(W) \;\le\;
        \tfrac12 \ln \frac{|W^T \Sigma_Y W|}{|W^T \Sigma_{\rm noise} W|}

    Compare l'information tirée de ``Y_m = W^T Y`` à celle d'**aucune**
    observation (``EIG(∅) = 0``). Entièrement calculable, aucun terme inconnu.
    """
    return BoundResult(
        lower=log_ratio(diagnostics.Sigma_signal, diagnostics.Sigma_Y_given_theta, design),
        upper=log_ratio(diagnostics.Sigma_Y, diagnostics.Sigma_noise, design),
        certified=diagnostics.certified,
    )


@jaxtyped(typechecker=beartype)
def conservative_bounds(
    diagnostics: DiagnosticMatrices,
    design: Int[Array, " n_sensors"] | None = None,
    eig_full: Float[Array, ""] | None = None,
) -> BoundResult:
    r"""Corollaire 2 -- équations (17)-(18).

    Compare l'information tirée de ``Y_m`` à celle du dataset **complet**
    ``EIG(I_p)``, d'où « conservatif » : la stratégie cherche à retenir un maximum
    de l'information disponible.

    Parameters
    ----------
    diagnostics : DiagnosticMatrices
    design : Int[Array, " n_sensors"] | None
    eig_full : Float[Array, ""] | None
        ``EIG(I_p)``, **inconnu** en pratique. ``None`` (défaut) -> il est encadré
        par le corollaire 1 appliqué à ``W = I_p``, ce qui garde les bornes
        entièrement calculables **et certifiées**. Voir Notes.

    Notes
    -----
    ⚠️ **Pourquoi ne pas estimer ``eig_full`` par Monte-Carlo.** Le prototype
    NumPy injecte un ``eig_offset`` estimé par MC dans (17)-(18) -- ce qui
    **décertifie** la borne : un encadrement garanti additionné d'une estimation
    bruitée n'est plus un encadrement.

    Inutile de l'estimer. Cor. 1 à ``W = I_p`` borne ``EIG(I_p)`` gratuitement :

    .. math::
        \tfrac12 \ln \tfrac{|\Sigma_{\rm signal}|}{|\Sigma_{Y|\theta}|}
        \;\le\; \mathrm{EIG}(I_p) \;\le\;
        \tfrac12 \ln \tfrac{|\Sigma_Y|}{|\Sigma_{\rm noise}|}

    En substituant la borne INF dans (17) et la borne SUP dans (18), les bornes
    conservatives restent valides. Plus lâches que si ``EIG(I_p)`` était connu --
    mais certifiées, ce qui est le point du module.

    ``eig_full`` reste acceptée pour comparer avec l'approche du prototype.
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
