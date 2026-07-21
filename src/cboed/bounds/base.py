from dataclasses import dataclass, field

import jax
from jax import Array
from jaxtyping import Float


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiagnosticMatrices:
    r"""``Sigma_Y``, ``Sigma_Y_given_theta``, ``Sigma_signal``, ``Sigma_noise``.

    They come from **two distinct sources**, never a single one:

    ===================  ==========================  ====================
    Matrices             Route                       Alternative
    ===================  ==========================  ====================
    ``Sigma_Y``,         §3.1 sample-based (26)(27)  none
    ``Sigma_Y_given_theta``
    ``Sigma_signal``,    §3.3 gradient, Prop. 4      §3.2 approximation
    ``Sigma_noise``
    ===================  ==========================  ====================

    Attributes
    ----------
    Sigma_Y : Float[Array, "n_obs n_obs"]
        ``Sigma_obs + Cov(u(eta))``.
    Sigma_Y_given_theta : Float[Array, "n_obs n_obs"]
        ``Sigma_obs + E[Cov(u(eta)|theta)]``.
    Sigma_signal : Float[Array, "n_obs n_obs"]
        Bound on the Fisher information: ``Sigma_signal^{-1} ⪰ I_Y``.
    Sigma_noise : Float[Array, "n_obs n_obs"]
        ``Sigma_noise^{-1} ⪰ E[I_{Y|theta}]``.
    certified : bool
        Is the Loewner order required by Theorem 2.1 **guaranteed**?

        ``True`` for the gradient route (Prop. 4). ``False`` for the
        approximation route (§3.2), whose inequality points the **wrong way**
        (``(Sigma^{(N,F)}_signal)^{-1} ⪯ I_Y``): consistent as ``N → ∞`` and
        ``F → L²``, but with no guarantee at finite ``N``. The paper is
        explicit -- it *cannot be safely used in Theorem 2.1*.

    Notes
    -----
    ⚠️ ``certified`` is not decorative. A certified bound must **refuse** an
    uncertified diagnostic, or degrade its own result -- never stay silent.
    That is the only honest way to let two routes coexist when only one of
    them certifies.

    ⚠️ Not to be confused with ``BoundResult.is_certified`` ("is the gap below
    the tolerance?"). Two useful, distinct notions.
    """

    Sigma_Y: Float[Array, "n_obs n_obs"]
    Sigma_Y_given_theta: Float[Array, "n_obs n_obs"]
    Sigma_signal: Float[Array, "n_obs n_obs"]
    Sigma_noise: Float[Array, "n_obs n_obs"]
    certified: bool = field(metadata=dict(static=True))
