from dataclasses import dataclass

from jax import Array
from jaxtyping import Float


@dataclass(frozen=True)
class DiagnosticMatrices:
    r"""``Sigma_Y``, ``Sigma_Y_given_theta``, ``Sigma_signal``, ``Sigma_noise``.

    Elles proviennent de **deux sources distinctes**, jamais d'une seule :

    ===================  ==========================  ====================
    Matrices             Voie                        Alternative
    ===================  ==========================  ====================
    ``Sigma_Y``,         §3.1 sample-based (26)(27)  aucune
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
        Borne sur l'information de Fisher : ``Sigma_signal^{-1} ⪰ I_Y``.
    Sigma_noise : Float[Array, "n_obs n_obs"]
        ``Sigma_noise^{-1} ⪰ E[I_{Y|theta}]``.
    certified : bool
        L'ordre de Loewner exigé par le théorème 2.1 est-il **garanti** ?

        ``True`` pour la voie gradient (Prop. 4). ``False`` pour la voie
        approximation (§3.2), dont l'inégalité part dans le **mauvais sens**
        (``(Sigma^{(N,F)}_signal)^{-1} ⪯ I_Y``) : consistante quand ``N → ∞`` et
        ``F → L²``, mais sans garantie à ``N`` fini. Le papier est explicite --
        elle *cannot be safely used in Theorem 2.1*.

    Notes
    -----
    ⚠️ ``certified`` n'est pas décoratif. Une borne certifiée doit **refuser** un
    diagnostic non certifié, ou dégrader son propre résultat -- jamais se taire.
    C'est le seul moyen honnête de faire coexister deux voies dont une seule
    certifie.

    ⚠️ À ne pas confondre avec ``BoundResult.is_certified`` (« le gap est-il sous
    la tolérance ? »). Deux notions utiles, distinctes.
    """

    Sigma_Y: Float[Array, "n_obs n_obs"]
    Sigma_Y_given_theta: Float[Array, "n_obs n_obs"]
    Sigma_signal: Float[Array, "n_obs n_obs"]
    Sigma_noise: Float[Array, "n_obs n_obs"]
    certified: bool
