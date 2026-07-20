r"""Matrices diagnostiques par gradients -- Proposition 4, §3.3.

Le module sépare **les moments** (la partie chère : `N` jacobiennes du modèle direct)
de **l'assemblage** (algèbre linéaire sur des matrices `q x q`). Deux voies
d'assemblage existent pour la même quantité -- elles sont oracles l'une de l'autre.

Moments (31)-(34)
-----------------
    L(u)  = E[Jac u(eta)]^T                                              (q, p)
    H(u)  = E[(Jac u - E[Jac u])^T Sigma_obs^{-1} (Jac u - E[Jac u])]    (q, q)
    J(h)  = E[Jac h(eta)^T Sigma_xi^{-1} Jac h(eta)]                     (q, q)
    I_eta = Cov(grad_eta ln pi(eta))                                     (q, q)

Assemblage (35)-(36)
--------------------
    Sigma_signal = Sigma_obs + L^T (H + I_eta)^{-1} L
    Sigma_noise  = Sigma_obs + L^T (H + I_eta + J(h))^{-1} L

`J(h)` est la seule différence entre les deux.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.priors.base import Prior


@jaxtyped(typechecker=beartype)
def psd_sqrt(A: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Racine PSD par ``eigh``, valeurs propres écrêtées à zéro.

    Pas ``cholesky`` : les matrices concernées peuvent être singulières (covariance
    postérieure dégénérée, ``Sigma_xi = 0``), et LAPACK y rend des ``nan``.
    """
    ev, P = jnp.linalg.eigh(0.5 * (A + A.T))
    return (P * jnp.sqrt(jnp.clip(ev, 0.0))) @ P.T


@jaxtyped(typechecker=beartype)
def _expected_quadratic(
    Jacs: Float[Array, "n_samples a b"],
    chol: tuple[Float[Array, "a a"], bool],
) -> Float[Array, "b b"]:
    """``E[J^T M^{-1} J]`` sur l'échantillon, ``M`` donnée par sa factorisation."""

    def quad(J: Float[Array, "a b"]) -> Float[Array, "b b"]:
        return J.T @ jsp.linalg.cho_solve(chol, J)

    out = jnp.mean(jax.vmap(quad)(Jacs), axis=0)
    return 0.5 * (out + out.T)


# =============================================================================
# Moments -- la partie chere : N jacobiennes du modele direct
# =============================================================================


@jaxtyped(typechecker=beartype)
def expected_jacobian_moments(
    u,
    etas: Float[Array, "n_samples n_eta"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> tuple[Float[Array, "n_eta n_obs"], Float[Array, "n_eta n_eta"]]:
    r"""``(L(u), H(u))`` -- équations (31)-(32).

    Parameters
    ----------
    u : Callable
        Modèle direct ``eta -> observations``, sans design.
    etas : Float[Array, "n_samples n_eta"]
        Tirages du prior.
    Sigma_obs : Float[Array, "n_obs n_obs"]

    Notes
    -----
    ``H(u)`` est calculée en **deux passes** (moyenne, puis quadratique centrée) et
    non par ``E[J^T S^{-1} J] - Jbar^T S^{-1} Jbar``. Les deux sont algébriquement
    égales, mais la seconde perd toute précision quand la moyenne domine la variance
    -- notamment quand la jacobienne est constante, où ``H(u)`` doit valoir zéro
    **exactement**.

    Mémoire ``O(N p q)`` : les jacobiennes sont matérialisées.
    """
    Jacs = jax.vmap(jax.jacfwd(u))(etas)
    J_bar = jnp.mean(Jacs, axis=0)
    chol_obs = jsp.linalg.cho_factor(Sigma_obs, lower=True)
    return J_bar.T, _expected_quadratic(Jacs - J_bar, chol_obs)


@jaxtyped(typechecker=beartype)
def qoi_fisher_moment(
    h,
    etas: Float[Array, "n_samples n_eta"],
    Sigma_xi: Float[Array, "n_param n_param"],
) -> Float[Array, "n_eta n_eta"]:
    r"""``J(h) = E[Jac h^T Sigma_xi^{-1} Jac h]`` -- équation (33).

    Bon marché : ``h`` est explicite, aucun modèle direct à évaluer.

    ``Sigma_xi`` doit être strictement définie positive. Quand ``Sigma_xi -> 0``,
    ``J(h) -> inf`` : le cas ``xi = 0`` n'est pas atteignable ici, il relève de
    :func:`gradient_diagnostics_standard`.
    """
    Jacs_h = jax.vmap(jax.jacfwd(h))(etas)
    chol_xi = jsp.linalg.cho_factor(Sigma_xi, lower=True)
    return _expected_quadratic(Jacs_h, chol_xi)


@jaxtyped(typechecker=beartype)
def fisher_information_prior(prior_eta: Prior) -> Float[Array, "n_eta n_eta"]:
    r"""``I_eta = Cov(grad log pi(eta))`` -- équation (34), cas gaussien.

    Pour un prior gaussien, ``grad log pi = -Gamma^{-1}(eta - m)``, donc
    ``Cov(grad log pi) = Gamma^{-1} Gamma Gamma^{-1} = Gamma^{-1}``. Exact, sans
    échantillonnage.

    Voir :func:`fisher_information_prior_mc` pour un prior quelconque.
    """
    q = prior_eta.mu.shape[0]
    return prior_eta.prior_precision_matmul(jnp.eye(q, dtype=prior_eta.mu.dtype))


@jaxtyped(typechecker=beartype)
def fisher_information_prior_mc(
    prior_eta: Prior, key: PRNGKeyArray, n_samples: int
) -> Float[Array, "n_eta n_eta"]:
    """``I_eta`` par covariance empirique des scores. Valable pour tout prior.

    Oracle de :func:`fisher_information_prior` : leur accord prouve que l'hypothèse
    gaussienne tient.
    """
    scores = jax.vmap(prior_eta.grad_log_prior)(prior_eta.sample(key, n_samples))
    centered = scores - jnp.mean(scores, axis=0)
    out = centered.T @ centered / (n_samples - 1)
    return 0.5 * (out + out.T)


# =============================================================================
# Assemblage -- bon marche : algebre lineaire q x q
# =============================================================================


@jaxtyped(typechecker=beartype)
def assemble(
    L: Float[Array, "n_eta n_obs"],
    A: Float[Array, "n_eta n_eta"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma_obs + L^T A^{-1} L``, par ``cho_solve``. ``A`` SDP.

    Forme commune à (35) et (36) : seule ``A`` change.
    ``A = H + I_eta`` -> ``Sigma_signal``. ``A = H + I_eta + J(h)`` -> ``Sigma_noise``.
    """
    chol = jsp.linalg.cho_factor(A, lower=True)
    out = Sigma_obs + L.T @ jsp.linalg.cho_solve(chol, L)
    return 0.5 * (out + out.T)


@jaxtyped(typechecker=beartype)
def assemble_misfit(
    L: Float[Array, "n_eta n_obs"],
    H: Float[Array, "n_eta n_eta"],
    Sigma_eta: Float[Array, "n_eta n_eta"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
    extra: Float[Array, "n_eta n_eta"] | None = None,
) -> Float[Array, "n_obs n_obs"]:
    r"""Idem, **préconditionné par le prior** -- sans former ``Sigma_eta^{-1}``.

    .. math::
        (H + \Sigma_\eta^{-1})^{-1}
        = \Sigma_\eta^{1/2}(I + \Sigma_\eta^{1/2} H \Sigma_\eta^{1/2})^{-1}
          \Sigma_\eta^{1/2}

    Chemin **indépendant** de :func:`assemble` avec ``A = H + I_eta`` : oracle.

    Parameters
    ----------
    extra : Float[Array, "n_eta n_eta"] | None
        Terme additionnel dans ``A`` (``J(h)`` pour ``Sigma_noise``). L'identité de
        préconditionnement ne s'applique qu'à ``Sigma_eta^{-1}`` : ``extra`` est donc
        absorbé dans le Hessien, ``H + extra`` jouant le rôle de ``H``.

    Notes
    -----
    N'utilise que ``Sigma_eta^{1/2}``, jamais ``Sigma_eta^{-1}``. C'est ce qui
    survivra quand la précision du prior ne sera plus formable.

    ⚠️ ``A_mis`` **est déjà l'inverse** : le terme est ``A_mis @ L``, pas
    ``solve(A_mis, L)``. Le prototype NumPy faisait le second -- erreur relative de
    248 %, et le résultat restait SDP, donc silencieux.
    """
    Hx = H if extra is None else H + extra
    S_sqrt = psd_sqrt(Sigma_eta)
    H_mis = S_sqrt @ Hx @ S_sqrt
    n = H_mis.shape[0]
    A_inv = S_sqrt @ jnp.linalg.solve(jnp.eye(n, dtype=H_mis.dtype) + H_mis, S_sqrt)
    out = Sigma_obs + L.T @ (A_inv @ L)
    return 0.5 * (out + out.T)


# =============================================================================
# Orchestration
# =============================================================================


@jaxtyped(typechecker=beartype)
def gradient_diagnostics(
    u,
    h,
    prior_eta: Prior,
    Sigma_obs: Float[Array, "n_obs n_obs"],
    Sigma_xi: Float[Array, "n_param n_param"],
    key: PRNGKeyArray,
    n_samples: int,
) -> tuple[Float[Array, "n_obs n_obs"], Float[Array, "n_obs n_obs"]]:
    r"""``(Sigma_signal, Sigma_noise)`` en cadre goal-oriented -- Prop. 4.

    Vérifient ``Sigma_signal^{-1} ⪰ I_Y`` et ``Sigma_noise^{-1} ⪰ E[I_{Y|theta}]``.

    Rend **deux des quatre** matrices : ``Sigma_Y`` et ``Sigma_Y_given_theta``
    viennent de §3.1, sans alternative.
    """
    etas = prior_eta.sample(key, n_samples)
    L, H = expected_jacobian_moments(u, etas, Sigma_obs)
    J_h = qoi_fisher_moment(h, etas, Sigma_xi)
    I_eta = fisher_information_prior(prior_eta)
    return assemble(L, H + I_eta, Sigma_obs), assemble(L, H + I_eta + J_h, Sigma_obs)


@jaxtyped(typechecker=beartype)
def gradient_diagnostics_standard(
    u,
    prior_theta: Prior,
    Sigma_obs: Float[Array, "n_obs n_obs"],
    key: PRNGKeyArray,
    n_samples: int,
) -> tuple[Float[Array, "n_obs n_obs"], Float[Array, "n_obs n_obs"]]:
    r"""``(Sigma_signal, Sigma_noise)`` en cadre standard ``Y = u(theta) + eps``.

    Prop. 2 donne ``E[I_{Y|theta}] = Sigma_obs^{-1}``, donc
    ``Sigma_noise = Sigma_obs`` **exactement** -- posé, pas calculé.

    Ce n'est **pas** :func:`gradient_diagnostics` avec ``h = id`` et ``Sigma_xi``
    minuscule : cette limite est singulière (``J(h) = Sigma_xi^{-1} -> inf``), il
    reste ``O(Sigma_xi)``, et l'encadrement s'inverse.
    """
    thetas = prior_theta.sample(key, n_samples)
    L, H = expected_jacobian_moments(u, thetas, Sigma_obs)
    I_theta = fisher_information_prior(prior_theta)
    return assemble(L, H + I_theta, Sigma_obs), Sigma_obs
