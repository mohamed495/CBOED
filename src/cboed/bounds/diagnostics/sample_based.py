r"""Matrices diagnostiques par échantillonnage -- §3.1.

Fournit ``Sigma_Y`` et ``Sigma_{Y|theta}``. **Sans alternative** : contrairement à
``Sigma_signal``/``Sigma_noise``, qui ont deux voies (§3.2 et §3.3), ces deux-là ne
s'obtiennent que par échantillonnage.

Estimateur par différences appariées
------------------------------------
Plutôt que la covariance empirique usuelle, on exploite
``Cov(u(eta)) = ½ E[(u(eta) - u(eta'))^⊗2]`` pour ``eta'`` copie indépendante :

.. math::
    \Sigma_Y^{(N)} = \Sigma_{\rm obs}
    + \frac{1}{2N} \sum_{i=1}^N (u(\eta^{(i)}) - u(\eta'^{(i)}))^{\otimes 2}

Non biaisé, et sans moyenne empirique à retrancher -- donc sans la cancellation qui
guette ``E[X X^T] - \bar{X}\bar{X}^T``. Coût : ``2N`` évaluations du modèle direct.

``Sigma_{Y|theta}`` suit la même forme avec ``eta, eta'`` **conditionnellement
indépendants sachant theta** (27) : ``eta ~ pi_eta``, ``theta ~ pi_{theta|eta}``,
``eta' ~ pi_{eta|theta}``. Le dernier tirage est le goulot du papier -- il demande en
général du MCMC. La remarque 3.1 le donne en forme fermée quand ``h`` est linéaire.

Notes
-----
⚠️ **Pick-freeze (Rem. 3.2) ne s'applique pas au banc.** Il exige ``eta_1 ⊥ eta_2`` ;
les deux moitiés du champ viennent du même GP, donc ``Sigma_{theta eta} ≠ 0``. C'est
la remarque 3.1 qui s'applique.

⚠️ Contrairement à Prop. 4, cette voie est **régulière en ``Sigma_xi = 0``** : la
remarque 3.1 reste valide sans bruit (``eta|theta`` dégénère proprement en ``delta``).
Là où la voie gradient a une limite singulière, la voie échantillon n'en a pas.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.priors.base import Prior


@jaxtyped(typechecker=beartype)
def _paired_covariance(
    diffs: Float[Array, "n_samples n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``(1/2N) sum_i d_i d_i^T`` -- l'estimateur (26)/(27)."""
    out = 0.5 * diffs.T @ diffs / diffs.shape[0]
    return 0.5 * (out + out.T)


@jaxtyped(typechecker=beartype)
def _psd_sqrt(A: Float[Array, "n n"]) -> Float[Array, "n n"]:
    r"""Racine PSD par ``eigh``, valeurs propres écrêtées à zéro.

    Pas ``cholesky`` : la covariance postérieure de la remarque 3.1 **dégénère** en
    zéro quand ``Sigma_xi -> 0`` et ``B = I`` (``eta|theta`` devient un Dirac). C'est
    un cas nominal, pas un accident -- et LAPACK rend des ``nan`` sur une PSD
    singulière.
    """
    ev, P = jnp.linalg.eigh(A)
    return P @ jnp.diag(jnp.sqrt(jnp.clip(ev, 0.0, None)))


@jaxtyped(typechecker=beartype)
def sample_Sigma_Y(
    u,
    prior_eta: Prior,
    Sigma_obs: Float[Array, "n_obs n_obs"],
    key: PRNGKeyArray,
    n_samples: int,
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma_Y = Sigma_obs + Cov(u(eta))`` -- équation (26).

    Parameters
    ----------
    u : Callable
        Modèle direct ``eta -> observations``, sans design.
    prior_eta : Prior
    Sigma_obs : Float[Array, "n_obs n_obs"]
    key : PRNGKeyArray
    n_samples : int
        Nombre de **paires**. Coût : ``2 * n_samples`` évaluations de ``u``.
    """
    k1, k2 = jax.random.split(key)
    eta = prior_eta.sample(k1, n_samples)
    eta_prime = prior_eta.sample(k2, n_samples)
    diffs = jax.vmap(u)(eta) - jax.vmap(u)(eta_prime)
    return Sigma_obs + _paired_covariance(diffs)


@jaxtyped(typechecker=beartype)
def sample_Sigma_Y_given_theta(
    u,
    prior_eta: Prior,
    B: Float[Array, "n_param n_eta"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
    Sigma_xi: Float[Array, "n_param n_param"],
    key: PRNGKeyArray,
    n_samples: int,
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma_{Y|theta} = Sigma_obs + E[Cov(u(eta)|theta)]`` -- (27) via Rem. 3.1.

    Parameters
    ----------
    B : Float[Array, "n_param n_eta"]
        Jacobienne de ``h``, **supposée constante** : la forme fermée de la remarque
        3.1 n'existe que pour ``h`` linéaire. Pour ``h`` non linéaire, il faut du
        MCMC ciblant ``pi_{eta|theta}`` -- hors périmètre.
    Sigma_xi : Float[Array, "n_param n_param"]
        Covariance de ``xi``. **Peut être nulle** : cf. Rem. 3.1.

    Notes
    -----
    Le gain de Kalman ``K = Sigma_eta B^T (B Sigma_eta B^T + Sigma_xi)^{-1}`` et la
    covariance postérieure ``Sigma_pos = Sigma_eta - K B Sigma_eta`` **ne dépendent
    pas de theta** : factorisés une fois, hors de la boucle. Seule la moyenne en
    dépend.
    """
    k_eta, k_xi, k_pos = jax.random.split(key, 3)

    Sigma_eta = prior_eta.Sigma()
    m_eta = prior_eta.mu

    # -- Rem. 3.1 : eta|theta gaussien, covariance independante de theta ----
    S = B @ Sigma_eta @ B.T + Sigma_xi
    K = jnp.linalg.solve(S, B @ Sigma_eta).T  # (q, d)
    Sigma_pos = Sigma_eta - K @ B @ Sigma_eta
    L_pos = _psd_sqrt(0.5 * (Sigma_pos + Sigma_pos.T))

    # -- eta ~ pi_eta, theta ~ pi_{theta|eta}, eta' ~ pi_{eta|theta} --------
    eta = prior_eta.sample(k_eta, n_samples)
    L_xi = _psd_sqrt(Sigma_xi)
    z_xi = jax.random.normal(k_xi, (n_samples, Sigma_xi.shape[0]))
    theta = eta @ B.T + z_xi @ L_xi.T

    z_pos = jax.random.normal(k_pos, (n_samples, m_eta.shape[0]))
    eta_prime = m_eta + (theta - m_eta @ B.T) @ K.T + z_pos @ L_pos.T

    diffs = jax.vmap(u)(eta) - jax.vmap(u)(eta_prime)
    return Sigma_obs + _paired_covariance(diffs)


@jaxtyped(typechecker=beartype)
def sample_diagnostics_standard(
    u,
    prior_theta: Prior,
    Sigma_obs: Float[Array, "n_obs n_obs"],
    key: PRNGKeyArray,
    n_samples: int,
) -> tuple[Float[Array, "n_obs n_obs"], Float[Array, "n_obs n_obs"]]:
    r"""``(Sigma_Y, Sigma_Y_given_theta)`` dans le cas standard ``Y = u(theta) + eps``.

    Prop. 2 avec ``h = id`` et ``xi = 0`` donne ``E[Cov(u(theta)|theta)] = 0``, donc
    ``Sigma_Y_given_theta = Sigma_obs`` **exactement**. Aucun échantillonnage : c'est
    une égalité à poser, pas une limite à approcher.

    Symétrique de ``gradient_diagnostics_standard``, et pour la même raison.
    """
    return sample_Sigma_Y(u, prior_theta, Sigma_obs, key, n_samples), Sigma_obs
