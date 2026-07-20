r"""Matrices diagnostiques par approximation -- §3.2.

Troisième voie vers ``Sigma_signal`` et ``Sigma_noise``, à côté de :mod:`sample_based`
(§3.1) et :mod:`gradient_based` (§3.3). Toutes trois produisent les **mêmes** deux
matrices ; elles diffèrent par le coût et par ce qu'elles garantissent.

Le principe (29)-(30)
---------------------
``E[u(eta)|Y]`` est la projection orthogonale de ``u(eta)`` sur ``L^2_{pi_Y}``. Donnée
une classe d'approximation ``F``, le **débruiteur**

.. math::
    f \in \arg\min_{f \in F} E[\|u(\eta) - f(Y)\|^2]

approche cette espérance conditionnelle : ``f(u(eta) + eps) ~ u(eta)``, il retire le
bruit. On en déduit (Prop. 3)

.. math::
    \Sigma^{(N,F)}_{\rm signal} =
    \left(\Sigma_{\rm obs}^{-1}
      - \Sigma_{\rm obs}^{-1} R_f \Sigma_{\rm obs}^{-1}\right)^{-1},
    \qquad
    R_f = \frac1N \sum_i (u(\eta^{(i)}) - f(Y^{(i)}))^{\otimes 2}

Coût et garantie
----------------
* **Coût marginal nul.** ``f`` est appris sur les paires ``(eta, Y)`` déjà tirées pour
  ``Sigma_Y`` (§3.1). Aucune jacobienne (contrairement à §3.3), aucun MCMC. C'est ce
  qui passe à l'échelle.
* **Estimateur, pas borne certifiée.** ``E[(u - f(Y))^{⊗2}] ⪰ E[Cov(u|Y)]`` (égalité
  ssi ``f = E[u|Y]``), donc ``R_f`` **majore** la covariance conditionnelle. La matrice
  est bien définie (Prop. 3, si ``F`` contient les affines) mais ne rentre pas dans le
  Thm 2.1 comme garantie -- elle l'approche.

Le débruiteur affine
--------------------
Prop. 3 exige que ``F`` contienne les fonctions affines. Le débruiteur **affine** est
donc le plancher : ``f(Y) = A Y + b`` avec ``A = Cov(u, Y) Cov(Y)^{-1}``, en forme
fermée sur les mêmes paires. À ``lambda = 0`` le modèle est linéaire, ``E[u|Y]`` est
exactement affine, et ``Sigma^{(F)}_signal`` **égale** la voie gradient -- c'est
l'oracle inter-modules.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped


@jax.jit
@jaxtyped(typechecker=beartype)
def affine_denoiser(
    u_samples: Float[Array, "n_samples n_obs"],
    features: Float[Array, "n_samples n_feat"],
) -> tuple[Float[Array, "n_obs n_feat"], Float[Array, " n_obs"]]:
    r"""Débruiteur affine ``f(Y) = A Y + b`` par moindres carrés -- forme fermée.

    .. math::
        A = \mathrm{Cov}(u, Y)\,\mathrm{Cov}(Y)^{-1}, \qquad b = E[u] - A\,E[Y]

    Parameters
    ----------
    u_samples : Float[Array, "n_samples n_obs"]
        Les ``u(eta^{(i)})`` -- la cible à débruiter.
    features : Float[Array, "n_samples n_feat"]
        Les entrées du débruiteur. ``Y`` pour ``f`` (n_feat = n_obs) ; ``(Y, theta)``
        concaténés pour ``g`` (n_feat = n_obs + n_param).

    Notes
    -----
    Régression sur les **mêmes** paires que ``Sigma_Y`` : coût marginal nul.

    ``Cov(Y)`` est régularisée par un jitter relatif : à ``sigma_obs`` petit et ``N``
    modeste, la covariance empirique des features peut être quasi singulière.
    """
    n = u_samples.shape[0]
    u_bar, f_bar = u_samples.mean(0), features.mean(0)
    U, Fc = u_samples - u_bar, features - f_bar

    cov_uf = U.T @ Fc / n
    cov_ff = Fc.T @ Fc / n
    jitter = 1e-8 * jnp.trace(cov_ff) / cov_ff.shape[0]
    cov_ff = cov_ff + jitter * jnp.eye(cov_ff.shape[0])

    A = jsp.linalg.solve(cov_ff, cov_uf.T, assume_a="pos").T
    return A, u_bar - A @ f_bar


@jax.jit
@jaxtyped(typechecker=beartype)
def denoiser_residual(
    u_samples: Float[Array, "n_samples n_obs"],
    features: Float[Array, "n_samples n_feat"],
    A: Float[Array, "n_obs n_feat"],
    b: Float[Array, " n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``R_f = (1/N) sum (u - f(Y))^{⊗2}`` -- le résidu du débruiteur.

    ``R_f`` **majore** ``E[Cov(u|Y)]`` (égalité ssi ``f = E[u|Y]``). C'est ce qui rend
    §3.2 non certifié : la matrice est bien définie mais l'ordre de Loewner du Thm 2.1
    n'est pas garanti.
    """
    resid = u_samples - (features @ A.T + b)
    out = resid.T @ resid / resid.shape[0]
    return 0.5 * (out + out.T)


@jax.jit
@jaxtyped(typechecker=beartype)
def _assemble_from_residual(
    R: Float[Array, "n_obs n_obs"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``(Sigma_obs^{-1} - Sigma_obs^{-1} R Sigma_obs^{-1})^{-1}`` -- Prop. 3.

    ⚠️ Bien définie **ssi** ``R ≺ Sigma_obs`` (Prop. 3, garanti si ``F`` contient les
    affines et ``N`` assez grand). Sinon la parenthèse cesse d'être SDP et l'inverse
    n'existe pas -- c'est le prix de l'absence de certification.
    """
    chol = jsp.linalg.cho_factor(Sigma_obs, lower=True)
    inner = jsp.linalg.cho_solve(chol, R)
    inner = jsp.linalg.cho_solve(chol, inner.T).T
    prec = jnp.linalg.inv(Sigma_obs) - inner
    out = jnp.linalg.inv(0.5 * (prec + prec.T))
    return 0.5 * (out + out.T)


@jax.jit
@jaxtyped(typechecker=beartype)
def approximation_signal(
    u_samples: Float[Array, "n_samples n_obs"],
    Y_samples: Float[Array, "n_samples n_obs"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma^{(N,F)}_signal`` par débruiteur affine ``f : Y -> u(eta)``.

    Parameters
    ----------
    u_samples, Y_samples : Float[Array, "n_samples n_obs"]
        Paires ``(u(eta^{(i)}), Y^{(i)})`` -- **celles déjà tirées pour ``Sigma_Y``**.
    Sigma_obs : Float[Array, "n_obs n_obs"]
    """
    A, b = affine_denoiser(u_samples, Y_samples)
    R_f = denoiser_residual(u_samples, Y_samples, A, b)
    return _assemble_from_residual(R_f, Sigma_obs)


@jax.jit
@jaxtyped(typechecker=beartype)
def approximation_noise(
    u_samples: Float[Array, "n_samples n_obs"],
    Y_samples: Float[Array, "n_samples n_obs"],
    theta_samples: Float[Array, "n_samples n_param"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma^{(N,G)}_noise`` par débruiteur affine ``g : (Y, theta) -> u(eta)``.

    ``g`` voit ``theta`` en plus de ``Y`` : il débruite mieux, donc ``R_g ⪯ R_f``, donc
    ``Sigma_noise ⪯ Sigma_signal`` -- l'écart **est** ``gap_h``.
    """
    features = jnp.concatenate([Y_samples, theta_samples], axis=1)
    A, b = affine_denoiser(u_samples, features)
    R_g = denoiser_residual(u_samples, features, A, b)
    return _assemble_from_residual(R_g, Sigma_obs)
