r"""Matrices diagnostiques par approximation -- §3.2.

Troisième voie vers ``Sigma_signal`` et ``Sigma_noise``, à côté de :mod:`sample_based`
(§3.1) et :mod:`gradient_based` (§3.3). Les trois produisent les **mêmes** matrices ;
elles diffèrent par le coût et par ce qu'elles garantissent.

Prop. 3 :

.. math::
    \Sigma^{(N,F)}_{\rm signal} =
    \left(\Sigma_{\rm obs}^{-1} - \Sigma_{\rm obs}^{-1} R_f \Sigma_{\rm obs}^{-1}\right)^{-1},
    \qquad R_f = \frac1N \sum_i (u(\eta^{(i)}) - f(Y^{(i)}))^{\otimes 2}

où ``f`` est un débruiteur (:mod:`cboed.bounds.diagnostics.denoisers`).

* **Coût marginal nul.** ``f`` est appris sur les paires ``(eta, Y)`` déjà tirées pour
  ``Sigma_Y``. Aucune jacobienne, aucun MCMC.
* **Estimateur, pas borne.** ``R_f`` majore ``E[Cov(u|Y)]`` : bien défini (Prop. 3) mais
  pas garanti au sens du Thm 2.1.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped

from cboed.bounds.diagnostics.denoisers import Denoiser


@jax.jit
@jaxtyped(typechecker=beartype)
def denoiser_residual(
    denoiser: Denoiser,
    u_samples: Float[Array, "n_samples n_obs"],
    features: Float[Array, "n_samples n_feat"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``R_f = (1/N) sum (u - f(features))^{⊗2}``.

    Majore ``E[Cov(u|Y)]`` (égalité ssi ``f = E[u|Y]``) -- la raison pour laquelle
    §3.2 n'est pas certifié.
    """
    resid = u_samples - jax.vmap(denoiser)(features)
    out = resid.T @ resid / resid.shape[0]
    return 0.5 * (out + out.T)


@jaxtyped(typechecker=beartype)
def assemble_from_residual(
    R: Float[Array, "n_obs n_obs"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
    n_samples: int,
) -> Float[Array, "n_obs n_obs"]:
    r"""``(Sigma_obs^{-1} - Sigma_obs^{-1} R Sigma_obs^{-1})^{-1}`` -- Prop. 3.

    ⚠️ **§3.2 est le moins fiable dans le regime lineaire** -- l'inverse de §3.3.
    Quand le debruiteur est quasi exact (a ``lambda = 0``, l'affine l'est),
    ``R -> Sigma_obs`` par en dessous avec une marge de l'ordre de ``1e-6`` : la
    precision devient quasi singuliere. Le papier le dit : bien definie *avec haute
    probabilite pour N assez grand*. Mesure : a ``N = 1e4`` la marge est noyee dans
    le bruit MC (``lambda_min < 0``) ; a ``N = 1e5`` elle emerge. La voie
    approximation demande donc **beaucoup** d'echantillons pres du lineaire.

    La garde distingue le bruit MC (tolere, ecretage spectral) d'une violation
    franche (``ValueError``). Le seuil est ``~10 sigma_MC`` avec
    ``sigma_MC ~ trace(R)/n / sqrt(N)`` : il n'attrape que ce qui NE PEUT PAS venir
    de l'estimation.

    Raises
    ------
    ValueError
        Si ``R`` depasse ``Sigma_obs`` bien au-dela du bruit d'estimation -- le
        debruiteur *degrade* le residu au-dela du bruit d'observation.

    Notes
    -----
    ⚠️ Pas de ``jit`` : le ``if`` ci-dessous teste une valeur qui depend de ``R``
    (donc tracee sous jit), et ``ValueError`` exige un bool concret. Meme
    limitation pour :func:`approximation_signal`/:func:`approximation_noise`, qui
    appellent cette fonction.
    """
    n = R.shape[0]
    tol = 10.0 * jnp.trace(R) / n / jnp.sqrt(n_samples)
    gap_eig = jnp.min(jnp.linalg.eigvalsh(Sigma_obs - R))
    if gap_eig <= -tol:
        raise ValueError(
            f"R depasse franchement Sigma_obs "
            f"(lambda_min(Sigma_obs - R) = {float(gap_eig):.2e}, "
            f"tolerance MC = {float(-tol):.2e}). Le debruiteur degrade le residu -- "
            "§3.2 n'est pas applicable, ou l'entrainement a diverge."
        )
    chol = jsp.linalg.cho_factor(Sigma_obs, lower=True)
    inner = jsp.linalg.cho_solve(chol, R)
    inner = jsp.linalg.cho_solve(chol, inner.T).T
    prec = jnp.linalg.inv(Sigma_obs) - inner
    out = jnp.linalg.inv(0.5 * (prec + prec.T))
    return 0.5 * (out + out.T)


@jaxtyped(typechecker=beartype)
def approximation_signal(
    denoiser: Denoiser,
    u_samples: Float[Array, "n_samples n_obs"],
    Y_samples: Float[Array, "n_samples n_obs"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma^{(N,F)}_signal`` via un débruiteur ``f : Y -> u(eta)`` déjà entraîné."""
    R_f = denoiser_residual(denoiser, u_samples, Y_samples)
    return assemble_from_residual(R_f, Sigma_obs, u_samples.shape[0])


@jaxtyped(typechecker=beartype)
def approximation_noise(
    denoiser: Denoiser,
    u_samples: Float[Array, "n_samples n_obs"],
    Y_samples: Float[Array, "n_samples n_obs"],
    theta_samples: Float[Array, "n_samples n_param"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma^{(N,G)}_noise`` via ``g : (Y, theta) -> u(eta)`` déjà entraîné.

    ``g`` voit ``theta`` en plus de ``Y``, débruite mieux : ``R_g ⪯ R_f``, donc
    ``Sigma_noise ⪯ Sigma_signal`` -- l'écart est ``gap_h``.
    """
    features = jnp.concatenate([Y_samples, theta_samples], axis=1)
    R_g = denoiser_residual(denoiser, u_samples, features)
    return assemble_from_residual(R_g, Sigma_obs, u_samples.shape[0])
