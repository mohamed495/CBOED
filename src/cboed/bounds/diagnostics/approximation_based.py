r"""Compute approximation-based diagnostic matrices -- §3.2.

Third route to ``Sigma_signal`` and ``Sigma_noise``, alongside :mod:`sample_based`
(§3.1) and :mod:`gradient_based` (§3.3). All three produce the **same** matrices;
they differ in cost and in what they guarantee.

Prop. 3:

.. math::
    \Sigma^{(N,F)}_{\rm signal} =
    \left(\Sigma_{\rm obs}^{-1} - \Sigma_{\rm obs}^{-1} R_f \Sigma_{\rm obs}^{-1}\right)^{-1},
    \qquad R_f = \frac1N \sum_i (u(\eta^{(i)}) - f(Y^{(i)}))^{\otimes 2}

where ``f`` is a denoiser (:mod:`cboed.bounds.diagnostics.denoisers`).

* **Zero marginal cost.** ``f`` is learned on the ``(eta, Y)`` pairs already
  drawn for ``Sigma_Y``. No Jacobian, no MCMC.
* **Estimator, not a bound.** ``R_f`` upper-bounds ``E[Cov(u|Y)]``: well
  defined (Prop. 3) but not guaranteed in the sense of Thm 2.1.
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
    r"""Compute the denoiser's empirical residual covariance ``R_f``.

    .. math::
        R_f = \frac1N \sum_i (u^{(i)} - f(\mathrm{features}^{(i)}))^{\otimes 2}

    Parameters
    ----------
    denoiser : Denoiser
        Trained approximation ``f`` of ``E[u|features]``.
    u_samples : Float[Array, "n_samples n_obs"]
        The ``u(eta^{(i)})`` -- the target to denoise.
    features : Float[Array, "n_samples n_feat"]
        The denoiser's inputs (``Y`` for ``f``, ``(Y, theta)`` concatenated
        for ``g``).

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        ``R_f``, symmetrized.

    Notes
    -----
    ``R_f`` **upper-bounds** ``E[Cov(u|Y)]`` (equality iff ``f = E[u|Y]``) --
    the reason §3.2 is not certified.
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
    r"""Assemble ``Sigma^{(N,F)}_signal`` (or ``_noise``) from a residual -- Prop. 3.

    .. math::
        \Sigma^{(N,F)} = \left(\Sigma_{\rm obs}^{-1}
            - \Sigma_{\rm obs}^{-1} R \Sigma_{\rm obs}^{-1}\right)^{-1}

    Parameters
    ----------
    R : Float[Array, "n_obs n_obs"]
        Denoiser residual covariance (``R_f`` or ``R_g``, from
        :func:`denoiser_residual`).
    Sigma_obs : Float[Array, "n_obs n_obs"]
        Observation noise covariance.
    n_samples : int
        Number of samples ``N`` used to estimate ``R`` -- sets the tolerance
        of the guard below.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        The assembled matrix, symmetrized.

    Raises
    ------
    ValueError
        If ``R`` exceeds ``Sigma_obs`` well beyond the estimation noise -- the
        denoiser *degrades* the residual beyond the observation noise.

    Notes
    -----
    ⚠️ **§3.2 is least reliable in the linear regime** -- the opposite of
    §3.3. When the denoiser is nearly exact (at ``lambda = 0``, the affine one
    is), ``R -> Sigma_obs`` from below with a margin on the order of
    ``1e-6``: the precision becomes nearly singular. The paper says so: well
    defined *with high probability for N large enough*. Measured: at
    ``N = 1e4`` the margin is swamped by MC noise (``lambda_min < 0``); at
    ``N = 1e5`` it emerges. The approximation route therefore needs **a lot**
    of samples near the linear regime.

    The guard distinguishes MC noise (tolerated, spectral clipping) from an
    outright violation (``ValueError``). The threshold is ``~10 sigma_MC``
    with ``sigma_MC ~ trace(R)/n / sqrt(N)``: it only catches what CANNOT come
    from estimation noise.

    ⚠️ No ``jit``: the ``if`` below tests a value that depends on ``R`` (thus
    traced under jit), and ``ValueError`` requires a concrete bool. Same
    limitation for :func:`approximation_signal`/:func:`approximation_noise`,
    which call this function.
    """
    n = R.shape[0]
    tol = 10.0 * jnp.trace(R) / n / jnp.sqrt(n_samples)
    gap_eig = jnp.min(jnp.linalg.eigvalsh(Sigma_obs - R))
    if gap_eig <= -tol:
        raise ValueError(
            f"R clearly exceeds Sigma_obs "
            f"(lambda_min(Sigma_obs - R) = {float(gap_eig):.2e}, "
            f"MC tolerance = {float(-tol):.2e}). The denoiser is degrading the "
            "residual -- §3.2 is not applicable, or training has diverged."
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
    r"""Compute ``Sigma^{(N,F)}_signal`` via an already-trained denoiser ``f : Y -> u(eta)``.

    Parameters
    ----------
    denoiser : Denoiser
        Trained approximation ``f`` of ``E[u|Y]``.
    u_samples : Float[Array, "n_samples n_obs"]
        The ``u(eta^{(i)})`` -- the target to denoise, drawn jointly with
        ``Y_samples`` (the same pairs already used for ``Sigma_Y``).
    Y_samples : Float[Array, "n_samples n_obs"]
        Observations ``Y^{(i)}``, the denoiser's inputs.
    Sigma_obs : Float[Array, "n_obs n_obs"]
        Observation noise covariance.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        ``Sigma^{(N,F)}_signal``, an estimator (not certified) of the signal
        diagnostic matrix.
    """
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
    r"""Compute ``Sigma^{(N,G)}_noise`` via an already-trained ``g : (Y, theta) -> u(eta)``.

    Parameters
    ----------
    denoiser : Denoiser
        Trained approximation ``g`` of ``E[u|Y, theta]``.
    u_samples : Float[Array, "n_samples n_obs"]
        The ``u(eta^{(i)})`` -- the target to denoise.
    Y_samples : Float[Array, "n_samples n_obs"]
        Observations ``Y^{(i)}``.
    theta_samples : Float[Array, "n_samples n_param"]
        QoI draws ``theta^{(i)}``, concatenated with ``Y_samples`` to form
        the denoiser's inputs.
    Sigma_obs : Float[Array, "n_obs n_obs"]
        Observation noise covariance.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        ``Sigma^{(N,G)}_noise``, an estimator (not certified) of the noise
        diagnostic matrix.

    Notes
    -----
    ``g`` sees ``theta`` in addition to ``Y``, so it denoises better:
    ``R_g ⪯ R_f``, hence ``Sigma_noise ⪯ Sigma_signal`` -- the gap is
    ``gap_h``.
    """
    features = jnp.concatenate([Y_samples, theta_samples], axis=1)
    R_g = denoiser_residual(denoiser, u_samples, features)
    return assemble_from_residual(R_g, Sigma_obs, u_samples.shape[0])
