r"""Approximation-based diagnostic matrices -- §3.2.

Third route to ``Sigma_signal`` and ``Sigma_noise``, alongside :mod:`sample_based`
(§3.1) and :mod:`gradient_based` (§3.3). All three produce the **same** two
matrices; they differ in cost and in what they guarantee.

The principle (29)-(30)
------------------------
``E[u(eta)|Y]`` is the orthogonal projection of ``u(eta)`` onto ``L^2_{pi_Y}``.
Given an approximation class ``F``, the **denoiser**

.. math::
    f \in \arg\min_{f \in F} E[\|u(\eta) - f(Y)\|^2]

approximates this conditional expectation: ``f(u(eta) + eps) ~ u(eta)``, it
removes the noise. From this follows (Prop. 3)

.. math::
    \Sigma^{(N,F)}_{\rm signal} =
    \left(\Sigma_{\rm obs}^{-1}
      - \Sigma_{\rm obs}^{-1} R_f \Sigma_{\rm obs}^{-1}\right)^{-1},
    \qquad
    R_f = \frac1N \sum_i (u(\eta^{(i)}) - f(Y^{(i)}))^{\otimes 2}

Cost and guarantee
-------------------
* **Zero marginal cost.** ``f`` is learned on the ``(eta, Y)`` pairs already
  drawn for ``Sigma_Y`` (§3.1). No Jacobian (unlike §3.3), no MCMC. That is
  what makes it scale.
* **Estimator, not a certified bound.** ``E[(u - f(Y))^{⊗2}] ⪰ E[Cov(u|Y)]``
  (equality iff ``f = E[u|Y]``), so ``R_f`` **upper-bounds** the conditional
  covariance. The matrix is well defined (Prop. 3, if ``F`` contains the
  affines) but does not enter Thm 2.1 as a guarantee -- it only approximates
  it.

The affine denoiser
--------------------
Prop. 3 requires that ``F`` contain the affine functions. The **affine**
denoiser is therefore the floor: ``f(Y) = A Y + b`` with
``A = Cov(u, Y) Cov(Y)^{-1}``, in closed form on the same pairs. At
``lambda = 0`` the model is linear, ``E[u|Y]`` is exactly affine, and
``Sigma^{(F)}_signal`` **equals** the gradient route -- the cross-module
oracle.
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
    r"""Affine denoiser ``f(Y) = A Y + b`` via least squares -- closed form.

    .. math::
        A = \mathrm{Cov}(u, Y)\,\mathrm{Cov}(Y)^{-1}, \qquad b = E[u] - A\,E[Y]

    Parameters
    ----------
    u_samples : Float[Array, "n_samples n_obs"]
        The ``u(eta^{(i)})`` -- the target to denoise.
    features : Float[Array, "n_samples n_feat"]
        The denoiser's inputs. ``Y`` for ``f`` (n_feat = n_obs); ``(Y, theta)``
        concatenated for ``g`` (n_feat = n_obs + n_param).

    Notes
    -----
    Regression on the **same** pairs as ``Sigma_Y``: zero marginal cost.

    ``Cov(Y)`` is regularized with a relative jitter: at small ``sigma_obs``
    and modest ``N``, the empirical covariance of the features can be nearly
    singular.
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
    r"""``R_f = (1/N) sum (u - f(Y))^{⊗2}`` -- the denoiser's residual.

    ``R_f`` **upper-bounds** ``E[Cov(u|Y)]`` (equality iff ``f = E[u|Y]``).
    This is what makes §3.2 uncertified: the matrix is well defined but the
    Loewner order of Thm 2.1 is not guaranteed.
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

    ⚠️ Well defined **iff** ``R ≺ Sigma_obs`` (Prop. 3, guaranteed if ``F``
    contains the affines and ``N`` is large enough). Otherwise the
    parenthesized term stops being SDP and the inverse does not exist -- the
    price of not being certified.
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
    r"""``Sigma^{(N,F)}_signal`` via the affine denoiser ``f : Y -> u(eta)``.

    Parameters
    ----------
    u_samples, Y_samples : Float[Array, "n_samples n_obs"]
        Pairs ``(u(eta^{(i)}), Y^{(i)})`` -- **the ones already drawn for
        ``Sigma_Y``**.
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
    r"""``Sigma^{(N,G)}_noise`` via the affine denoiser ``g : (Y, theta) -> u(eta)``.

    ``g`` sees ``theta`` in addition to ``Y``: it denoises better, so
    ``R_g ⪯ R_f``, hence ``Sigma_noise ⪯ Sigma_signal`` -- the gap **is**
    ``gap_h``.
    """
    features = jnp.concatenate([Y_samples, theta_samples], axis=1)
    A, b = affine_denoiser(u_samples, features)
    R_g = denoiser_residual(u_samples, features, A, b)
    return _assemble_from_residual(R_g, Sigma_obs)
