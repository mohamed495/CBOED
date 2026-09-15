r"""Sampling from the conditional law ``eta | theta``.

The conditional law is Gaussian when ``theta = B eta + xi`` with a constant
matrix ``B``. For a nonlinear quantity of interest, no Gaussian closed form
exists; the sampler below uses MALA (Metropolis-Adjusted Langevin Algorithm)
to target the conditional density numerically.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray

from cboed.priors.base import Prior


def _psd_sqrt(matrix: Array) -> Array:
    """Return a symmetric PSD square root, including singular matrices."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (matrix + matrix.T))
    return (eigenvectors * jnp.sqrt(jnp.clip(eigenvalues, 0.0))) @ eigenvectors.T


def _sample_linear(
    prior_eta: Prior,
    theta: Array,
    B: Array,
    Sigma_xi: Array,
    key: PRNGKeyArray,
    n_samples: int,
) -> Array:
    """Sample the exact Gaussian conditional for a constant ``B``."""
    Sigma_eta = prior_eta.Sigma()
    mean_eta = prior_eta.mu
    system = B @ Sigma_eta @ B.T + Sigma_xi
    gain = jnp.linalg.solve(system, B @ Sigma_eta).T
    Sigma_pos = Sigma_eta - gain @ B @ Sigma_eta
    L_pos = _psd_sqrt(Sigma_pos)
    z = jax.random.normal(key, (n_samples, mean_eta.shape[0]))
    mean_pos = mean_eta + (theta - mean_eta @ B.T) @ gain.T
    return mean_pos + z @ L_pos.T


def _mala_samples(
    prior_eta: Prior,
    h,
    Sigma_xi: Array,
    theta: Array,
    key: PRNGKeyArray,
    n_samples: int,
    n_warmup: int,
    step_size: float,
    thinning: int,
) -> Array:
    """Run independent MALA chains targeting ``pi(eta | theta)``."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    if thinning <= 0:
        raise ValueError(f"thinning must be > 0, got {thinning}")

    precision_xi = jnp.linalg.inv(Sigma_xi)
    step_size_sq = step_size**2
    n_steps = n_warmup + n_samples * thinning
    n_eta = prior_eta.mu.shape[0]

    def log_target(eta):
        residual = theta - h(eta)
        return prior_eta.log_prior(eta) - 0.5 * residual @ precision_xi @ residual

    grad_log_target = jax.grad(log_target)

    def chain(initial, chain_key):
        def transition(state, transition_key):
            noise_key, uniform_key = jax.random.split(transition_key)
            gradient = grad_log_target(state)
            proposal = state + 0.5 * step_size_sq * gradient
            proposal = proposal + step_size * jax.random.normal(noise_key, (n_eta,))

            proposal_gradient = grad_log_target(proposal)
            forward_residual = proposal - state - 0.5 * step_size_sq * gradient
            reverse_residual = state - proposal - 0.5 * step_size_sq * proposal_gradient
            log_acceptance = (
                log_target(proposal)
                - log_target(state)
                - 0.5
                * (reverse_residual @ reverse_residual - forward_residual @ forward_residual)
                / step_size_sq
            )
            accepted = jnp.log(jax.random.uniform(uniform_key)) < log_acceptance
            new_state = jnp.where(accepted, proposal, state)
            return new_state, new_state

        keys = jax.random.split(chain_key, n_steps)
        _, states = jax.lax.scan(transition, initial, keys)
        return states[n_warmup::thinning]

    initial_key, chain_key = jax.random.split(key)
    initial = prior_eta.sample(initial_key, 1)[0]
    return chain(initial, chain_key)


@partial(jax.jit, static_argnums=(0, 4, 6, 7, 8, 9))
def sample_eta_given_theta(
    prior_eta: Prior,
    theta: Array,
    key: PRNGKeyArray,
    B: Array | None = None,
    h=None,
    Sigma_xi: Array | None = None,
    n_samples: int = 1,
    n_warmup: int = 100,
    step_size: float = 1e-3,
    thinning: int = 1,
) -> Array:
    r"""Sample ``eta | theta`` for a linear or nonlinear QoI.

    Parameters
    ----------
    prior_eta : Prior
        Prior distribution of ``eta``.
    theta : Array
        Observed QoI, shape ``(n_qoi,)``.
    key : PRNGKeyArray
        JAX random key.
    B : Array or None, optional
        Constant linear QoI matrix. If provided, the exact Gaussian
        conditional is used.
    h : callable or None, optional
        Nonlinear QoI map. Required when ``B`` is ``None``.
    Sigma_xi : Array
        QoI noise covariance.
    n_samples : int, optional
        Number of returned samples.
    n_warmup : int, optional
        Number of MALA warmup transitions. Used only for nonlinear QoIs.
    step_size : float, optional
        MALA proposal step size. Used only for nonlinear QoIs.
    thinning : int, optional
        Number of MALA transitions between returned samples.

    Returns
    -------
    Array
        Samples with shape ``(n_samples, n_eta)``.

    Notes
    -----
    The linear branch is an exact Gaussian conditional draw. The nonlinear
    branch is an approximate MALA draw from the specified conditional target.
    """
    if Sigma_xi is None:
        raise ValueError("Sigma_xi is required")
    if B is not None:
        return _sample_linear(prior_eta, theta, B, Sigma_xi, key, n_samples)
    if h is None:
        raise ValueError("h is required when B is None")
    return _mala_samples(
        prior_eta, h, Sigma_xi, theta, key, n_samples, n_warmup, step_size, thinning
    )
