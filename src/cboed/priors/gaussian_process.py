"""Construct Gaussian priors: Gaussian process plus inferential facade."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.priors.base import KernelBase, Prior


class GaussianProcess:
    r"""Construct a Gaussian process on a 1D grid.

    States **where the covariance comes from**: evaluates the kernel on the
    grid and stabilizes the Gram matrix with a nugget. The
    :class:`GaussianPrior` facade states **how to use it for inference** --
    do not merge the two.

    Parameters
    ----------
    kernel : KernelBase
        Covariance kernel.
    mu : Float[Array, " n_param"]
        Prior mean. Its length fixes the grid size.
    domain : tuple[float, float], default=(0.0, 1.0)
        Spatial interval.
    boundary_values : Float[Array, "2"] | None, default=None
        Fixed values at the left and right boundaries. When provided, the
        Gaussian process is first constructed on the full grid, including
        both boundary points, and then conditioned on those values. The
        resulting ``mu`` and ``Sigma`` contain only the interior points.
    jitter : float, default=1e-8
        **Relative** nugget: the diagonal receives ``jitter * tr(K)/n``.

    Notes
    -----
    The jitter is relative, not absolute: an absolute nugget does not mean
    the same thing depending on ``sigma``, and vanishes when the signal
    variance is large. ``tr(K)/n`` equals ``sigma**2`` for a stationary
    kernel while remaining well defined for a kernel that is not.

    This is **not** just a numerical trick: on a fine grid an RBF kernel is
    effectively rank-deficient (super-exponential spectral decay), and the
    nugget then carries the last modes -- it modifies the prior. Hence
    ``test_jitter_does_not_move_eig``.

    Examples
    --------
    >>> gp = GaussianProcess(Matern32(length_scale=0.2, sigma=1.0), jnp.zeros(64))
    >>> gp.Sigma.shape
    (64, 64)
    """

    def __init__(
        self,
        kernel: KernelBase,
        mu: Float[Array, " n_param"],
        domain: tuple[float, float] = (0.0, 1.0),
        jitter: float = 1e-8,
        boundary_values: Float[Array, "2"] | None = None,
        mu_boundary: Float[Array, "2"] | None = None,
    ) -> None:
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")
        if boundary_values is not None:
            boundary_values = jnp.asarray(boundary_values)
            if boundary_values.shape != (2,):
                raise ValueError(
                    "boundary_values must contain exactly the left and right "
                    f"boundary values, got shape {boundary_values.shape}"
                )
            if mu_boundary is None:
                mu_boundary = jnp.zeros(2)
            else:
                mu_boundary = jnp.asarray(mu_boundary)

                if mu_boundary.shape != (2,):
                    raise ValueError(
                        "mu_boundary must contain exactly the left and right "
                        f"boundary means, got shape {mu_boundary.shape}"
                    )
        self.kernel = kernel
        self.domain = domain
        self.jitter = jitter
        self.boundary_values = boundary_values
        self.mu_boundary = mu_boundary

        if boundary_values is None:
            self.mu = mu
            self.x = jnp.linspace(domain[0], domain[1], len(mu))
            self.Sigma = self._build_covariance(self.x)
        else:
            x_full = jnp.linspace(domain[0], domain[1], len(mu) + 2)
            Sigma_full = self._build_covariance(x_full)
            self.x = x_full[1:-1]
            self.mu, self.Sigma = self._condition_on_boundaries(
                mu, mu_boundary, Sigma_full, boundary_values
            )

    def _build_covariance(self, x: Float[Array, " n_param"]) -> Float[Array, "n_param n_param"]:
        """Evaluate the kernel Gram matrix on `x` and add the relative nugget.

        Parameters
        ----------
        x : Float[Array, " n_param"]
            Grid locations.

        Returns
        -------
        Float[Array, "n_param n_param"]
            Stabilized Gram matrix ``K(x, x) + jitter * (tr(K)/n) * I``.
        """
        K = self.kernel(x, x)
        n = x.shape[0]
        scale = jnp.trace(K) / n
        return K + self.jitter * scale * jnp.eye(n, dtype=K.dtype)

    @staticmethod
    def _condition_on_boundaries(
        mu: Float[Array, " n_interior"],
        mu_boundary: Float[Array, "2"],
        Sigma: Float[Array, " n_full n_full"],
        boundary_values: Float[Array, "2"],
    ) -> tuple[Float[Array, " n_interior"], Float[Array, " n_interior n_interior"]]:
        r"""Condition a Gaussian vector on its two boundary values.

        Parameters
        ----------
        mu : Float[Array, "n_interior"]
            Prior mean on the interior grid. The boundary mean is assumed to
            be zero, consistently with the default Gaussian-process mean.
        Sigma : Float[Array, "n_full n_full"]
            Covariance on the full grid, including both boundaries.
        boundary_values : Float[Array, "2"]
            Fixed values at the left and right boundaries.

        Returns
        -------
        tuple[Float[Array, "n_interior"], Float[Array, "n_interior n_interior"]]
            Conditional mean and covariance on the interior grid.

        Notes
        -----
        The covariance is computed with the Schur complement. Linear solves
        are used instead of explicitly forming ``Sigma_ee^{-1}``.
        """
        boundary = jnp.array([0, Sigma.shape[0] - 1])
        Sigma_ii = Sigma[1:-1, 1:-1]
        Sigma_ie = Sigma[1:-1, boundary]
        Sigma_ee = Sigma[jnp.ix_(boundary, boundary)]

        correction = Sigma_ie @ jnp.linalg.solve(Sigma_ee, Sigma_ie.T)
        conditional_covariance = Sigma_ii - correction
        conditional_mean = mu + Sigma_ie @ jnp.linalg.solve(Sigma_ee, boundary_values - mu_boundary)
        return conditional_mean, conditional_covariance


class GaussianPrior(Prior):
    r"""Wrap a :class:`GaussianProcess` as an inferential facade.

    Factorizes ``Gamma_prior`` **once** and exposes the actions inference
    needs. No inversion at construction: the dense oracles (``Sigma()``,
    ``hessian()``) are inherited from :class:`Prior` and materialize on
    demand.

    Parameters
    ----------
    prior : GaussianProcess
        Passed as a keyword: ``GaussianPrior(prior=gp)``.
    """

    def __init__(self, **hyperparameters) -> None:
        self._hyperparameters = hyperparameters
        Sigma = self.prior.Sigma
        # Single factorization: both solves *and* sampling derive from it.
        self._chol = jsp.linalg.cho_factor(Sigma, lower=True)
        # cho_factor leaves residue in the other triangle: tril() gives the
        # clean factor expected by sampling.
        self._L = jnp.tril(self._chol[0])

    @property
    def prior(self) -> GaussianProcess:
        return self._hyperparameters["prior"]

    @property
    def mu(self) -> Float[Array, " n_param"]:
        return self.prior.mu

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_prior(self, theta: Float[Array, " n_param"]) -> Float[Array, ""]:
        """Evaluate the Gaussian log-density ``log N(theta; mu, Gamma_prior)``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Parameter at which the prior is evaluated.

        Returns
        -------
        Float[Array, ""]
            ``log p(theta)``, via the cached Cholesky factor of
            ``Gamma_prior``.
        """
        n = theta.shape[0]
        r = theta - self.mu
        quad = r @ jsp.linalg.cho_solve(self._chol, r)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(self._chol[0])))
        return -0.5 * (n * jnp.log(2 * jnp.pi) + logdet + quad)

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def grad_log_prior(self, theta: Float[Array, " n_param"]) -> Float[Array, " n_param"]:
        """Compute the gradient of the Gaussian log-density.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Parameter at which the gradient is evaluated.

        Returns
        -------
        Float[Array, " n_param"]
            ``-Gamma_prior^{-1} (theta - mu)``, via the cached Cholesky
            factor.
        """
        return -jsp.linalg.cho_solve(self._chol, theta - self.mu)

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_det_precision(self) -> Float[Array, ""]:
        """Compute the log-determinant of the prior precision.

        Returns
        -------
        Float[Array, ""]
            ``log det Gamma_prior^{-1} = -2 sum log diag L``, where ``L`` is
            the cached Cholesky factor of ``Gamma_prior``.
        """
        return -2.0 * jnp.sum(jnp.log(jnp.diag(self._chol[0])))

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def prior_cov_matmul(self, B: Float[Array, "n_param k"]) -> Float[Array, "n_param k"]:
        """Apply the prior covariance to a batch of vectors.

        Parameters
        ----------
        B : Float[Array, "n_param k"]
            Batch of ``k`` vectors, as columns.

        Returns
        -------
        Float[Array, "n_param k"]
            ``Gamma_prior @ B``.
        """
        return self.prior.Sigma @ B

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def prior_precision_matmul(self, B: Float[Array, "n_param k"]) -> Float[Array, "n_param k"]:
        """Apply the prior precision to a batch of vectors.

        Parameters
        ----------
        B : Float[Array, "n_param k"]
            Batch of ``k`` vectors, as columns.

        Returns
        -------
        Float[Array, "n_param k"]
            ``Gamma_prior^{-1} @ B``, via ``cho_solve`` over the ``k``
            columns -- never via explicit inversion.
        """
        return jsp.linalg.cho_solve(self._chol, B)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, key: PRNGKeyArray, n_samples: int = 1) -> Float[Array, "n_samples n_param"]:
        """Draw samples ``theta ~ N(mu, Gamma_prior)`` via the cached Cholesky factor.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key.
        n_samples : int, default=1
            Number of samples to draw.

        Returns
        -------
        Float[Array, "n_samples n_param"]
            Samples, one per row: ``mu + z @ L.T`` with ``z`` standard
            normal.
        """
        z = jax.random.normal(key, (n_samples, self.mu.shape[0]))
        return self.mu + z @ self._L.T
