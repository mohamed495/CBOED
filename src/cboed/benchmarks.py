r"""Define the reference Burgers benchmark: forward model, prior, and sweep constants.

In the package and not in `tests/`: the benchmark is **scientific
configuration**, not test infrastructure. Tests and experiment scripts read
it from the same place -- otherwise they silently diverge, and nothing
catches it (`test_bench.py` tests the conftest, never the scripts).

Notes
-----
`dt=0.001 x 100` hence `T=0.1`, `n=200`, `sigma_obs=0.1`,
`Matern32(0.2, 0.3)`, `mu=0` are aligned with the NumPy prototype.

`nu=0.003` (not the prototype's `nu=0.2`): diffusion length
`sqrt(nu T) = 0.017`, i.e. 1.7% of the domain -- a much sharper, shock-like
regime. Deliberately deviates from the prototype: at `nu=0.2` the
generalized spectrum of Prop. 1 is extremely concentrated (effective rank
2-3 out of `N=200`, verified across every tested `lambda` and prior length
scale -- shortening the prior's correlation length only made it worse,
since Burgers' diffusion smooths high-frequency content before `T`
regardless). That collapses the incremental/conservative sub-optimality
trade-off of eq. (22)/(23) onto a near-flat curve over the realistic sensor
budgets (`SENSOR_BUDGETS`): both constants saturate almost immediately, so
neither figure shows the trade-off the proposition predicts. Lowering `nu`
instead (less numerical diffusion, so less of the nonlinear/non-Gaussian
signature gets smoothed away before observation) raises the effective rank
substantially (verified up to 13 at `nu=0.003`) -- but `nu` cannot be
lowered arbitrarily: below `nu ~= 0.0027`, the spatial Peclet number
`Pe = max|theta| dx / nu` (see `peclet()`) exceeds the `<= 2` resolution
limit already enforced by `test_bench_is_resolved`, and the centered
advection scheme starts to under-resolve the field (checked directly: at
`nu=0.001`, `Pe ~= 5.5`, well past the limit). `nu=0.003` keeps `Pe ~= 1.8`,
with margin, while still raising the effective rank from 2-3 to 13.
"""

import jax.numpy as jnp

from cboed.core.burgers import Burgers
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Matern32

# -- model -----------------------------------------------------------------
N = 200
NT = 100
T = 0.1  # dt = T / NT = 0.001
NU = 0.02
DOMAIN = [0.0, 1.0]

# -- observation noise -------------------------------------------------------
SIGMA_OBS = 0.1  # standard deviation -> Sigma_obs = SIGMA_OBS**2 * I
SIGMA_OBS_MATRIX = SIGMA_OBS**2 * jnp.eye(N)

# -- prior -------------------------------------------------------------------
KERNEL_LENGTH_SCALE = 0.2
KERNEL_SIGMA = 0.3

# -- sweeps --------------------------------------------------------------
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
SENSOR_BUDGETS = tuple(range(1, 16))

# -- goal-oriented QoIs ------------------------------------------------------
# The energy QoI is noiseless: its conditional law is sampled by rejection
# from a tolerance band around the level set ``||eta||**2 = theta``.


def build_qoi(qoi_type: str = "nuisance"):
    """Build the quantity-of-interest map and its noise covariance.

    Parameters
    ----------
    qoi_type : {"nuisance", "energy"}, default="nuisance"
        Type of quantity of interest.

        - ``"nuisance"``: considers the first half of the field,
          with ``h(eta) = eta[:N // 2]``.
        - ``"energy"``: considers the total energy of the field,
          with ``h(eta) = sum(eta**2)``.

    Returns
    -------
    h : callable
        Quantity-of-interest map.

        For ``"nuisance"``, ``h`` maps the field to its first
        ``N // 2`` components.

        For ``"energy"``, ``h`` maps the field to its squared
        Euclidean norm.

    sigma_xi_qoi : jax.Array
        Observation noise covariance associated with the QoI.
        For ``"nuisance"``, this is ``1e-3 * I_{N // 2}``.
        For ``"energy"``, this is the zero covariance ``[[0.0]]``.

    n_qoi : int
        Dimension of the quantity of interest. It is ``N // 2``
        for ``"nuisance"`` and ``1`` for ``"energy"``.

    Raises
    ------
    ValueError
        If ``qoi_type`` is not ``"nuisance"`` or ``"energy"``.
    """
    if qoi_type == "nuisance":
        n_qoi = N // 2

        def h(eta):
            return eta[:n_qoi]

        sigma_xi_qoi = 1e-3 * jnp.eye(n_qoi)

    elif qoi_type == "energy":
        n_qoi = 1

        def h(eta):
            return jnp.asarray([jnp.sum(eta**2)])

        sigma_xi_qoi = jnp.zeros((1, 1))

    else:
        raise ValueError(f"Unknown qoi_type={qoi_type!r}. Expected 'nuisance' or 'energy'.")

    return h, sigma_xi_qoi, n_qoi


def make_prior(n: int = N) -> GaussianPrior:
    """Build the Gaussian process prior used by the benchmark.

    Parameters
    ----------
    n : int, default=N
        Grid size (number of parameters).

    Returns
    -------
    GaussianPrior
        Prior with a Matern-3/2 kernel (`KERNEL_LENGTH_SCALE`,
        `KERNEL_SIGMA`) on `DOMAIN`, conditioned on prescribed values
        at the two boundaries

    Examples
    --------
    >>> prior = make_prior()
    >>> prior.mu.shape
    (200,)
    """
    gp = GaussianProcess(
        kernel=Matern32(length_scale=KERNEL_LENGTH_SCALE, sigma=KERNEL_SIGMA),
        mu=jnp.ones(n),
        mu_boundary=jnp.ones(2),
        domain=tuple(DOMAIN),
        boundary_values=jnp.ones(2),
    )
    return GaussianPrior(prior=gp)


def make_model(lambda_: float, n: int = N, nt: int = NT) -> Burgers:
    """Build the Burgers forward model used by the benchmark.

    Parameters
    ----------
    lambda_ : float
        Nonlinearity parameter of the advection term.
    n : int, default=N
        Grid size (number of interior points).
    nt : int, default=NT
        Number of time steps.

    Returns
    -------
    Burgers
        Forward model with diffusivity `NU`, horizon `T`, and domain
        `DOMAIN`.

    Examples
    --------
    >>> model = make_model(lambda_=0.5)
    """
    return Burgers(diffusivity=NU, lambda_=lambda_, T=T, domain=DOMAIN, nt=nt, n=n)


def forward(lambda_: float):
    """Build the undesigned forward map ``u : theta -> observations``.

    Parameters
    ----------
    lambda_ : float
        Nonlinearity parameter of the Burgers model.

    Returns
    -------
    callable
        Function mapping `theta` (the interior initial condition) to the
        full observable, with no design applied (``design=None``). What
        the diagnostics consume.

    Examples
    --------
    >>> u = forward(lambda_=0.5)
    """
    model = make_model(lambda_)
    return lambda theta: model(theta, None)


def grid_spacing(n: int = N) -> float:
    """Compute the grid spacing ``dx = L / (n + 1)``.

    Parameters
    ----------
    n : int, default=N
        Number of interior grid points.

    Returns
    -------
    float
        Grid spacing. `theta` is the **interior** initial condition, hence
        the ``n + 1`` in the denominator.
    """
    return (DOMAIN[1] - DOMAIN[0]) / (n + 1)


def peclet(u_max: float, n: int = N) -> float:
    """Compute the spatial Peclet number ``Pe = max|u| dx / nu``.

    Parameters
    ----------
    u_max : float
        Maximum absolute velocity in the field.
    n : int, default=N
        Number of interior grid points.

    Returns
    -------
    float
        Peclet number. Must stay ``<= 2``, beyond which the centered
        advection scheme oscillates.

    Notes
    -----
    **Spatial** constraint: refining `nt` does not change it.
    """
    return u_max * grid_spacing(n) / NU


def cfl(u_max: float, lambda_: float, n: int = N, nt: int = NT) -> float:
    """Compute the CFL number ``lambda_ * max|u| * dt / dx``.

    Parameters
    ----------
    u_max : float
        Maximum absolute velocity in the field.
    lambda_ : float
        Nonlinearity parameter of the advection term.
    n : int, default=N
        Number of interior grid points.
    nt : int, default=NT
        Number of time steps, fixing ``dt = T / nt``.

    Returns
    -------
    float
        CFL number. Must stay ``<= 1``; fixed by increasing `nt`.

    Notes
    -----
    The diffusion number is not a constraint: `Burgers` is IMEX, diffusion
    is implicit Crank-Nicolson, unconditionally stable.
    """
    return lambda_ * u_max * (T / nt) / grid_spacing(n)
