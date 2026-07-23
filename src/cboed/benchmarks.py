r"""Define the reference Burgers benchmark: forward model, prior, and sweep constants.

In the package and not in `tests/`: the benchmark is **scientific
configuration**, not test infrastructure. Tests and experiment scripts read
it from the same place -- otherwise they silently diverge, and nothing
catches it (`test_bench.py` tests the conftest, never the scripts).

Notes
-----
Aligned with the NumPy prototype: `dt=0.001 x 100` hence `T=0.1`, `n=200`,
`nu=0.2`, `sigma_obs=0.1`, `Matern32(0.2, 0.3)`, `mu=0`.

Diffusion length `sqrt(nu T) = 0.141`, i.e. 14% of the domain: the field
keeps its structure long enough for advection to act. (A benchmark with
`T=1, nu=0.05` gives 0.22 -- the field is smooth before it can do anything
nonlinear.)
"""

import jax.numpy as jnp

from cboed.core.burgers import Burgers
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Matern32

# -- model -----------------------------------------------------------------
N = 200
NT = 100
T = 0.1  # dt = T / NT = 0.001
NU = 0.2
DOMAIN = [0.0, 1.0]

# -- observation noise -------------------------------------------------------
SIGMA_OBS = 0.1  # standard deviation -> Sigma_obs = SIGMA_OBS**2 * I
SIGMA_OBS_MATRIX = SIGMA_OBS**2 * jnp.eye(N)

# -- prior -------------------------------------------------------------------
KERNEL_LENGTH_SCALE = 0.2
KERNEL_SIGMA = 0.3

# -- sweeps --------------------------------------------------------------
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
SENSOR_BUDGETS = (5, 10, 15, 20, 25)

# -- goal-oriented: QoI = first half of the field ---------------------------
# Sigma_xi = 0 exactly is a singular limit (qoi_fisher_moment diverges,
# cf. bounds/diagnostics/gradient_based.py): nonzero jitter, chosen small
# relative to the prior variance (KERNEL_SIGMA**2 = 0.09).
N_QOI = N // 2
SIGMA_XI_QOI = 1e-3 * jnp.eye(N_QOI)


def qoi_projection(n_qoi: int = N_QOI):
    """Build the QoI projection ``h : eta -> eta[:n_qoi]`` (first half of the field).

    Parameters
    ----------
    n_qoi : int, default=N_QOI
        Number of leading field components kept as the quantity of
        interest.

    Returns
    -------
    callable
        Function mapping a field ``eta`` to its first `n_qoi` components.
    """
    return lambda eta: eta[:n_qoi]


def make_prior(n: int = N) -> GaussianPrior:
    """Build the zero-mean Gaussian process prior used by the benchmark.

    Parameters
    ----------
    n : int, default=N
        Grid size (number of parameters).

    Returns
    -------
    GaussianPrior
        Prior with a Matern-3/2 kernel (`KERNEL_LENGTH_SCALE`,
        `KERNEL_SIGMA`) on `DOMAIN`, zero mean.

    Examples
    --------
    >>> prior = make_prior()
    >>> prior.mu.shape
    (200,)
    """
    gp = GaussianProcess(
        kernel=Matern32(length_scale=KERNEL_LENGTH_SCALE, sigma=KERNEL_SIGMA),
        mu=jnp.zeros(n),
        domain=tuple(DOMAIN),
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
