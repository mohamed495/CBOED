r"""Bancs de référence.

Dans le paquet et non dans `tests/` : le banc est de la **configuration
scientifique**, pas de l'infrastructure de test. Les tests et les scripts
d'expérience le lisent au même endroit -- sinon ils divergent en silence, et rien ne
le voit (`test_bench.py` teste le conftest, jamais les scripts).

Aligné sur le prototype NumPy : `dt=0.001 x 100` donc `T=0.1`, `n=200`, `nu=0.02`,
`sigma_obs=0.1`, `Matern32(0.2, 1.0)`, `mu=0`.

Longueur de diffusion `sqrt(nu T) = 0.045`, soit 4.5 % du domaine : le champ garde sa
structure assez longtemps pour que l'advection agisse. (Un banc `T=1, nu=0.05` donne
0.22 -- le champ est lisse avant d'avoir pu faire quoi que ce soit de non linéaire.)
"""

import jax.numpy as jnp

from cboed.core.burgers import Burgers
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Matern32

# -- modele --------------------------------------------------------------
N = 200
NT = 100
T = 0.1  # dt = T / NT = 0.001
NU = 0.2
DOMAIN = [0.0, 1.0]

# -- bruit d'observation -------------------------------------------------
SIGMA_OBS = 0.1  # ecart-type -> Sigma_obs = SIGMA_OBS**2 * I
SIGMA_OBS_MATRIX = SIGMA_OBS**2 * jnp.eye(N)

# -- prior ---------------------------------------------------------------
KERNEL_LENGTH_SCALE = 0.2
KERNEL_SIGMA = 0.3

# -- balayages -----------------------------------------------------------
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
SENSOR_BUDGETS = (5, 10, 15, 20, 25)

# -- goal-oriented : QoI = premiere moitie du champ -----------------------
# Sigma_xi = 0 exactement est une limite singuliere (qoi_fisher_moment diverge,
# cf. bounds/diagnostics/gradient_based.py) : jitter non nul, choisi petit
# devant la variance du prior (KERNEL_SIGMA**2 = 1).
N_QOI = N // 2
SIGMA_XI_QOI = 1e-3 * jnp.eye(N_QOI)


def qoi_projection(n_qoi: int = N_QOI):
    """``h : eta -> eta[:n_qoi]`` -- premiere moitie du champ."""
    return lambda eta: eta[:n_qoi]


def make_prior(n: int = N) -> GaussianPrior:
    """Prior GP a moyenne nulle."""
    gp = GaussianProcess(
        kernel=Matern32(length_scale=KERNEL_LENGTH_SCALE, sigma=KERNEL_SIGMA),
        mu=jnp.zeros(n),
        domain=tuple(DOMAIN),
    )
    return GaussianPrior(prior=gp)


def make_model(lambda_: float, n: int = N, nt: int = NT) -> Burgers:
    return Burgers(diffusivity=NU, lambda_=lambda_, T=T, domain=DOMAIN, nt=nt, n=n)


def forward(lambda_: float):
    """`u : theta -> observations`, sans design. Ce que consomment les diagnostiques."""
    model = make_model(lambda_)
    return lambda theta: model(theta, None)


def grid_spacing(n: int = N) -> float:
    """`dx = L / (n+1)` : `theta` est la condition initiale **interieure**."""
    return (DOMAIN[1] - DOMAIN[0]) / (n + 1)


def peclet(u_max: float, n: int = N) -> float:
    """`Pe = max|u| dx / nu <= 2` -- au-dela, le schema centre oscille.

    Contrainte **spatiale** : raffiner `nt` n'y change rien.
    """
    return u_max * grid_spacing(n) / NU


def cfl(u_max: float, lambda_: float, n: int = N, nt: int = NT) -> float:
    """`lambda max|u| dt / dx <= 1`. Se corrige en montant `nt`.

    Le nombre de diffusion n'est pas une contrainte : `Burgers` est IMEX, la
    diffusion est en Crank-Nicolson implicite, inconditionnellement stable.
    """
    return lambda_ * u_max * (T / nt) / grid_spacing(n)
