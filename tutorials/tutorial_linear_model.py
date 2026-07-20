import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess


# -------------------------------------------------------------------------
# Paramètres
# -------------------------------------------------------------------------

N = 200
NT = 100

T = 0.1
NU = 0.2
DOMAIN = [0.0, 1.0]

SIGMA_OBS = 0.1
SIGMA_OBS_MATRIX = SIGMA_OBS**2 * jnp.eye(N)


KERNEL_LENGTH_SCALE = 0.2
KERNEL_SIGMA = 1.0


# -------------------------------------------------------------------------
# Paramètre vrai
# -------------------------------------------------------------------------

x = jnp.linspace(0.0, 1.0, N)


def gaussian_bump(
    x,
    center,
    width,
    amplitude,
):
    return amplitude * jnp.exp(
        -0.5*((x-center)/width)**2
    )


theta_true = (
    gaussian_bump(
        x,
        center=0.3,
        width=0.08,
        amplitude=1.0,
    )
    -
    0.5 * gaussian_bump(
        x,
        center=0.7,
        width=0.12,
        amplitude=1.0,
    )
)


# -------------------------------------------------------------------------
# Modèle direct
# -------------------------------------------------------------------------

model = AdvectionDiffusion(
    diffusivity=NU,
    velocity=1.0,
    T=T,
    domain=DOMAIN,
    nt=NT,
    n=N,
)


# -------------------------------------------------------------------------
# Prior
# -------------------------------------------------------------------------

prior = GaussianProcess(
    kernel=kernel.Matern32(
        length_scale=KERNEL_LENGTH_SCALE,
        sigma=KERNEL_SIGMA,
    ),
    mu=jnp.zeros(model.n),
)


gaussian_prior = GaussianPrior(
    prior=prior
)


# -------------------------------------------------------------------------
# Likelihood
# -------------------------------------------------------------------------

likelihood = GaussianLikelihood(
    model=model,
    prior=prior,
    Sigma_obs=SIGMA_OBS_MATRIX,
)


inference = LinearModel(
    prior=gaussian_prior,
    likelihood=likelihood,
)


# -------------------------------------------------------------------------
# Génération des données
# -------------------------------------------------------------------------

key = jax.random.key(42)

key, noise_key = jax.random.split(key)

y = (
    model(theta_true)
    +
    jax.random.multivariate_normal(
        key=noise_key,
        mean=jnp.zeros(N),
        cov=SIGMA_OBS_MATRIX,
    )
)


# -------------------------------------------------------------------------
# Reconstruction itérative
# -------------------------------------------------------------------------

theta = jnp.zeros(N)   # moyenne du prior

n_iter = 10

history = []

for k in range(n_iter):

    theta_new = inference._mu(
        y=y,
        theta=theta,
    )

    error = jnp.linalg.norm(theta_new-theta)

    history.append(error)

    print(
        f"iteration {k}: update norm = {error:.3e}"
    )

    theta = theta_new


theta_rec = theta


# -------------------------------------------------------------------------
# Erreur finale
# -------------------------------------------------------------------------

relative_error = (
    jnp.linalg.norm(theta_rec-theta_true)
    /
    jnp.linalg.norm(theta_true)
)

print(
    "relative error =",
    relative_error
)


# -------------------------------------------------------------------------
# Reconstruction
# -------------------------------------------------------------------------

plt.figure(figsize=(8,4))

plt.plot(
    x,
    theta_true,
    label=r"$\theta_{\rm true}$",
    linewidth=2,
)

plt.plot(
    x,
    theta_rec,
    "--",
    label=r"$\theta_{\rm rec}$",
    linewidth=2,
)

plt.xlabel("x")
plt.ylabel(r"$\theta(x)$")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "iterative_reconstruction.png",
    dpi=300,
)

plt.close()


# -------------------------------------------------------------------------
# Convergence
# -------------------------------------------------------------------------

plt.figure(figsize=(6,3))

plt.semilogy(
    history,
    marker="o",
)

plt.xlabel("iteration")
plt.ylabel(r"$||\theta_{k+1}-\theta_k||$")
plt.grid()

plt.tight_layout()
plt.savefig(
    "convergence.png",
    dpi=300,
)


plt.figure(figsize=(8,4))

plt.plot(
    x,
    theta_true,
    label=r"$\theta_{\mathrm{true}}$",
    linewidth=2,
)

plt.plot(
    x,
    theta_rec,
    "--",
    label=r"$\theta_{\mathrm{rec}}$",
    linewidth=2,
)

plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel(r"$\theta(x)$")

plt.tight_layout()
plt.savefig("reconstruction.png", dpi=300)
plt.close()

plt.close()