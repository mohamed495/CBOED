r"""Verification visuelle rapide -- cas goal-oriented, QoI = premiere moitie de theta.

Meme structure que le test rapide du cas standard (reconstruction + spectre
log-generalise), mais l'objet d'interet n'est plus le champ entier ``eta``
(200 points) : c'est ``theta = h(eta) = eta[:n_qoi]``, via
:class:`cboed.inference.goal_oriented.GoalOrientedModel`.

``Sigma_xi`` (bruit sur ``theta = h(eta) + xi``) est fixe a un jitter non nul
plutot qu'a zero exact : ``qoi_fisher_moment`` diverge quand ``Sigma_xi -> 0``
(cf. ``bounds/diagnostics/gradient_based.py``), y compris pour une projection
et pas seulement pour ``h = identite``.

Echantillons reduits (N_SAMPLES, N_GRADIENT) : verification de plomberie, pas
un resultat de production.
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cboed.benchmarks import N, SIGMA_OBS_MATRIX, forward, make_model, make_prior
from cboed.bounds.base import DiagnosticMatrices
from cboed.bounds.diagnostics.gradient_based import gradient_diagnostics, gradient_diagnostics_standard
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y, sample_Sigma_Y_given_theta
from cboed.bounds.quasi_optimality import quasi_optimality
from cboed.inference.goal_oriented import GoalOrientedModel
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.optim.greedy_schur import greedy_schur
from cboed.viz import fields as vf
from cboed.viz import spectrum as vs
from cboed.viz.style import save, use_style

OUT = Path(__file__).parent / "figures_go_quick"

LAMBDA = 0.0
N_QOI = N // 2
SIGMA_XI = 1e-3 * jnp.eye(N_QOI)  # jitter -- cf. docstring du module

N_SAMPLES = 300  # Sigma_Y, Sigma_Y_given_theta (paires MC)
N_GRADIENT = 60  # Sigma_signal, Sigma_noise (jacobiennes)
M_DESIGN = 10


def main() -> None:
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)

    prior = make_prior()
    model = make_model(LAMBDA)
    u = forward(LAMBDA)
    h = lambda eta: eta[:N_QOI]  # noqa: E731 -- projection QoI, premiere moitie

    likelihood = GaussianLikelihood(model=model, Sigma_obs=SIGMA_OBS_MATRIX)
    inference = LinearModel(prior=prior, likelihood=likelihood)
    go = GoalOrientedModel(inner=inference, h=h, Sigma_theta=SIGMA_XI)

    keys = jr.split(jr.key(0), 7)
    k_grad, k_sampleY, k_sampleYth, k_true, k_noise, k_prior_qoi, k_post_qoi = keys

    # -- design : glouton incremental standard (Sigma_signal du champ entier) --
    print("[design] Sigma_signal (echelle standard) + greedy_schur...")
    Sigma_signal_std, _ = gradient_diagnostics_standard(
        u, prior, SIGMA_OBS_MATRIX, k_grad, N_GRADIENT
    )
    design = greedy_schur(Sigma_signal_std, SIGMA_OBS_MATRIX, M_DESIGN).design
    n_qoi_sensors = int(jnp.sum(design < N_QOI))
    print(f"  {n_qoi_sensors}/{M_DESIGN} capteurs tombent dans la QoI (x < 0.5)")

    # -- reconstruction QoI : theta = h(eta), prior/posterior/verite --------
    print("[reconstruction QoI]")
    theta_true = prior.sample(k_true, 1)[0]
    y = model(theta_true, design) + jr.normal(k_noise, (len(design),)) * jnp.sqrt(
        jnp.diag(SIGMA_OBS_MATRIX)[design]
    )
    mu_post_full = inference._mu(y, prior.mu, design)

    Sigma_theta_prior = go.prior_covariance_qoi(prior.mu)
    Sigma_theta_post = go.posterior_covariance_qoi(prior.mu, design)

    L_prior = jnp.linalg.cholesky(Sigma_theta_prior + 1e-10 * jnp.eye(N_QOI))
    L_post = jnp.linalg.cholesky(Sigma_theta_post + 1e-10 * jnp.eye(N_QOI))
    prior_qoi = prior.mu[:N_QOI] + jr.normal(k_prior_qoi, (200, N_QOI)) @ L_prior.T
    post_qoi = mu_post_full[:N_QOI] + jr.normal(k_post_qoi, (200, N_QOI)) @ L_post.T

    X = np.linspace(0.0, 1.0, N + 2)[1:-1]
    x_qoi = X[:N_QOI]
    sensors_qoi = np.asarray(design)[np.asarray(design) < N_QOI]

    save(
        vf.plot_reconstruction(
            x_qoi,
            np.asarray(prior_qoi),
            np.asarray(post_qoi),
            np.asarray(theta_true[:N_QOI]),
            sensors=sensors_qoi if sensors_qoi.size else None,
        ),
        OUT / "01_reconstruction_qoi.png",
    )

    # -- spectre log-generalise, diagnostiques goal-oriented -----------------
    print("[diagnostiques GO] Sigma_signal, Sigma_noise, Sigma_Y, Sigma_Y_given_theta...")
    Sigma_signal_go, Sigma_noise_go = gradient_diagnostics(
        u, h, prior, SIGMA_OBS_MATRIX, SIGMA_XI, k_grad, N_GRADIENT
    )
    Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k_sampleY, N_SAMPLES)
    B = jnp.eye(N)[:N_QOI]
    Sigma_Y_given_theta = sample_Sigma_Y_given_theta(
        u, prior, B, SIGMA_OBS_MATRIX, SIGMA_XI, k_sampleYth, N_SAMPLES
    )

    dg = DiagnosticMatrices(
        Sigma_Y=Sigma_Y,
        Sigma_Y_given_theta=Sigma_Y_given_theta,
        Sigma_signal=Sigma_signal_go,
        Sigma_noise=Sigma_noise_go,
        certified=True,
    )
    q = quasi_optimality(dg)
    n_below_one = int(jnp.sum(q.alpha < 1.0 - 1e-6) + jnp.sum(q.beta < 1.0 - 1e-6))
    if n_below_one:
        print(f"  attention : {n_below_one} valeurs propres < 1 (Prop. 1 les exige >= 1)")

    save(
        vs.plot_log_generalized_spectrum(
            np.asarray(q.alpha), np.asarray(q.beta), title=rf"goal-oriented, $\lambda = {LAMBDA}$"
        ),
        OUT / "02_log_spectrum_go.png",
    )

    print(f"\n-> {OUT.resolve()}")


if __name__ == "__main__":
    main()
