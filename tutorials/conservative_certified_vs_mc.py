#!/usr/bin/env python
r"""Bornes conservatives : ``eig_full`` certifie (defaut) vs estime par Monte-Carlo.

``conservative_bounds`` (Cor. 2) a besoin d'un point de reference ``EIG(I_p)``
(l'information du dataset complet). Par defaut (``eig_full=None``), il est
encadre gratuitement par le corollaire 1 applique au design complet -- jamais
estime, ce qui garde la borne certifiee.

Le prototype NumPy fait l'inverse : il injecte un ``eig_offset`` estime par
Monte-Carlo imbrique (NMC) dans la meme formule -- ce qui decertifie la borne.
Ce script compare les deux sur le meme jeu de diagnostics.

Ce que ca montre
-----------------
Le NMC (:class:`cboed.estimators.nmc.NestedMonteCarloEIG`) sous-estime
systematiquement ``log p(y)`` a ``n_inner`` fini (biais de Jensen sur le
``logsumexp``), donc surestime l'EIG. En dimension d'observation elevee
(``n_obs = 200`` ici), la convergence en ``n_inner`` est tres lente : meme
``n_inner = 8000`` reste loin de la bande certifiee dans le banc par defaut.
Injecter cette estimation dans (17)-(18) deplace tout l'encadrement
conservatif hors de la bande garantie par le theoreme.

Usage
-----
    pixi run -e test python tutorials/conservative_certified_vs_mc.py
    pixi run -e test python tutorials/conservative_certified_vs_mc.py --n-inner 500 2000 8000
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from cboed.benchmarks import SIGMA_OBS_MATRIX, forward, make_model, make_prior
from cboed.bounds.base import DiagnosticMatrices
from cboed.bounds.bounds import conservative_bounds, incremental_bounds
from cboed.bounds.diagnostics.gradient_based import (
    assemble,
    expected_jacobian_moments,
    fisher_information_prior,
)
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y
from cboed.estimators.nmc import NestedMonteCarloEIG
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.optim.greedy_schur import greedy_schur
from cboed.viz.style import COLORS, save, use_style


def compute_diagnostics(lambda_: float, n_samples: int, n_gradient: int, key):
    """Diagnostics standard (``Sigma_Y_given_theta = Sigma_noise = Sigma_obs``)."""
    prior, u = make_prior(), forward(lambda_)
    k_sample, k_grad = jr.split(key)

    Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k_sample, n_samples)
    thetas = prior.sample(k_grad, n_gradient)
    L, H = expected_jacobian_moments(u, thetas, SIGMA_OBS_MATRIX)
    I_eta = fisher_information_prior(prior)
    Sigma_signal = assemble(L, H + I_eta, SIGMA_OBS_MATRIX)

    dg = DiagnosticMatrices(
        Sigma_Y=Sigma_Y,
        Sigma_Y_given_theta=SIGMA_OBS_MATRIX,
        Sigma_signal=Sigma_signal,
        Sigma_noise=SIGMA_OBS_MATRIX,
        certified=True,
    )
    return prior, Sigma_signal, dg


def eig_full_convergence(prior, lambda_: float, n_outer: int, n_inners: list[int], key):
    """``EIG(I_p)`` par NMC a plusieurs ``n_inner`` -- illustre le biais en 1/M."""
    model = make_model(lambda_)
    likelihood = GaussianLikelihood(model=model, Sigma_obs=SIGMA_OBS_MATRIX)
    nmc = NestedMonteCarloEIG(likelihood=likelihood, prior=prior)

    estimates = {}
    for n_inner in n_inners:
        k = jr.fold_in(key, n_inner)
        estimates[n_inner] = float(nmc.estimate(k, design=None, n_outer=n_outer, n_inner=n_inner))
    return estimates


def plot_comparison(ms, rows, eig_full_mc, full_cert, n_inner, lambda_):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.fill_between(
        ms,
        rows["def_low"],
        rows["def_up"],
        color=COLORS["conservative"],
        alpha=0.25,
        label="conservatif certifie (defaut, Cor. 1 @ $I_p$)",
    )
    ax.plot(ms, rows["def_low"], color=COLORS["conservative"], lw=1.2)
    ax.plot(ms, rows["def_up"], color=COLORS["conservative"], lw=1.2)

    ax.fill_between(
        ms,
        rows["mc_low"],
        rows["mc_up"],
        color=COLORS["exact"],
        alpha=0.20,
        label=f"conservatif, $eig_{{full}}$ = EIG($I_p$) MC, n_inner={n_inner} (non certifie)",
    )
    ax.plot(ms, rows["mc_low"], color=COLORS["exact"], lw=1.2, ls="--")
    ax.plot(ms, rows["mc_up"], color=COLORS["exact"], lw=1.2, ls="--")

    ax.axhline(
        eig_full_mc, color="0.3", lw=1.0, ls=":", label=f"EIG($I_p$) MC = {eig_full_mc:.2f}"
    )
    ax.axhspan(float(full_cert.lower), float(full_cert.upper), color="0.85", alpha=0.4, zorder=0)

    ax.set_xlabel("nombre de capteurs $m$")
    ax.set_ylabel("gain d'information (nats)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(rf"conservatif : certifie vs $eig_{{full}}$ MC -- $\lambda = {lambda_}$")
    fig.tight_layout()
    return fig


def plot_convergence(estimates, full_cert, lambda_):
    """``EIG(I_p)`` estime en fonction de ``n_inner``, bande certifiee en reference."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    n_inners = sorted(estimates)
    ax.semilogx(
        n_inners,
        [estimates[n] for n in n_inners],
        "o-",
        ms=5,
        lw=1.6,
        color=COLORS["exact"],
        label="EIG($I_p$) estime (NMC)",
    )
    ax.axhspan(
        float(full_cert.lower),
        float(full_cert.upper),
        color=COLORS["conservative"],
        alpha=0.25,
        label="bande certifiee (Cor. 1 @ $I_p$)",
    )
    ax.set_xlabel("$n_{inner}$")
    ax.set_ylabel("EIG($I_p$) estime (nats)")
    ax.legend(fontsize=8)
    ax.set_title(rf"convergence du NMC vers la bande certifiee -- $\lambda = {lambda_}$")
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    p.add_argument("--n-samples", type=int, default=300, help="Sigma_Y (paires MC)")
    p.add_argument("--n-gradient", type=int, default=60, help="Sigma_signal (jacobiennes)")
    p.add_argument("--n-outer", type=int, default=150, help="NMC : boucle externe")
    p.add_argument(
        "--n-inner",
        type=int,
        nargs="+",
        default=[150, 500, 2000, 8000],
        help="NMC : tailles de boucle interne testees (la derniere sert a la comparaison)",
    )
    p.add_argument("--m-max", type=int, default=25)
    p.add_argument("--out", default="figures_conservative_mc")
    args = p.parse_args()

    use_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    key = jr.key(0)
    k_diag, k_mc = jr.split(key)

    print(f"[diagnostics] lambda={args.lambda_}, N_samples={args.n_samples}, N_grad={args.n_gradient}")
    prior, Sigma_signal, dg = compute_diagnostics(
        args.lambda_, args.n_samples, args.n_gradient, k_diag
    )
    full_cert = incremental_bounds(dg, None)
    print(f"  Cor. 1 @ I_p (certifie) = [{float(full_cert.lower):.4f}, {float(full_cert.upper):.4f}]")

    print(f"[NMC] EIG(I_p) pour n_inner = {args.n_inner}")
    estimates = eig_full_convergence(prior, args.lambda_, args.n_outer, args.n_inner, k_mc)
    for n_inner, val in estimates.items():
        print(f"  n_inner={n_inner:>6d}  EIG_hat={val:.4f}")

    save(
        plot_convergence(estimates, full_cert, args.lambda_),
        out / "convergence_nmc_vs_certifie.png",
    )

    n_inner_best = max(estimates)
    eig_full_mc = estimates[n_inner_best]

    design = greedy_schur(Sigma_signal, SIGMA_OBS_MATRIX, args.m_max).design
    ms = np.arange(1, args.m_max + 1)
    rows = {k: [] for k in ("def_low", "def_up", "mc_low", "mc_up")}
    for m in ms:
        c_def = conservative_bounds(dg, design[:m])
        c_mc = conservative_bounds(dg, design[:m], eig_full=jnp.asarray(eig_full_mc))
        rows["def_low"].append(float(c_def.lower))
        rows["def_up"].append(float(c_def.upper))
        rows["mc_low"].append(float(c_mc.lower))
        rows["mc_up"].append(float(c_mc.upper))

    save(
        plot_comparison(ms, rows, eig_full_mc, full_cert, n_inner_best, args.lambda_),
        out / "conservative_certifie_vs_mc.png",
    )

    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
