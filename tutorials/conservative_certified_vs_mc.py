#!/usr/bin/env python
r"""Conservative bounds: certified ``eig_full`` (default) vs Monte-Carlo estimate.

``conservative_bounds`` (Cor. 2) needs a reference point ``EIG(I_p)``
(the information of the full dataset). By default (``eig_full=None``), it is
bounded for free by Corollary 1 applied to the full design -- never
estimated, which keeps the bound certified.

The NumPy prototype does the opposite: it injects an ``eig_offset`` estimated
by nested Monte-Carlo (NMC) into the same formula -- which decertifies the
bound. This script compares the two on the same set of diagnostics.

What this shows
-----------------
NMC (:class:`cboed.estimators.nmc.NestedMonteCarloEIG`) systematically
underestimates ``log p(y)`` at finite ``n_inner`` (Jensen bias on the
``logsumexp``), and therefore overestimates the EIG. At high observation
dimension (``n_obs = 200`` here), convergence in ``n_inner`` is very slow:
even ``n_inner = 8000`` stays far from the certified band on the default
benchmark. Injecting this estimate into (17)-(18) shifts the entire
conservative bracket outside the band guaranteed by the theorem.

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
    """Standard diagnostics (``Sigma_Y_given_theta = Sigma_noise = Sigma_obs``)."""
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
    """``EIG(I_p)`` by NMC at several ``n_inner`` -- illustrates the 1/M bias."""
    model = make_model(lambda_)
    likelihood = GaussianLikelihood(model=model, Sigma_obs=SIGMA_OBS_MATRIX)
    nmc = NestedMonteCarloEIG(likelihood=likelihood, prior=prior)

    estimates = {}
    for n_inner in n_inners:
        k = jr.fold_in(key, n_inner)
        estimates[n_inner] = float(nmc.estimate(k, design=None, n_outer=n_outer, n_inner=n_inner, chunk_size=25))
    return estimates


def plot_comparison(ms, rows, eig_full_mc, full_cert, n_inner, lambda_):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.fill_between(
        ms,
        rows["def_low"],
        rows["def_up"],
        color=COLORS["conservative"],
        alpha=0.25,
        label="certified conservative (default, Cor. 1 @ $I_p$)",
    )
    ax.plot(ms, rows["def_low"], color=COLORS["conservative"], lw=1.2)
    ax.plot(ms, rows["def_up"], color=COLORS["conservative"], lw=1.2)

    ax.fill_between(
        ms,
        rows["mc_low"],
        rows["mc_up"],
        color=COLORS["exact"],
        alpha=0.20,
        label=f"conservative, $eig_{{full}}$ = EIG($I_p$) MC, n_inner={n_inner} (not certified)",
    )
    ax.plot(ms, rows["mc_low"], color=COLORS["exact"], lw=1.2, ls="--")
    ax.plot(ms, rows["mc_up"], color=COLORS["exact"], lw=1.2, ls="--")

    ax.axhline(
        eig_full_mc, color="0.3", lw=1.0, ls=":", label=f"EIG($I_p$) MC = {eig_full_mc:.2f}"
    )
    ax.axhspan(float(full_cert.lower), float(full_cert.upper), color="0.85", alpha=0.4, zorder=0)

    ax.set_xlabel("number of sensors $m$")
    ax.set_ylabel("information gain (nats)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(rf"conservative: certified vs $eig_{{full}}$ MC -- $\lambda = {lambda_}$")
    fig.tight_layout()
    return fig


def plot_convergence(estimates, full_cert, lambda_):
    """``EIG(I_p)`` estimated as a function of ``n_inner``, certified band as reference."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    n_inners = sorted(estimates)
    ax.semilogx(
        n_inners,
        [estimates[n] for n in n_inners],
        "o-",
        ms=5,
        lw=1.6,
        color=COLORS["exact"],
        label="EIG($I_p$) estimate (NMC)",
    )
    ax.axhspan(
        float(full_cert.lower),
        float(full_cert.upper),
        color=COLORS["conservative"],
        alpha=0.25,
        label="certified band (Cor. 1 @ $I_p$)",
    )
    ax.set_xlabel("$n_{inner}$")
    ax.set_ylabel("EIG($I_p$) estimate (nats)")
    ax.legend(fontsize=8)
    ax.set_title(rf"NMC convergence toward the certified band -- $\lambda = {lambda_}$")
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    p.add_argument("--n-samples", type=int, default=300, help="Sigma_Y (MC pairs)")
    p.add_argument("--n-gradient", type=int, default=60, help="Sigma_signal (Jacobians)")
    p.add_argument("--n-outer", type=int, default=150, help="NMC: outer loop")
    p.add_argument(
        "--n-inner",
        type=int,
        nargs="+",
        default=[150, 500, 2000, 8000],
        help="NMC: inner loop sizes tested (the last one is used for the comparison)",
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
    print(f"  Cor. 1 @ I_p (certified) = [{float(full_cert.lower):.4f}, {float(full_cert.upper):.4f}]")

    print(f"[NMC] EIG(I_p) for n_inner = {args.n_inner}")
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
