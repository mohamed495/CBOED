#!/usr/bin/env python
r"""Full protocol for the paper -- standard and goal-oriented.

Sweeps ``lambda in {0.0, 0.05, 0.2, 0.75, 1.0}``, standard and goal-oriented
cases (QoI = first half of the field), and compares three routes to
``Sigma_signal``/``Sigma_noise`` (gradient, affine approximation, affine+network
approximation). Each repetition redraws every MC quantity with a fresh key
**and** reselects the design via greedy -- as the NumPy prototype does with
its ``N_REPEATS``.

Three figures
-------------
1. Reconstruction (``lambda=0``): two separate figures -- (a) standard case,
   prior/posterior/``theta_true`` on the full field; (b) GO case, same thing
   but restricted to the QoI, posterior via ``GoalOrientedModel``.
2. Spectrum ``log(alpha_i)``, ``log(beta_i)``, their sum (``lambda > 0``,
   gradient method only, standard and GO) -- Prop. 1.
3. Boxplots of the bounds (incremental and conservative) over the
   ``N_repeats``, one series per method. The conservative bound uses
   ``eig_full`` estimated by nested Monte-Carlo (``NestedMonteCarloEIG`` in
   standard, ``GoalOrientedNestedMonteCarloEIG`` in GO) -- not the certified
   default.

Fragile methods
------------------
The affine approximation alone raises ``ValueError`` (Prop. 3 not satisfied)
as soon as ``lambda`` moves away from 0 at realistic scale -- expected, not a
protocol error. A method that fails for a given ``(lambda, case, repeat)`` is
simply absent from that point: no crash, a diagnostic print.

Default scale: reduced (pipeline validation, not publication-grade figures).
Increase ``--n-samples``, ``--n-gradient``, ``--net-steps``, ``--nmc-*`` for
production.

Usage
-----
    pixi run -e gpu python tutorials/paper_protocol.py --lambdas 0.0 0.75
    pixi run -e gpu python tutorials/paper_protocol.py --n-repeats 10
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cboed.benchmarks import (
    DOMAIN,
    N,
    N_QOI,
    SENSOR_BUDGETS,
    SIGMA_OBS_MATRIX,
    SIGMA_XI_QOI,
    forward,
    make_model,
    make_prior,
    qoi_projection,
)
from cboed.bounds.base import DiagnosticMatrices
from cboed.bounds.bounds import conservative_bounds, incremental_bounds
from cboed.bounds.diagnostics.approximation_based import approximation_noise, approximation_signal
from cboed.bounds.diagnostics.denoisers import AffineDenoiser, ResidualDenoiser
from cboed.bounds.diagnostics.gradient_based import gradient_diagnostics, gradient_diagnostics_standard
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y, sample_Sigma_Y_given_theta
from cboed.bounds.quasi_optimality import quasi_optimality
from cboed.estimators.nmc import NestedMonteCarloEIG
from cboed.estimators.nmc_go import GoalOrientedNestedMonteCarloEIG
from cboed.inference.goal_oriented import GoalOrientedModel
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.optim.greedy_schur import greedy_schur
from cboed.viz import bounds as vb
from cboed.viz import fields as vf
from cboed.viz import spectrum as vs
from cboed.viz.style import save, use_style

LAMBDAS_PROTOCOL = (0.0, 0.05, 0.2, 0.75, 1.0)
CASES = ("standard", "go")
METHODS = ("gradient", "affine", "affine_nn")

QOI_H = qoi_projection(N_QOI)
B_QOI = jnp.eye(N)[:N_QOI]
X = np.linspace(DOMAIN[0], DOMAIN[1], N + 2)[1:-1]
X_QOI = X[:N_QOI]


# =============================================================================
# Setup per (lambda, case)
# =============================================================================


def build_case(lambda_: float, case: str):
    prior = make_prior()
    model = make_model(lambda_)
    u = forward(lambda_)
    likelihood = GaussianLikelihood(model=model, Sigma_obs=SIGMA_OBS_MATRIX)
    inference = LinearModel(prior=prior, likelihood=likelihood)
    go = None
    if case == "go":
        go = GoalOrientedModel(inner=inference, h=QOI_H, Sigma_theta=SIGMA_XI_QOI)
    return prior, model, u, likelihood, inference, go


def paired_samples(u, prior, key, n_samples):
    """``(u(eta), Y = u(eta) + eps, eta)`` -- for the affine/NN denoisers."""
    k_eta, k_eps = jr.split(key)
    eta = prior.sample(k_eta, n_samples)
    u_vals = jax.vmap(u)(eta)
    L = jnp.linalg.cholesky(SIGMA_OBS_MATRIX)
    Y = u_vals + jr.normal(k_eps, u_vals.shape) @ L.T
    return u_vals, Y, eta


# =============================================================================
# Diagnostics per repetition -- the three methods
# =============================================================================


def compute_repeat(lambda_: float, case: str, key, n_samples: int, n_gradient: int, net_steps: int):
    """``(Sigma_Y, Sigma_Y_given_theta, {method: (Sigma_signal, Sigma_noise) | None})``."""
    prior, model, u, likelihood, inference, go = build_case(lambda_, case)
    k_pairs, k_Y, k_Yth, k_grad, k_net_f, k_net_g = jr.split(key, 6)

    u_vals, Y, eta = paired_samples(u, prior, k_pairs, n_samples)
    Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k_Y, n_samples)

    if case == "standard":
        Sigma_Y_given_theta = SIGMA_OBS_MATRIX
        theta_for_noise = eta
        Sigma_signal_g, Sigma_noise_g = gradient_diagnostics_standard(
            u, prior, SIGMA_OBS_MATRIX, k_grad, n_gradient
        )
    else:
        Sigma_Y_given_theta = sample_Sigma_Y_given_theta(
            u, prior, B_QOI, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, k_Yth, n_samples
        )
        theta_for_noise = eta[:, :N_QOI]
        Sigma_signal_g, Sigma_noise_g = gradient_diagnostics(
            u, QOI_H, prior, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, k_grad, n_gradient
        )

    methods: dict = {"gradient": (Sigma_signal_g, Sigma_noise_g)}
    features_g = jnp.concatenate([Y, theta_for_noise], axis=1)

    try:
        d_f = AffineDenoiser.fit(u_vals, Y)
        d_g = AffineDenoiser.fit(u_vals, features_g)
        Sigma_signal_a = approximation_signal(d_f, u_vals, Y, SIGMA_OBS_MATRIX)
        Sigma_noise_a = approximation_noise(d_g, u_vals, Y, theta_for_noise, SIGMA_OBS_MATRIX)
        methods["affine"] = (Sigma_signal_a, Sigma_noise_a)
    except ValueError as e:
        print(f"    [affine] failed lambda={lambda_} case={case}: {e}")
        methods["affine"] = None

    try:
        d_f_nn = ResidualDenoiser.fit(u_vals, Y, k_net_f, steps=net_steps)
        d_g_nn = ResidualDenoiser.fit(u_vals, features_g, k_net_g, steps=net_steps)
        Sigma_signal_nn = approximation_signal(d_f_nn, u_vals, Y, SIGMA_OBS_MATRIX)
        Sigma_noise_nn = approximation_noise(d_g_nn, u_vals, Y, theta_for_noise, SIGMA_OBS_MATRIX)
        methods["affine_nn"] = (Sigma_signal_nn, Sigma_noise_nn)
    except ValueError as e:
        print(f"    [affine+NN] failed lambda={lambda_} case={case}: {e}")
        methods["affine_nn"] = None

    return Sigma_Y, Sigma_Y_given_theta, methods


def estimate_eig_full(lambda_: float, case: str, key, nmc_n_outer: int, nmc_n_inner: int,
                       nmc_n_inner_theta: int, nmc_n_inner_marginal: int, nmc_chunk_size: int | None = None):
    """``EIG(I_p)`` by nested MC -- used for the conservative bound's ``eig_full``.

    ``nmc_chunk_size``: bounds peak memory to ``chunk_size x n_inner``
    (instead of ``n_outer x n_inner``) by processing the outer loop in
    sequential batches -- see ``cboed.estimators.base.chunked_vmap``.
    Essential on GPU as soon as ``n_outer x n_inner`` (or
    ``n_inner_theta``/``n_inner_marginal`` in GO) exceeds available memory --
    this is the cause of the OOM observed with the large default ``--nmc-*``
    values.
    """
    prior, model, u, likelihood, inference, go = build_case(lambda_, case)
    if case == "standard":
        est = NestedMonteCarloEIG(likelihood=likelihood, prior=prior)
        return est.estimate(key, n_outer=nmc_n_outer, n_inner=nmc_n_inner, chunk_size=nmc_chunk_size)
    est = GoalOrientedNestedMonteCarloEIG(likelihood=likelihood, prior_eta=prior, B=B_QOI, Sigma_xi=SIGMA_XI_QOI)
    return est.estimate(
        key, n_outer=nmc_n_outer, n_inner_theta=nmc_n_inner_theta, n_inner_marginal=nmc_n_inner_marginal,
        chunk_size=nmc_chunk_size,
    )


STRATEGY_LABELS = ("iEIG design", "cEIG design")


def strategies_for_method(Sigma_Y, Sigma_Y_given_theta, Sigma_signal, Sigma_noise, eig_full_mc, certified, budgets):
    """Two designs (iEIG, cEIG), four bounds each -- paper protocol Sec. 2.

    Same structure as ``make_figures.py::fig_bounds``: the design chosen by
    optimizing (Sigma_signal, Sigma_Y_given_theta) serves as the reference for
    the "iEIG design" panel, the one optimizing (Sigma_Y, Sigma_noise) for
    "cEIG design" -- each displaying its four bounds (inc + cons).

    Parameters
    ----------
    eig_full_mc : float or None
        ``eig_full`` for the conservative bound (Cor. 2). ``None`` (default,
        as in ``make_figures.py``) -> bounded by Cor. 1 at the full design,
        certified. A value -> the provided MC estimate -- decertifies the
        bound, and at lambda=0 the NMC bias (see conservative_certified_vs_mc.py)
        can fully disconnect the conservative bound from the incremental one,
        which should otherwise coincide exactly (Rem. 2.2, zero gap in the
        linear case).
    """
    dg = DiagnosticMatrices(
        Sigma_Y=Sigma_Y, Sigma_Y_given_theta=Sigma_Y_given_theta,
        Sigma_signal=Sigma_signal, Sigma_noise=Sigma_noise, certified=certified,
    )
    m_max = max(budgets)
    designs = {
        "iEIG design": greedy_schur(Sigma_signal, Sigma_Y_given_theta, m_max).design,
        "cEIG design": greedy_schur(Sigma_Y, Sigma_noise, m_max).design,
    }
    eig_full_arg = None if eig_full_mc is None else jnp.asarray(eig_full_mc)
    out = {}
    for label, W in designs.items():
        rows = {k: [] for k in ("inc_low", "inc_up", "cons_low", "cons_up")}
        for m in budgets:
            inc = incremental_bounds(dg, W[:m])
            cons = conservative_bounds(dg, W[:m], eig_full=eig_full_arg)
            rows["inc_low"].append(float(inc.lower))
            rows["inc_up"].append(float(inc.upper))
            rows["cons_low"].append(float(cons.lower))
            rows["cons_up"].append(float(cons.upper))
        out[label] = {k: np.array(v) for k, v in rows.items()}
    return out


# =============================================================================
# Compute + cache, per (lambda, case)
# =============================================================================


def compute_lambda_case(lambda_, case, n_repeats, n_samples, n_gradient, net_steps,
                         nmc_n_outer, nmc_n_inner, nmc_n_inner_theta, nmc_n_inner_marginal, budgets, base_seed,
                         eig_full_mode="certified", nmc_chunk_size=None):
    """Repetition loop -- 'once' diagnostics (repeat 0) + bounds per repetition.

    Parameters
    ----------
    eig_full_mode : {"certified", "mc"}
        ``"certified"`` (default, as in ``make_figures.py``): conservative
        bound bracketed by Cor. 1 at the full design, never estimated.
        ``"mc"``: estimates ``eig_full`` via NMC -- decertifies the bound,
        costs one extra NMC per repetition, and strongly biases the result at
        small scale (see ``conservative_certified_vs_mc.py``).
    """
    per_method = {
        m: {label: {k: [] for k in ("inc_low", "inc_up", "cons_low", "cons_up")} for label in STRATEGY_LABELS}
        for m in METHODS
    }
    once = None  # (Sigma_Y, Sigma_Y_given_theta, methods_diag) from repeat 0 -- for fig 1/2

    for r in range(n_repeats):
        key = jr.fold_in(jr.key(base_seed), r)
        k_diag, k_eig = jr.split(key)
        print(f"    repeat {r + 1}/{n_repeats} ...", flush=True)
        Sigma_Y, Sigma_Y_given_theta, methods_diag = compute_repeat(
            lambda_, case, k_diag, n_samples, n_gradient, net_steps
        )
        eig_full_mc = None
        if eig_full_mode == "mc":
            eig_full_mc = estimate_eig_full(
                lambda_, case, k_eig, nmc_n_outer, nmc_n_inner, nmc_n_inner_theta, nmc_n_inner_marginal,
                nmc_chunk_size=nmc_chunk_size,
            )
        if r == 0:
            once = (Sigma_Y, Sigma_Y_given_theta, methods_diag)

        for method in METHODS:
            diag = methods_diag.get(method)
            if diag is None:
                continue
            Sigma_signal, Sigma_noise = diag
            res = strategies_for_method(
                Sigma_Y, Sigma_Y_given_theta, Sigma_signal, Sigma_noise,
                eig_full_mc, certified=(method == "gradient"), budgets=budgets,
            )
            for label, d in res.items():
                for k, v in d.items():
                    per_method[method][label][k].append(v)

    per_method = {
        m: {label: {k: np.stack(v) for k, v in d.items()} for label, d in strat.items()}
        for m, strat in per_method.items()
        if strat[STRATEGY_LABELS[0]]["inc_low"]
    }
    return once, per_method


CACHE_SCHEMA_VERSION = 2  # bump if per_method's structure changes -- invalidates stale caches


def cache_path(cache_dir, lambda_, case, eig_full_mode):
    return cache_dir / f"protocol_v{CACHE_SCHEMA_VERSION}_{eig_full_mode}_lambda_{lambda_:.2f}_{case}.npz"


def load_or_compute(lambda_, case, cache_dir, force, **kwargs):
    path = cache_path(cache_dir, lambda_, case, kwargs.get("eig_full_mode", "certified"))
    if path.exists() and not force:
        print(f"  cache  {path.name}")
        data = dict(np.load(path, allow_pickle=True))
        once = (data["once_Sigma_Y"], data["once_Sigma_Y_given_theta"], data["once_methods"].item())
        per_method = data["per_method"].item()
        return once, per_method
    print(f"  computing lambda={lambda_} case={case} ...", flush=True)
    once, per_method = compute_lambda_case(lambda_, case, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        once_Sigma_Y=np.asarray(once[0]),
        once_Sigma_Y_given_theta=np.asarray(once[1]),
        once_methods=np.array({k: (np.asarray(v[0]), np.asarray(v[1])) if v is not None else None
                                for k, v in once[2].items()}, dtype=object),
        per_method=np.array(per_method, dtype=object),
    )
    return once, per_method


# =============================================================================
# Figure 1 -- reconstruction, lambda=0: standard (full field) and GO (QoI)
# =============================================================================


def fig_reconstruction_standard(once_standard_lambda0, out: Path, m_design: int = 10):
    """Full field, prior/posterior/``theta_true`` -- no QoI zone (this is the standard case)."""
    Sigma_Y, Sigma_Y_given_theta, methods_diag = once_standard_lambda0
    Sigma_signal, Sigma_noise = methods_diag["gradient"]

    prior, model, u, likelihood, inference, _ = build_case(0.0, "standard")
    dg = DiagnosticMatrices(
        Sigma_Y=jnp.asarray(Sigma_Y), Sigma_Y_given_theta=jnp.asarray(Sigma_Y_given_theta),
        Sigma_signal=jnp.asarray(Sigma_signal), Sigma_noise=jnp.asarray(Sigma_noise), certified=True,
    )
    design = greedy_schur(dg.Sigma_signal, dg.Sigma_Y_given_theta, m_design).design

    k_true, k_noise, k_prior, k_post = jr.split(jr.key(42), 4)
    theta_true = prior.sample(k_true, 1)[0]
    y = model(theta_true, design) + jr.normal(k_noise, (len(design),)) * jnp.sqrt(
        jnp.diag(SIGMA_OBS_MATRIX)[design]
    )
    mu_post = inference._mu(y, prior.mu, design)
    Gamma_post = inference._cov(prior.mu, design)

    post = mu_post + jr.normal(k_post, (200, N)) @ np.linalg.cholesky(
        np.asarray(Gamma_post) + 1e-10 * np.eye(N)
    ).T

    save(
        vf.plot_reconstruction(
            X, np.asarray(prior.sample(k_prior, 200)), np.asarray(post), np.asarray(theta_true),
            sensors=np.asarray(design),
        ),
        out / "01a_reconstruction_standard_lambda_0.00.png",
    )


def fig_reconstruction_go(once_go_lambda0, out: Path, m_design: int = 10):
    """Restricted to the QoI (first half of the field) -- posterior via GoalOrientedModel,
    same construction as :func:`make_figures_go.fig_reconstruction`.
    """
    Sigma_Y, Sigma_Y_given_theta, methods_diag = once_go_lambda0
    Sigma_signal, Sigma_noise = methods_diag["gradient"]

    prior, model, u, likelihood, inference, go = build_case(0.0, "go")
    dg = DiagnosticMatrices(
        Sigma_Y=jnp.asarray(Sigma_Y), Sigma_Y_given_theta=jnp.asarray(Sigma_Y_given_theta),
        Sigma_signal=jnp.asarray(Sigma_signal), Sigma_noise=jnp.asarray(Sigma_noise), certified=True,
    )
    design = greedy_schur(dg.Sigma_signal, dg.Sigma_Y_given_theta, m_design).design

    k_true, k_noise, k_prior, k_post = jr.split(jr.key(42), 4)
    theta_true = prior.sample(k_true, 1)[0]
    y = model(theta_true, design) + jr.normal(k_noise, (len(design),)) * jnp.sqrt(
        jnp.diag(SIGMA_OBS_MATRIX)[design]
    )
    mu_post_full = inference._mu(y, prior.mu, design)

    Sigma_theta_prior = go.prior_covariance_qoi(prior.mu)
    Sigma_theta_post = go.posterior_covariance_qoi(prior.mu, design)

    L_prior = jnp.linalg.cholesky(Sigma_theta_prior + 1e-10 * jnp.eye(N_QOI))
    L_post = jnp.linalg.cholesky(Sigma_theta_post + 1e-10 * jnp.eye(N_QOI))
    prior_qoi = prior.mu[:N_QOI] + jr.normal(k_prior, (200, N_QOI)) @ L_prior.T
    post_qoi = mu_post_full[:N_QOI] + jr.normal(k_post, (200, N_QOI)) @ L_post.T

    design_np = np.asarray(design)
    sensors_qoi = design_np[design_np < N_QOI]

    save(
        vf.plot_reconstruction(
            X_QOI, np.asarray(prior_qoi), np.asarray(post_qoi), np.asarray(theta_true[:N_QOI]),
            sensors=sensors_qoi if sensors_qoi.size else None,
        ),
        out / "01b_reconstruction_go_lambda_0.00.png",
    )


# =============================================================================
# Figure 2 -- spectrum log(alpha), log(beta), sum -- gradient, lambda>0
# =============================================================================


def fig_spectrum(all_once, out: Path):
    for case in CASES:
        alpha_by_lambda, beta_by_lambda = {}, {}
        inc_by_lambda, cons_by_lambda = {}, {}
        ms_dense = None
        for lambda_ in LAMBDAS_PROTOCOL:
            if lambda_ == 0.0:
                continue
            once = all_once.get((lambda_, case))
            if once is None:
                continue
            Sigma_Y, Sigma_Y_given_theta, methods_diag = once
            Sigma_signal, Sigma_noise = methods_diag["gradient"]
            dg = DiagnosticMatrices(
                Sigma_Y=jnp.asarray(Sigma_Y), Sigma_Y_given_theta=jnp.asarray(Sigma_Y_given_theta),
                Sigma_signal=jnp.asarray(Sigma_signal), Sigma_noise=jnp.asarray(Sigma_noise), certified=True,
            )
            q = quasi_optimality(dg)
            alpha_by_lambda[lambda_] = np.asarray(q.alpha)
            beta_by_lambda[lambda_] = np.asarray(q.beta)

            # Eq. (22)/(23): two distinct partial sums over the SAME spectral
            # terms -- first m (incremental) vs first d-m (conservative), not
            # the raw per-mode ln(alpha_i)+ln(beta_i) plotted above.
            if ms_dense is None:
                ms_dense = np.arange(1, q.alpha.shape[0])
            inc_by_lambda[lambda_] = np.array(
                [q.suboptimality(int(m), "incremental") for m in ms_dense]
            )
            cons_by_lambda[lambda_] = np.array(
                [q.suboptimality(int(m), "conservative") for m in ms_dense]
            )

        if not alpha_by_lambda:
            continue
        save(
            vs.plot_spectrum_vs_lambda(alpha_by_lambda, beta_by_lambda, title=f"gradient, {case}"),
            out / f"02_spectrum_vs_lambda_{case}.png",
        )
        save(
            vs.plot_suboptimality_vs_lambda(
                ms_dense, inc_by_lambda, cons_by_lambda, title=f"gradient, {case}"
            ),
            out / f"02b_suboptimality_vs_lambda_{case}.png",
        )


# =============================================================================
# Figure 3 -- boxplots of the bounds per method
# =============================================================================


def fig_boxplots(per_method_all, budgets, out: Path):
    """One figure per (lambda, case, method) -- same layout as ``07_bounds_lambda``
    (2 panels, iEIG design / cEIG design), boxplot at each budget instead of
    a continuous band.
    """
    for (lambda_, case), per_method in per_method_all.items():
        for method, per_strategy in per_method.items():
            save(
                vb.plot_two_strategies_boxplot(
                    budgets, per_strategy, title=rf"{method}, {case}, $\lambda={lambda_}$"
                ),
                out / f"03_boxplot_{method}_{case}_lambda_{lambda_:.2f}.png",
            )


# =============================================================================
# Orchestration
# =============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS_PROTOCOL))
    p.add_argument("--cases", type=str, nargs="+", default=list(CASES), choices=CASES)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=20000)
    p.add_argument("--n-gradient", type=int, default=5000)
    p.add_argument("--net-steps", type=int, default=500)
    p.add_argument("--nmc-n-outer", type=int, default=5000)
    p.add_argument("--nmc-n-inner", type=int, default=50000)
    p.add_argument("--nmc-n-inner-theta", type=int, default=5000)
    p.add_argument("--nmc-n-inner-marginal", type=int, default=50000)
    p.add_argument("--budgets", type=int, nargs="+", default=list(SENSOR_BUDGETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--eig-full-mode", choices=("certified", "mc"), default="certified",
        help="Conservative bound: 'certified' (default, as in make_figures.py, Cor. 1 at the"
             " full design) or 'mc' (NMC -- decertified, strongly biased at small scale).",
    )
    p.add_argument(
        "--nmc-chunk-size", type=int, default=200,
        help="Bounds the NMC's peak memory (eig-full-mode=mc) to chunk_size x n_inner instead"
             " of n_outer x n_inner, by processing the outer loop in sequential batches"
             " (cboed.estimators.base.chunked_vmap). Necessary on GPU as soon as"
             " n_outer x n_inner (or n_inner_theta/n_inner_marginal in GO) saturates memory --"
             " this is the cause of the OOM with the default --nmc-* values without chunking."
             " Increase if the GPU has headroom (faster), decrease if it still OOMs.",
    )
    p.add_argument("--out", default="figures_protocol")
    p.add_argument("--cache", default=".cache_protocol")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    use_style()
    out, cache_dir = Path(args.out), Path(args.cache)

    all_once: dict = {}
    per_method_all: dict = {}

    print("Compute + figures (per lambda x case, as they complete)")
    for lambda_ in args.lambdas:
        for case in args.cases:
            once, per_method = load_or_compute(
                lambda_, case, cache_dir, args.force,
                n_repeats=args.n_repeats, n_samples=args.n_samples, n_gradient=args.n_gradient,
                net_steps=args.net_steps, nmc_n_outer=args.nmc_n_outer, nmc_n_inner=args.nmc_n_inner,
                nmc_n_inner_theta=args.nmc_n_inner_theta, nmc_n_inner_marginal=args.nmc_n_inner_marginal,
                budgets=args.budgets, base_seed=args.seed, eig_full_mode=args.eig_full_mode,
                nmc_chunk_size=args.nmc_chunk_size,
            )
            all_once[(lambda_, case)] = once
            per_method_all[(lambda_, case)] = per_method

            # Figures for this (lambda, case) right away -- no waiting on the
            # rest of the sweep. fig_spectrum is redrawn each time with
            # everything available so far (one curve per lambda>0 already
            # computed): it fills in over the course of the sweep rather
            # than appearing all at once at the end.
            print(f"  figures lambda={lambda_} case={case} ...", flush=True)
            if lambda_ == 0.0 and case == "standard":
                fig_reconstruction_standard(once, out)
            if lambda_ == 0.0 and case == "go":
                fig_reconstruction_go(once, out)
            fig_spectrum(all_once, out)
            fig_boxplots({(lambda_, case): per_method}, args.budgets, out)

    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
