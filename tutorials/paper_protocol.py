#!/usr/bin/env python
r"""Full protocol for the paper -- standard and goal-oriented.

Sweeps ``lambda in {0.0, 0.05, 0.2, 0.75, 1.0}``, standard and goal-oriented
cases (the QoI is selected by ``--go-type``), and compares three routes to
``Sigma_signal``/``Sigma_noise`` (gradient, affine approximation, affine+network
approximation). Each repetition redraws every MC quantity with a fresh key
**and** reselects the design via greedy -- as the NumPy prototype does with
its ``N_REPEATS``.

Figures, mapped to the paper's narrative arc
---------------------------------------------
1. Setup (``01a``/``01b``, ``lambda=0``) -- two separate figures, full field
   in both: (a) standard case, prior/posterior/``theta_true``; (b) GO case,
   same posterior machinery (``inference._mu``/``_cov``, the full field, not
   just the QoI submatrix) but with the QoI region shaded (``qoi_span``), so
   the contrast is visible in one figure -- the posterior contracts inside
   the shaded zone and stays close to the prior outside it, since the GO
   design was never asked to inform anything else.
2. Why (``02``/``02b``, ``lambda > 0``, gradient method, standard and GO) --
   Prop. 1. ``02``: the raw per-mode spectrum ``log(alpha_i)``, ``log(beta_i)``,
   their sum. ``02b``: the two eq. (22)/(23) sub-optimality constants vs
   budget ``m`` (same spectral terms, summed the other way) -- this is the
   figure that shows the incremental/conservative trade-off is real at the
   sensor budgets actually used, not just in the abstract.
3. Certification payoff (``03``): boxplots of the certified bounds
   (incremental and conservative) over the ``N_repeats``, one series per
   method. The conservative bound uses ``eig_full`` estimated by nested
   Monte-Carlo (``NestedMonteCarloEIG`` in standard, ``GoalOrientedNestedMonteCarloEIG``
   in GO) -- not the certified default.
4. Cost of certification -- not produced by this script:
   ``compare_signal_noise_methods.py`` (``--case standard|go``) compares the
   gradient route (Prop. 4, certified) against the affine/affine+NN routes
   (Prop. 3, cheaper but not certified, and not always applicable).

Fragile methods
------------------
The affine approximation alone raises ``ValueError`` (Prop. 3 not satisfied)
as soon as ``lambda`` moves away from 0 at realistic scale -- expected, not a
protocol error. A method that fails for a given ``(lambda, case, repeat)`` is
simply absent from that point: no crash, a ``logger.warning`` message.

Default scale: reduced (pipeline validation, not publication-grade figures).
Increase ``--n-samples``, ``--n-gradient``, ``--net-steps``, ``--nmc-*`` for
production.

Usage
-----
    pixi run -e gpu python tutorials/paper_protocol.py --lambdas 0.0 0.75
    pixi run -e gpu python tutorials/paper_protocol.py --n-repeats 10
"""

import argparse
import logging
from pathlib import Path

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jr # type: ignore
import numpy as np

jax.config.update("jax_enable_x64", False)

from cboed.benchmarks import (
    DOMAIN,
    N,
    SENSOR_BUDGETS,
    SIGMA_OBS_MATRIX,
    forward,
    make_model,
    make_prior,
    build_qoi,
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
from cboed.viz import designs as vd
from cboed.viz import fields as vf
from cboed.viz import spectrum as vs
from cboed.viz.style import save, use_style

logger = logging.getLogger(__name__)

LAMBDAS_PROTOCOL = (0.0, 0.05, 0.2, 0.75, 1.0)
CASES = ("standard", "go")
METHODS = ("gradient", "affine", "affine_nn")

X = np.linspace(DOMAIN[0], DOMAIN[1], N + 2)[1:-1]
QOI_TYPE = None
QOI_H = None
SIGMA_XI_QOI = None
N_QOI = None
B_QOI = None
X_QOI = None


def configure_qoi(qoi_type: str):
    """Configure the protocol QoI after command-line parsing."""
    global QOI_TYPE, QOI_H, SIGMA_XI_QOI, N_QOI, B_QOI, X_QOI
    QOI_TYPE = qoi_type
    QOI_H, SIGMA_XI_QOI, N_QOI = build_qoi(qoi_type)
    if qoi_type == "nuisance":
        B_QOI = jnp.eye(N)[:N_QOI]
        X_QOI = X[:N_QOI]
    else:
        B_QOI = None
        X_QOI = None


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


def compute_repeat(lambda_: float, case: str, key, n_samples: int, n_gradient: int, net_steps: int,
                    n_gradient_chunk_size: int | None = None,
                    qoi_mcmc_warmup: int = 100, qoi_mcmc_step_size: float = 1e-3,
                    qoi_mcmc_thinning: int = 1, qoi_method: str = "mala",
                    delta_theta: float = 1.0, qoi_max_trials: int = 1000):
    """``(Sigma_Y, Sigma_Y_given_theta, {method: (Sigma_signal, Sigma_noise) | None})``.

    ``n_gradient_chunk_size`` : bounds the gradient route's peak memory
    (``expected_jacobian_moments``/``qoi_fisher_moment``) to
    ``n_gradient_chunk_size`` Jacobians at a time instead of ``n_gradient`` --
    see ``cboed.bounds.base.chunked_vmap``. At full-field scale, a single
    fused ``vmap`` over ``n_gradient`` (n_obs, n_obs) Jacobians plus their
    quadratic form can need several GiB even though the final matrices are
    small -- this is what caused the GPU OOM inside ``greedy_schur`` (the
    first point downstream that forces JAX to materialize the result).
    """
    prior, model, u, likelihood, inference, go = build_case(lambda_, case)
    if QOI_TYPE == "energy" and lambda_ != 0.0:
        raise ValueError(
            "The noiseless energy protocol uses the gradient simplification "
            "only available when the forward model is linear (lambda=0)."
        )
    k_pairs, k_Y, k_Yth, k_grad, k_net_f, k_net_g = jr.split(key, 6)

    u_vals, Y, eta = paired_samples(u, prior, k_pairs, n_samples)
    Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k_Y, n_samples)

    if case == "standard":
        Sigma_Y_given_theta = SIGMA_OBS_MATRIX
        theta_for_noise = eta
        Sigma_signal_g, Sigma_noise_g = gradient_diagnostics_standard(
            u, prior, SIGMA_OBS_MATRIX, k_grad, n_gradient, chunk_size=n_gradient_chunk_size
        )
    else:
        Sigma_Y_given_theta = sample_Sigma_Y_given_theta(
            u, prior, B_QOI, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, k_Yth, n_samples,
            h=QOI_H if QOI_TYPE == "energy" else None,
            n_warmup=qoi_mcmc_warmup, step_size=qoi_mcmc_step_size,
            thinning=qoi_mcmc_thinning,
            method=qoi_method, delta_theta=delta_theta, max_trials=qoi_max_trials,
        )
        if not bool(jnp.all(jnp.isfinite(Sigma_Y_given_theta))):
            raise ValueError(
                "Conditional QoI sampling produced non-finite Sigma_Y_given_theta; "
                "increase --delta-theta or --qoi-max-trials."
            )
        if QOI_TYPE == "energy":
            theta_for_noise = jax.vmap(QOI_H)(eta)
        else:
            theta_for_noise = eta[:, :N_QOI]
        if QOI_TYPE == "energy":
            Sigma_signal_g, Sigma_noise_g = Sigma_Y, SIGMA_OBS_MATRIX
        else:
            Sigma_signal_g, Sigma_noise_g = gradient_diagnostics(
                u, QOI_H, prior, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, k_grad, n_gradient,
                chunk_size=n_gradient_chunk_size,
            )

    methods: dict = {"gradient": (Sigma_signal_g, Sigma_noise_g)}
    features_g = jnp.concatenate([Y, theta_for_noise], axis=1)

    if lambda_ == 0.0:
        # Rem. 2.2: the forward model is exactly linear at lambda=0, so the
        # affine denoiser's *ideal* (population) fit already equals the
        # gradient route -- verified directly (max abs diff ~1e-12, and the
        # gradient route needs no MC there either: the Jacobian is constant,
        # so it is exact regardless of n_gradient). Fitting an affine map
        # from finite samples only adds estimation noise on top of an
        # already-exact answer -- and that noise (a ~40000-parameter
        # regression here) is the same order of magnitude as Prop. 3's
        # actual margin at lambda=0 (~1e-5, confirmed via the closed-form
        # Woodbury residual), so the finite-sample fit spuriously reports
        # "R exceeds Sigma_obs" at a rate that has nothing to do with
        # lambda=0 specifically failing. Skip the fit entirely and reuse the
        # exact value instead of chasing a razor-thin margin no realistic
        # sample size clears reliably.
        methods["affine"] = (Sigma_signal_g, Sigma_noise_g)
        methods["affine_nn"] = (Sigma_signal_g, Sigma_noise_g)
        return Sigma_Y, Sigma_Y_given_theta, methods

    try:
        d_f = AffineDenoiser.fit(u_vals, Y)
        d_g = AffineDenoiser.fit(u_vals, features_g)
        Sigma_signal_a = approximation_signal(d_f, u_vals, Y, SIGMA_OBS_MATRIX)
        Sigma_noise_a = approximation_noise(d_g, u_vals, Y, theta_for_noise, SIGMA_OBS_MATRIX)
        methods["affine"] = (Sigma_signal_a, Sigma_noise_a)
    except ValueError as e:
        logger.warning("[affine] failed lambda=%s case=%s: %s", lambda_, case, e)
        methods["affine"] = None

    try:
        d_f_nn = ResidualDenoiser.fit(u_vals, Y, k_net_f, steps=net_steps)
        d_g_nn = ResidualDenoiser.fit(u_vals, features_g, k_net_g, steps=net_steps)
        Sigma_signal_nn = approximation_signal(d_f_nn, u_vals, Y, SIGMA_OBS_MATRIX)
        Sigma_noise_nn = approximation_noise(d_g_nn, u_vals, Y, theta_for_noise, SIGMA_OBS_MATRIX)
        methods["affine_nn"] = (Sigma_signal_nn, Sigma_noise_nn)
    except ValueError as e:
        logger.warning("[affine+NN] failed lambda=%s case=%s: %s", lambda_, case, e)
        methods["affine_nn"] = None

    return Sigma_Y, Sigma_Y_given_theta, methods


def estimate_eig_full(lambda_: float, case: str, key, nmc_n_outer: int, nmc_n_inner: int,
                       nmc_n_inner_theta: int, nmc_n_inner_marginal: int, nmc_chunk_size: int | None = None,
                       nmc_inner_chunk_size: int | None = None,
                       qoi_mcmc_warmup: int = 100, qoi_mcmc_step_size: float = 1e-3,
                       qoi_mcmc_thinning: int = 1, qoi_method: str = "mala",
                       delta_theta: float = 1.0, qoi_max_trials: int = 1000):
    """``EIG(I_p)`` by nested MC -- used for the conservative bound's ``eig_full``.

    ``nmc_chunk_size`` processes outer samples in sequential batches and
    ``nmc_inner_chunk_size`` limits the number of inner forward solves held
    at once. Together, they bound the dominant batch to
    ``nmc_chunk_size x nmc_inner_chunk_size`` rather than
    ``n_outer x n_inner``.
    """
    prior, model, u, likelihood, inference, go = build_case(lambda_, case)
    if case == "standard":
        est = NestedMonteCarloEIG(likelihood=likelihood, prior=prior)
        return est.estimate(
            key,
            n_outer=nmc_n_outer,
            n_inner=nmc_n_inner,
            chunk_size=nmc_chunk_size,
            inner_chunk_size=nmc_inner_chunk_size,
        )
    est = GoalOrientedNestedMonteCarloEIG(
        likelihood=likelihood, prior_eta=prior, B=B_QOI, h=QOI_H,
        Sigma_xi=SIGMA_XI_QOI, n_warmup=qoi_mcmc_warmup,
        step_size=qoi_mcmc_step_size, thinning=qoi_mcmc_thinning,
        conditional_method=qoi_method, delta_theta=delta_theta,
        max_trials=qoi_max_trials,
    )
    return est.estimate(
        key, n_outer=nmc_n_outer, n_inner_theta=nmc_n_inner_theta, n_inner_marginal=nmc_n_inner_marginal,
        chunk_size=nmc_chunk_size, inner_chunk_size=nmc_inner_chunk_size,
    )


STRATEGY_LABELS = ("i-SNR design", "c-SNR design")


def strategies_for_method(Sigma_Y, Sigma_Y_given_theta, Sigma_signal, Sigma_noise, eig_full_mc, certified, budgets):
    """Two designs (i-SNR, c-SNR), four bounds each -- paper protocol Sec. 2.

    Same structure as ``make_figures.py::fig_bounds``: the design chosen by
    optimizing (Sigma_signal, Sigma_Y_given_theta) serves as the reference for
    the "i-SNR design" panel, the one optimizing (Sigma_Y, Sigma_noise) for
    "c-SNR design" -- each displaying its four bounds (inc + cons).

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
        "i-SNR design": greedy_schur(Sigma_signal, Sigma_Y_given_theta, m_max).design,
        "c-SNR design": greedy_schur(Sigma_Y, Sigma_noise, m_max).design,
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
    return out, designs


# =============================================================================
# Compute + cache, per (lambda, case)
# =============================================================================


def compute_lambda_case(lambda_, case, n_repeats, n_samples, n_gradient, net_steps,
                         nmc_n_outer, nmc_n_inner, nmc_n_inner_theta, nmc_n_inner_marginal, budgets, base_seed,
                         eig_full_mode="certified", nmc_chunk_size=None, nmc_inner_chunk_size=None,
                         n_gradient_chunk_size=None, qoi_mcmc_warmup=100,
                         qoi_mcmc_step_size=1e-3, qoi_mcmc_thinning=1,
                         qoi_method="mala", delta_theta=1.0, qoi_max_trials=1000):
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
    selection_counts = np.zeros((2, len(budgets), N), dtype=int)
    once = None  # (Sigma_Y, Sigma_Y_given_theta, methods_diag) from repeat 0 -- for fig 1/2
    eig_full_mc = None
    if eig_full_mode == "mc":
        logger.info("estimating EIG(I_p) by NMC once for lambda=%s, case=%s ...", lambda_, case)
        eig_full_mc = estimate_eig_full(
            lambda_, case, jr.fold_in(jr.key(base_seed), 10_000), nmc_n_outer, nmc_n_inner,
            nmc_n_inner_theta, nmc_n_inner_marginal, nmc_chunk_size=nmc_chunk_size,
            nmc_inner_chunk_size=nmc_inner_chunk_size,
            qoi_mcmc_warmup=qoi_mcmc_warmup, qoi_mcmc_step_size=qoi_mcmc_step_size,
            qoi_mcmc_thinning=qoi_mcmc_thinning,
            qoi_method=qoi_method, delta_theta=delta_theta, qoi_max_trials=qoi_max_trials,
        )

    for r in range(n_repeats):
        key = jr.fold_in(jr.key(base_seed), r)
        k_diag, _ = jr.split(key)
        logger.info("repeat %d/%d ...", r + 1, n_repeats)
        Sigma_Y, Sigma_Y_given_theta, methods_diag = compute_repeat(
            lambda_, case, k_diag, n_samples, n_gradient, net_steps,
            n_gradient_chunk_size=n_gradient_chunk_size,
            qoi_mcmc_warmup=qoi_mcmc_warmup, qoi_mcmc_step_size=qoi_mcmc_step_size,
            qoi_mcmc_thinning=qoi_mcmc_thinning,
            qoi_method=qoi_method, delta_theta=delta_theta, qoi_max_trials=qoi_max_trials,
        )
        if r == 0:
            once = (Sigma_Y, Sigma_Y_given_theta, methods_diag)

        for method in METHODS:
            diag = methods_diag.get(method)
            if diag is None:
                continue
            Sigma_signal, Sigma_noise = diag
            res, designs = strategies_for_method(
                Sigma_Y, Sigma_Y_given_theta, Sigma_signal, Sigma_noise,
                eig_full_mc,
                certified=(method == "gradient" and eig_full_mc is None),
                budgets=budgets,
            )
            for label, d in res.items():
                for k, v in d.items():
                    per_method[method][label][k].append(v)
            if method == "gradient":
                for strategy, label in enumerate(STRATEGY_LABELS):
                    for budget_index, m in enumerate(budgets):
                        selection_counts[strategy, budget_index, np.asarray(designs[label][:m])] += 1

    per_method = {
        m: {label: {k: np.stack(v) for k, v in d.items()} for label, d in strat.items()}
        for m, strat in per_method.items()
        if strat[STRATEGY_LABELS[0]]["inc_low"]
    }
    return once, per_method, selection_counts / n_repeats


CACHE_SCHEMA_VERSION = 4  # bump if cached protocol outputs change -- invalidates stale caches


def cache_path(cache_dir, lambda_, case, eig_full_mode, budgets, qoi_type=None):
    qoi_type = QOI_TYPE if qoi_type is None else qoi_type
    budget_key = "-".join(str(int(budget)) for budget in budgets)
    return cache_dir / (
        f"protocol_v{CACHE_SCHEMA_VERSION}_{qoi_type}_{eig_full_mode}_"
        f"budgets_{budget_key}_lambda_{lambda_:.2f}_{case}.npz"
    )


def load_or_compute(lambda_, case, cache_dir, force, **kwargs):
    path = cache_path(
        cache_dir,
        lambda_,
        case,
        kwargs.get("eig_full_mode", "certified"),
        kwargs["budgets"],
    )
    if path.exists() and not force:
        logger.info("cache  %s", path.name)
        data = dict(np.load(path, allow_pickle=True))
        once = (data["once_Sigma_Y"], data["once_Sigma_Y_given_theta"], data["once_methods"].item())
        per_method = data["per_method"].item()
        return once, per_method, data["selection_frequency"]
    logger.info("computing lambda=%s case=%s ...", lambda_, case)
    once, per_method, selection_frequency = compute_lambda_case(lambda_, case, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        once_Sigma_Y=np.asarray(once[0]),
        once_Sigma_Y_given_theta=np.asarray(once[1]),
        once_methods=np.array({k: (np.asarray(v[0]), np.asarray(v[1])) if v is not None else None
                                for k, v in once[2].items()}, dtype=object),
        per_method=np.array(per_method, dtype=object),
        selection_frequency=selection_frequency,
    )
    return once, per_method, selection_frequency


# =============================================================================
# Figure 1 -- reconstruction, lambda=0: standard (full field) and GO (QoI)
# =============================================================================


def fig_reconstruction_standard(once_standard_lambda0, out: Path, m_design: int = 5, fmt: str = "png"):
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
            sensors=np.asarray(design), sensor_order=np.arange(1, len(design) + 1),
        ),
        out / f"01a_reconstruction_standard_lambda_0.00.{fmt}",
    )


def fig_reconstruction_go(once_go_lambda0, out: Path, m_design: int = 5, fmt: str = "png"):
    """Full field, GO design -- shows the QoI zone shaded *and* the rest of the
    field, so the contrast is visible: the posterior should contract nicely
    inside the QoI (what the design was chosen to inform) and stay close to
    the prior outside it (the GO design has no reason to inform what it was
    never asked to reconstruct).
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
    mu_post = inference._mu(y, prior.mu, design)
    Gamma_post = inference._cov(prior.mu, design)

    post = mu_post + jr.normal(k_post, (200, N)) @ np.linalg.cholesky(
        np.asarray(Gamma_post) + 1e-10 * np.eye(N)
    ).T
    qoi_span = None if QOI_TYPE == "energy" else (float(X[0]), float(X[N_QOI - 1]))

    save(
        vf.plot_reconstruction(
            X, np.asarray(prior.sample(k_prior, 200)), np.asarray(post), np.asarray(theta_true),
            sensors=np.asarray(design), sensor_order=np.arange(1, len(design) + 1), qoi_span=qoi_span,
        ),
        out / f"01b_reconstruction_go_lambda_0.00.{fmt}",
    )


def fig_histogram_energy_posterior(
    once_go_lambda0, out: Path, m_design: int = 5, n_samples: int = 10000,
    histogram_x_max: float | None = 200.0, fmt: str = "png"
):
    """Plot ``theta | Y_m`` for the incremental and conservative GO designs.

    This figure is restricted to ``energy`` at ``lambda=0``. The forward
    model is then linear, so ``eta | Y_m`` is sampled exactly from its Gaussian
    posterior before applying the nonlinear map ``theta = ||eta||**2``.
    """
    if QOI_TYPE != "energy":
        return

    Sigma_Y, Sigma_Y_given_theta, methods_diag = once_go_lambda0
    Sigma_signal, Sigma_noise = methods_diag["gradient"]
    dg = DiagnosticMatrices(
        Sigma_Y=jnp.asarray(Sigma_Y),
        Sigma_Y_given_theta=jnp.asarray(Sigma_Y_given_theta),
        Sigma_signal=jnp.asarray(Sigma_signal),
        Sigma_noise=jnp.asarray(Sigma_noise),
        certified=True,
    )
    designs = {
        "INC": greedy_schur(dg.Sigma_signal, dg.Sigma_Y_given_theta, m_design).design,
        "CONS": greedy_schur(dg.Sigma_Y, dg.Sigma_noise, m_design).design,
    }

    prior, model, _, _, inference, _ = build_case(0.0, "go")
    k_true, k_noise, k_prior, k_post = jr.split(jr.key(43), 4)
    eta_true = prior.sample(k_true, 1)[0]
    y_full = model(eta_true, None)
    y_full = y_full + jr.normal(k_noise, (N,)) * jnp.sqrt(jnp.diag(SIGMA_OBS_MATRIX))

    theta_samples = {}
    for label, design in designs.items():
        y = y_full[design]
        mu_post = inference._mu(y, prior.mu, design)
        Gamma_post = inference._cov(prior.mu, design)
        L_post = jnp.linalg.cholesky(Gamma_post + 1e-10 * jnp.eye(N))
        eta_post = mu_post + jr.normal(k_post, (n_samples, N)) @ L_post.T
        theta_samples[label] = np.asarray(jnp.sum(eta_post**2, axis=1))

    prior_samples = prior.sample(k_prior, n_samples)
    prior_theta = np.asarray(jnp.sum(prior_samples**2, axis=1))
    save(
        vf.plot_energy_posterior_histograms(
            theta_samples,
            float(jnp.sum(eta_true**2)),
            prior_theta_samples=prior_theta,
            x_max=histogram_x_max,
        ),
        out / f"04_histogram_energy_posterior_lambda_0.00_m_{m_design:02d}.{fmt}",
    )


# =============================================================================
# Figure 2 -- spectrum log(alpha), log(beta), sum -- gradient, lambda>0
# =============================================================================


def fig_spectrum(all_once, budgets, out: Path, fmt: str = "png"):
    """``budgets``: the same sensor budgets used everywhere else in the protocol
    (``args.budgets``, cf. ``fig_boxplots``/``strategies_for_method``) -- the
    sub-optimality constants (22)/(23) are evaluated at these ``m``, not some
    unrelated dense range, so this figure is directly comparable to the bounds
    figures at the same budgets.
    """
    ms = np.asarray(budgets)
    for case in CASES:
        alpha_by_lambda, beta_by_lambda = {}, {}
        inc_by_lambda, cons_by_lambda = {}, {}
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
            # the raw per-mode ln(alpha_i)+ln(beta_i) plotted above. Evaluated
            # at the same budgets ``m`` as the bounds figures.
            inc_by_lambda[lambda_] = np.array(
                [q.suboptimality(int(m), "incremental") for m in ms]
            )
            cons_by_lambda[lambda_] = np.array(
                [q.suboptimality(int(m), "conservative") for m in ms]
            )

        if not alpha_by_lambda:
            continue
        save(
            vs.plot_spectrum_vs_lambda(alpha_by_lambda, beta_by_lambda, title=f"gradient, {case}"),
            out / f"02_spectrum_vs_lambda_{case}.{fmt}",
        )
        save(
            vs.plot_suboptimality_vs_lambda(
                ms, inc_by_lambda, cons_by_lambda, title=f"gradient, {case}"
            ),
            out / f"02b_suboptimality_vs_lambda_{case}.{fmt}",
        )


# =============================================================================
# Figure 3 -- selection stability across repeated diagnostics
# =============================================================================


def fig_selection_stability(lambda_, case, selection_frequency, budgets, n_repeats, out: Path, fmt: str = "png"):
    """Selection frequencies of the two gradient-route SNR designs."""
    save(
        vd.plot_selection_frequency(
            X,
            selection_frequency,
            budgets,
            n_repeats,
            title=rf"Sensor-selection stability -- {case}, $\lambda={lambda_}$",
        ),
        out / f"03_selection_stability_{case}_lambda_{lambda_:.2f}.{fmt}",
    )


# =============================================================================
# Figure 4 -- boxplots of the bounds per method
# =============================================================================

def _shared_ylim(per_method_all, method, case, pad_frac=0.05, q=1.0):
    """Global (y_min, y_max) for one (method, case) across every lambda already
    computed -- so the (a)-(d) panels of the same method/case share an axis and
    the *heights* become comparable (a bound sitting lower in one panel than
    another is otherwise invisible once each panel autoscales independently).

    ``q`` : quantile (0 < q <= 1) applied to the pooled values before taking
    the extremes, e.g. ``q=0.99`` clips the worst 1% of a single stray point
    (e.g. a very negative ``cons_low`` at high lambda) so it does not blow up
    the shared scale for every other panel. ``q=1.0`` (default) uses the true
    min/max -- set it lower only if one lambda visibly dominates the range.
    """
    vals = []
    for (lam, c), per_method in per_method_all.items():
        if c != case or method not in per_method:
            continue
        for strat in per_method[method].values():
            for arr in strat.values():
                vals.append(np.asarray(arr).ravel())
    if not vals:
        return None
    pooled = np.concatenate(vals)
    lo, hi = np.quantile(
        pooled,
        [(1 - q) / 2, 1 - (1 - q) / 2]
    )

    return (0, int(np.ceil(hi + 1)))

def fig_boxplots(per_method_all, budgets, out: Path, fmt: str = "png"):
    """One figure per (lambda, case, method) -- same layout as ``07_bounds_lambda``
    (2 panels, i-SNR design / c-SNR design), boxplot at each budget instead of
    a continuous band.
    """
    for (lambda_, case), per_method in per_method_all.items():
        for method, per_strategy in per_method.items():
            ylim = _shared_ylim(per_method_all, method, case)
            save(
                vb.plot_two_strategies_boxplot(
                    budgets, per_strategy, title=rf"{method}, {case}, $\lambda={lambda_}$",
                    ylim=ylim,
                ),
                out / f"03_boxplot_{method}_{case}_lambda_{lambda_:.2f}.{fmt}",
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
        help="Outer NMC batch size for eig-full-mode=mc. Together with --nmc-inner-chunk-size,"
             " bounds the dominant GPU batch. Increase if the GPU has headroom (faster),"
             " decrease if it still OOMs.",
    )
    p.add_argument(
        "--go-type",
        choices=("nuisance", "energy"),
        default="nuisance",
    )
    p.add_argument("--qoi-mcmc-warmup", type=int, default=100)
    p.add_argument("--qoi-mcmc-step-size", type=float, default=1e-3)
    p.add_argument("--qoi-mcmc-thinning", type=int, default=1)
    p.add_argument("--n-histogram", type=int, default=10000)
    p.add_argument(
        "--histogram-x-max", type=float, default=200.0,
        help="Upper display limit for the energy histogram axis; samples are not truncated.",
    )
    p.add_argument(
        "--qoi-method", choices=("mala", "rejection"), default="mala",
        help="Conditional sampler for nonlinear QoIs; energy uses rejection.",
    )
    p.add_argument(
        "--delta-theta", type=float, default=1.0,
        help="Energy tolerance for rejection sampling (default: 1.0).",
    )
    p.add_argument("--qoi-max-trials", type=int, default=1000)
    p.add_argument(
        "--nmc-inner-chunk-size", type=int, default=128,
        help="Inner NMC batch size for eig-full-mode=mc. Inner likelihoods are accumulated"
             " by a stable streaming logsumexp, so GPU memory scales with this value rather"
             " than --nmc-n-inner (or the GO inner sample counts).",
    )
    p.add_argument(
        "--n-gradient-chunk-size", type=int, default=500,
        help="Bounds the gradient route's peak memory (Prop. 4, expected_jacobian_moments/"
             "qoi_fisher_moment) to this many Jacobians at a time instead of all --n-gradient"
             " at once (cboed.bounds.base.chunked_vmap). At full-field scale a single fused"
             " vmap over --n-gradient (n_obs, n_obs) Jacobians can need several GiB even though"
             " the final matrices are small -- this caused a GPU OOM inside greedy_schur (the"
             " first point downstream that forces materialization). Increase if the GPU has"
             " headroom (faster), decrease if it still OOMs.",
    )
    p.add_argument("--out", default="figures_protocol")
    p.add_argument("--cache", default=".cache_protocol")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--format", choices=("png", "pdf"), default="png",
        help="Figure file format. 'pdf' is vector (text/lines stay crisp at any print size) --"
             " use it for figures going into the paper; 'png' (default) for quick inspection.",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity. INFO (default) shows per-repeat/per-figure progress;"
             " DEBUG adds noisier third-party (JAX/matplotlib) messages too.",
    )
    args = p.parse_args()
    configure_qoi(args.go_type)
    if args.go_type == "energy" and any(lambda_ != 0.0 for lambda_ in args.lambdas):
        raise ValueError("The noiseless energy protocol currently requires --lambdas 0.0.")
    if args.go_type == "energy" and args.qoi_method != "rejection":
        raise ValueError("The energy protocol requires --qoi-method rejection.")

    use_style()
    out, cache_dir = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")],
    )

    all_once: dict = {}
    per_method_all: dict = {}

    logger.info("Compute + figures (per lambda x case, as they complete)")
    for lambda_ in args.lambdas:
        for case in args.cases:
            once, per_method, selection_frequency = load_or_compute(
                lambda_, case, cache_dir, args.force,
                n_repeats=args.n_repeats, n_samples=args.n_samples, n_gradient=args.n_gradient,
                net_steps=args.net_steps, nmc_n_outer=args.nmc_n_outer, nmc_n_inner=args.nmc_n_inner,
                nmc_n_inner_theta=args.nmc_n_inner_theta, nmc_n_inner_marginal=args.nmc_n_inner_marginal,
                budgets=args.budgets, base_seed=args.seed, eig_full_mode=args.eig_full_mode,
                nmc_chunk_size=args.nmc_chunk_size, nmc_inner_chunk_size=args.nmc_inner_chunk_size,
                n_gradient_chunk_size=args.n_gradient_chunk_size,
                qoi_mcmc_warmup=args.qoi_mcmc_warmup,
                qoi_mcmc_step_size=args.qoi_mcmc_step_size,
                qoi_mcmc_thinning=args.qoi_mcmc_thinning,
                qoi_method=args.qoi_method,
                delta_theta=args.delta_theta,
                qoi_max_trials=args.qoi_max_trials,
            )
            all_once[(lambda_, case)] = once
            per_method_all[(lambda_, case)] = per_method

            # Figures for this (lambda, case) right away -- no waiting on the
            # rest of the sweep. fig_spectrum is redrawn each time with
            # everything available so far (one curve per lambda>0 already
            # computed): it fills in over the course of the sweep rather
            # than appearing all at once at the end.
            logger.info("figures lambda=%s case=%s ...", lambda_, case)
            if lambda_ == 0.0 and case == "standard":
                fig_reconstruction_standard(once, out, fmt=args.format)
            if lambda_ == 0.0 and case == "go":
                fig_reconstruction_go(once, out, fmt=args.format)
                fig_histogram_energy_posterior(
                    once, out, m_design=max(args.budgets),
                    n_samples=args.n_histogram,
                    histogram_x_max=args.histogram_x_max,
                    fmt=args.format,
                )
            fig_spectrum(all_once, args.budgets, out, fmt=args.format)
            fig_selection_stability(
                lambda_, case, selection_frequency, args.budgets, args.n_repeats, out, fmt=args.format
            )
            fig_boxplots({(lambda_, case): per_method}, args.budgets, out, fmt=args.format)
    logger.info("-> %s", out.resolve())


if __name__ == "__main__":
    main()
