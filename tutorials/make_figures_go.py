#!/usr/bin/env python
r"""Produces the same figures as ``make_figures.py``, goal-oriented case.

QoI ``theta = h(eta) = eta[:N_QOI]`` (first half of the field), via
:class:`cboed.inference.goal_oriented.GoalOrientedModel`. Same
compute/draw separation, same ``.npz`` cache (different folder to avoid
colliding with the standard case).

What changes relative to ``make_figures.py``
------------------------------------------------
``Sigma_signal`` and ``Sigma_Y`` do not depend on ``h`` -- identical to the
standard case (same formulas, see ``bounds/diagnostics/gradient_based.py``).
Only the following change:

- ``Sigma_noise``            : ``assemble(L, H + I_eta + J(h), Sigma_obs)``,
  ``J(h)`` nonzero (vs. ``Sigma_noise = Sigma_obs`` exactly in the standard case).
- ``Sigma_Y_given_theta``    : estimated by ``sample_Sigma_Y_given_theta`` with
  ``B`` = projection matrix (vs. ``Sigma_obs`` exactly in the standard case).
- reconstruction/contraction : restricted to the QoI, via
  ``GoalOrientedModel.prior_covariance_qoi`` / ``posterior_covariance_qoi``.
- the greedy EIG criterion (``fig_greedy``) : ``EIG(inference=go)``, not
  ``EIG(inference=inference)``.

``fig_linear_vs_nonlinear`` is kept for parallelism with the standard case,
but its content (``H(u)``, spectra of ``Sigma_Y``/``Sigma_signal``) is
**identical in substance** to the standard case: neither depends on ``h``.
Nothing goal-oriented is demonstrated there.

Usage
-----
    pixi run -e test python tutorials/make_figures_go.py
    pixi run -e test python tutorials/make_figures_go.py --lambdas 0.0 --force
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cboed.benchmarks import (
    DOMAIN,
    LAMBDAS,
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
from cboed.bounds.diagnostics.gradient_based import (
    assemble,
    expected_jacobian_moments,
    fisher_information_prior,
    qoi_fisher_moment,
)
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y, sample_Sigma_Y_given_theta
from cboed.bounds.quasi_optimality import quasi_optimality
from cboed.criteria.optimality import EIG
from cboed.inference.goal_oriented import GoalOrientedModel
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.optim.greedy import GreedyOptimizer
from cboed.optim.greedy_batch import GreedyBatchReopt
from cboed.optim.greedy_schur import greedy_schur
from cboed.viz import bounds as vb
from cboed.viz import designs as vd
from cboed.viz import fields as vf
from cboed.viz import matrices as vm
from cboed.viz import spectrum as vs
from cboed.viz.style import save, use_style

QOI_H = qoi_projection(N_QOI)
B_QOI = jnp.eye(N)[:N_QOI]  # Jacobian of QOI_H, constant

M_GREEDY = 8  # see make_figures.py: GreedyBatchReopt ~O(m^2 * n_candidates)

N_SAMPLES = 20_000
N_GRADIENT = 5000
M_MAX = max(SENSOR_BUDGETS)
X = np.linspace(DOMAIN[0], DOMAIN[1], N + 2)[1:-1]
X_QOI = X[:N_QOI]


def _make_go(lambda_: float):
    """Prior, model, GoalOrientedModel -- assembled for each figure (cheap)."""
    prior = make_prior()
    model = make_model(lambda_)
    likelihood = GaussianLikelihood(model=model, Sigma_obs=SIGMA_OBS_MATRIX)
    inference = LinearModel(prior=prior, likelihood=likelihood)
    go = GoalOrientedModel(inner=inference, h=QOI_H, Sigma_theta=SIGMA_XI_QOI)
    return prior, model, inference, go


def fig_greedy(d: dict, out: Path) -> None:
    """Three greedy strategies, **a single GO criterion** -- at `lambda = 0` only.

    Same protocol as ``make_figures.py::fig_greedy``, but the criterion is
    ``EIG(inference=go)`` (information on the QoI), not the full-field EIG.

    Expected:
        naive == schur          (`greedy_schur` optimizes (Sigma_signal, Sigma_Y_given_theta)
                                as an oracle for the GO black-box criterion, if the
                                equality of Rem. 2.2 also holds goal-oriented at lambda=0)
        batch >= naive
    """
    _, _, _, go = _make_go(0.0)
    prior = make_prior()
    criterion = EIG(inference=go)
    ms = np.arange(1, M_GREEDY + 1)

    r_naive = GreedyOptimizer(criterion=criterion).run(prior.mu, M_GREEDY, N)
    r_batch = GreedyBatchReopt(criterion=criterion).run(prior.mu, M_GREEDY, N)
    r_schur = greedy_schur(
        jnp.asarray(d["Sigma_signal"]), jnp.asarray(d["Sigma_Y_given_theta"]), M_GREEDY
    )

    designs = {
        "naive (black-box, GO)": np.asarray(r_naive.design),
        "batch (reoptimized, GO)": np.asarray(r_batch.design),
        "schur $O(mp^2)$ (GO)": np.asarray(r_schur.design),
    }
    save(
        vd.plot_sensor_positions(X, designs, m=M_GREEDY, title=r"$\lambda = 0$, goal-oriented"),
        out / "15_greedy_designs_go.png",
    )

    scores = {
        label: np.array([float(criterion.evaluate(prior.mu, jnp.asarray(W[:m]))) for m in ms])
        for label, W in designs.items()
    }
    save(
        vd.plot_greedy_comparison(ms, scores, r"$\lambda = 0$, goal-oriented -- exact QoI EIG"),
        out / "16_greedy_scores_go.png",
    )

    costs = {
        "naive": np.array([m * N for m in ms]),
        "batch (~)": np.array([m * N + 3 * N * m * (m + 1) // 2 for m in ms]),
        "schur": np.array([m * N**2 * 1e-6 for m in ms]),
    }
    save(
        vd.plot_greedy_cost(ms, costs, "criterion evaluations (schur: $p^2$ flops)"),
        out / "17_greedy_cost_go.png",
    )


# =============================================================================
# Compute + cache
# =============================================================================


def compute(lambda_: float):
    """``Sigma_signal`` and ``Sigma_Y`` identical to the standard case; ``Sigma_noise``
    and ``Sigma_Y_given_theta`` reflect the projection onto the QoI.
    """
    prior, u = make_prior(), forward(lambda_)
    k_sample, k_sample_yth, k_grad = jr.split(jr.key(0), 3)

    Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k_sample, N_SAMPLES)
    Sigma_Y_given_theta = sample_Sigma_Y_given_theta(
        u, prior, B_QOI, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, k_sample_yth, N_SAMPLES
    )

    thetas = prior.sample(k_grad, N_GRADIENT)
    L, H = expected_jacobian_moments(u, thetas, SIGMA_OBS_MATRIX)
    I_eta = fisher_information_prior(prior)
    J_h = qoi_fisher_moment(QOI_H, thetas, SIGMA_XI_QOI)

    Sigma_signal = assemble(L, H + I_eta, SIGMA_OBS_MATRIX)
    Sigma_noise = assemble(L, H + I_eta + J_h, SIGMA_OBS_MATRIX)

    return {
        "Sigma_Y": np.asarray(Sigma_Y),
        "Sigma_Y_given_theta": np.asarray(Sigma_Y_given_theta),
        "Sigma_signal": np.asarray(Sigma_signal),
        "Sigma_noise": np.asarray(Sigma_noise),
        "L": np.asarray(L),
        "H": np.asarray(H),
        "I_eta": np.asarray(I_eta),
        "J_h": np.asarray(J_h),
    }


def load(lambda_: float, cache_dir: Path, force: bool) -> dict:
    path = cache_dir / f"diag_go_lambda_{lambda_:.2f}_N{N_SAMPLES}.npz"
    if path.exists() and not force:
        print(f"  cache  {path.name}")
        return dict(np.load(path))
    print(f"  computing lambda={lambda_} (goal-oriented) ...", flush=True)
    data = compute(lambda_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def as_diagnostics(d: dict) -> DiagnosticMatrices:
    return DiagnosticMatrices(
        Sigma_Y=jnp.asarray(d["Sigma_Y"]),
        Sigma_Y_given_theta=jnp.asarray(d["Sigma_Y_given_theta"]),
        Sigma_signal=jnp.asarray(d["Sigma_signal"]),
        Sigma_noise=jnp.asarray(d["Sigma_noise"]),
        certified=True,
    )


# =============================================================================
# Figures
# =============================================================================


def fig_reconstruction(lambda_: float, design, out: Path) -> None:
    """Prior, posterior, ``theta_true`` -- restricted to the QoI (first half)."""
    prior, model, inference, go = _make_go(lambda_)
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

    sensors_qoi = np.asarray(design)[np.asarray(design) < N_QOI]

    save(
        vf.plot_reconstruction(
            X_QOI,
            np.asarray(prior_qoi),
            np.asarray(post_qoi),
            np.asarray(theta_true[:N_QOI]),
            sensors=sensors_qoi if sensors_qoi.size else None,
            laplace_warning=lambda_ > 0,
        ),
        out / f"01_reconstruction_go_lambda_{lambda_:.2f}.png",
    )
    save(
        vf.plot_contraction(
            X_QOI,
            np.asarray(Sigma_theta_prior.diagonal()) ** 0.5,
            np.asarray(Sigma_theta_post.diagonal()) ** 0.5,
            sensors=sensors_qoi if sensors_qoi.size else None,
        ),
        out / f"02_contraction_go_lambda_{lambda_:.2f}.png",
    )


def fig_matrices(lambda_: float, d: dict, out: Path) -> None:
    save(
        vm.plot_diagnostics(as_diagnostics(d), rf"goal-oriented, $\lambda = {lambda_}$"),
        out / f"03_diagnostics_go_lambda_{lambda_:.2f}.png",
    )
    save(
        vm.plot_moments(
            d["L"], d["H"], d["I_eta"], d["J_h"], title=rf"goal-oriented, $\lambda = {lambda_}$"
        ),
        out / f"04_moments_go_lambda_{lambda_:.2f}.png",
    )


def fig_linear_vs_nonlinear(data: dict, out: Path) -> None:
    """``H(u)`` and spectra -- identical in substance to the standard case (h-independent).

    Kept for parallelism with ``make_figures.py``, not to demonstrate
    anything goal-oriented: neither ``H(u)`` nor ``Sigma_signal`` depends
    on ``h``.
    """
    lams = sorted(data)
    save(
        vm.plot_matrix_comparison(
            [data[lam]["H"] for lam in lams],
            [rf"$H(u)$, $\lambda = {lam}$" for lam in lams],
            reference=0,
            title=r"$H(u)$ (goal-oriented -- identique au standard, h-independant)",
        ),
        out / "05_H_vs_lambda_go.png",
    )
    save(
        vm.plot_spectrum_comparison(
            [data[lam]["Sigma_Y"] for lam in lams] + [data[lam]["Sigma_signal"] for lam in lams],
            [rf"$\Sigma_Y$, $\lambda={lam}$" for lam in lams]
            + [rf"$\Sigma_{{signal}}$, $\lambda={lam}$" for lam in lams],
            title="Spectres (goal-oriented -- identiques au standard)",
        ),
        out / "06_spectra_go.png",
    )


def fig_bounds(lambda_: float, d: dict, out: Path):
    """Two designs, four bounds each -- paper protocol, GO diagnostics."""
    dg = as_diagnostics(d)
    strategies = {
        "iEIG$\\geq$ (19) GO": (dg.Sigma_signal, dg.Sigma_Y_given_theta),
        "cEIG$\\geq$ (20) GO": (dg.Sigma_Y, dg.Sigma_noise),
    }
    ms = np.arange(1, M_MAX + 1)
    per_strategy, designs, widths = {}, {}, {}

    for label, (A, B) in strategies.items():
        W = greedy_schur(A, B, M_MAX).design
        designs[label] = np.asarray(W)
        rows = {k: [] for k in ("inc_low", "inc_up", "cons_low", "cons_up")}
        for m in ms:
            i, c = incremental_bounds(dg, W[:m]), conservative_bounds(dg, W[:m])
            rows["inc_low"].append(float(i.lower))
            rows["inc_up"].append(float(i.upper))
            rows["cons_low"].append(float(c.lower))
            rows["cons_up"].append(float(c.upper))
        per_strategy[label] = {k: np.array(v) for k, v in rows.items()}
        widths[f"width inc -- {label}"] = (
            per_strategy[label]["inc_up"] - per_strategy[label]["inc_low"]
        )
        widths[f"width cons -- {label}"] = (
            per_strategy[label]["cons_up"] - per_strategy[label]["cons_low"]
        )

    save(
        vb.plot_two_strategies(ms, per_strategy, rf"$\lambda = {lambda_}$, goal-oriented"),
        out / f"07_bounds_go_lambda_{lambda_:.2f}.png",
    )
    save(
        vb.plot_width_vs_m(ms, widths, rf"$\lambda = {lambda_}$, goal-oriented"),
        out / f"08_widths_go_lambda_{lambda_:.2f}.png",
    )
    return designs


def fig_spectrum(lambda_: float, d: dict, out: Path) -> None:
    dg = as_diagnostics(d)
    q = quasi_optimality(dg)
    ms = np.arange(1, M_MAX + 1)

    save(
        vs.plot_alpha_spectrum(
            np.asarray(q.alpha),
            np.asarray(q.beta),
            q.effective_rank,
            title=rf"goal-oriented, $\lambda = {lambda_}$",
        ),
        out / f"09_alpha_go_lambda_{lambda_:.2f}.png",
    )
    save(
        vs.plot_log_generalized_spectrum(
            np.asarray(q.alpha),
            np.asarray(q.beta),
            title=rf"goal-oriented, $\lambda = {lambda_}$",
        ),
        out / f"09b_log_spectrum_go_lambda_{lambda_:.2f}.png",
    )
    save(
        vs.plot_gap_decomposition(
            np.asarray(q.alpha), np.asarray(q.beta), rf"goal-oriented, $\lambda = {lambda_}$"
        ),
        out / f"10_gap_decomposition_go_lambda_{lambda_:.2f}.png",
    )
    W = greedy_schur(dg.Sigma_signal, dg.Sigma_Y_given_theta, M_MAX).design
    eig_scale = float(incremental_bounds(dg, W).upper)
    save(
        vs.plot_suboptimality(
            ms,
            [q.suboptimality(int(m), "incremental") for m in ms],
            [q.suboptimality(int(m), "conservative") for m in ms],
            eig_scale=eig_scale,
            title=rf"goal-oriented, $\lambda = {lambda_}$",
        ),
        out / f"11_suboptimality_go_lambda_{lambda_:.2f}.png",
    )


def fig_designs(lambda_: float, d: dict, designs: dict, out: Path) -> None:
    save(
        vd.plot_sensor_positions(X, designs, m=M_MAX, title=rf"goal-oriented, $\lambda = {lambda_}$"),
        out / f"12_sensors_go_lambda_{lambda_:.2f}.png",
    )
    save(
        vd.plot_design_on_field(
            X,
            np.diag(d["Sigma_Y"] - d["Sigma_signal"]),
            designs,
            title=r"sensors (GO) and $\mathrm{diag}(\Sigma_Y - \Sigma_{signal})$",
        ),
        out / f"13_design_on_gap_go_lambda_{lambda_:.2f}.png",
    )


def fig_gap_vs_lambda(data: dict, out: Path) -> None:
    lams = sorted(data)
    gaps = [quasi_optimality(as_diagnostics(data[lam])).total_gap for lam in lams]
    save(
        vb.plot_gap_vs_parameter(
            lams,
            gaps,
            mc_floor=abs(gaps[0]) if lams[0] == 0.0 else None,
            title="gap($I_p$) goal-oriented -- non-Gaussianity of $Y$ conditioned on the QoI",
        ),
        out / "14_gap_vs_lambda_go.png",
    )


# =============================================================================
# Orchestration
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    p.add_argument("--out", default="figures_go")
    p.add_argument("--cache", default=".cache")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    use_style()
    out, cache = Path(args.out), Path(args.cache)

    print("Diagnostics (goal-oriented)")
    data = {lam: load(lam, cache, args.force) for lam in args.lambdas}

    print("Figures")
    for lam in args.lambdas:
        d = data[lam]
        design = greedy_schur(
            jnp.asarray(d["Sigma_signal"]), jnp.asarray(d["Sigma_Y_given_theta"]), 10
        ).design
        fig_reconstruction(lam, design, out)
        fig_matrices(lam, d, out)
        designs = fig_bounds(lam, d, out)
        fig_spectrum(lam, d, out)
        fig_designs(lam, d, designs, out)
        print(f"  lambda={lam} ok")
    if 0.0 in args.lambdas:
        fig_greedy(data[0.0], out)

    if len(args.lambdas) > 1:
        fig_linear_vs_nonlinear(data, out)
        fig_gap_vs_lambda(data, out)

    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
