# r"""Figure orchestration for λ=0.0 -- linear-Gaussian case (with MC estimates).

# Generates the matrices estimated by Monte Carlo:
#   - Σ_Y : Cov(u(θ)) via sample_Sigma_Y()
#   - Σ_signal, Σ_noise : via gradient_diagnostics_standard()
#   - Σ_Y|θ = Σ_obs (exact in the standard case)

# The results are visualized in 5 figures.
# Output: outputs/lambda_0p0/
# """

# from pathlib import Path

# import jax.numpy as jnp
# import jax.random as jr

# from cboed.benchmarks import SIGMA_OBS_MATRIX, forward, make_model, make_prior
# from cboed.bounds.diagnostics.gradient_based import gradient_diagnostics_standard
# from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y, sample_Sigma_Y_given_theta
# from cboed.viz.matrices import plot_matrix_comparison
# from cboed.viz.style import COLORS, save, use_style


# def main():
#     """Full pipeline λ=0.0."""
#     use_style()
#     output_dir = Path("outputs/lambda_0p0")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     print("=" * 70)
#     print("Figure orchestration | λ = 0.0 (linear-Gaussian, MC)")
#     print("=" * 70)

#     # -----------------------------------------------------------------------
#     # 1. Configuration
#     # -----------------------------------------------------------------------
#     lambda_ = 0.0
#     n_samples = 1000  # MC estimates
#     prior = make_prior()
#     model = make_model(lambda_)
#     u = forward(lambda_)  # Forward model without design

#     print(f"\n[Setup]")
#     print(f"  λ = {lambda_}")
#     print(f"  Prior: GP Matern32(σ={prior.prior.kernel.sigma:.2f}, ℓ={prior.prior.kernel.length_scale:.2f})")
#     print(f"  n_samples (MC) = {n_samples}")

#     # -----------------------------------------------------------------------
#     # 2. Estimate the matrices by Monte Carlo
#     # -----------------------------------------------------------------------
#     print(f"\n[MC estimation]")
#     key = jr.key(42)
#     k1, k2, k3 = jr.split(key, 3)

#     # Sigma_Y via paired-differences sampling
#     Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k1, n_samples)
#     print(f"  Σ_Y estimated (pairs MC)")

#     # Sigma_signal, Sigma_noise via gradient
#     Sigma_signal, Sigma_noise = gradient_diagnostics_standard(
#         u, prior, SIGMA_OBS_MATRIX, k2, n_samples
#     )
#     print(f"  Σ_signal, Σ_noise estimated (gradient)")

#     # Sigma_Y_given_theta via sampling with change of variables
#     # In the standard case: B = Identity, Sigma_xi ≈ 0
#     n_param = prior.mu.shape[0]
#     B = jnp.eye(n_param)
#     Sigma_xi = 1e-8 * jnp.eye(n_param)
#     Sigma_Y_given_theta = sample_Sigma_Y_given_theta(
#         u, prior, B, SIGMA_OBS_MATRIX, Sigma_xi, k3, n_samples
#     )
#     print(f"  Σ_Y|θ estimated (sample)")

#     # Prior covariance
#     Sigma_theta = prior.Sigma()

#     # -----------------------------------------------------------------------
#     # 3. Spectra
#     # -----------------------------------------------------------------------
#     print(f"\n[Spectra]")
#     eigs_theta = jnp.sort(jnp.linalg.eigvalsh(Sigma_theta))[::-1]
#     eigs_signal = jnp.sort(jnp.linalg.eigvalsh(Sigma_signal))[::-1]
#     eigs_Y = jnp.sort(jnp.linalg.eigvalsh(Sigma_Y))[::-1]
#     eigs_noise = jnp.sort(jnp.linalg.eigvalsh(Sigma_noise))[::-1]

#     print(f"  λ_max(Σ_θ) = {eigs_theta[0]:.3e}, λ_min = {eigs_theta[-1]:.3e}")
#     print(f"  λ_max(Σ_signal) = {eigs_signal[0]:.3e}, λ_min = {eigs_signal[-1]:.3e}")
#     print(f"  λ_max(Σ_Y) = {eigs_Y[0]:.3e}, λ_min = {eigs_Y[-1]:.3e}")

#     # -----------------------------------------------------------------------
#     # 4. Figures
#     # -----------------------------------------------------------------------
#     print(f"\n[Figures]")

#     # Figure 1: Σ_theta alone
#     fig1, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(figsize=(6, 5))
#     im1 = ax.imshow(Sigma_theta, cmap="viridis")
#     ax.set_title(r"$\Sigma_\theta$ (prior GP)")
#     ax.set_xlabel("Parameter index j")
#     ax.set_ylabel("Parameter index i")
#     __import__("matplotlib.pyplot", fromlist=["colorbar"]).colorbar(im1, ax=ax, label="covariance")
#     fig1.tight_layout()
#     path1 = save(fig1, output_dir / "01_Sigma_theta.png")
#     print(f"  → {path1.name}")

#     # Figure 2: Σ_Y and Σ_signal (numerator)
#     fig2 = plot_matrix_comparison(
#         [Sigma_Y, Sigma_signal],
#         labels=[r"$\Sigma_Y$ (MC)", r"$\Sigma_{\mathrm{signal}}$ (gradient)"],
#         title=rf"Numerator: $\Sigma_Y$ vs $\Sigma_{{\mathrm{{signal}}}}$ | $\lambda = {lambda_:.2f}$"
#     )
#     path2 = save(fig2, output_dir / "02_numerator.png")
#     print(f"  → {path2.name}")

#     # Figure 3: Σ_Y|θ and Σ_noise (denominator)
#     fig3 = plot_matrix_comparison(
#         [Sigma_Y_given_theta, Sigma_noise],
#         labels=[r"$\Sigma_{Y|\theta}$ (exact)", r"$\Sigma_{\mathrm{noise}}$ (gradient)"],
#         title=rf"Denominator: $\Sigma_{{Y|\theta}}$ vs $\Sigma_{{\mathrm{{noise}}}}$ | $\lambda = {lambda_:.2f}$"
#     )
#     path3 = save(fig3, output_dir / "03_denominator.png")
#     print(f"  → {path3.name}")

#     # Figure 4: Spectra
#     fig4, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(figsize=(8, 5))
#     n_show = 50
#     ax.semilogy(range(n_show), eigs_theta[:n_show], "o-", label=r"$\Sigma_\theta$",
#                 color=COLORS["prior"], lw=1.5, markersize=4)
#     ax.semilogy(range(n_show), eigs_signal[:n_show], "s-", label=r"$\Sigma_{\mathrm{signal}}$",
#                 color=COLORS["Sigma_signal"], lw=1.5, markersize=4)
#     ax.semilogy(range(n_show), eigs_Y[:n_show], "^-", label=r"$\Sigma_Y$",
#                 color=COLORS["Sigma_Y"], lw=1.5, markersize=4)
#     ax.semilogy(range(n_show), eigs_noise[:n_show], "d-", label=r"$\Sigma_{\mathrm{noise}}$",
#                 color=COLORS["Sigma_noise"], lw=1.5, markersize=4)
#     ax.set_xlabel("Mode index $i$")
#     ax.set_ylabel("Eigenvalue")
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)
#     ax.set_title(rf"Spectra (MC+gradient) | $\lambda = {lambda_:.2f}$")
#     fig4.tight_layout()
#     path4 = save(fig4, output_dir / "04_spectra.png")
#     print(f"  → {path4.name}")

#     # Figure 5: Jacobian
#     J = model.jacobian(prior.mu, None)
#     fig5, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(figsize=(6, 5))
#     im5 = ax.imshow(J, aspect="auto", cmap="viridis")
#     ax.set_xlabel("Observed index")
#     ax.set_ylabel("Parameter index")
#     ax.set_title(rf"Jacobian $J$ (Burgers, $\lambda = {lambda_:.2f}$)")
#     __import__("matplotlib.pyplot", fromlist=["colorbar"]).colorbar(im5, ax=ax)
#     fig5.tight_layout()
#     path5 = save(fig5, output_dir / "05_jacobian.png")
#     print(f"  → {path5.name}")

#     # -----------------------------------------------------------------------
#     # 5. Summary
#     # -----------------------------------------------------------------------
#     print(f"\n[Summary]")
#     print(f"  MC estimates with {n_samples} sample pairs")
#     print(f"  Figures saved to: {output_dir.resolve()}")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
r"""Produces all figures.

Separation: this script **computes** (with a disk cache) and calls ``cboed.viz``,
which only draws. No figure is built here, no computation is done there.

The cache
--------
The four diagnostic matrices are **all** the cost: once computed, the
bounds are ``slogdet`` of submatrices, the greedy is ``O(m p^2)``, the spectrum
an ``eigvalsh``. One ``.npz`` per ``(lambda, N)`` and replotting becomes instant.

``--force`` to recompute.

Usage
-----
    pixi run -e test python scripts/make_figures.py
    pixi run -e test python scripts/make_figures.py --lambdas 0.0 1.0 --force
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cboed.benchmarks import (
    DOMAIN,
    LAMBDAS,
    SENSOR_BUDGETS,
    SIGMA_OBS_MATRIX,
    N,
    forward,
    make_model,
    make_prior,
)
from cboed.bounds.base import DiagnosticMatrices
from cboed.bounds.bounds import conservative_bounds, incremental_bounds
from cboed.bounds.diagnostics.gradient_based import (
    assemble,
    expected_jacobian_moments,
    fisher_information_prior,
)
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y
from cboed.bounds.quasi_optimality import quasi_optimality
from cboed.criteria.optimality import EIG
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

M_GREEDY = 8  # GreedyBatchReopt is O(m^2 * n_candidates): ~23,000 evaluations at m=8

N_SAMPLES = 20_000
N_GRADIENT = 500
M_MAX = max(SENSOR_BUDGETS)
X = np.linspace(DOMAIN[0], DOMAIN[1], N + 2)[1:-1]  # INTERIOR points


def fig_greedy(d: dict, out: Path) -> None:
    """Three greedy strategies, **a single criterion** -- at `lambda = 0` only.

    At `lambda = 0` the model is linear, so `EIG.evaluate` is exact and
    `incremental_lower = incremental_upper = EIG` (Rem. 2.2). The three strategies
    then optimize the **same** quantity: the figure compares *algorithms*.

    At `lambda > 0` it would no longer do so -- `greedy_schur` maximizes a bound,
    `GreedyOptimizer` maximizes Laplace. Two criteria, not two algorithms.

    Expected:
        naive == schur          (`greedy.py` is the oracle of `greedy_schur.py`)
        batch >= naive          (reoptimization corrects earlier-step errors;
                                at equality, the simple greedy was already
                                optimal on this problem)
    """
    prior, model = make_prior(), make_model(0.0)
    inference = LinearModel(
        prior=prior,
        likelihood=GaussianLikelihood(model=model, Sigma_obs=SIGMA_OBS_MATRIX),
    )
    criterion = EIG(inference=inference)
    ms = np.arange(1, M_GREEDY + 1)

    r_naive = GreedyOptimizer(criterion=criterion).run(prior.mu, M_GREEDY, N)
    r_batch = GreedyBatchReopt(criterion=criterion).run(prior.mu, M_GREEDY, N)
    r_schur = greedy_schur(
        jnp.asarray(d["Sigma_signal"]), jnp.asarray(d["Sigma_Y_given_theta"]), M_GREEDY
    )

    designs = {
        "naive (black-box)": np.asarray(r_naive.design),
        "batch (reoptimized)": np.asarray(r_batch.design),
        "schur $O(m p^2)$": np.asarray(r_schur.design),
    }
    save(
        vd.plot_sensor_positions(X, designs, m=M_GREEDY, title=r"$\lambda = 0$"),
        out / "15_greedy_designs.png",
    )

    # all evaluated at the exact EIG: otherwise we would be comparing criteria
    scores = {
        label: np.array([float(criterion.evaluate(prior.mu, jnp.asarray(W[:m]))) for m in ms])
        for label, W in designs.items()
    }
    save(
        vd.plot_greedy_comparison(ms, scores, r"$\lambda = 0$ -- exact EIG"),
        out / "16_greedy_scores.png",
    )

    # cost: the number of criterion evaluations, not wall-clock time
    costs = {
        "naive": np.array([m * N for m in ms]),
        "batch (~)": np.array([m * N + 3 * N * m * (m + 1) // 2 for m in ms]),
        "schur": np.array([m * N**2 * 1e-6 for m in ms]),  # p^2 flops, not an evaluation
    }
    save(
        vd.plot_greedy_cost(ms, costs, "criterion evaluations (schur: $p^2$ flops)"),
        out / "17_greedy_cost.png",
    )


# =============================================================================
# Compute + cache
# =============================================================================


def compute(lambda_: float):
    """The four matrices and the moments. All the model's cost is here.

    Standard setting: Prop. 2 sets ``Sigma_{Y|theta} = Sigma_noise = Sigma_obs``
    exactly -- no :func:`gradient_diagnostics` with ``h = id`` and a tiny
    ``Sigma_xi``, whose limit is singular.
    """
    prior, u = make_prior(), forward(lambda_)
    k_sample, k_grad = jr.split(jr.key(0))

    Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k_sample, N_SAMPLES)

    thetas = prior.sample(k_grad, N_GRADIENT)
    L, H = expected_jacobian_moments(u, thetas, SIGMA_OBS_MATRIX)
    I_eta = fisher_information_prior(prior)
    Sigma_signal = assemble(L, H + I_eta, SIGMA_OBS_MATRIX)

    return {
        "Sigma_Y": np.asarray(Sigma_Y),
        "Sigma_Y_given_theta": np.asarray(SIGMA_OBS_MATRIX),
        "Sigma_signal": np.asarray(Sigma_signal),
        "Sigma_noise": np.asarray(SIGMA_OBS_MATRIX),
        "L": np.asarray(L),
        "H": np.asarray(H),
        "I_eta": np.asarray(I_eta),
    }


def load(lambda_: float, cache_dir: Path, force: bool) -> dict:
    path = cache_dir / f"diag_lambda_{lambda_:.2f}_N{N_SAMPLES}.npz"
    if path.exists() and not force:
        print(f"  cache  {path.name}")
        return dict(np.load(path))
    print(f"  computing lambda={lambda_} ...", flush=True)
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
    """Prior, posterior, ``theta_true``.

    The posterior is computed explicitly (Laplace linearized at ``mu_prior``)
    rather than read from ``LinearModel._mu``: this is an implementation
    detail, and writing out the update makes visible what it approximates.
    """
    prior, model = make_prior(), make_model(lambda_)
    k_true, k_noise, k_prior, k_post = jr.split(jr.key(42), 4)

    theta_true = prior.sample(k_true, 1)[0]
    y = model(theta_true, design) + jr.normal(k_noise, (len(design),)) * jnp.sqrt(
        jnp.diag(SIGMA_OBS_MATRIX)[design]
    )

    J = model.jacobian(prior.mu, design)
    S = SIGMA_OBS_MATRIX[jnp.ix_(design, design)]
    prec = prior.prior_precision_matmul(jnp.eye(N)) + J.T @ jnp.linalg.solve(S, J)
    Gamma_post = jnp.linalg.inv(prec)
    residual = y - model(prior.mu, design)
    mu_post = prior.mu + Gamma_post @ J.T @ jnp.linalg.solve(S, residual)

    post = (
        mu_post
        + jr.normal(k_post, (200, N))
        @ np.linalg.cholesky(np.asarray(Gamma_post) + 1e-10 * np.eye(N)).T
    )

    save(
        vf.plot_reconstruction(
            X,
            np.asarray(prior.sample(k_prior, 200)),
            np.asarray(post),
            np.asarray(theta_true),
            sensors=np.asarray(design),
            laplace_warning=lambda_ > 0,
        ),
        out / f"01_reconstruction_lambda_{lambda_:.2f}.png",
    )
    save(
        vf.plot_contraction(
            X,
            np.asarray(prior.Sigma().diagonal()) ** 0.5,
            np.asarray(jnp.diag(Gamma_post)) ** 0.5,
            sensors=np.asarray(design),
        ),
        out / f"02_contraction_lambda_{lambda_:.2f}.png",
    )


def fig_matrices(lambda_: float, d: dict, out: Path) -> None:
    save(
        vm.plot_diagnostics(as_diagnostics(d), rf"$\lambda = {lambda_}$"),
        out / f"03_diagnostics_lambda_{lambda_:.2f}.png",
    )
    save(
        vm.plot_moments(d["L"], d["H"], d["I_eta"], title=rf"$\lambda = {lambda_}$"),
        out / f"04_moments_lambda_{lambda_:.2f}.png",
    )


def fig_linear_vs_nonlinear(data: dict, out: Path) -> None:
    """``H(u)`` at several ``lambda`` -- the visual validation.

    ``H(u) = 0`` **exactly** in the linear case (constant Jacobian). This is
    the only quantity that distinguishes Prop. 4 from a linear-Gaussian
    computation: seeing it empty at ``lambda=0`` and populated afterward
    validates the branch.
    """
    lams = sorted(data)
    save(
        vm.plot_matrix_comparison(
            [data[lam]["H"] for lam in lams],
            [rf"$H(u)$, $\lambda = {lam}$" for lam in lams],
            reference=0,
            title=r"$H(u)$: zero in the linear case, structured otherwise",
        ),
        out / "05_H_vs_lambda.png",
    )
    save(
        vm.plot_spectrum_comparison(
            [data[lam]["Sigma_Y"] for lam in lams] + [data[lam]["Sigma_signal"] for lam in lams],
            [rf"$\Sigma_Y$, $\lambda={lam}$" for lam in lams]
            + [rf"$\Sigma_{{signal}}$, $\lambda={lam}$" for lam in lams],
            title="Spectra: the gap drives the log-dets, hence the bounds",
        ),
        out / "06_spectra.png",
    )


def fig_bounds(lambda_: float, d: dict, out: Path):
    """Two designs, four bounds each -- the paper protocol Sec. 2."""
    dg = as_diagnostics(d)
    strategies = {
        "iEIG$\\geq$ (19)": (dg.Sigma_signal, dg.Sigma_Y_given_theta),
        "cEIG$\\geq$ (20)": (dg.Sigma_Y, dg.Sigma_noise),
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
        vb.plot_two_strategies(ms, per_strategy, rf"$\lambda = {lambda_}$"),
        out / f"07_bounds_lambda_{lambda_:.2f}.png",
    )
    save(
        vb.plot_width_vs_m(ms, widths, rf"$\lambda = {lambda_}$"),
        out / f"08_widths_lambda_{lambda_:.2f}.png",
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
            title=rf"$\lambda = {lambda_}$",
        ),
        out / f"09_alpha_lambda_{lambda_:.2f}.png",
    )
    save(
        vs.plot_log_generalized_spectrum(
            np.asarray(q.alpha), np.asarray(q.beta), title=rf"$\lambda = {lambda_}$"
        ),
        out / f"09b_log_spectrum_lambda_{lambda_:.2f}.png",
    )
    save(
        vs.plot_gap_decomposition(
            np.asarray(q.alpha), np.asarray(q.beta), rf"$\lambda = {lambda_}$"
        ),
        out / f"10_gap_decomposition_lambda_{lambda_:.2f}.png",
    )
    # EIG scale: the incremental UPPER bound at the max budget
    W = greedy_schur(dg.Sigma_signal, dg.Sigma_Y_given_theta, M_MAX).design
    eig_scale = float(incremental_bounds(dg, W).upper)
    save(
        vs.plot_suboptimality(
            ms,
            [q.suboptimality(int(m), "incremental") for m in ms],
            [q.suboptimality(int(m), "conservative") for m in ms],
            eig_scale=eig_scale,
            title=rf"$\lambda = {lambda_}$",
        ),
        out / f"11_suboptimality_lambda_{lambda_:.2f}.png",
    )


def fig_designs(lambda_: float, d: dict, designs: dict, out: Path) -> None:
    save(
        vd.plot_sensor_positions(X, designs, m=M_MAX, title=rf"$\lambda = {lambda_}$"),
        out / f"12_sensors_lambda_{lambda_:.2f}.png",
    )
    # where the non-Gaussianity resides: the diagonal of Sigma_Y - Sigma_signal
    save(
        vd.plot_design_on_field(
            X,
            np.diag(d["Sigma_Y"] - d["Sigma_signal"]),
            designs,
            title=r"sensors and $\mathrm{diag}(\Sigma_Y - \Sigma_{signal})$",
        ),
        out / f"13_design_on_gap_lambda_{lambda_:.2f}.png",
    )


def fig_gap_vs_lambda(data: dict, out: Path) -> None:
    lams = sorted(data)
    gaps = [quasi_optimality(as_diagnostics(data[lam])).total_gap for lam in lams]
    save(
        vb.plot_gap_vs_parameter(
            lams,
            gaps,
            mc_floor=abs(gaps[0]) if lams[0] == 0.0 else None,
            title="gap($I_p$) -- non-Gaussianity of $Y$",
        ),
        out / "14_gap_vs_lambda.png",
    )


# =============================================================================
# Orchestration
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    p.add_argument("--out", default="figures")
    p.add_argument("--cache", default=".cache")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    use_style()
    out, cache = Path(args.out), Path(args.cache)

    print("Diagnostics")
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
