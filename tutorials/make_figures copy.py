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


# =============================================================================
# 2. Non-Gaussianity: log(alpha), log(beta)
# =============================================================================

def fig_alpha_beta_lambda(data, out):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15,4),
        sharey=False
    )

    for lam, d in sorted(data.items()):

        q = quasi_optimality(
            as_diagnostics(d)
        )

        alpha = np.asarray(q.alpha)
        beta = np.asarray(q.beta)

        axes[0].plot(
            np.log(alpha),
            label=rf"$\lambda={lam}$"
        )

        axes[1].plot(
            np.log(beta),
            label=rf"$\lambda={lam}$"
        )

        axes[2].plot(
            np.log(alpha)+np.log(beta),
            label=rf"$\lambda={lam}$"
        )


    axes[0].set_title(r"$\log(\alpha_i)$")
    axes[1].set_title(r"$\log(\beta_i)$")
    axes[2].set_title(
        r"$\log(\alpha_i)+\log(\beta_i)$"
    )

    for ax in axes:
        ax.set_xlabel("mode $i$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("value")

    fig.tight_layout()

    save(
        fig,
        out / "02_alpha_beta_spectrum.png"
    )

# =============================================================================
# 2. Non-Gaussianity: log(alpha), log(beta)
# =============================================================================

def fig_alpha_beta_lambda(data, out):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15,4),
        sharey=False
    )

    for lam, d in sorted(data.items()):

        q = quasi_optimality(
            as_diagnostics(d)
        )

        alpha = np.asarray(q.alpha)
        beta = np.asarray(q.beta)

        axes[0].plot(
            np.log(alpha),
            label=rf"$\lambda={lam}$"
        )

        axes[1].plot(
            np.log(beta),
            label=rf"$\lambda={lam}$"
        )

        axes[2].plot(
            np.log(alpha)+np.log(beta),
            label=rf"$\lambda={lam}$"
        )


    axes[0].set_title(r"$\log(\alpha_i)$")
    axes[1].set_title(r"$\log(\beta_i)$")
    axes[2].set_title(
        r"$\log(\alpha_i)+\log(\beta_i)$"
    )

    for ax in axes:
        ax.set_xlabel("mode $i$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("value")

    fig.tight_layout()

    save(
        fig,
        out / "02_alpha_beta_spectrum.png"
    )

# =============================================================================
# 3. Bounds and optimal design
# =============================================================================

def fig_optimal_bounds(lambda_, d, out):

    dg = as_diagnostics(d)

    ms = np.arange(
        1,
        M_MAX+1
    )

    strategies = {
        "incremental":
        (
            dg.Sigma_signal,
            dg.Sigma_Y_given_theta
        ),

        "conservative":
        (
            dg.Sigma_Y,
            dg.Sigma_noise
        )
    }


    results = {}
    designs = {}


    for name, (A,B) in strategies.items():

        W = greedy_schur(
            A,
            B,
            M_MAX
        ).design

        designs[name] = np.asarray(W)


        inc_low=[]
        inc_up=[]
        cons_low=[]
        cons_up=[]


        for m in ms:

            inc = incremental_bounds(
                dg,
                W[:m]
            )

            cons = conservative_bounds(
                dg,
                W[:m]
            )


            inc_low.append(float(inc.lower))
            inc_up.append(float(inc.upper))

            cons_low.append(float(cons.lower))
            cons_up.append(float(cons.upper))


        results[name] = {
            "inc_low":np.array(inc_low),
            "inc_up":np.array(inc_up),
            "cons_low":np.array(cons_low),
            "cons_up":np.array(cons_up),
        }


    save(
        vb.plot_two_strategies(
            ms,
            results,
            title=rf"$\lambda={lambda_}$"
        ),
        out / f"03_bounds_lambda_{lambda_}.png"
    )


    save(
        vd.plot_sensor_positions(
            X,
            designs,
            m=M_MAX,
            title=rf"Optimal designs $\lambda={lambda_}$"
        ),
        out / f"04_designs_lambda_{lambda_}.png"
    )

    