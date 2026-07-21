r"""Visualization of the fields (model solutions).

For λ=0.0 and different θ drawn from the prior, plots:
  - The solutions u(x, t) (spatio-temporal)
  - The initial conditions θ
  - The observations at the sensors
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from cboed.benchmarks import make_model, make_prior
from cboed.viz.style import save, use_style


def main():
    """Field visualization pipeline for λ=0.0."""
    use_style()
    output_dir = Path("outputs/fields")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Fields (model solutions) | λ = 0.0")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Configuration
    # -----------------------------------------------------------------------
    lambda_ = 0.0
    prior = make_prior()
    model = make_model(lambda_)

    print(f"\n[Setup] λ = {lambda_:.2f}, n = {model.n}, nt = {model.nt}, T = {model.T}")
    print(
        f"  Domain: {model.domain}, dx = {(model.domain[1] - model.domain[0]) / (model.n + 1):.4f}"
    )

    # -----------------------------------------------------------------------
    # 2. Sample a few θ from the prior
    # -----------------------------------------------------------------------
    print("\n[Priors samples]")
    key = jr.key(42)
    n_samples = 4
    thetas = prior.sample(key, n_samples)
    print(f"  {n_samples} samples from the GP prior")

    # -----------------------------------------------------------------------
    # 3. Integrate the fields
    # -----------------------------------------------------------------------
    print("\n[Integration]")

    # Spatial grid
    x = jnp.linspace(model.domain[0], model.domain[1], model.n + 2)[1:-1]  # Exclude boundaries (∂)

    # Integrate the solutions
    solutions = []
    for i, theta in enumerate(thetas):
        # model(theta, None) returns the observations at the last time step
        u_obs = model(theta, None)
        solutions.append(u_obs)

    print(f"  {n_samples} solutions computed (final observations)")

    # -----------------------------------------------------------------------
    # 4. Figures
    # -----------------------------------------------------------------------
    print("\n[Figures]")

    # Figure 1: Initial conditions (θ) vs observations (sol)
    fig1, axes = plt.subplots(n_samples, 2, figsize=(10, 8))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, (ax_row, theta, sol) in enumerate(zip(axes, thetas, solutions)):
        # Column 1: initial condition θ
        ax_row[0].plot(x, theta, "o-", markersize=3, alpha=0.7, color="steelblue")
        ax_row[0].fill_between(x, theta, alpha=0.2, color="steelblue")
        ax_row[0].set_ylabel(r"$\theta(x)$")
        ax_row[0].set_title(f"Initial condition θ_{idx + 1}")
        ax_row[0].grid(True, alpha=0.3)
        ax_row[0].set_ylim(-3, 3)

        # Column 2: final observations
        ax_row[1].plot(x, sol, "o-", markersize=3, alpha=0.7, color="orangered")
        ax_row[1].fill_between(x, sol, alpha=0.2, color="orangered")
        ax_row[1].set_ylabel(r"$y(x, t=T)$")
        ax_row[1].set_title(f"Observations θ_{idx + 1}")
        ax_row[1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Position $x$")
    axes[-1, 1].set_xlabel("Position $x$")

    fig1.suptitle(rf"Initial conditions vs observations | $\lambda = {lambda_:.2f}$", fontsize=11)
    fig1.tight_layout()
    path1 = save(fig1, output_dir / "01_theta_vs_observations.png")
    print(f"  → {path1.name}")

    # Figure 2: Distribution of the observations for each θ
    fig2, axes = plt.subplots(1, n_samples, figsize=(12, 3.5))
    if n_samples == 1:
        axes = [axes]

    for idx, (ax, sol) in enumerate(zip(axes, solutions)):
        ax.hist(sol, bins=30, alpha=0.7, color="coral", edgecolor="black")
        ax.set_xlabel("Observation value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Obs. distribution θ_{idx + 1}")
        ax.grid(True, alpha=0.2, axis="y")

    fig2.suptitle(rf"Observation histograms | $\lambda = {lambda_:.2f}$", fontsize=11)
    fig2.tight_layout()
    path2 = save(fig2, output_dir / "02_observations_distribution.png")
    print(f"  → {path2.name}")

    # Figure 3: Comparison θ vs observations (scatter)
    fig3, axes = plt.subplots(1, n_samples, figsize=(12, 3.5))
    if n_samples == 1:
        axes = [axes]

    for idx, (ax, theta, sol) in enumerate(zip(axes, thetas, solutions)):
        ax.scatter(theta, sol, alpha=0.5, s=20)
        ax.plot([-3, 3], [-3, 3], "k--", alpha=0.3, lw=1)
        ax.set_xlabel(r"$\theta(x)$ (initial condition)")
        ax.set_ylabel("Final observations")
        ax.set_title(f"Relation θ_{idx + 1} → obs")
        ax.grid(True, alpha=0.2)

    fig3.suptitle(rf"Linear mapping: θ → obs | $\lambda = {lambda_:.2f}$", fontsize=11)
    fig3.tight_layout()
    path3 = save(fig3, output_dir / "03_mapping_theta_to_obs.png")
    print(f"  → {path3.name}")

    # Figure 4: All θ and observations on the same figure
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = plt.cm.tab10(jnp.linspace(0, 1, n_samples))
    for idx, (theta, sol) in enumerate(zip(thetas, solutions)):
        ax1.plot(x, theta, "o-", markersize=2, label=f"θ_{idx + 1}", color=colors[idx], alpha=0.7)
        ax2.plot(x, sol, "s-", markersize=2, label=f"obs_{idx + 1}", color=colors[idx], alpha=0.7)

    ax1.set_xlabel("Position $x$")
    ax1.set_ylabel(r"Value")
    ax1.set_title("Initial conditions")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel("Position $x$")
    ax2.set_ylabel(r"Value")
    ax2.set_title("Final observations")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.2)

    fig4.suptitle(rf"All θ and observations | $\lambda = {lambda_:.2f}$", fontsize=11)
    fig4.tight_layout()
    path4 = save(fig4, output_dir / "04_all_samples.png")
    print(f"  → {path4.name}")

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    print("\n[Summary]")
    print(f"  {n_samples} solutions plotted")
    print(f"  Grid: x ∈ [{x[0]:.2f}, {x[-1]:.2f}]")
    print(f"  Figures saved to: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
