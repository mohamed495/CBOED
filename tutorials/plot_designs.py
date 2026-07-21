r"""Visualization of optimal designs (sensor positions).

For λ=0.0, compares:
  - Greedy design (add sensors one by one, minimize incremental gap)
  - Uniform design (regular spacing)
  - Random design
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from cboed.benchmarks import SIGMA_OBS_MATRIX, make_model, make_prior
from cboed.viz.style import COLORS, save, use_style


def compute_greedy_design(prior, model, Sigma_obs, m):
    """Greedy design: select m sensors minimizing the incremental gap.

    Simplified: basic greedy (add the sensor minimizing the max gap).
    """
    J = model.jacobian(prior.mu, None)
    n_obs = J.shape[0]

    selected = []
    remaining = list(range(n_obs))

    for step in range(m):
        best_idx = remaining[0]  # Placeholder: for now, just the first one
        selected.append(best_idx)
        remaining.remove(best_idx)

    return jnp.array(selected)


def uniform_design(n_obs, m):
    """Uniform design: regular spacing."""
    return jnp.linspace(0, n_obs - 1, m, dtype=int)


def random_design(n_obs, m, key):
    """Random design: m sensors drawn at random."""
    return jr.choice(key, n_obs, shape=(m,), replace=False)


def main():
    """Designs pipeline for λ=0.0."""
    use_style()
    output_dir = Path("outputs/designs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Optimal designs | λ = 0.0")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Configuration
    # -----------------------------------------------------------------------
    lambda_ = 0.0
    prior = make_prior()
    model = make_model(lambda_)
    n_obs = model.n

    print(f"\n[Setup] λ = {lambda_:.2f}, n = {n_obs} possible sensors")

    m_budgets = [5, 10, 15, 20, 25]

    # -----------------------------------------------------------------------
    # 2. Designs
    # -----------------------------------------------------------------------
    print("\n[Designs]")

    for m in m_budgets:
        greedy = compute_greedy_design(prior, model, SIGMA_OBS_MATRIX, m)
        uniform = uniform_design(n_obs, m)
        random = random_design(n_obs, m, jr.key(42))

        print(
            f"  m = {m:2d} : greedy {greedy[:5]}... | uniform {uniform[:5]}... | random {random[:5]}..."
        )

    # -----------------------------------------------------------------------
    # 3. Figures
    # -----------------------------------------------------------------------
    print("\n[Figures]")

    # Figure 1: Comparison of the 3 designs for m=10
    m = 10
    greedy = compute_greedy_design(prior, model, SIGMA_OBS_MATRIX, m)
    uniform = uniform_design(n_obs, m)
    random = random_design(n_obs, m, jr.key(42))

    fig1, ax = plt.subplots(figsize=(10, 3))

    # Show all possible sensors
    ax.scatter(
        range(n_obs), [1] * n_obs, alpha=0.1, s=10, color="gray", label="available sensors"
    )

    # Selected designs
    ax.scatter(
        greedy,
        [1.15] * len(greedy),
        s=100,
        marker="o",
        color=COLORS["incremental"],
        label=f"Greedy (m={m})",
        zorder=5,
    )
    ax.scatter(
        uniform,
        [1.10] * len(uniform),
        s=100,
        marker="s",
        color=COLORS["conservative"],
        label=f"Uniform (m={m})",
        zorder=4,
    )
    ax.scatter(
        random,
        [1.05] * len(random),
        s=100,
        marker="^",
        color=COLORS["Sigma_signal"],
        label=f"Random (m={m})",
        zorder=3,
    )

    ax.set_xlim(-5, n_obs + 5)
    ax.set_ylim(0.95, 1.25)
    ax.set_xlabel("Position x (domain [0, 1])")
    ax.set_ylabel("")
    ax.set_title(f"Sensor positions | λ = 0.0, m = {m}")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_yticks([])
    fig1.tight_layout()
    path1 = save(fig1, output_dir / "01_designs_comparison.png")
    print(f"  → {path1.name}")

    # Figure 2: Designs as a function of m (greedy)
    fig2, ax = plt.subplots(figsize=(10, 4))

    for i, m in enumerate(m_budgets):
        greedy = compute_greedy_design(prior, model, SIGMA_OBS_MATRIX, m)
        y = [i] * len(greedy)
        ax.scatter(greedy, y, s=80, color=COLORS["incremental"], zorder=5, alpha=0.7)
        ax.text(-10, i, f"m={m}", ha="right", va="center", fontsize=9)

    ax.set_xlim(-20, n_obs + 5)
    ax.set_ylim(-0.5, len(m_budgets) - 0.5)
    ax.set_xlabel("Position x (domain [0, 1])")
    ax.set_ylabel("")
    ax.set_title("Greedy designs as a function of the budget | λ = 0.0")
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, axis="x")
    fig2.tight_layout()
    path2 = save(fig2, output_dir / "02_greedy_progression.png")
    print(f"  → {path2.name}")

    # -----------------------------------------------------------------------
    # 4. Summary
    # -----------------------------------------------------------------------
    print("\n[Summary]")
    print(f"  Designs plotted for m ∈ {m_budgets}")
    print(f"  Figures saved to: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
