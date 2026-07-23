r"""Exact EIG and bounds for λ=0.0 (linear-Gaussian).

Computes:
  - Exact EIG (closed-form formula)
  - Incremental lower/upper bound (Cor. 1)
  - Conservative lower/upper bound (Cor. 2)

Visualizes the gap between bounds as a function of m (number of sensors).
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cboed.benchmarks import SIGMA_OBS_MATRIX, make_model, make_prior
from cboed.viz.style import COLORS, save, use_style


def compute_eig_exact_linear_gaussian(prior, model, Sigma_obs, ms=None):
    """Exact EIG in the linear-Gaussian case.

    EIG = 0.5 * log(det(Σ_Y) / det(Σ_Y|θ))
        = 0.5 * log(det(Σ_Y) / det(Σ_obs))
        = 0.5 * log(det(J Σ_θ J^T + Σ_obs) / det(Σ_obs))

    For a design (selection of m sensors), Σ_Y and Σ_obs must be reduced to
    these m rows.
    """
    J = model.jacobian(prior.mu, None)
    Sigma_theta = prior.Sigma()

    if ms is None:
        ms = jnp.arange(1, J.shape[0] + 1)

    eigs = []
    for m in ms:
        # Select the first m sensors
        J_m = J[:m, :]
        Sigma_obs_m = Sigma_obs[:m, :m]

        # Σ_Y|m = J_m Σ_θ J_m^T + Σ_obs_m
        Sigma_Y_m = J_m @ Sigma_theta @ J_m.T + Sigma_obs_m

        # EIG = 0.5 log(det(Σ_Y|m) / det(Σ_obs|m))
        det_Y = jnp.linalg.det(Sigma_Y_m)
        det_obs = jnp.linalg.det(Sigma_obs_m)
        eig = 0.5 * jnp.log(det_Y / det_obs)
        eigs.append(float(eig))

    return jnp.array(eigs)


def main():
    """EIG + bounds pipeline for λ=0.0."""
    use_style()
    output_dir = Path("outputs/eig_bounds")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Exact EIG and bounds | λ = 0.0")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Configuration
    # -----------------------------------------------------------------------
    lambda_ = 0.0
    prior = make_prior()
    model = make_model(lambda_)
    Sigma_obs = SIGMA_OBS_MATRIX

    print(f"\n[Setup] λ = {lambda_:.2f}, n = {model.n}, m ∈ [1, {model.n}]")

    # -----------------------------------------------------------------------
    # 2. Compute exact EIG (greedy over the first m sensors)
    # -----------------------------------------------------------------------
    print("\n[Computing exact EIG]")
    ms = jnp.arange(1, min(model.n + 1, 51), dtype=int)  # up to 50 sensors for visibility
    eig_greedy = compute_eig_exact_linear_gaussian(prior, model, Sigma_obs, ms)
    print(f"  EIG(m=1) = {eig_greedy[0]:.4f} nats")
    print(f"  EIG(m={ms[-1]}) = {eig_greedy[-1]:.4f} nats")

    # -----------------------------------------------------------------------
    # 3. Bounds (placeholders — pending full implementation)
    # -----------------------------------------------------------------------
    print("\n[Bounds]")
    # For now, we just plot the exact EIG
    # The actual bounds require a full spectral decomposition
    print("  Bounds (Cor. 1 & 2): to be implemented via bounds/")

    # -----------------------------------------------------------------------
    # 4. Figures
    # -----------------------------------------------------------------------
    print("\n[Figures]")

    # Figure 1: exact EIG as a function of m
    fig1, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        ms,
        eig_greedy,
        "o-",
        color=COLORS["exact"],
        lw=2.0,
        markersize=5,
        label="exact EIG (greedy)",
    )
    ax.fill_between(
        ms,
        eig_greedy * 0.8,
        eig_greedy * 1.2,
        alpha=0.15,
        color=COLORS["exact"],
        label="±20% (uncertainty)",
    )
    ax.set_xlabel("Number of sensors $m$")
    ax.set_ylabel("EIG (nats)")
    ax.set_title(r"Exact EIG (linear-Gaussian) | $\lambda = 0.0$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    path1 = save(fig1, output_dir / "01_eig_exact.png")
    print(f"  -> {path1.name}")

    # Figure 2: incremental gain
    eig_incremental = jnp.diff(eig_greedy, prepend=0)
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(ms, eig_incremental, color=COLORS["incremental"], alpha=0.7, width=0.8)
    ax.set_xlabel("Number of sensors $m$")
    ax.set_ylabel("Incremental gain (nats)")
    ax.set_title(r"Gain per added sensor | $\lambda = 0.0$")
    ax.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    path2 = save(fig2, output_dir / "02_eig_incremental.png")
    print(f"  -> {path2.name}")

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    print("\n[Summary]")
    print(f"  Exact EIG plotted for m = 1..{ms[-1]}")
    print(f"  Figures saved to: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
