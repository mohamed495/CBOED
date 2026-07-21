r"""Sweep over λ — plots the estimated matrices for each lambda.

For each λ ∈ LAMBDAS, estimates the matrices by Monte Carlo:
  - Σ_Y : Cov(u(θ)) via sample_Sigma_Y()
  - Σ_signal, Σ_noise : via gradient_diagnostics_standard()
  - Σ_Y|θ = Σ_obs (exact in the standard case)

Generates 5 figures per lambda.
Output: outputs/lambda_*/
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from cboed.benchmarks import LAMBDAS, SIGMA_OBS_MATRIX, forward, make_model, make_prior
from cboed.bounds.diagnostics.gradient_based import gradient_diagnostics
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y, sample_Sigma_Y_given_theta
from cboed.viz.matrices import plot_matrix_comparison
from cboed.viz.style import COLORS, save, use_style


def process_lambda(lambda_, n_samples=1000):
    """Pipeline for a given lambda."""
    use_style()
    output_dir = Path(f"outputs/lambda_{lambda_:.2f}".replace(".", "p"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"λ = {lambda_:.2f}")
    print(f"{'=' * 70}")

    # -----------------------------------------------------------------------
    # 1. Configuration
    # -----------------------------------------------------------------------
    prior = make_prior()
    model = make_model(lambda_)
    u = forward(lambda_)  # Forward model without design

    print(f"[Setup] n = {model.n}, n_samples (MC) = {n_samples}")

    # -----------------------------------------------------------------------
    # 2. Estimate the matrices by Monte Carlo
    # -----------------------------------------------------------------------
    print("[MC estimation]")
    key = jr.key(42)

    # Sigma_Y via paired-differences sampling
    k1, k2, k3 = jr.split(key, 3)
    Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k1, n_samples)
    print("  Σ_Y estimated (pairs MC)")

    n_param = prior.mu.shape[0]
    Sigma_xi = 1e-8 * jnp.eye(n_param)

    h = lambda x: x

    # Sigma_signal, Sigma_noise via gradient (Prop. 4)
    Sigma_signal, Sigma_noise = gradient_diagnostics(
        u, h, prior, SIGMA_OBS_MATRIX, Sigma_xi, k2, n_samples
    )

    print("  Σ_signal, Σ_noise estimated (gradient)")

    # Sigma_Y_given_theta via sampling with change of variables
    # In the standard case: B = Identity, Sigma_xi ≈ 0

    B = jnp.eye(n_param)

    Sigma_Y_given_theta = sample_Sigma_Y_given_theta(
        u, prior, B, SIGMA_OBS_MATRIX, Sigma_xi, k3, n_samples
    )
    print("  Σ_Y|θ estimated (sample)")

    # Prior and prior covariance
    Sigma_theta = prior.Sigma()

    # -----------------------------------------------------------------------
    # 3. Spectra
    # -----------------------------------------------------------------------
    eigs_theta = jnp.sort(jnp.linalg.eigvalsh(Sigma_theta))[::-1]
    eigs_signal = jnp.sort(jnp.linalg.eigvalsh(Sigma_signal))[::-1]
    eigs_Y = jnp.sort(jnp.linalg.eigvalsh(Sigma_Y))[::-1]
    eigs_noise = jnp.sort(jnp.linalg.eigvalsh(Sigma_noise))[::-1]

    print("[Spectra]")
    print(f"  λ_max(Σ_Y) = {eigs_Y[0]:.3e}, λ_min = {eigs_Y[-1]:.3e}")
    print(f"  λ_max(Σ_signal) = {eigs_signal[0]:.3e}, λ_min = {eigs_signal[-1]:.3e}")

    # -----------------------------------------------------------------------
    # 4. Figures
    # -----------------------------------------------------------------------
    print("[Figures]")

    # Figure 1: Σ_theta alone
    fig1, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(figsize=(6, 5))
    im1 = ax.imshow(Sigma_theta, cmap="viridis")
    ax.set_title(r"$\Sigma_\theta$ (prior GP)")
    ax.set_xlabel("Parameter index j")
    ax.set_ylabel("Parameter index i")
    __import__("matplotlib.pyplot", fromlist=["colorbar"]).colorbar(im1, ax=ax, label="covariance")
    fig1.tight_layout()
    path1 = save(fig1, output_dir / "01_Sigma_theta.png")
    print(f"  → {path1.name}")

    # Figure 2: Σ_Y and Σ_signal (numerator)
    fig2 = plot_matrix_comparison(
        [Sigma_Y, Sigma_signal],
        labels=[r"$\Sigma_Y$ (observed, MC)", r"$\Sigma_{\mathrm{signal}}$ (gradient)"],
        title=f"Numerator: $\\Sigma_Y$ vs $\\Sigma_{{\\mathrm{{signal}}}}$ | $\\lambda = {lambda_:.2f}$",
    )
    path2 = save(fig2, output_dir / "02_numerator.png")
    print(f"  → {path2.name}")

    # Figure 3: Σ_Y|θ and Σ_noise (denominator)
    fig3 = plot_matrix_comparison(
        [Sigma_Y_given_theta, Sigma_noise],
        labels=[r"$\Sigma_{Y|\theta}$ (exact)", r"$\Sigma_{\mathrm{noise}}$ (gradient)"],
        title=f"Denominator: $\\Sigma_{{Y|\\theta}}$ vs $\\Sigma_{{\\mathrm{{noise}}}}$ | $\\lambda = {lambda_:.2f}$",
    )
    path3 = save(fig3, output_dir / "03_denominator.png")
    print(f"  → {path3.name}")

    # Figure 4: Spectra
    fig4, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(figsize=(8, 5))
    n_show = 50
    ax.semilogy(
        range(n_show),
        eigs_theta[:n_show],
        "o-",
        label=r"$\Sigma_\theta$",
        color=COLORS["prior"],
        lw=1.5,
        markersize=4,
    )
    ax.semilogy(
        range(n_show),
        eigs_signal[:n_show],
        "s-",
        label=r"$\Sigma_{\mathrm{signal}}$",
        color=COLORS["Sigma_signal"],
        lw=1.5,
        markersize=4,
    )
    ax.semilogy(
        range(n_show),
        eigs_Y[:n_show],
        "^-",
        label=r"$\Sigma_Y$",
        color=COLORS["Sigma_Y"],
        lw=1.5,
        markersize=4,
    )
    ax.semilogy(
        range(n_show),
        eigs_noise[:n_show],
        "d-",
        label=r"$\Sigma_{\mathrm{noise}}$",
        color=COLORS["Sigma_noise"],
        lw=1.5,
        markersize=4,
    )
    ax.set_xlabel("Mode index $i$")
    ax.set_ylabel("Eigenvalue")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Spectra (estimated) | $\\lambda = {lambda_:.2f}$")
    fig4.tight_layout()
    path4 = save(fig4, output_dir / "04_spectra.png")
    print(f"  → {path4.name}")

    # Figure 5: Jacobian
    J = model.jacobian(prior.mu, None)
    fig5, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(figsize=(6, 5))
    im5 = ax.imshow(J, aspect="auto", cmap="viridis")
    ax.set_xlabel("Observed index")
    ax.set_ylabel("Parameter index")
    ax.set_title(f"Jacobian $J$ (Burgers, $\\lambda = {lambda_:.2f}$)")
    __import__("matplotlib.pyplot", fromlist=["colorbar"]).colorbar(im5, ax=ax)
    fig5.tight_layout()
    path5 = save(fig5, output_dir / "05_jacobian.png")
    print(f"  → {path5.name}")


def main():
    """Full pipeline: all lambdas."""
    print("=" * 70)
    print("Sweep over λ — Matrices for each lambda")
    print("=" * 70)

    for lambda_ in LAMBDAS:
        process_lambda(lambda_)

    print(f"\n{'=' * 70}")
    print("✓ All sweeps completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
