#!/usr/bin/env python
r"""Compare three routes to ``Sigma_signal``/``Sigma_noise`` on the Burgers benchmark.

- gradient (Sec. 3.3, ``gradient_based.py``): Jacobians, the reference route
  for the rest of the pipeline.
- affine approximation (Sec. 3.2, ``approximation_based.py`` + ``AffineDenoiser``).
- affine approximation + network (Sec. 3.2, ``approximation_based.py`` +
  ``ResidualDenoiser``).

The three methods differ in cost and guarantee (see docstrings of
``bounds/diagnostics/*``), not in what they estimate: all three target the
same pair of matrices. The gradient serves as the reference in the figures,
not because it is ground truth -- it is the only certified route (Prop. 4).

What this script shows (N=200, lambda=0.5, default benchmark)
---------------------------------------------------------------
The assembly from Prop. 3, ``(Sigma_obs^{-1} - Sigma_obs^{-1} R Sigma_obs^{-1})^{-1}``,
is ill-conditioned as soon as ``R`` (the denoiser's residual) approaches
``Sigma_obs`` -- from above (residual too large, the inverse stops existing:
``ValueError`` from ``assemble_from_residual``) or from below (residual very
close but below: the inverse exists but strongly amplifies the estimation
noise on ``R``).

At small ``N`` (a few thousand), the affine denoiser (``200x200`` parameters)
overfits and produces an artificially small ``R`` -- silently: no error, but
``Sigma_signal`` then bears no relation to the gradient (~90-99% relative gap
measured). At ``N=20000``, the overfitting disappears and the guard triggers:
the affine alone no longer suffices to satisfy Prop. 3 at ``lambda=0.5``
(``u`` nonlinear). The network (``ResidualDenoiser``) reduces the residual
below ``Sigma_obs`` where the affine fails -- but the residual gap with the
gradient stays substantial (~75% measured with default settings): the gap
``Sigma_obs - R`` stays small, so the inversion remains ill-conditioned even
when formally valid. See the ``max_eig(R - Sigma_obs)`` diagnostic print for
each route before reading the figures.

Also runs in the goal-oriented (GO) case (``--case go``): the noise route
then targets the QoI (``theta = eta[:N_QOI]``, the first half of the field)
rather than the full ``eta``, using ``gradient_diagnostics``/``Sigma_xi``
instead of ``gradient_diagnostics_standard``/``Sigma_obs`` for the gradient
reference -- mirroring ``paper_protocol.py``'s ``compute_repeat``. The signal
route (``f: Y -> u``) is identical in both cases: only the noise route
(``g: (Y, theta) -> u``) depends on ``theta``.

Usage
-----
    pixi run -e test python tutorials/compare_signal_noise_methods.py
    pixi run -e test python tutorials/compare_signal_noise_methods.py --lambda 0.5 --n-samples 20000 --net-steps 500
    pixi run -e test python tutorials/compare_signal_noise_methods.py --case go
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cboed.benchmarks import N_QOI, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, forward, make_prior, qoi_projection
from cboed.bounds.diagnostics.approximation_based import (
    approximation_noise,
    approximation_signal,
    denoiser_residual,
)
from cboed.bounds.diagnostics.denoisers import AffineDenoiser, ResidualDenoiser
from cboed.bounds.diagnostics.gradient_based import gradient_diagnostics, gradient_diagnostics_standard
from cboed.viz.matrices import plot_matrix_comparison, plot_spectrum_comparison
from cboed.viz.style import save, use_style

QOI_H = qoi_projection(N_QOI)


def paired_samples(u, prior, Sigma_obs, key, n):
    """``(u(eta), Y = u(eta) + eps, eta)`` -- the same pairs for all routes."""
    k_eta, k_eps = jr.split(key)
    eta = prior.sample(k_eta, n)
    u_vals = jax.vmap(u)(eta)
    L = jnp.linalg.cholesky(Sigma_obs)
    Y = u_vals + jr.normal(k_eps, u_vals.shape) @ L.T
    return u_vals, Y, eta


def report_gap(label, R, Sigma_obs):
    """``max_eig(R - Sigma_obs)`` -- conditioning diagnostic for Prop. 3.

    Negative and far from zero: well-conditioned assembly. Positive: Prop. 3
    does not apply (``assemble_from_residual`` will raise ``ValueError``).
    Negative but close to zero: formally valid, numerically fragile.
    """
    gap = float(jnp.max(jnp.linalg.eigvalsh(R - Sigma_obs)))
    print(f"  max_eig(R - Sigma_obs) [{label}] = {gap:+.3e}")
    return gap


def try_assemble(name, fn, *args):
    """Assemble ``Sigma_signal``/``Sigma_noise``, or report failure without crashing.

    Prop. 3 is not always applicable (see the module docstring): a denoiser
    too weak against the nonlinearity raises ``ValueError``. This is not an
    accident to hide -- it is the information this script wants to show.
    """
    try:
        return fn(*args)
    except ValueError as e:
        print(f"  [{name}] Prop. 3 not applicable: {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    p.add_argument(
        "--case", choices=("standard", "go"), default="standard",
        help="'standard': theta = eta (full field). 'go': theta = eta[:N_QOI] (QoI only),"
             " matching paper_protocol.py's goal-oriented case.",
    )
    p.add_argument(
        "--n-samples", type=int, default=20_000, help="(u, Y) pairs for the denoisers"
    )
    p.add_argument("--n-gradient", type=int, default=300, help="Jacobians for the gradient route")
    p.add_argument("--net-steps", type=int, default=500, help="Adam steps, ResidualDenoiser")
    p.add_argument("--out", default="figures_conservative_mc")
    args = p.parse_args()

    use_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prior = make_prior()
    u = forward(args.lambda_)
    k_pairs, k_grad, k_net_f, k_net_g = jr.split(jr.key(0), 4)

    print(f"[pairs] case={args.case}, lambda={args.lambda_}, n_samples={args.n_samples}")
    u_vals, Y, eta = paired_samples(u, prior, SIGMA_OBS_MATRIX, k_pairs, args.n_samples)
    theta_for_noise = eta if args.case == "standard" else eta[:, :N_QOI]
    features_g = jnp.concatenate([Y, theta_for_noise], axis=1)

    print(f"[gradient] n_gradient={args.n_gradient}")
    if args.case == "standard":
        Sigma_signal_grad, Sigma_noise_grad = gradient_diagnostics_standard(
            u, prior, SIGMA_OBS_MATRIX, k_grad, args.n_gradient
        )
    else:
        Sigma_signal_grad, Sigma_noise_grad = gradient_diagnostics(
            u, QOI_H, prior, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, k_grad, args.n_gradient
        )

    print("[affine approximation]")
    denoiser_f_affine = AffineDenoiser.fit(u_vals, Y)
    denoiser_g_affine = AffineDenoiser.fit(u_vals, features_g)
    report_gap("affine, f: Y->u", denoiser_residual(denoiser_f_affine, u_vals, Y), SIGMA_OBS_MATRIX)
    report_gap(
        "affine, g: (Y,theta)->u",
        denoiser_residual(denoiser_g_affine, u_vals, features_g),
        SIGMA_OBS_MATRIX,
    )
    Sigma_signal_affine = try_assemble(
        "affine signal", approximation_signal, denoiser_f_affine, u_vals, Y, SIGMA_OBS_MATRIX
    )
    Sigma_noise_affine = try_assemble(
        "affine noise",
        approximation_noise,
        denoiser_g_affine,
        u_vals,
        Y,
        theta_for_noise,
        SIGMA_OBS_MATRIX,
    )

    print(f"[affine approximation + network] steps={args.net_steps}")
    denoiser_f_nn = ResidualDenoiser.fit(u_vals, Y, k_net_f, steps=args.net_steps)
    denoiser_g_nn = ResidualDenoiser.fit(u_vals, features_g, k_net_g, steps=args.net_steps)
    report_gap("affine+NN, f: Y->u", denoiser_residual(denoiser_f_nn, u_vals, Y), SIGMA_OBS_MATRIX)
    report_gap(
        "affine+NN, g: (Y,theta)->u",
        denoiser_residual(denoiser_g_nn, u_vals, features_g),
        SIGMA_OBS_MATRIX,
    )
    Sigma_signal_nn = try_assemble(
        "affine+NN signal", approximation_signal, denoiser_f_nn, u_vals, Y, SIGMA_OBS_MATRIX
    )
    Sigma_noise_nn = try_assemble(
        "affine+NN noise", approximation_noise, denoiser_g_nn, u_vals, Y, theta_for_noise, SIGMA_OBS_MATRIX
    )

    signals = [("gradient (Sec. 3.3)", Sigma_signal_grad)]
    if Sigma_signal_affine is not None:
        signals.append(("affine approximation (Sec. 3.2)", Sigma_signal_affine))
    if Sigma_signal_nn is not None:
        signals.append(("affine+NN approximation (Sec. 3.2)", Sigma_signal_nn))

    noises = [("gradient (Sec. 3.3)", Sigma_noise_grad)]
    if Sigma_noise_affine is not None:
        noises.append(("affine approximation (Sec. 3.2)", Sigma_noise_affine))
    if Sigma_noise_nn is not None:
        noises.append(("affine+NN approximation (Sec. 3.2)", Sigma_noise_nn))

    print("[figures]")
    if len(signals) > 1:
        save(
            plot_matrix_comparison(
                [S for _, S in signals],
                [lbl for lbl, _ in signals],
                reference=0,
                title=rf"$\Sigma_{{signal}}$: gradient vs approximation -- $\lambda={args.lambda_}$",
            ),
            out / f"signal_comparison_lambda_{args.lambda_:.2f}.png",
        )
        save(
            plot_spectrum_comparison(
                [S for _, S in signals],
                [f"$\\Sigma_{{signal}}$ {lbl}" for lbl, _ in signals],
                title=rf"Spectra $\Sigma_{{signal}}$ -- $\lambda={args.lambda_}$",
            ),
            out / f"signal_spectra_lambda_{args.lambda_:.2f}.png",
        )
    if len(noises) > 1:
        save(
            plot_matrix_comparison(
                [S for _, S in noises],
                [lbl for lbl, _ in noises],
                reference=0,
                title=rf"$\Sigma_{{noise}}$: gradient vs approximation -- {args.case}, $\lambda={args.lambda_}$",
            ),
            out / f"noise_comparison_{args.case}_lambda_{args.lambda_:.2f}.png",
        )
        save(
            plot_spectrum_comparison(
                [S for _, S in noises],
                [f"$\\Sigma_{{noise}}$ {lbl}" for lbl, _ in noises],
                title=rf"Spectra $\Sigma_{{noise}}$ -- {args.case}, $\lambda={args.lambda_}$",
            ),
            out / f"noise_spectra_{args.case}_lambda_{args.lambda_:.2f}.png",
        )

    print("[relative gaps vs gradient]")
    for label, S in signals[1:]:
        rel = np.linalg.norm(np.asarray(S) - np.asarray(Sigma_signal_grad)) / np.linalg.norm(
            np.asarray(Sigma_signal_grad)
        )
        print(f"  ||Sigma_signal({label}) - gradient|| / ||.|| = {rel:.3e}")
    for label, S in noises[1:]:
        rel = np.linalg.norm(np.asarray(S) - np.asarray(Sigma_noise_grad)) / np.linalg.norm(
            np.asarray(Sigma_noise_grad)
        )
        print(f"  ||Sigma_noise({label}) - gradient|| / ||.|| = {rel:.3e}")

    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
