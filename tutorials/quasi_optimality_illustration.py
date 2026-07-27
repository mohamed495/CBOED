#!/usr/bin/env python
r"""Didactic illustration of Proposition 1 -- a synthetic spectrum, not the real benchmark.

On the actual Burgers benchmark (``benchmarks.py``), the generalized spectrum
``alpha_i``/``beta_i`` is extremely concentrated (effective rank 2-3 out of
``N=200``, verified across every tested ``lambda`` and GP length scale --
shortening the prior's correlation length only makes it worse, since Burgers'
diffusion smooths out high-frequency content before the observation time
``T``). At that scale, the incremental and conservative sub-optimality
constants of eq. (22)/(23) both saturate almost immediately and coincide
over the entire realistic sensor-budget range: not a useful figure to explain
*why* incremental wins at small budgets and conservative at large ones.

This script builds a small, hand-chosen spectrum instead -- a smooth
geometric decay in ``log(alpha_i)) = log(beta_i)``, symmetric so the
crossover sits visibly inside the plotted range -- purely to illustrate the
proposition itself. It has no connection to the physical benchmark and
should never be presented as a measurement; it is a diagram, not a result.

Usage
-----
    pixi run -e test python tutorials/quasi_optimality_illustration.py
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cboed.bounds.quasi_optimality import QuasiOptimality
from cboed.viz.spectrum import plot_alpha_spectrum, plot_suboptimality
from cboed.viz.style import save, use_style


def make_synthetic_spectrum(p: int, tau: float, peak: float) -> QuasiOptimality:
    r"""Build a symmetric, smoothly decaying illustrative spectrum.

    ``log(alpha_i) = log(beta_i) = peak * exp(-(i-1)/tau)`` for
    ``i = 1, ..., p``: a single decay rate ``tau`` controls the effective
    rank, and the symmetric choice ``alpha_i = beta_i`` places the crossover
    exactly at ``p // 2 + 1`` (the same invariant :meth:`QuasiOptimality.crossover`
    always returns), with the sub-optimality curves visibly meeting there
    rather than off the edge of the plot.

    Parameters
    ----------
    p : int
        Number of modes (dimension of the synthetic pencil).
    tau : float
        Decay rate, in modes. Larger ``tau`` spreads the gap over more modes
        (higher effective rank).
    peak : float
        Value of ``log(alpha_1) = log(beta_1)``, i.e. the largest per-mode
        contribution; sets the overall scale of the total gap.

    Returns
    -------
    QuasiOptimality
        Synthetic spectrum with ``alpha = beta = exp(peak * exp(-(i-1)/tau))``.
    """
    i = jnp.arange(1, p + 1)
    t = peak * jnp.exp(-(i - 1) / tau)
    return QuasiOptimality(alpha=jnp.exp(t), beta=jnp.exp(t))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p_ = p.add_argument
    p_("--p", type=int, default=30, help="Number of synthetic modes.")
    p_("--tau", type=float, default=6.0, help="Decay rate (modes) -- controls effective rank.")
    p_("--peak", type=float, default=0.65, help="log(alpha_1) = log(beta_1), sets the total gap scale.")
    p_("--out", default="figures_quasi_optimality_illustration")
    args = p.parse_args()

    use_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q = make_synthetic_spectrum(args.p, args.tau, args.peak)
    print(f"[synthetic spectrum] p={args.p}  effective_rank={q.effective_rank}  crossover={q.crossover()}")
    print(f"[synthetic spectrum] total_gap={q.total_gap:.3f} nats")

    save(
        plot_alpha_spectrum(
            np.asarray(q.alpha), np.asarray(q.beta), effective_rank=q.effective_rank,
            title="Illustrative spectrum (synthetic, not the benchmark)",
        ),
        out / "spectrum_illustration.png",
    )

    ms = np.arange(1, args.p)
    inc = np.array([q.suboptimality(int(m), "incremental") for m in ms])
    cons = np.array([q.suboptimality(int(m), "conservative") for m in ms])
    save(
        plot_suboptimality(
            ms, inc, cons, eig_scale=q.total_gap,
            title="Sub-optimality trade-off (illustrative, Prop. 1)",
        ),
        out / "suboptimality_illustration.png",
    )

    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
