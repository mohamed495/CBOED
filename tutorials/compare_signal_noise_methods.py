#!/usr/bin/env python
r"""Compare trois voies vers ``Sigma_signal``/``Sigma_noise`` sur le banc Burgers.

- gradient (§3.3, ``gradient_based.py``) : jacobiennes, la voie de reference du
  reste du pipeline.
- approximation affine (§3.2, ``approximation_based.py`` + ``AffineDenoiser``).
- approximation affine + reseau (§3.2, ``approximation_based.py`` +
  ``ResidualDenoiser``).

Les trois methodes different par le cout et la garantie (voir docstrings de
``bounds/diagnostics/*``), pas par ce qu'elles estiment : les trois visent la
meme paire de matrices. Le gradient sert de reference dans les figures, pas
parce que c'est la verite -- c'est la seule voie certifiee (Prop. 4).

Ce que ce script montre (N=200, lambda=0.5, banc par defaut)
-------------------------------------------------------------
L'assemblage de Prop. 3, ``(Sigma_obs^{-1} - Sigma_obs^{-1} R Sigma_obs^{-1})^{-1}``,
est mal conditionne des que ``R`` (le residu du debruiteur) s'approche de
``Sigma_obs`` -- par valeurs superieures (residu trop grand, l'inverse cesse
d'exister : ``ValueError`` de ``assemble_from_residual``) ou par valeurs
inferieures (residu tres proche mais en dessous : l'inverse existe mais
amplifie fortement le bruit d'estimation sur ``R``).

A petit ``N`` (quelques milliers), le debruiteur affine (``200x200`` parametres)
sur-apprend et produit un ``R`` artificiellement petit -- silencieusement : pas
d'erreur, mais ``Sigma_signal`` est alors sans rapport avec le gradient (~90-99%
d'ecart relatif mesure). A ``N=20000``, le sur-apprentissage disparait et le
garde-fou se declenche : l'affine seul ne suffit plus a satisfaire Prop. 3 a
``lambda=0.5`` (``u`` non lineaire). Le reseau (``ResidualDenoiser``) reduit le
residu sous ``Sigma_obs`` la ou l'affine echoue -- mais l'ecart residuel avec le
gradient reste substantiel (~75% mesure avec les reglages par defaut) : le gap
``Sigma_obs - R`` reste faible, donc l'inversion reste mal conditionnee meme
quand elle est formellement valide. Voir le print de diagnostic ``max_eig(R -
Sigma_obs)`` pour chaque voie avant de lire les figures.

Usage
-----
    pixi run -e test python tutorials/compare_signal_noise_methods.py
    pixi run -e test python tutorials/compare_signal_noise_methods.py --lambda 0.5 --n-samples 20000 --net-steps 500
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cboed.benchmarks import SIGMA_OBS_MATRIX, forward, make_prior
from cboed.bounds.diagnostics.approximation_based import (
    approximation_noise,
    approximation_signal,
    denoiser_residual,
)
from cboed.bounds.diagnostics.denoisers import AffineDenoiser, ResidualDenoiser
from cboed.bounds.diagnostics.gradient_based import gradient_diagnostics_standard
from cboed.viz.matrices import plot_matrix_comparison, plot_spectrum_comparison
from cboed.viz.style import save, use_style


def paired_samples(u, prior, Sigma_obs, key, n):
    """``(u(eta), Y = u(eta) + eps, eta)`` -- les memes paires pour toutes les voies."""
    k_eta, k_eps = jr.split(key)
    eta = prior.sample(k_eta, n)
    u_vals = jax.vmap(u)(eta)
    L = jnp.linalg.cholesky(Sigma_obs)
    Y = u_vals + jr.normal(k_eps, u_vals.shape) @ L.T
    return u_vals, Y, eta


def report_gap(label, R, Sigma_obs):
    """``max_eig(R - Sigma_obs)`` -- diagnostic du conditionnement de Prop. 3.

    Negatif et loin de zero : assemblage bien conditionne. Positif : Prop. 3
    ne s'applique pas (``assemble_from_residual`` va lever ``ValueError``).
    Negatif mais proche de zero : formellement valide, numeriquement fragile.
    """
    gap = float(jnp.max(jnp.linalg.eigvalsh(R - Sigma_obs)))
    print(f"  max_eig(R - Sigma_obs) [{label}] = {gap:+.3e}")
    return gap


def try_assemble(name, fn, *args):
    """Assemble ``Sigma_signal``/``Sigma_noise``, ou rapporte l'echec sans planter.

    Prop. 3 n'est pas toujours applicable (cf. docstring du module) : un debruiteur
    trop faible face a la non-linearite leve ``ValueError``. Ce n'est pas un
    accident a masquer -- c'est l'information que ce script veut montrer.
    """
    try:
        return fn(*args)
    except ValueError as e:
        print(f"  [{name}] Prop. 3 non applicable : {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    p.add_argument(
        "--n-samples", type=int, default=20_000, help="paires (u, Y) pour les debruiteurs"
    )
    p.add_argument("--n-gradient", type=int, default=300, help="jacobiennes pour le gradient")
    p.add_argument("--net-steps", type=int, default=500, help="pas d'Adam, ResidualDenoiser")
    p.add_argument("--out", default="figures_conservative_mc")
    args = p.parse_args()

    use_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prior = make_prior()
    u = forward(args.lambda_)
    k_pairs, k_grad, k_net_f, k_net_g = jr.split(jr.key(0), 4)

    print(f"[paires] lambda={args.lambda_}, n_samples={args.n_samples}")
    u_vals, Y, eta = paired_samples(u, prior, SIGMA_OBS_MATRIX, k_pairs, args.n_samples)
    features_g = jnp.concatenate([Y, eta], axis=1)

    print(f"[gradient] n_gradient={args.n_gradient}")
    Sigma_signal_grad, Sigma_noise_grad = gradient_diagnostics_standard(
        u, prior, SIGMA_OBS_MATRIX, k_grad, args.n_gradient
    )

    print("[approximation affine]")
    denoiser_f_affine = AffineDenoiser.fit(u_vals, Y)
    denoiser_g_affine = AffineDenoiser.fit(u_vals, features_g)
    report_gap("affine, f: Y->u", denoiser_residual(denoiser_f_affine, u_vals, Y), SIGMA_OBS_MATRIX)
    report_gap(
        "affine, g: (Y,eta)->u",
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
        eta,
        SIGMA_OBS_MATRIX,
    )

    print(f"[approximation affine + reseau] steps={args.net_steps}")
    denoiser_f_nn = ResidualDenoiser.fit(u_vals, Y, k_net_f, steps=args.net_steps)
    denoiser_g_nn = ResidualDenoiser.fit(u_vals, features_g, k_net_g, steps=args.net_steps)
    report_gap("affine+NN, f: Y->u", denoiser_residual(denoiser_f_nn, u_vals, Y), SIGMA_OBS_MATRIX)
    report_gap(
        "affine+NN, g: (Y,eta)->u",
        denoiser_residual(denoiser_g_nn, u_vals, features_g),
        SIGMA_OBS_MATRIX,
    )
    Sigma_signal_nn = try_assemble(
        "affine+NN signal", approximation_signal, denoiser_f_nn, u_vals, Y, SIGMA_OBS_MATRIX
    )
    Sigma_noise_nn = try_assemble(
        "affine+NN noise", approximation_noise, denoiser_g_nn, u_vals, Y, eta, SIGMA_OBS_MATRIX
    )

    signals = [("gradient (§3.3)", Sigma_signal_grad)]
    if Sigma_signal_affine is not None:
        signals.append(("approximation affine (§3.2)", Sigma_signal_affine))
    if Sigma_signal_nn is not None:
        signals.append(("approximation affine+NN (§3.2)", Sigma_signal_nn))

    noises = [("gradient (§3.3)", Sigma_noise_grad)]
    if Sigma_noise_affine is not None:
        noises.append(("approximation affine (§3.2)", Sigma_noise_affine))
    if Sigma_noise_nn is not None:
        noises.append(("approximation affine+NN (§3.2)", Sigma_noise_nn))

    print("[figures]")
    if len(signals) > 1:
        save(
            plot_matrix_comparison(
                [S for _, S in signals],
                [lbl for lbl, _ in signals],
                reference=0,
                title=rf"$\Sigma_{{signal}}$ : gradient vs approximation -- $\lambda={args.lambda_}$",
            ),
            out / f"signal_comparison_lambda_{args.lambda_:.2f}.png",
        )
        save(
            plot_spectrum_comparison(
                [S for _, S in signals],
                [f"$\\Sigma_{{signal}}$ {lbl}" for lbl, _ in signals],
                title=rf"Spectres $\Sigma_{{signal}}$ -- $\lambda={args.lambda_}$",
            ),
            out / f"signal_spectra_lambda_{args.lambda_:.2f}.png",
        )
    if len(noises) > 1:
        save(
            plot_matrix_comparison(
                [S for _, S in noises],
                [lbl for lbl, _ in noises],
                reference=0,
                title=rf"$\Sigma_{{noise}}$ : gradient vs approximation -- $\lambda={args.lambda_}$",
            ),
            out / f"noise_comparison_lambda_{args.lambda_:.2f}.png",
        )
        save(
            plot_spectrum_comparison(
                [S for _, S in noises],
                [f"$\\Sigma_{{noise}}$ {lbl}" for lbl, _ in noises],
                title=rf"Spectres $\Sigma_{{noise}}$ -- $\lambda={args.lambda_}$",
            ),
            out / f"noise_spectra_lambda_{args.lambda_:.2f}.png",
        )

    print("[ecarts relatifs vs gradient]")
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
