#!/usr/bin/env python
r"""Protocole complet pour le papier -- standard et goal-oriented.

Balaie ``lambda in {0.0, 0.05, 0.2, 0.75, 1.0}``, cas standard et goal-oriented
(QoI = premiere moitie du champ), et compare trois voies vers
``Sigma_signal``/``Sigma_noise`` (gradient, approximation affine, approximation
affine+reseau). Chaque repetition retire toutes les quantites MC avec une
nouvelle cle **et** re-selectionne le design par glouton -- comme le fait le
prototype NumPy avec son ``N_REPEATS``.

Trois figures
-------------
1. Reconstruction (``lambda=0``, cas standard uniquement) : prior/posterieur/
   ``theta_vrai`` sur le champ complet, zone QoI ombree.
2. Spectre ``log(alpha_i)``, ``log(beta_i)``, leur somme (``lambda > 0``,
   methode gradient uniquement, standard et GO) -- Prop. 1.
3. Boxplots des bornes (incrementale et conservative) sur les ``N_repeats``,
   une serie par methode. La borne conservative utilise ``eig_full`` estime
   par Monte-Carlo imbrique (``NestedMonteCarloEIG`` en standard,
   ``GoalOrientedNestedMonteCarloEIG`` en GO) -- pas le defaut certifie.

Methodes fragiles
------------------
L'approximation affine seule leve ``ValueError`` (Prop. 3 non satisfaite) des
que ``lambda`` s'eloigne de 0 a l'echelle reelle -- attendu, pas une erreur du
protocole. Une methode qui echoue pour un ``(lambda, case, repeat)`` donne est
simplement absente de ce point : pas de crash, un print de diagnostic.

Echelle par defaut : reduite (validation du pipeline, pas des chiffres de
publication). Augmenter ``--n-samples``, ``--n-gradient``, ``--net-steps``,
``--nmc-*`` pour la production.

Usage
-----
    pixi run -e gpu python tutorials/paper_protocol.py --lambdas 0.0 0.75
    pixi run -e gpu python tutorials/paper_protocol.py --n-repeats 10
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cboed.benchmarks import (
    DOMAIN,
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
from cboed.bounds.diagnostics.approximation_based import approximation_noise, approximation_signal
from cboed.bounds.diagnostics.denoisers import AffineDenoiser, ResidualDenoiser
from cboed.bounds.diagnostics.gradient_based import gradient_diagnostics, gradient_diagnostics_standard
from cboed.bounds.diagnostics.sample_based import sample_Sigma_Y, sample_Sigma_Y_given_theta
from cboed.bounds.quasi_optimality import quasi_optimality
from cboed.estimators.nmc import NestedMonteCarloEIG
from cboed.estimators.nmc_go import GoalOrientedNestedMonteCarloEIG
from cboed.inference.goal_oriented import GoalOrientedModel
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.optim.greedy_schur import greedy_schur
from cboed.viz import bounds as vb
from cboed.viz import fields as vf
from cboed.viz import spectrum as vs
from cboed.viz.style import save, use_style

LAMBDAS_PROTOCOL = (0.0, 0.05, 0.2, 0.75, 1.0)
CASES = ("standard", "go")
METHODS = ("gradient", "affine", "affine_nn")

QOI_H = qoi_projection(N_QOI)
B_QOI = jnp.eye(N)[:N_QOI]
X = np.linspace(DOMAIN[0], DOMAIN[1], N + 2)[1:-1]


# =============================================================================
# Setup par (lambda, cas)
# =============================================================================


def build_case(lambda_: float, case: str):
    prior = make_prior()
    model = make_model(lambda_)
    u = forward(lambda_)
    likelihood = GaussianLikelihood(model=model, Sigma_obs=SIGMA_OBS_MATRIX)
    inference = LinearModel(prior=prior, likelihood=likelihood)
    go = None
    if case == "go":
        go = GoalOrientedModel(inner=inference, h=QOI_H, Sigma_theta=SIGMA_XI_QOI)
    return prior, model, u, likelihood, inference, go


def paired_samples(u, prior, key, n_samples):
    """``(u(eta), Y = u(eta) + eps, eta)`` -- pour les debruiteurs affine/NN."""
    k_eta, k_eps = jr.split(key)
    eta = prior.sample(k_eta, n_samples)
    u_vals = jax.vmap(u)(eta)
    L = jnp.linalg.cholesky(SIGMA_OBS_MATRIX)
    Y = u_vals + jr.normal(k_eps, u_vals.shape) @ L.T
    return u_vals, Y, eta


# =============================================================================
# Diagnostics par repetition -- les trois methodes
# =============================================================================


def compute_repeat(lambda_: float, case: str, key, n_samples: int, n_gradient: int, net_steps: int):
    """``(Sigma_Y, Sigma_Y_given_theta, {methode: (Sigma_signal, Sigma_noise) | None})``."""
    prior, model, u, likelihood, inference, go = build_case(lambda_, case)
    k_pairs, k_Y, k_Yth, k_grad, k_net_f, k_net_g = jr.split(key, 6)

    u_vals, Y, eta = paired_samples(u, prior, k_pairs, n_samples)
    Sigma_Y = sample_Sigma_Y(u, prior, SIGMA_OBS_MATRIX, k_Y, n_samples)

    if case == "standard":
        Sigma_Y_given_theta = SIGMA_OBS_MATRIX
        theta_for_noise = eta
        Sigma_signal_g, Sigma_noise_g = gradient_diagnostics_standard(
            u, prior, SIGMA_OBS_MATRIX, k_grad, n_gradient
        )
    else:
        Sigma_Y_given_theta = sample_Sigma_Y_given_theta(
            u, prior, B_QOI, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, k_Yth, n_samples
        )
        theta_for_noise = eta[:, :N_QOI]
        Sigma_signal_g, Sigma_noise_g = gradient_diagnostics(
            u, QOI_H, prior, SIGMA_OBS_MATRIX, SIGMA_XI_QOI, k_grad, n_gradient
        )

    methods: dict = {"gradient": (Sigma_signal_g, Sigma_noise_g)}
    features_g = jnp.concatenate([Y, theta_for_noise], axis=1)

    try:
        d_f = AffineDenoiser.fit(u_vals, Y)
        d_g = AffineDenoiser.fit(u_vals, features_g)
        Sigma_signal_a = approximation_signal(d_f, u_vals, Y, SIGMA_OBS_MATRIX)
        Sigma_noise_a = approximation_noise(d_g, u_vals, Y, theta_for_noise, SIGMA_OBS_MATRIX)
        methods["affine"] = (Sigma_signal_a, Sigma_noise_a)
    except ValueError as e:
        print(f"    [affine] echec lambda={lambda_} case={case}: {e}")
        methods["affine"] = None

    try:
        d_f_nn = ResidualDenoiser.fit(u_vals, Y, k_net_f, steps=net_steps)
        d_g_nn = ResidualDenoiser.fit(u_vals, features_g, k_net_g, steps=net_steps)
        Sigma_signal_nn = approximation_signal(d_f_nn, u_vals, Y, SIGMA_OBS_MATRIX)
        Sigma_noise_nn = approximation_noise(d_g_nn, u_vals, Y, theta_for_noise, SIGMA_OBS_MATRIX)
        methods["affine_nn"] = (Sigma_signal_nn, Sigma_noise_nn)
    except ValueError as e:
        print(f"    [affine+NN] echec lambda={lambda_} case={case}: {e}")
        methods["affine_nn"] = None

    return Sigma_Y, Sigma_Y_given_theta, methods


def estimate_eig_full(lambda_: float, case: str, key, nmc_n_outer: int, nmc_n_inner: int,
                       nmc_n_inner_theta: int, nmc_n_inner_marginal: int):
    """``EIG(I_p)`` par MC imbrique -- utilise pour ``eig_full`` de la borne conservative."""
    prior, model, u, likelihood, inference, go = build_case(lambda_, case)
    if case == "standard":
        est = NestedMonteCarloEIG(likelihood=likelihood, prior=prior)
        return est.estimate(key, n_outer=nmc_n_outer, n_inner=nmc_n_inner)
    est = GoalOrientedNestedMonteCarloEIG(likelihood=likelihood, prior_eta=prior, B=B_QOI, Sigma_xi=SIGMA_XI_QOI)
    return est.estimate(
        key, n_outer=nmc_n_outer, n_inner_theta=nmc_n_inner_theta, n_inner_marginal=nmc_n_inner_marginal
    )


STRATEGY_LABELS = ("iEIG design", "cEIG design")


def strategies_for_method(Sigma_Y, Sigma_Y_given_theta, Sigma_signal, Sigma_noise, eig_full_mc, certified, budgets):
    """Deux designs (iEIG, cEIG), quatre bornes chacun -- protocole du papier §2.

    Meme structure que ``make_figures.py::fig_bounds`` : le design choisi en
    optimisant (Sigma_signal, Sigma_Y_given_theta) sert de reference pour le
    panneau "iEIG design", celui optimisant (Sigma_Y, Sigma_noise) pour
    "cEIG design" -- chacun affiche ses quatre bornes (inc + cons).

    Parameters
    ----------
    eig_full_mc : float or None
        ``eig_full`` pour la borne conservative (Cor. 2). ``None`` (defaut,
        comme ``make_figures.py``) -> encadre par Cor. 1 au design complet,
        certifie. Une valeur -> l'estimation MC fournie -- decertifie la
        borne, et a lambda=0 le biais du NMC (voir conservative_certified_vs_mc.py)
        peut deconnecter completement le conservatif de l'incremental, qui
        eux doivent coincider exactement (Rem. 2.2, gap nul en lineaire).
    """
    dg = DiagnosticMatrices(
        Sigma_Y=Sigma_Y, Sigma_Y_given_theta=Sigma_Y_given_theta,
        Sigma_signal=Sigma_signal, Sigma_noise=Sigma_noise, certified=certified,
    )
    m_max = max(budgets)
    designs = {
        "iEIG design": greedy_schur(Sigma_signal, Sigma_Y_given_theta, m_max).design,
        "cEIG design": greedy_schur(Sigma_Y, Sigma_noise, m_max).design,
    }
    eig_full_arg = None if eig_full_mc is None else jnp.asarray(eig_full_mc)
    out = {}
    for label, W in designs.items():
        rows = {k: [] for k in ("inc_low", "inc_up", "cons_low", "cons_up")}
        for m in budgets:
            inc = incremental_bounds(dg, W[:m])
            cons = conservative_bounds(dg, W[:m], eig_full=eig_full_arg)
            rows["inc_low"].append(float(inc.lower))
            rows["inc_up"].append(float(inc.upper))
            rows["cons_low"].append(float(cons.lower))
            rows["cons_up"].append(float(cons.upper))
        out[label] = {k: np.array(v) for k, v in rows.items()}
    return out


# =============================================================================
# Calcul + cache, par (lambda, cas)
# =============================================================================


def compute_lambda_case(lambda_, case, n_repeats, n_samples, n_gradient, net_steps,
                         nmc_n_outer, nmc_n_inner, nmc_n_inner_theta, nmc_n_inner_marginal, budgets, base_seed,
                         eig_full_mode="certified"):
    """Boucle de repetition -- diagnostics 'une fois' (repeat 0) + bornes par repetition.

    Parameters
    ----------
    eig_full_mode : {"certified", "mc"}
        ``"certified"`` (defaut, comme ``make_figures.py``) : borne conservative
        encadree par Cor. 1 au design complet, jamais estimee. ``"mc"`` : estime
        ``eig_full`` par NMC -- decertifie la borne, coute un NMC en plus par
        repetition, et biaise fortement le resultat a petite echelle (voir
        ``conservative_certified_vs_mc.py``).
    """
    per_method = {
        m: {label: {k: [] for k in ("inc_low", "inc_up", "cons_low", "cons_up")} for label in STRATEGY_LABELS}
        for m in METHODS
    }
    once = None  # (Sigma_Y, Sigma_Y_given_theta, methods_diag) du repeat 0 -- pour fig 1/2

    for r in range(n_repeats):
        key = jr.fold_in(jr.key(base_seed), r)
        k_diag, k_eig = jr.split(key)
        print(f"    repeat {r + 1}/{n_repeats} ...", flush=True)
        Sigma_Y, Sigma_Y_given_theta, methods_diag = compute_repeat(
            lambda_, case, k_diag, n_samples, n_gradient, net_steps
        )
        eig_full_mc = None
        if eig_full_mode == "mc":
            eig_full_mc = estimate_eig_full(
                lambda_, case, k_eig, nmc_n_outer, nmc_n_inner, nmc_n_inner_theta, nmc_n_inner_marginal
            )
        if r == 0:
            once = (Sigma_Y, Sigma_Y_given_theta, methods_diag)

        for method in METHODS:
            diag = methods_diag.get(method)
            if diag is None:
                continue
            Sigma_signal, Sigma_noise = diag
            res = strategies_for_method(
                Sigma_Y, Sigma_Y_given_theta, Sigma_signal, Sigma_noise,
                eig_full_mc, certified=(method == "gradient"), budgets=budgets,
            )
            for label, d in res.items():
                for k, v in d.items():
                    per_method[method][label][k].append(v)

    per_method = {
        m: {label: {k: np.stack(v) for k, v in d.items()} for label, d in strat.items()}
        for m, strat in per_method.items()
        if strat[STRATEGY_LABELS[0]]["inc_low"]
    }
    return once, per_method


CACHE_SCHEMA_VERSION = 2  # bump si la structure de per_method change -- invalide les caches perimes


def cache_path(cache_dir, lambda_, case, eig_full_mode):
    return cache_dir / f"protocol_v{CACHE_SCHEMA_VERSION}_{eig_full_mode}_lambda_{lambda_:.2f}_{case}.npz"


def load_or_compute(lambda_, case, cache_dir, force, **kwargs):
    path = cache_path(cache_dir, lambda_, case, kwargs.get("eig_full_mode", "certified"))
    if path.exists() and not force:
        print(f"  cache  {path.name}")
        data = dict(np.load(path, allow_pickle=True))
        once = (data["once_Sigma_Y"], data["once_Sigma_Y_given_theta"], data["once_methods"].item())
        per_method = data["per_method"].item()
        return once, per_method
    print(f"  calcul lambda={lambda_} case={case} ...", flush=True)
    once, per_method = compute_lambda_case(lambda_, case, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        once_Sigma_Y=np.asarray(once[0]),
        once_Sigma_Y_given_theta=np.asarray(once[1]),
        once_methods=np.array({k: (np.asarray(v[0]), np.asarray(v[1])) if v is not None else None
                                for k, v in once[2].items()}, dtype=object),
        per_method=np.array(per_method, dtype=object),
    )
    return once, per_method


# =============================================================================
# Figure 1 -- reconstruction, lambda=0, standard, zone QoI
# =============================================================================


def fig_reconstruction(once_standard_lambda0, out: Path, m_design: int = 10):
    Sigma_Y, Sigma_Y_given_theta, methods_diag = once_standard_lambda0
    Sigma_signal, Sigma_noise = methods_diag["gradient"]

    prior, model, u, likelihood, inference, _ = build_case(0.0, "standard")
    dg = DiagnosticMatrices(
        Sigma_Y=jnp.asarray(Sigma_Y), Sigma_Y_given_theta=jnp.asarray(Sigma_Y_given_theta),
        Sigma_signal=jnp.asarray(Sigma_signal), Sigma_noise=jnp.asarray(Sigma_noise), certified=True,
    )
    design = greedy_schur(dg.Sigma_signal, dg.Sigma_Y_given_theta, m_design).design

    k_true, k_noise, k_prior, k_post = jr.split(jr.key(42), 4)
    theta_true = prior.sample(k_true, 1)[0]
    y = model(theta_true, design) + jr.normal(k_noise, (len(design),)) * jnp.sqrt(
        jnp.diag(SIGMA_OBS_MATRIX)[design]
    )
    mu_post = inference._mu(y, prior.mu, design)
    Gamma_post = inference._cov(prior.mu, design)

    post = mu_post + jr.normal(k_post, (200, N)) @ np.linalg.cholesky(
        np.asarray(Gamma_post) + 1e-10 * np.eye(N)
    ).T
    qoi_span = (float(X[0]), float(X[N_QOI - 1]))

    save(
        vf.plot_reconstruction(
            X, np.asarray(prior.sample(k_prior, 200)), np.asarray(post), np.asarray(theta_true),
            sensors=np.asarray(design), qoi_span=qoi_span,
        ),
        out / "01_reconstruction_standard_lambda_0.00.png",
    )


# =============================================================================
# Figure 2 -- spectre log(alpha), log(beta), somme -- gradient, lambda>0
# =============================================================================


def fig_spectrum(all_once, out: Path):
    for case in CASES:
        alpha_by_lambda, beta_by_lambda = {}, {}
        for lambda_ in LAMBDAS_PROTOCOL:
            if lambda_ == 0.0:
                continue
            once = all_once.get((lambda_, case))
            if once is None:
                continue
            Sigma_Y, Sigma_Y_given_theta, methods_diag = once
            Sigma_signal, Sigma_noise = methods_diag["gradient"]
            dg = DiagnosticMatrices(
                Sigma_Y=jnp.asarray(Sigma_Y), Sigma_Y_given_theta=jnp.asarray(Sigma_Y_given_theta),
                Sigma_signal=jnp.asarray(Sigma_signal), Sigma_noise=jnp.asarray(Sigma_noise), certified=True,
            )
            q = quasi_optimality(dg)
            alpha_by_lambda[lambda_] = np.asarray(q.alpha)
            beta_by_lambda[lambda_] = np.asarray(q.beta)

        if not alpha_by_lambda:
            continue
        save(
            vs.plot_spectrum_vs_lambda(alpha_by_lambda, beta_by_lambda, title=f"gradient, {case}"),
            out / f"02_spectrum_vs_lambda_{case}.png",
        )


# =============================================================================
# Figure 3 -- boxplots des bornes par methode
# =============================================================================


def fig_boxplots(per_method_all, budgets, out: Path):
    """Une figure par (lambda, cas, methode) -- meme mise en page que ``07_bounds_lambda``
    (2 panneaux, design iEIG / design cEIG), boxplot a chaque budget au lieu
    d'une bande continue.
    """
    for (lambda_, case), per_method in per_method_all.items():
        for method, per_strategy in per_method.items():
            save(
                vb.plot_two_strategies_boxplot(
                    budgets, per_strategy, title=rf"{method}, {case}, $\lambda={lambda_}$"
                ),
                out / f"03_boxplot_{method}_{case}_lambda_{lambda_:.2f}.png",
            )


# =============================================================================
# Orchestration
# =============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS_PROTOCOL))
    p.add_argument("--cases", type=str, nargs="+", default=list(CASES), choices=CASES)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-gradient", type=int, default=200)
    p.add_argument("--net-steps", type=int, default=500)
    p.add_argument("--nmc-n-outer", type=int, default=100)
    p.add_argument("--nmc-n-inner", type=int, default=300)
    p.add_argument("--nmc-n-inner-theta", type=int, default=200)
    p.add_argument("--nmc-n-inner-marginal", type=int, default=300)
    p.add_argument("--budgets", type=int, nargs="+", default=list(SENSOR_BUDGETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--eig-full-mode", choices=("certified", "mc"), default="certified",
        help="Borne conservative : 'certified' (defaut, comme make_figures.py, Cor. 1 au design"
             " complet) ou 'mc' (NMC -- decertifie, biaise fortement a petite echelle).",
    )
    p.add_argument("--out", default="figures_protocol")
    p.add_argument("--cache", default=".cache_protocol")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    use_style()
    out, cache_dir = Path(args.out), Path(args.cache)

    all_once: dict = {}
    per_method_all: dict = {}

    print("Calcul (par lambda x cas)")
    for lambda_ in args.lambdas:
        for case in args.cases:
            once, per_method = load_or_compute(
                lambda_, case, cache_dir, args.force,
                n_repeats=args.n_repeats, n_samples=args.n_samples, n_gradient=args.n_gradient,
                net_steps=args.net_steps, nmc_n_outer=args.nmc_n_outer, nmc_n_inner=args.nmc_n_inner,
                nmc_n_inner_theta=args.nmc_n_inner_theta, nmc_n_inner_marginal=args.nmc_n_inner_marginal,
                budgets=args.budgets, base_seed=args.seed, eig_full_mode=args.eig_full_mode,
            )
            all_once[(lambda_, case)] = once
            per_method_all[(lambda_, case)] = per_method

    print("Figures")
    if (0.0, "standard") in all_once:
        fig_reconstruction(all_once[(0.0, "standard")], out)
    fig_spectrum(all_once, out)
    fig_boxplots(per_method_all, args.budgets, out)

    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
