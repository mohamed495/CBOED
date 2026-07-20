r"""Débruiteurs pour §3.2 -- l'espace d'approximation ``F``.

Prop. 3 exige que ``F`` contienne les fonctions affines. On construit donc les
débruiteurs par **somme** :

    f(Y) = affine(Y) + reseau(Y)
           |            |
           forme fermee  apprend le residu non lineaire

L'affine capture toute la partie linéaire (exact à ``lambda = 0``, où ``E[u|Y]`` est
affine) ; le réseau ne corrige que ce que l'affine rate. Un réseau partant de zéro
convergerait bien plus lentement -- même principe que le préconditionnement : résoudre
la partie facile exactement, ne laisser au solveur que le reste.

Tous les débruiteurs sont des modules ``equinox`` -- des pytrees JAX, donc ``jit`` /
``grad`` / ``vmap`` s'appliquent directement.
"""

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jaxtyping import Array, Float, PRNGKeyArray


@runtime_checkable
class Denoiser(Protocol):
    """Une approximation ``f`` de ``E[u|features]``. ``features`` = ``Y`` (pour ``f``)
    ou ``(Y, theta)`` concaténés (pour ``g``).
    """

    def __call__(self, features: Float[Array, " n_feat"]) -> Float[Array, " n_obs"]:
        ...


# =============================================================================
# Affine -- forme fermee, le plancher de Prop. 3
# =============================================================================


class AffineDenoiser(eqx.Module):
    r"""``f(Y) = A Y + b`` -- moindres carrés en forme fermée.

    ``A = Cov(u, Y) Cov(Y)^{-1}``, ``b = E[u] - A E[Y]``. Exact quand ``E[u|Y]`` est
    affine, c'est-à-dire à ``lambda = 0``.
    """

    A: Float[Array, "n_obs n_feat"]
    b: Float[Array, " n_obs"]

    def __call__(self, features):
        return self.A @ features + self.b

    @staticmethod
    def fit(
        u_samples: Float[Array, "n_samples n_obs"],
        features: Float[Array, "n_samples n_feat"],
    ) -> "AffineDenoiser":
        """Régression sur les paires déjà tirées pour ``Sigma_Y`` -- coût marginal nul."""
        n = u_samples.shape[0]
        u_bar, f_bar = u_samples.mean(0), features.mean(0)
        U, Fc = u_samples - u_bar, features - f_bar
        cov_uf = U.T @ Fc / n
        cov_ff = Fc.T @ Fc / n
        jit = 1e-8 * jnp.trace(cov_ff) / cov_ff.shape[0]
        cov_ff = cov_ff + jit * jnp.eye(cov_ff.shape[0])
        A = jsp.linalg.solve(cov_ff, cov_uf.T, assume_a="pos").T
        return AffineDenoiser(A=A, b=u_bar - A @ f_bar)


# =============================================================================
# Residuel -- affine + reseau
# =============================================================================


class _MLP(eqx.Module):
    """MLP simple, sortie initialisée à zéro pour que le résidu parte de ``0``."""

    layers: list

    def __init__(self, n_in, n_out, width, depth, key):
        keys = jax.random.split(key, depth)
        sizes = [n_in] + [width] * (depth - 1) + [n_out]
        self.layers = [eqx.nn.Linear(a, b, key=k)
                       for a, b, k in zip(sizes[:-1], sizes[1:], keys, strict=True)]
        # derniere couche a zero : NN(Y) = 0 au depart, l'affine domine
        last = self.layers[-1]
        self.layers[-1] = eqx.tree_at(lambda m: (m.weight, m.bias), last,
                                      (jnp.zeros_like(last.weight),
                                       jnp.zeros_like(last.bias)))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


class ResidualDenoiser(eqx.Module):
    r"""``f(Y) = affine(Y) + net(Y)`` -- affine figé, réseau appris sur le résidu.

    L'affine est ``fit`` en forme fermée puis **gelé** : le réseau apprend
    ``u - affine(Y)``, qui est ``~0`` à ``lambda = 0`` et petit à ``lambda`` modéré.

    ``F`` contient les affines par construction (``net`` initialisé à zéro), donc
    Prop. 3 s'applique.
    """

    affine: AffineDenoiser
    net: _MLP

    def __call__(self, features):
        return self.affine(features) + self.net(features)

    @staticmethod
    def fit(
        u_samples: Float[Array, "n_samples n_obs"],
        features: Float[Array, "n_samples n_feat"],
        key: PRNGKeyArray,
        *,
        width: int = 0,
        depth: int = 3,
        steps: int = 2000,
        lr: float = 1e-3,
    ) -> "ResidualDenoiser":
        """Affine en forme fermée, puis ``steps`` pas d'Adam sur le résidu.

        Pas de validation ni d'early-stopping : le module reste simple, le script
        décide ``steps``. ``width = 0`` -> ``2 * n_obs``.
        """
        n_feat, n_obs = features.shape[1], u_samples.shape[1]
        width = width or 2 * n_obs
        affine = AffineDenoiser.fit(u_samples, features)
        target = u_samples - jax.vmap(affine)(features)   # le residu a apprendre

        net = _MLP(n_feat, n_obs, width, depth, key)
        opt = optax.adam(lr)
        state = opt.init(eqx.filter(net, eqx.is_array))

        @eqx.filter_jit
        def step(net, state):
            def loss(net):
                pred = jax.vmap(net)(features)
                return jnp.mean(jnp.sum((pred - target) ** 2, axis=1))
            val, grads = eqx.filter_value_and_grad(loss)(net)
            updates, state = opt.update(grads, state)
            return eqx.apply_updates(net, updates), state, val

        for _ in range(steps):
            net, state, _ = step(net, state)
        return ResidualDenoiser(affine=affine, net=net)
