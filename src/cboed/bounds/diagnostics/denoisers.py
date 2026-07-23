r"""Construct denoisers for §3.2 -- the approximation space ``F``.

Prop. 3 requires that ``F`` contain the affine functions. We therefore build
denoisers as a **sum**:

    f(Y) = affine(Y) + network(Y)
           |               |
           closed form     learns the non-linear residual

The affine part captures the entire linear component (exact at
``lambda = 0``, where ``E[u|Y]`` is affine); the network only corrects what
the affine part misses. A network starting from scratch would converge much
more slowly -- the same idea as preconditioning: solve the easy part exactly,
leave only the rest to the solver.

All denoisers are ``equinox`` modules -- JAX pytrees, so ``jit`` / ``grad`` /
``vmap`` apply directly.
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
    """Structural interface for a denoiser ``f`` (or ``g``).

    Any callable ``features -> observations`` of this shape can serve as a
    denoiser in :mod:`cboed.bounds.diagnostics.approximation_based`, e.g.
    :class:`AffineDenoiser` or :class:`ResidualDenoiser`.

    An approximation ``f`` of ``E[u|features]``, with ``features`` = ``Y``
    (for ``f``) or ``(Y, theta)`` concatenated (for ``g``).
    """

    def __call__(self, features: Float[Array, " n_feat"]) -> Float[Array, " n_obs"]: ...


# =============================================================================
# Affine -- closed form, the floor of Prop. 3
# =============================================================================


class AffineDenoiser(eqx.Module):
    r"""Affine denoiser ``f(Y) = A Y + b``, fit by least squares in closed form.

    .. math::
        A = \mathrm{Cov}(u, Y)\,\mathrm{Cov}(Y)^{-1}, \qquad b = E[u] - A\,E[Y]

    Exact when ``E[u|Y]`` is affine, i.e. at ``lambda = 0``.

    Attributes
    ----------
    A : Float[Array, "n_obs n_feat"]
        Linear coefficient.
    b : Float[Array, " n_obs"]
        Intercept.
    """

    A: Float[Array, "n_obs n_feat"]
    b: Float[Array, " n_obs"]

    def __call__(self, features):
        return self.A @ features + self.b

    @staticmethod
    @jax.jit
    def fit(
        u_samples: Float[Array, "n_samples n_obs"],
        features: Float[Array, "n_samples n_feat"],
    ) -> "AffineDenoiser":
        r"""Fit ``A`` and ``b`` by regression on the pairs already drawn for ``Sigma_Y``.

        Zero marginal cost: no new samples are drawn.

        Parameters
        ----------
        u_samples : Float[Array, "n_samples n_obs"]
            The ``u(eta^{(i)})`` -- the target to denoise.
        features : Float[Array, "n_samples n_feat"]
            The denoiser's inputs (``Y`` for ``f``; ``(Y, theta)`` concatenated
            for ``g``).

        Returns
        -------
        AffineDenoiser
            Fitted denoiser.

        Notes
        -----
        ``Cov(features)`` is regularized with a relative jitter
        (``1e-8 * trace(Cov(features)) / n_feat``): at small ``sigma_obs``
        and modest sample count, the empirical covariance can be nearly
        singular.
        """
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
# Residual -- affine + network
# =============================================================================


class _MLP(eqx.Module):
    """Simple MLP, output initialized to zero so the residual starts at ``0``."""

    layers: list

    def __init__(self, n_in, n_out, width, depth, key):
        keys = jax.random.split(key, depth)
        sizes = [n_in] + [width] * (depth - 1) + [n_out]
        self.layers = [
            eqx.nn.Linear(a, b, key=k) for a, b, k in zip(sizes[:-1], sizes[1:], keys, strict=True)
        ]
        # last layer at zero: NN(Y) = 0 at the start, the affine part dominates
        last = self.layers[-1]
        self.layers[-1] = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            last,
            (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
        )

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


class ResidualDenoiser(eqx.Module):
    r"""``f(Y) = affine(Y) + net(Y)`` -- affine frozen, network learned on the residual.

    The affine part is ``fit`` in closed form then **frozen**: the network
    learns ``u - affine(Y)``, which is ``~0`` at ``lambda = 0`` and small at
    moderate ``lambda``.

    ``F`` contains the affines by construction (``net`` initialized to zero),
    so Prop. 3 applies.

    Attributes
    ----------
    affine : AffineDenoiser
        Closed-form affine part, fit once and left frozen during training.
    net : _MLP
        Network trained on the residual ``u - affine(features)``.
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
        r"""Fit the affine part in closed form, then train the network on its residual.

        Parameters
        ----------
        u_samples : Float[Array, "n_samples n_obs"]
            The ``u(eta^{(i)})`` -- the target to denoise.
        features : Float[Array, "n_samples n_feat"]
            The denoiser's inputs (``Y`` for ``f``; ``(Y, theta)`` concatenated
            for ``g``).
        key : PRNGKeyArray
            Random key for the network's initialization.
        width : int, optional
            Hidden layer width. ``0`` (default) -> ``2 * n_obs``.
        depth : int, optional
            Number of layers, by default 3.
        steps : int, optional
            Number of Adam optimization steps, by default 2000.
        lr : float, optional
            Adam learning rate, by default 1e-3.

        Returns
        -------
        ResidualDenoiser
            Fitted denoiser: frozen affine part plus a network trained to
            predict ``u - affine(features)``.

        Notes
        -----
        No validation or early stopping: the module stays simple, the caller
        decides ``steps``.
        """
        n_feat, n_obs = features.shape[1], u_samples.shape[1]
        width = width or 2 * n_obs
        affine = AffineDenoiser.fit(u_samples, features)
        target = u_samples - jax.vmap(affine)(features)  # residual to learn

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
