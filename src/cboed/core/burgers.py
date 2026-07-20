from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jaxtyping import Float, Int

from cboed.core.base import ForwardModel
from cboed.core.linear_operator import LinearizedOperator, compose
from cboed.core.selection import selection_operator


class Burgers(ForwardModel):
    r"""Burgers 1D visqueux, IMEX (diffusion implicite, advection explicite).

    .. math::
        \partial_t u + \lambda u \partial_x u = \nu \partial_{xx} u

    lambda_ : paramètre de non-linéarité (0 = diffusion pure, 1 = Burgers complet).
    Carte G : theta (CI intérieure) -> état final. Non-linéaire sauf à lambda=0.
    """

    def __init__(self, diffusivity, lambda_, T, domain, nt, n):
        super().__init__(diffusivity=diffusivity, lambda_=lambda_, T=T, domain=domain)
        self.nt = nt
        self.n = n

    @property
    def diffusivity(self) -> float:
        return self._hyperparameters["diffusivity"]

    @property
    def lambda_(self) -> float:
        return self._hyperparameters["lambda_"]

    @property
    def T(self) -> float:
        return self._hyperparameters["T"]

    @property
    def domain(self) -> tuple[float, float]:
        return self._hyperparameters["domain"]

    @property
    def dx(self) -> float:
        return (self.domain[1] - self.domain[0]) / (self.n + 1)

    @property
    def dt(self) -> float:
        return self.T / self.nt

    @property
    def n_parameters(self) -> int:
        return self.n

    @property
    def n_obs(self) -> int:
        return self.n

    @property
    def dim(self) -> int:
        return 1

    def _diffusion_factor(self):
        """Matrice implicite A (diffusion Crank-Nicolson), constante. Factorisée."""
        r = self.diffusivity * self.dt / (2 * self.dx**2)
        A = (
            jnp.diag((1 + 2 * r) * jnp.ones(self.n))
            + jnp.diag(-r * jnp.ones(self.n - 1), 1)
            + jnp.diag(-r * jnp.ones(self.n - 1), -1)
        )
        return jsp.linalg.lu_factor(A), r

    def _nonlinear_flux(self, u):
        r"""Terme d'advection non-linéaire λ u ∂ₓu, différences centrées.

        Forme conservative : λ ∂ₓ(u²/2). Sur l'intérieur, bords nuls.
        """
        u_pad = jnp.zeros(self.n + 2).at[1:-1].set(u)
        flux = 0.5 * u_pad**2
        # ∂ₓ centré : (flux[i+1] - flux[i-1]) / (2 dx)
        dflux = (flux[2:] - flux[:-2]) / (2 * self.dx)
        return self.lambda_ * dflux

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, U0):
        """Avance U0 (vecteur complet n+2) sur nt pas. IMEX."""
        lu, r = self._diffusion_factor()

        def step(U, _):
            u_int = U[1:-1]
            # diffusion explicite (partie CN droite) + advection non-linéaire explicite
            expl = (
                u_int
                + r * (U[2:] - 2 * u_int + U[:-2])  # diffusion explicite
                - self.dt * self._nonlinear_flux(u_int)  # advection non-linéaire
            )
            u_new = jsp.linalg.lu_solve(lu, expl)
            U = U.at[1:-1].set(u_new)
            return U, None

        U_final, _ = jax.lax.scan(step, U0, xs=None, length=self.nt)
        return U_final

    def _forward(self, theta: Float[Array, " n_param"]) -> Float[Array, " n_obs"]:
        U0 = jnp.zeros(self.n + 2).at[1:-1].set(theta)
        return self.solve(U0)[1:-1]

    def __call__(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, " n_obs"]:
        y_full = self._forward(theta)
        return y_full if design is None else y_full[design]

    def linearize(self, theta0: Float[Array, " n_param"]) -> LinearizedOperator:
        """Opérateur tangent au point theta0. Dépend de theta0 (non-linéaire)."""
        y0, tangent = jax.linearize(self._forward, theta0)
        transpose_fn = jax.linear_transpose(tangent, theta0)
        return LinearizedOperator(
            tangent,
            lambda w: transpose_fn(w)[0],
            (y0.shape[0], theta0.shape[0]),
        )

    def jacobian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> LinearizedOperator:
        G = self.linearize(theta)
        if design is None:
            return G
        H = selection_operator(design, self.n)
        return compose(H, G)

    def jacobian(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, "n_obs n_param"]:
        op = self.jacobian_operator(theta, design)
        return jnp.asarray(jax.vmap(op.matvec)(jnp.eye(self.n_parameters)).T)
