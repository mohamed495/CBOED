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
    r"""1D viscous Burgers equation, solved with an IMEX scheme.

    .. math::
        \partial_t u + \lambda u \partial_x u = \nu \partial_{xx} u

    Diffusion is treated implicitly (Crank-Nicolson), advection explicitly
    (conservative, centered differences). ``lambda_`` is the non-linearity
    parameter: ``0`` gives pure (linear) diffusion, ``1`` the full Burgers
    equation. The forward map ``G : theta -> y`` sends the interior initial
    condition ``theta`` to the final state; it is non-linear for
    ``lambda_ != 0`` and linear at ``lambda_ = 0``.

    Parameters
    ----------
    diffusivity : float
        Viscosity ``nu``.
    lambda_ : float
        Non-linearity parameter (``0``: pure diffusion, ``1``: full Burgers).
    T : float
        Final time.
    domain : tuple[float, float]
        Spatial domain bounds ``(x_min, x_max)``.
    nt : int
        Number of time steps.
    n : int
        Number of interior spatial points (``= n_parameters = n_obs``).
    """

    def __init__(self, diffusivity, lambda_, T, domain, nt, n):
        super().__init__(diffusivity=diffusivity, lambda_=lambda_, T=T, domain=domain)
        self.nt = nt
        self.n = n

    @property
    def diffusivity(self) -> float:
        """Viscosity ``nu``."""
        return self._hyperparameters["diffusivity"]

    @property
    def lambda_(self) -> float:
        """Non-linearity parameter (``0``: pure diffusion, ``1``: full Burgers)."""
        return self._hyperparameters["lambda_"]

    @property
    def T(self) -> float:
        """Final time."""
        return self._hyperparameters["T"]

    @property
    def domain(self) -> tuple[float, float]:
        """Spatial domain bounds ``(x_min, x_max)``."""
        return self._hyperparameters["domain"]

    @property
    def dx(self) -> float:
        """Spatial step, ``(x_max - x_min) / (n + 1)``."""
        return (self.domain[1] - self.domain[0]) / (self.n + 1)

    @property
    def dt(self) -> float:
        """Time step, ``T / nt``."""
        return self.T / self.nt

    @property
    def n_parameters(self) -> int:
        """Dimension ``d`` of the parameter ``theta``."""
        return self.n

    @property
    def n_obs(self) -> int:
        """Dimension ``p`` of the complete observable ``Y = u(theta)``."""
        return self.n

    @property
    def dim(self) -> int:
        """Spatial dimension -- always ``1`` for this model."""
        return 1

    def _diffusion_factor(self):
        """Build and LU-factorize the constant implicit diffusion matrix ``A``.

        Returns
        -------
        lu : tuple
            LU factorization of ``A`` (Crank-Nicolson diffusion operator),
            as returned by ``jax.scipy.linalg.lu_factor``.
        r : float
            Diffusion number ``nu * dt / (2 * dx^2)``.
        """
        r = self.diffusivity * self.dt / (2 * self.dx**2)
        A = (
            jnp.diag((1 + 2 * r) * jnp.ones(self.n))
            + jnp.diag(-r * jnp.ones(self.n - 1), 1)
            + jnp.diag(-r * jnp.ones(self.n - 1), -1)
        )
        return jsp.linalg.lu_factor(A), r

    def _nonlinear_flux(self, u):
        r"""Explicit non-linear advection term ``lambda u d_x u``.

        Conservative form ``lambda d_x(u^2 / 2)``, centered differences, zero
        Dirichlet boundaries padded around the interior ``u``.

        Parameters
        ----------
        u : Float[Array, " n"]
            Interior state.

        Returns
        -------
        Float[Array, " n"]
            ``lambda * d_x(u^2 / 2)`` on the interior points.
        """
        u_pad = jnp.zeros(self.n + 2).at[1:-1].set(u)
        flux = 0.5 * u_pad**2
        # centered ∂ₓ : (flux[i+1] - flux[i-1]) / (2 dx)
        dflux = (flux[2:] - flux[:-2]) / (2 * self.dx)
        return self.lambda_ * dflux

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, U0):
        r"""Advance the full state ``U0`` over ``nt`` IMEX time steps.

        Diffusion is implicit (solved via the pre-factorized ``A`` from
        :meth:`_diffusion_factor`), advection is explicit
        (:meth:`_nonlinear_flux`). Boundary values of ``U0`` stay fixed at
        zero throughout (Dirichlet).

        Parameters
        ----------
        U0 : Float[Array, " n_plus_2"]
            Full initial state, including the two zero boundary points.

        Returns
        -------
        Float[Array, " n_plus_2"]
            Full state after ``nt`` steps.
        """
        lu, r = self._diffusion_factor()

        def step(U, _):
            u_int = U[1:-1]
            # explicit diffusion (right-hand CN part) + explicit non-linear advection
            expl = (
                u_int
                + r * (U[2:] - 2 * u_int + U[:-2])  # explicit diffusion
                - self.dt * self._nonlinear_flux(u_int)  # non-linear advection
            )
            u_new = jsp.linalg.lu_solve(lu, expl)
            U = U.at[1:-1].set(u_new)
            return U, None

        U_final, _ = jax.lax.scan(step, U0, xs=None, length=self.nt)
        return U_final

    def _forward(self, theta: Float[Array, " n_param"]) -> Float[Array, " n_obs"]:
        """Map ``theta`` to the final interior state (``n -> n``).

        Pads the interior initial condition with zero boundaries, advances
        via :meth:`solve`, then strips the padding.
        """
        U0 = jnp.zeros(self.n + 2).at[1:-1].set(theta)
        return self.solve(U0)[1:-1]

    def __call__(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, " n_obs"]:
        r"""Evaluate ``G(theta)``, optionally restricted to ``design``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Interior initial condition.
        design : Int[Array, " n_obs"] | None
            Indices of observed components. ``None`` returns the full final
            state ``y = G(theta)``.

        Returns
        -------
        Float[Array, " n_obs"]
            Final state (full, or restricted to ``design``).
        """
        y_full = self._forward(theta)
        return y_full if design is None else y_full[design]

    def linearize(self, theta0: Float[Array, " n_param"]) -> LinearizedOperator:
        r"""Build the tangent operator of :meth:`_forward` at ``theta0``.

        Unlike the linear advection-diffusion model, Burgers is non-linear
        (for ``lambda_ != 0``): the tangent depends on the point ``theta0``
        it is linearized at.

        Parameters
        ----------
        theta0 : Float[Array, " n_param"]
            Point at which the Jacobian is evaluated.

        Returns
        -------
        LinearizedOperator
            Matrix-free tangent operator, shape ``(n_obs, n_param)``,
            obtained via ``jax.linearize`` (forward pass) with the adjoint
            given by ``jax.linear_transpose`` of the tangent.
        """
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
        r"""Matrix-free Jacobian ``dG/dtheta``, optionally restricted to ``design``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Point at which the Jacobian is evaluated.
        design : Int[Array, " n_obs"] | None
            Indices of observed components. ``None`` returns the full
            tangent operator; otherwise it is composed with the selection
            operator ``H`` (:func:`cboed.core.selection.selection_operator`).

        Returns
        -------
        LinearizedOperator
            Tangent operator, never materialized as a dense matrix.
        """
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
        r"""Dense Jacobian ``dG/dtheta``, materialized column by column.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Point at which the Jacobian is evaluated.
        design : Int[Array, " n_obs"] | None
            Indices of observed components (see :meth:`jacobian_operator`).

        Returns
        -------
        Float[Array, "n_obs n_param"]
            Dense Jacobian matrix, obtained by applying the matrix-free
            operator from :meth:`jacobian_operator` to each canonical basis
            vector of the parameter space.
        """
        op = self.jacobian_operator(theta, design)
        return jnp.asarray(jax.vmap(op.matvec)(jnp.eye(self.n_parameters)).T)
