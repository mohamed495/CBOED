from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jaxtyping import Float, Int

from cboed.core.base import ForwardModel
from cboed.core.linear_operator import LinearizedOperator, compose
from cboed.core.selection import selection_operator


class AdvectionDiffusion(ForwardModel):
    r"""1D linear advection-diffusion, Crank-Nicolson, zero Dirichlet boundaries.

    Forward map ``G : theta -> y``, with ``theta`` the interior initial
    condition (``n`` degrees of freedom) and ``y`` the final state. Since the
    map is linear, :meth:`jacobian_operator` returns ``G`` itself as a
    matrix-free operator (never materialized), independent of ``theta``.

    Parameters
    ----------
    diffusivity : float
        Diffusion coefficient.
    velocity : float
        Advection velocity.
    T : float | int
        Final time.
    domain : Sequence[float]
        Spatial domain bounds ``(x_min, x_max)``.
    nt : int
        Number of time steps.
    n : int
        Number of interior spatial points (``= n_parameters = n_obs``).
    """

    def __init__(
        self,
        diffusivity: float,
        velocity: float,
        T: float | int,
        domain: Sequence[float],
        nt: int,
        n: int,
    ) -> None:
        super().__init__(diffusivity=diffusivity, velocity=velocity, T=T, domain=domain)
        self.nt = nt  # number of time step
        self.n = n  # number of interior points

    # ------------------------------------------------------------------
    # Hyperparameters (stored by the base class in self._hyperparameters)
    # ------------------------------------------------------------------

    @property
    def diffusivity(self) -> float:
        """Diffusion coefficient."""
        return self._hyperparameters["diffusivity"]

    @property
    def velocity(self) -> float:
        """Advection velocity."""
        return self._hyperparameters["velocity"]

    @property
    def T(self) -> float:
        """Final time."""
        return self._hyperparameters["T"]

    @property
    def domain(self) -> float:
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

    # ------------------------------------------------------------------
    # ForwardModel interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Spatial dimension -- always ``1`` for this model."""
        return 1

    @property
    def n_parameters(self) -> int:
        """Dimension ``d`` of the parameter ``theta``."""
        return self.n

    @property
    def n_obs(self) -> int:
        """Dimension ``p`` of the complete observable ``Y = u(theta)``."""
        return self.n

    def __call__(
        self,
        theta: Float[Array, " n_parameters"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, " ???"]:
        r"""Evaluate ``G(theta)``, optionally restricted to ``design``.

        Parameters
        ----------
        theta : Float[Array, " n_parameters"]
            Interior initial condition.
        design : Int[Array, " n_obs"] | None
            Indices of observed components. ``None`` returns the full final
            state ``Y = u(theta)``.

        Returns
        -------
        Array
            Full final state ``Y in R^p`` if ``design`` is ``None``,
            otherwise the restricted observation ``Y_m = W_m^T Y in R^m``.
        """
        y_full = self._forward(theta)  # Y ∈ ℝᵖ, full state
        if design is None:
            return y_full
        return y_full[design]  # Yₘ = Wₘᵀ Y ∈ ℝᵐ

    def jacobian_operator(
        self,
        theta: Float[Array, " n_parameters"],
        design: Float[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        r"""Matrix-free Jacobian ``dG/dtheta``, optionally restricted to ``design``.

        Parameters
        ----------
        theta : Float[Array, " n_parameters"]
            Point at which the Jacobian is evaluated.
        design : Float[Array, " n_sensors"] | None
            Indices of observed components. ``None`` returns the full
            tangent operator; otherwise it is composed with the selection
            operator ``H`` (:func:`cboed.core.selection.selection_operator`).

        Returns
        -------
        LinearizedOperator
            Tangent operator, never materialized as a dense matrix.

        Notes
        -----
        Since the map is linear, the result does not depend on ``theta``
        (see :meth:`linearize`).
        """
        G = self.linearize(theta)
        if design is None:
            return G
        H = selection_operator(design, self.n)
        return compose(H, G)

    def jacobian(
        self,
        theta: Float[Array, " n_parameters"],
        design: Float[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_obs n_parameters"]:
        r"""Dense Jacobian ``dG/dtheta``, materialized column by column.

        Parameters
        ----------
        theta : Float[Array, " n_parameters"]
            Point at which the Jacobian is evaluated (irrelevant here since
            the map is linear, but kept for interface compatibility with
            non-linear models such as :class:`cboed.core.burgers.Burgers`).
        design : Float[Array, " n_sensors"] | None
            Indices of observed components (see :meth:`jacobian_operator`).

        Returns
        -------
        Float[Array, "n_obs n_parameters"]
            Dense Jacobian matrix, obtained by applying the matrix-free
            operator from :meth:`jacobian_operator` to each canonical basis
            vector of the parameter space.
        """
        op = self.jacobian_operator(theta, design)
        return jnp.asarray(jax.vmap(op.matvec)(jnp.eye(self.n_parameters)).T)

    # ------------------------------------------------------------------
    # Numerical core: Crank-Nicolson step
    # ------------------------------------------------------------------

    def _factor(
        self,
    ) -> tuple[tuple[Float[Array, "n n"], Int[Array, " n"]], Float[Array, " 3"]]:
        """Build and LU-factorize the constant implicit matrix, and the explicit stencil.

        Returns
        -------
        lu : tuple
            LU factorization of the implicit Crank-Nicolson matrix ``A``.
        kernel_B : Float[Array, " 3"]
            Explicit right-hand-side stencil (reversed order, for use with
            ``jax.scipy.signal.convolve``).
        """
        r = self.diffusivity * self.dt / (2 * self.dx**2)
        c = self.velocity * self.dt / (4 * self.dx)

        # Right-hand-side stencil (reversed order for convolve)
        kernel_B = jnp.array([r - c, 1 - 2 * r, r + c])

        # Implicit matrix A
        A = (
            jnp.diag((1 + 2 * r) * jnp.ones(self.n))
            + jnp.diag(-(r - c) * jnp.ones(self.n - 1), 1)
            + jnp.diag(-(r + c) * jnp.ones(self.n - 1), -1)
        )
        return jsp.linalg.lu_factor(A), kernel_B

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, U0: Float[Array, " n_plus_2"]) -> Float[Array, " n_plus_2"]:
        r"""Advance the full state ``U0`` over ``nt`` Crank-Nicolson steps.

        The implicit matrix ``A`` is factorized once (:meth:`_factor`) and
        reused at every step; boundary values of ``U0`` stay fixed at zero
        throughout (Dirichlet).

        Parameters
        ----------
        U0 : Float[Array, " n_plus_2"]
            Full initial state, including the two zero boundary points.

        Returns
        -------
        Float[Array, " n_plus_2"]
            Full state after ``nt`` steps.
        """
        lu, kernel_B = self._factor()

        def step(U, _):
            rhs = jsp.signal.convolve(U, kernel_B, mode="valid")
            U = U.at[1:-1].set(jsp.linalg.lu_solve(lu, rhs))
            return U, None

        U_final, _ = jax.lax.scan(step, U0, xs=None, length=self.nt)
        return U_final

    # ------------------------------------------------------------------
    # Differentiable map and linearized operator
    # ------------------------------------------------------------------

    def _forward(self, theta: Float[Array, " n_param"]) -> Float[Array, " n_obs"]:
        """Map ``theta`` to the final interior state (``n -> n``).

        Pads the interior initial condition with zero boundaries, advances
        via :meth:`solve`, then strips the padding.
        """
        U0 = jnp.zeros(self.n + 2).at[1:-1].set(theta)
        return self.solve(U0)[1:-1]

    def linearize(self, theta0: Float[Array, " n_param"]) -> LinearizedOperator:
        r"""Build the tangent operator of :meth:`_forward` at ``theta0``.

        Parameters
        ----------
        theta0 : Float[Array, " n_param"]
            Point at which the Jacobian is evaluated.

        Returns
        -------
        LinearizedOperator
            Matrix-free tangent operator, shape ``(n_obs, n_param)``.

        Notes
        -----
        The map is linear, so the result does not depend on ``theta0`` (any
        point, e.g. ``jnp.zeros(n)``, works equally well). A single forward
        pass via ``jax.linearize`` gives the tangent; its adjoint is the
        transpose of the tangent, always valid since the tangent itself is
        linear.
        """
        y0, tangent = jax.linearize(self._forward, theta0)
        transpose_fn = jax.linear_transpose(tangent, theta0)
        matvec = tangent

        def rmatvec(w):
            return transpose_fn(w)[0]

        return LinearizedOperator(matvec, rmatvec, (y0.shape[0], theta0.shape[0]))
