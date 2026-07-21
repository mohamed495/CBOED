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
    """1D advection-diffusion, Crank-Nicolson, zero Dirichlet boundaries.

    Forward map G : theta -> final state, with theta = interior initial
    condition (n degrees of freedom). Since the map is linear, jacobian
    returns operator G as a matrix-free object (never materialized).
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
        return self._hyperparameters["diffusivity"]

    @property
    def velocity(self) -> float:
        return self._hyperparameters["velocity"]

    @property
    def T(self) -> float:
        return self._hyperparameters["T"]

    @property
    def domain(self) -> float:
        return self._hyperparameters["domain"]

    @property
    def dx(self) -> float:
        return (self.domain[1] - self.domain[0]) / (self.n + 1)

    @property
    def dt(self) -> float:
        return self.T / self.nt

    # ------------------------------------------------------------------
    # ForwardModel interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return 1

    @property
    def n_parameters(self) -> int:
        """Dimension d of parameter θ."""
        return self.n

    @property
    def n_obs(self) -> int:
        """Dimension p of the complete observable Y = u(θ)."""
        return self.n

    def __call__(
        self,
        theta: Float[Array, " n_parameters"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, " ???"]:
        """G(θ) : observed final state. Without a design, the full state Y = u(θ)."""
        y_full = self._forward(theta)  # Y ∈ ℝᵖ, full state
        if design is None:
            return y_full
        return y_full[design]  # Yₘ = Wₘᵀ Y ∈ ℝᵐ

    def jacobian_operator(
        self,
        theta: Float[Array, " n_parameters"],
        design: Float[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """dG/dtheta as a matrix-free operator.

        Parameters
        ----------
        theta : Float[Array, " n_parameters"]
            Point at which the Jacobian is evaluated.

        Returns
        -------
        LinearizedOperator
            Tangent operator, never materialized.

        Notes
        -----
        Since the map is linear, the result does not depend on theta.

        Example
        -----
        A [[1,2],
            [0,1]]
        x = [0,0]
        v = [1,1]
        f(x) = A@x
        jac = self.jacobian_operator(theta)
        res = jac(v)
        print(res) # res = [3,1]
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
        """dG/dtheta as a matrix-free operator (never materialized).
        Arguments:
            theta : scalar or ndarray
                where to evaluate the Jacobian
        return:
            J_theta(theta)

        Example:
            A [[1,2],
               [0,1]]
            x = [0,0]
            v = [1,1]
            f(x) = A@x
            jac = self.jacobian(theta)
            print(jac) # jac = A
        Since here, advection diffusion is linear we don't mind
        about the value of theta, only must be compatible to matrix-vector
        product operation
        """
        op = self.jacobian_operator(theta, design)
        return jnp.asarray(jax.vmap(op.matvec)(jnp.eye(self.n_parameters)).T)

    # ------------------------------------------------------------------
    # Numerical core: Crank-Nicolson step
    # ------------------------------------------------------------------

    def _factor(
        self,
    ) -> tuple[tuple[Float[Array, "n n"], Int[Array, " n"]], Float[Array, " 3"]]:
        """Builds A and the explicit kernel once. A is factorized (LU)."""
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
        """Advances U0 (full vector) over nt steps. A is factorized once."""
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
        """Map theta -> final state, on the interior (n -> n).

        Pads the interior initial condition with zero boundaries, advances,
        then strips the padding.
        """
        U0 = jnp.zeros(self.n + 2).at[1:-1].set(theta)
        return self.solve(U0)[1:-1]

    def linearize(self, theta0: Float[Array, " n_param"]) -> LinearizedOperator:
        """Tangent operator of _forward at point theta0.

        Linear map here: independent of theta0 (jnp.zeros(n) works fine).
        A single forward pass (jax.linearize); the adjoint is the transpose
        of the tangent (always valid: the tangent is linear).
        """
        y0, tangent = jax.linearize(self._forward, theta0)
        transpose_fn = jax.linear_transpose(tangent, theta0)
        matvec = tangent

        def rmatvec(w):
            return transpose_fn(w)[0]

        return LinearizedOperator(matvec, rmatvec, (y0.shape[0], theta0.shape[0]))
