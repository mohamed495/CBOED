from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jaxtyping import Float, Int

from cboed.core.base import ForwardModel
from cboed.core.linear_operator import LinearizedOperator


class AdvectionDiffusion(ForwardModel):
    """Advection-diffusion 1D, Crank-Nicolson, bords de Dirichlet nuls.

    Carte directe G : theta -> etat final, avec theta = condition initiale
    interieure (n degres de liberte). La carte etant lineaire, jacobian rend
    l'operateur G comme objet matrix-free (jamais materialise).
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
    # Hyperparametres (ranges par la base dans self._hyperparameters)
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
    # Interface ForwardModel
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return 1

    @property
    def n_parameters(self) -> int:
        return self.n

    @property
    def n_obs(self) -> int:
        return self.n  # etat complet interieur (pas de capteurs ici)

    def __call__(
        self,
        theta: Float[Array, " n_parameters"],
        xi: Float[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_obs"]:
        """G(theta) : etat final interieur a partir de la CI theta."""
        return self._forward(theta)

    def jacobian_operator(
        self,
        theta: Float[Array, " n_parameters"],
        xi: Float[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """dG/dtheta comme opérateur matrix-free.

        Parameters
        ----------
        theta : Float[Array, " n_parameters"]
            Point d'évaluation de la jacobienne.

        Returns
        -------
        LinearizedOperator
            Opérateur tangent, jamais matérialisé.

        Notes
        -----
        Since the map is linear étant, the result don't realy on theta.

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
        return self.linearize(theta)

    def jacobian(
        self,
        theta: Float[Array, " n_parameters"],
        xi: Float[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_obs n_parameters"]:
        """dG/dtheta comme operateur matrix-free (jamais materialise).
        Arguments :
            theta : scalar or ndarray
                where to evaluate the Jacobian
        return:
            J_theta(theta)

        Exemple :
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
        op = self.jacobian_operator(theta)
        return jnp.asarray(jax.vmap(op.matvec)(jnp.eye(self.n)).T)

    # ------------------------------------------------------------------
    # Coeur numerique : pas de Crank-Nicolson
    # ------------------------------------------------------------------

    def _factor(
        self,
    ) -> tuple[tuple[Float[Array, "n n"], Int[Array, " n"]], Float[Array, " 3"]]:
        """Construit A et le noyau explicite une fois. A est factorise (LU)."""
        r = self.diffusivity * self.dt / (2 * self.dx**2)
        c = self.velocity * self.dt / (4 * self.dx)

        # Stencil du membre de droite (ordre inverse pour convolve)
        kernel_B = jnp.array([r - c, 1 - 2 * r, r + c])

        # Matrice implicite A
        A = (
            jnp.diag((1 + 2 * r) * jnp.ones(self.n))
            + jnp.diag(-(r - c) * jnp.ones(self.n - 1), 1)
            + jnp.diag(-(r + c) * jnp.ones(self.n - 1), -1)
        )
        return jsp.linalg.lu_factor(A), kernel_B

    def solve(self, U0: Float[Array, " n_plus_2"]) -> Float[Array, " n_plus_2"]:
        """Avance U0 (vecteur complet) sur nt pas. A factorise une fois."""
        lu, kernel_B = self._factor()

        def step(U, _):
            rhs = jsp.signal.convolve(U, kernel_B, mode="valid")
            U = U.at[1:-1].set(jsp.linalg.lu_solve(lu, rhs))
            return U, None

        U_final, _ = jax.lax.scan(step, U0, xs=None, length=self.nt)
        return U_final

    # ------------------------------------------------------------------
    # Carte differentiable et operateur linearise
    # ------------------------------------------------------------------

    def _forward(self, theta: Float[Array, " n_param"]) -> Float[Array, " n_obs"]:
        """Carte theta -> etat final, sur l'interieur (n -> n).

        Pad la CI interieure avec des bords nuls, avance, puis depad.
        """
        U0 = jnp.zeros(self.n + 2).at[1:-1].set(theta)
        return self.solve(U0)[1:-1]

    def linearize(self, theta0: Float[Array, " n_param"]) -> LinearizedOperator:
        """Operateur tangent de _forward au point theta0.

        Carte lineaire ici : independant de theta0 (jnp.zeros(n) convient).
        Une seule passe forward (jax.linearize) ; l'adjoint est la transposee
        de la tangente (toujours licite : la tangente est lineaire).
        """
        y0, tangent = jax.linearize(self._forward, theta0)
        transpose_fn = jax.linear_transpose(tangent, theta0)
        matvec = tangent

        def rmatvec(w):
            return transpose_fn(w)[0]

        return LinearizedOperator(matvec, rmatvec, (y0.shape[0], theta0.shape[0]))
