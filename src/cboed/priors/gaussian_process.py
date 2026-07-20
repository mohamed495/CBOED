"""Priors gaussiens : processus gaussien + façade inférentielle."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.priors.base import KernelBase, Prior


class GaussianProcess:
    r"""Processus gaussien sur une grille 1D.

    Dit **d'où vient la covariance** : évalue le noyau sur la grille et
    stabilise la Gram par un nugget. La façade :class:`GaussianPrior` dit
    **comment l'utiliser pour inférer** -- ne pas fusionner les deux.

    Parameters
    ----------
    kernel : KernelBase
        Noyau de covariance.
    mu : Float[Array, " n_param"]
        Moyenne du prior. Sa longueur fixe la taille de la grille.
    domain : tuple[float, float], default=(0.0, 1.0)
        Intervalle spatial.
    jitter : float, default=1e-8
        Nugget **relatif** : la diagonale reçoit ``jitter * tr(K)/n``.

    Notes
    -----
    Le jitter est relatif et non absolu : un nugget absolu ne signifie pas la
    même chose selon ``sigma``, et disparaît quand la variance du signal est
    grande. ``tr(K)/n`` vaut ``sigma**2`` pour un noyau stationnaire tout en
    restant défini pour un noyau qui ne l'est pas.

    Ce n'est **pas** qu'une astuce numérique : sur grille fine un noyau RBF est
    effectivement de rang déficient (décroissance spectrale
    super-exponentielle), et le nugget porte alors les derniers modes -- il
    modifie le prior. D'où ``test_jitter_does_not_move_eig``.

    Examples
    --------
    >>> gp = GaussianProcess(Matern32(length_scale=0.2, sigma=1.0), jnp.zeros(64))
    >>> gp.Sigma.shape
    (64, 64)
    """

    def __init__(
        self,
        kernel: KernelBase,
        mu: Float[Array, " n_param"],
        domain: tuple[float, float] = (0.0, 1.0),
        jitter: float = 1e-8,
    ) -> None:
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")
        self.kernel = kernel
        self.mu = mu
        self.domain = domain
        self.jitter = jitter
        self.x = jnp.linspace(domain[0], domain[1], len(mu))
        self.Sigma = self._build_covariance(self.x)

    def _build_covariance(self, x: Float[Array, " n_param"]) -> Float[Array, "n_param n_param"]:
        """Gram + nugget relatif."""
        K = self.kernel(x, x)
        n = x.shape[0]
        scale = jnp.trace(K) / n
        return K + self.jitter * scale * jnp.eye(n, dtype=K.dtype)


class GaussianPrior(Prior):
    r"""Façade inférentielle sur un :class:`GaussianProcess`.

    Factorise ``Gamma_prior`` **une fois** et expose les actions dont
    l'inférence a besoin. Aucune inversion à la construction : les oracles
    denses (``Sigma()``, ``hessian()``) sont hérités de :class:`Prior` et
    matérialisent à la demande.

    Parameters
    ----------
    prior : GaussianProcess
        Passé en mot-clé : ``GaussianPrior(prior=gp)``.
    """

    def __init__(self, **hyperparameters) -> None:
        self._hyperparameters = hyperparameters
        Sigma = self.prior.Sigma
        # Unique factorisation : solves *et* échantillonnage en dérivent.
        self._chol = jsp.linalg.cho_factor(Sigma, lower=True)
        # cho_factor laisse des résidus dans l'autre triangle : tril() rend le
        # facteur propre attendu par l'échantillonnage.
        self._L = jnp.tril(self._chol[0])

    @property
    def prior(self) -> GaussianProcess:
        return self._hyperparameters["prior"]

    @property
    def mu(self) -> Float[Array, " n_param"]:
        return self.prior.mu

    @jaxtyped(typechecker=beartype)
    def log_prior(self, theta: Float[Array, " n_param"]) -> Float[Array, ""]:
        """Log-densité gaussienne."""
        n = theta.shape[0]
        r = theta - self.mu
        quad = r @ jsp.linalg.cho_solve(self._chol, r)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(self._chol[0])))
        return -0.5 * (n * jnp.log(2 * jnp.pi) + logdet + quad)

    @jaxtyped(typechecker=beartype)
    def grad_log_prior(self, theta: Float[Array, " n_param"]) -> Float[Array, " n_param"]:
        return -jsp.linalg.cho_solve(self._chol, theta - self.mu)

    @jaxtyped(typechecker=beartype)
    def log_det_precision(self) -> Float[Array, ""]:
        """``log det Gamma_prior^{-1} = -2 sum log diag L``."""
        return -2.0 * jnp.sum(jnp.log(jnp.diag(self._chol[0])))

    @jaxtyped(typechecker=beartype)
    def prior_cov_matmul(self, B: Float[Array, "n_param k"]) -> Float[Array, "n_param k"]:
        return self.prior.Sigma @ B

    @jaxtyped(typechecker=beartype)
    def prior_precision_matmul(self, B: Float[Array, "n_param k"]) -> Float[Array, "n_param k"]:
        """``Gamma_prior^{-1} @ B`` par ``cho_solve`` sur les k colonnes."""
        return jsp.linalg.cho_solve(self._chol, B)

    def sample(self, key: PRNGKeyArray, n_samples: int = 1) -> Float[Array, "n_samples n_param"]:
        z = jax.random.normal(key, (n_samples, self.mu.shape[0]))
        return self.mu + z @ self._L.T
