from abc import ABC, abstractmethod


class Likelihood(ABC):
    """p(y | theta, xi). Owns the observation operator and the noise model.

    The design xi enters here and nowhere else: it selects what is observed,
    leaving the prior and the forward dynamics untouched.
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    # @abstractmethod
    # def __call__(self, theta, xi=None):
    #     """Predictive mean of y given (theta, xi)."""
    #     ...

    @abstractmethod
    def jacobian(self, theta, xi=None):
        """d(mean)/dtheta at (theta, xi), as a matrix-free operator."""
        ...

    @abstractmethod
    def log_likelihood(self, y, theta, xi=None):
        """log p(y | theta, xi)."""
        ...

    # @abstractmethod
    # def sample(self, key, theta, xi=None, n_samples=1):
    #     """Draw y ~ p(. | theta, xi). Requires an explicit PRNG key."""
    #     ...
