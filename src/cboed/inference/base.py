from abc import ABC, abstractmethod


class InferenceModel(ABC):
    """Bayesian inference: a prior and a likelihood.

    Subclasses decide *how* the expected information gain is estimated
    (closed form, Laplace approximation, nested Monte Carlo...).
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @abstractmethod
    def posterior(self, y, xi):
        """Posterior p(theta | y, xi), in whatever form the
        strategy provides.
        """
        ...

    @abstractmethod
    def _mu(self, theta, xi=None):
        """Posterior p(theta | y, xi), in whatever form the
        strategy provides.
        """
        ...

    @abstractmethod
    def _cov(self, theta, xi=None):
        """Posterior p(theta | y, xi), in whatever form the
        strategy provides.
        """
        ...

    @abstractmethod
    def expected_information_gain(self, xi=None):
        """EIG of design xi."""
        ...
