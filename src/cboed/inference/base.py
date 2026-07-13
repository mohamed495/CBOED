from abc import ABC, abstractmethod


class InferenceModel(ABC):
    """Bayesian inference: a prior and a likelihood.

    Subclasses decide *how* the expected information gain is estimated
    (closed form, Laplace approximation, nested Monte Carlo...).
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @abstractmethod
    def posterior(self, y, design):
        """Posterior p(theta | y, design), in whatever form the
        strategy provides.
        """
        ...

    @abstractmethod
    def _mu(self, theta, design=None):
        """Posterior p(theta | y, design), in whatever form the
        strategy provides.
        """
        ...

    @abstractmethod
    def _cov(self, theta, design=None):
        """Posterior p(theta | y, design), in whatever form the
        strategy provides.
        """
        ...
