from abc import ABC, abstractmethod


class InferenceModel(ABC):
    """Bayesian inference: a prior and a likelihood.

    Subclasses decide *how* the expected information gain is estimated
    (closed form, Laplace approximation, nested Monte Carlo...).
    """

    def __init__(self, prior, likelihood, **hyperparameters):
        self._hyperparameters = hyperparameters
        self.prior = prior  # from cboed.priors
        self.likelihood = likelihood

    @abstractmethod
    def posterior(self, y, xi):
        """Posterior p(theta | y, xi), in whatever form the strategy provides."""
        ...

    @abstractmethod
    def expected_information_gain(self, xi):
        """EIG of design xi."""
        ...
