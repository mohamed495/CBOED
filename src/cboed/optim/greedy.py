"""Plain greedy design optimization -- the oracle for :mod:`cboed.optim.greedy_schur`."""

import jax.numpy as jnp

from cboed.optim.base import Optimizer, Result


class GreedyOptimizer(Optimizer):
    """Select a design by plain greedy maximization of a criterion.

    At each step, evaluates the criterion for every remaining candidate
    appended to the current selection and keeps the best one.

    Notes
    -----
    Treats the criterion as a black box -- no incremental structure is
    exploited, unlike :func:`cboed.optim.greedy_schur.greedy_schur`. Cost is
    ``n_candidates * n_sensors`` criterion evaluations. This class is the
    reference implementation against which the accelerated Schur-based
    greedy is validated.
    """

    def run(self, theta, n_sensors, n_candidates) -> Result:
        """Run the plain greedy selection.

        Parameters
        ----------
        theta
            Parameter value(s) at which the criterion is evaluated.
        n_sensors : int
            Number of sensors to select (budget).
        n_candidates : int
            Total number of candidate sensor locations to choose from.

        Returns
        -------
        Result
            `design`: selected indices in the order added.
            `scores`: best criterion score achieved after each addition.
        """
        selected: list[int] = []
        scores: list[float] = []

        for _ in range(n_sensors):
            best_score = -jnp.inf
            best_idx = None

            for i in range(n_candidates):
                if i in selected:
                    continue
                trial = jnp.array([*selected, i])
                score = self.criterion.evaluate(theta, trial)
                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(best_idx)
            scores.append(float(best_score))

        return Result(design=jnp.array(selected), scores=scores)
