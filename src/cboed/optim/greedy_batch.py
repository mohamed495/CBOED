# optim/greedy_batch.py
import jax.numpy as jnp

from cboed.optim.base import Optimizer, Result


class GreedyBatchReopt(Optimizer):
    """Greedy with full reoptimization at each step.

    On each addition, re-examines the whole design: for each sensor already
    chosen, tests whether it would be better to replace it with a free
    candidate. Cost ~k times plain greedy, corrects mistakes from earlier steps.
    """

    def run(self, theta, n_sensors, n_candidates) -> Result:
        """Run greedy selection with a swap-based reoptimization pass after each addition.

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
            `scores`: criterion value of the current design after each
            addition-and-reoptimization step.
        """
        selected = []
        scores = []

        for _ in range(n_sensors):
            # normal greedy step: add the best candidate
            best_i = self._best_addition(theta, selected, n_candidates)
            selected.append(best_i)

            # reoptimization: every position can be swapped
            selected = self._reoptimize(theta, selected, n_candidates)
            scores.append(float(self.criterion.evaluate(theta, jnp.array(selected))))

        return Result(design=jnp.array(selected), scores=scores)

    def _best_addition(self, theta, selected, n_candidates):
        """Return the free candidate index maximizing the criterion when appended to `selected`."""
        best_score, best_i = -jnp.inf, None
        for i in range(n_candidates):
            if i in selected:
                continue
            score = self.criterion.evaluate(theta, jnp.array([*selected, i]))
            if score > best_score:
                best_score, best_i = score, i
        return best_i

    def _reoptimize(self, theta, selected, n_candidates):
        """Repeatedly swap sensors for better free candidates until none improve the score.

        Parameters
        ----------
        theta
            Parameter value(s) at which the criterion is evaluated.
        selected : list of int
            Current selection, including the sensor just added.
        n_candidates : int
            Total number of candidate sensor locations to choose from.

        Returns
        -------
        list of int
            The selection after repeated swap passes, at a local fixed
            point: no single-sensor swap improves the criterion further.
        """
        improved = True
        while improved:
            improved = False
            for pos in range(len(selected)):
                current_score = self.criterion.evaluate(theta, jnp.array(selected))
                for i in range(n_candidates):
                    if i in selected:
                        continue
                    trial = selected.copy()
                    trial[pos] = i
                    score = self.criterion.evaluate(theta, jnp.array(trial))
                    if score > current_score:
                        selected = trial
                        current_score = score
                        improved = True
        return selected
