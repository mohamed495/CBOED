import jax.numpy as jnp

from cboed.optim.base import Optimizer, Result


class GreedyOptimizer(Optimizer):
    def run(self, theta, n_sensors, n_candidates) -> Result:
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
