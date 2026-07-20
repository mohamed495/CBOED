# optim/greedy_batch.py
import jax.numpy as jnp

from cboed.optim.base import Optimizer, Result


class GreedyBatchReopt(Optimizer):
    """Greedy avec réoptimisation complète à chaque étape.

    À chaque ajout, réexamine tout le design : pour chaque capteur déjà
    choisi, teste s'il vaut mieux le remplacer par un candidat libre.
    Coût ~k* le greedy simple, corrige les erreurs d'étapes antérieures.
    """

    def run(self, theta, n_sensors, n_candidates) -> Result:
        selected = []
        scores = []

        for _ in range(n_sensors):
            # étape greedy normale : ajouter le meilleur candidat
            best_i = self._best_addition(theta, selected, n_candidates)
            selected.append(best_i)

            # réoptimisation : chaque position peut être échangée
            selected = self._reoptimize(theta, selected, n_candidates)
            scores.append(float(self.criterion.evaluate(theta, jnp.array(selected))))

        return Result(design=jnp.array(selected), scores=scores)

    def _best_addition(self, theta, selected, n_candidates):
        best_score, best_i = -jnp.inf, None
        for i in range(n_candidates):
            if i in selected:
                continue
            score = self.criterion.evaluate(theta, jnp.array([*selected, i]))
            if score > best_score:
                best_score, best_i = score, i
        return best_i

    def _reoptimize(self, theta, selected, n_candidates):
        """Passe de swap : remplacer chaque capteur s'il existe mieux."""
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
