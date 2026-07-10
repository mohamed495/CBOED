from cboed.inference.base import (
    InferenceModel,
)


class linearModel(InferenceModel):
    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)

    @property
    def prior(self):
        return self._hyperparameters["prior"]

    @property
    def likelihood(self):
        return self._hyperparameters["likelihood"]
