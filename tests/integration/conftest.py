"""Fixtures partagees. Le banc vit dans `cboed.benchmarks`."""

import pytest  # type: ignore

from cboed.benchmarks import SIGMA_OBS_MATRIX, make_model, make_prior


@pytest.fixture(scope="session")
def bench():
    """`(make_model, prior, Sigma_obs)` -- le banc figé."""
    return make_model, make_prior(), SIGMA_OBS_MATRIX
