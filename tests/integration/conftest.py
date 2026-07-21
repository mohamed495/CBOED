"""Shared fixtures. The benchmark lives in `cboed.benchmarks`."""

import pytest  # type: ignore

from cboed.benchmarks import SIGMA_OBS_MATRIX, make_model, make_prior


@pytest.fixture(scope="session")
def bench():
    """`(make_model, prior, Sigma_obs)` -- the frozen benchmark."""
    return make_model, make_prior(), SIGMA_OBS_MATRIX
