import matplotlib.pyplot as plt
import numpy as np
import pytest

from cboed.viz.fields import plot_energy_posterior_histograms


def test_energy_posterior_histogram_has_one_panel_per_design():
    figure = plot_energy_posterior_histograms(
        {
            "INC": np.linspace(1.0, 2.0, 20),
            "CONS": np.linspace(1.2, 2.2, 20),
        },
        theta_true=1.5,
        prior_theta_samples=np.linspace(0.5, 3.0, 20),
        bins=8,
    )

    try:
        assert len(figure.axes) == 2
        assert all(axis.get_xlabel() == r"$\theta = \|\eta\|^2$" for axis in figure.axes)
    finally:
        plt.close(figure)


def test_energy_posterior_histogram_rejects_nonfinite_samples():
    with pytest.raises(ValueError, match="finite"):
        plot_energy_posterior_histograms({"INC": np.array([1.0, np.nan])}, theta_true=1.0)
