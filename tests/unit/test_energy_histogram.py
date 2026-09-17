import matplotlib.pyplot as plt
import numpy as np
import pytest

from cboed.viz.fields import plot_energy_posterior_histograms


def test_energy_posterior_histogram_overlays_designs():
    figure = plot_energy_posterior_histograms(
        {
            "INC": np.linspace(1.0, 2.0, 20),
            "CONS": np.linspace(1.2, 2.2, 20),
        },
        theta_true=1.5,
        prior_theta_samples=np.linspace(0.5, 3.0, 20),
        bins=8,
        x_max=2.5,
    )

    try:
        assert len(figure.axes) == 1
        assert all(axis.get_xlabel() == r"$\theta = \|\eta\|^2$" for axis in figure.axes)
        assert figure.axes[0].get_xlim() == (0.0, 2.5)
        legend_labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert legend_labels == ["prior", "INC posterior", "CONS posterior", r"$\theta_{\rm true}$"]
    finally:
        plt.close(figure)


def test_energy_posterior_histogram_rejects_nonfinite_samples():
    with pytest.raises(ValueError, match="finite"):
        plot_energy_posterior_histograms({"INC": np.array([1.0, np.nan])}, theta_true=1.0)
