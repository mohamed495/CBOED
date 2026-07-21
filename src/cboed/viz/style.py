r"""Shared style.

``viz`` module rule: functions take **arrays** and return ``Figure`` objects. No
computation, no disk reads, no direct model calls. Scripts compute (with caching)
and call ``viz``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot: no X server in the pixi env

import matplotlib.pyplot as plt

# -- palette -------------------------------------------------------------
# One color per object, held consistent across all figures.
COLORS = {
    "prior": "#925625",
    "posterior": "#2b6cb0",
    "truth": "#c0392b",
    "sensors": "#16a085",
    "Sigma_Y": "#2b6cb0",
    "Sigma_signal": "#c0392b",
    "Sigma_Y_given_theta": "#8e44ad",
    "Sigma_noise": "#d68910",
    "incremental": "#2b6cb0",
    "conservative": "#d68910",
    "exact": "#2c3e50",
}

#: Diverging and zero-centered -- for matrix differences.
CMAP_DIFF = "RdBu_r"
#: Sequential -- for SPD matrices.
CMAP_PSD = "viridis"

RC = {
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "image.origin": "upper",
}


def use_style() -> None:
    """Apply ``RC``. Call once per script."""
    plt.rcParams.update(RC)


def save(fig, path: str | Path) -> Path:
    """Save and close. Close explicitly: otherwise matplotlib keeps every
    figure in memory, and a sweep produces dozens of them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def symmetric_limits(*arrays) -> tuple[float, float]:
    """``(-v, v)`` with ``v = max|.|`` -- for a zero-centered diverging colormap.

    Without this, ``RdBu_r`` places white at the mean rather than at zero: a
    difference that is everywhere positive would appear to change sign.
    """
    v = max(float(abs(a).max()) for a in arrays)
    return -v, v
