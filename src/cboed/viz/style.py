r"""Set up the shared plotting style: palette, colormaps, rcParams, and small helpers.

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
    """Apply the shared rcParams (``RC``) to matplotlib.

    Notes
    -----
    Call once per script, before any figure is created.
    """
    plt.rcParams.update(RC)


def save(fig, path: str | Path) -> Path:
    """Save a figure to disk and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Destination path. Parent directories are created if missing.

    Returns
    -------
    path : Path
        The destination path, as a `Path`.

    Notes
    -----
    The figure is closed explicitly after saving: otherwise matplotlib keeps
    every figure in memory, and a sweep over many configurations produces
    dozens of them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def symmetric_limits(*arrays) -> tuple[float, float]:
    """Compute symmetric color limits ``(-v, v)`` with ``v = max|.|``.

    Parameters
    ----------
    *arrays : array_like
        One or more arrays; the limits use the maximum absolute value across
        all of them.

    Returns
    -------
    vmin, vmax : float
        ``(-v, v)``, suitable for a zero-centered diverging colormap.

    Notes
    -----
    Without this, a diverging colormap such as ``RdBu_r`` places white at the
    mean of the data rather than at zero: a difference that is everywhere
    positive would then appear to change sign.
    """
    v = max(float(abs(a).max()) for a in arrays)
    return -v, v
