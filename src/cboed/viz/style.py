r"""Style partagé.

Règle du module ``viz`` : les fonctions prennent des **tableaux** et rendent des
``Figure``. Aucun calcul, aucune lecture disque, aucun appel au modèle direct. Les
scripts calculent (avec cache) et appellent ``viz``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # avant pyplot : pas de serveur X dans l'env pixi

import matplotlib.pyplot as plt

# -- palette -------------------------------------------------------------
# Une couleur par objet, tenue dans toutes les figures.
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

#: Divergente et centrée en zéro -- pour les différences de matrices.
CMAP_DIFF = "RdBu_r"
#: Séquentielle -- pour les matrices SDP.
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
    """Applique ``RC``. À appeler une fois par script."""
    plt.rcParams.update(RC)


def save(fig, path: str | Path) -> Path:
    """Enregistre et ferme. Fermer explicitement : matplotlib garde sinon toutes les
    figures en mémoire, et un balayage en produit des dizaines.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def symmetric_limits(*arrays) -> tuple[float, float]:
    """``(-v, v)`` avec ``v = max|.|`` -- pour une colormap divergente centrée.

    Sans ça, ``RdBu_r`` place le blanc à la moyenne et non à zéro : une différence
    partout positive paraîtrait changer de signe.
    """
    v = max(float(abs(a).max()) for a in arrays)
    return -v, v
