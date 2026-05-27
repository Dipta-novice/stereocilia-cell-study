"""
================================================================================
VASCilia 2D Visualization Utilities
================================================================================
Simple 2D rendering helpers for flat TIFF images and mask overlays.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def plot_bundle_image_overlay(image: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              alpha: float = 0.9,
                              figsize: Tuple[float, float] = (10, 10),
                              show_ids: bool = True) -> plt.Figure:
    """Overlay colored bundle mask contours and id labels on a 2D image."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    if mask is not None and mask.shape != image.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap="gray")

    if mask is not None:
        labels = np.unique(mask)
        labels = labels[labels > 0]
        cmap = plt.get_cmap("tab20", max(len(labels), 1))
        for i, label_id in enumerate(labels):
            bundle = (mask == label_id)
            ys, xs = np.nonzero(bundle)
            if ys.size == 0:
                continue
            color = cmap(i % cmap.N)
            for contour in measure.find_contours(bundle.astype(float), 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color=color,
                        linewidth=1.4, alpha=alpha)
            if show_ids:
                ax.text(np.mean(xs), np.mean(ys), str(label_id), color="white",
                        fontsize=8, ha="center", va="center",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1,
                                  edgecolor="none"))

    ax.set_title("2D bundle contour overlay" if mask is not None else "Raw image")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_image_mask_overlay(image: np.ndarray,
                            mask: Optional[np.ndarray] = None,
                            alpha: float = 0.9,
                            figsize: Tuple[float, float] = (10, 10)) -> plt.Figure:
    """Backward-compatible wrapper for the 2D contour overlay."""
    return plot_bundle_image_overlay(image, mask, alpha=alpha, figsize=figsize)
