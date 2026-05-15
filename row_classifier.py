"""
================================================================================
VASCilia Hair Cell Row Classifier — IHC / OHC1 / OHC2 / OHC3
================================================================================
The organ of Corti has four parallel rows of hair cells along the cochlea:
    - 1 inner row  → IHC  (the ones that send signals to the brain)
    - 3 outer rows → OHC1 (closest to IHC), OHC2 (middle), OHC3 (farthest)

In the VASCilia 2D images, these rows are arranged along the **Y axis** after
proper rotation. This module:

  1. Estimates the principal axis of the bundle layout (PCA on centroids).
  2. Rotates the centroids so the row-axis is horizontal.
  3. Detects 4 clusters along the perpendicular axis (1-D K-Means).
  4. Labels them IHC / OHC1 / OHC2 / OHC3 by Y-order.

The IHC row is identified as the row that is:
   (a) numerically smallest (matches the metadata's `n_ihc` ~ 8-10), AND
   (b) most spatially separated from the others.
================================================================================
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _bundle_centroids(mask: np.ndarray,
                      voxel_size=(0.043, 0.043)) -> pd.DataFrame:
    """Return DataFrame with one row per bundle: bundle_id, cx, cy (microns)."""
    dy, dx = voxel_size
    labels = np.unique(mask)
    labels = labels[labels > 0]
    rows = []
    for L in labels:
        ys, xs = np.where(mask == L)
        if ys.size == 0:
            continue
        rows.append({
            "bundle_id": int(L),
            "cx": xs.mean() * dx,
            "cy": ys.mean() * dy,
            "n_pixels": int(ys.size),
        })
    return pd.DataFrame(rows)


def classify_rows(mask: np.ndarray,
                  voxel_size=(0.043, 0.043)) -> pd.DataFrame:
    """
    Cluster the bundles into 4 cochlear rows: IHC, OHC1, OHC2, OHC3.

    Returns a DataFrame with columns:
        bundle_id, cx, cy, row_label (string), row_index (0..3)
    """
    if mask.ndim != 2:
        raise ValueError(f"classify_rows expects a 2D mask, got shape {mask.shape}")

    df = _bundle_centroids(mask, voxel_size)
    if df.empty:
        return df

    # 2D footprint coords
    xy = df[["cx", "cy"]].values

    # Single-bundle / very small mask edge cases
    if len(xy) < 4:
        df["row_label"] = "UNK"
        df["row_index"] = -1
        return df

    # PCA → first component is "along the cochlea" (rows direction).
    # Second component is "across rows" — what we cluster along.
    pca = PCA(n_components=2)
    rotated = pca.fit_transform(xy)
    across = rotated[:, 1].reshape(-1, 1)  # perpendicular to row direction

    # K-Means with k=4 along the perpendicular axis
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(across)
    cluster_centers = km.cluster_centers_.flatten()
    # Order clusters by their across-axis position
    order = np.argsort(cluster_centers)
    cluster_rank = {c: r for r, c in enumerate(order)}

    df["row_index"] = [cluster_rank[c] for c in km.labels_]
    df["across"] = across.flatten()

    # Decide which row is IHC. The IHC row is typically the smallest cluster
    # AND the most separated from the others. We pick whichever end (top or
    # bottom in sorted order) has fewer members.
    counts = df["row_index"].value_counts().sort_index()
    end_top = counts.get(0, 0)
    end_bot = counts.get(3, 0)
    if end_top <= end_bot:
        ihc_idx = 0
        # 0=IHC, 1=OHC1, 2=OHC2, 3=OHC3
        name_map = {0: "IHC", 1: "OHC1", 2: "OHC2", 3: "OHC3"}
    else:
        ihc_idx = 3
        # 3=IHC, 2=OHC1, 1=OHC2, 0=OHC3
        name_map = {3: "IHC", 2: "OHC1", 1: "OHC2", 0: "OHC3"}

    df["row_label"] = df["row_index"].map(name_map)
    df["is_ihc"] = (df["row_label"] == "IHC")
    return df.drop(columns=["across"])


def counts_per_row(row_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row counts → tidy DataFrame for plotting."""
    counts = (row_df["row_label"].value_counts()
              .reindex(["IHC", "OHC1", "OHC2", "OHC3"], fill_value=0)
              .rename_axis("row").reset_index(name="count"))
    return counts


if __name__ == "__main__":
    # Synthetic test: 4 rows of bundles along Y
    rng = np.random.default_rng(0)
    mask = np.zeros((200, 400), dtype=np.int32)
    label = 1
    for row_y, n in [(40, 9), (80, 12), (120, 12), (160, 12)]:
        for k in range(n):
            x = 20 + k * 30 + rng.integers(-3, 3)
            mask[row_y-5:row_y+5, x-5:x+5] = label
            label += 1

    df = classify_rows(mask)
    print(df.head())
    print("\nCounts per row:")
    print(counts_per_row(df))
