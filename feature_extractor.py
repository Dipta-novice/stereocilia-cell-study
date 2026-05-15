"""
Functional 2D feature extraction helpers for VASCilia flat TIFF/PNG data.

The class-based implementation in feature_classes.py is the primary pipeline.
This module keeps a small function API for notebooks and scripts.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from skimage import feature, measure


PIXEL_SIZE_UM = (0.043, 0.043)  # (dy, dx)


def _validate_2d(image: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None) -> None:
    if image is not None and image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    if mask is not None and mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    if image is not None and mask is not None and image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}")


def extract_count_features(mask: np.ndarray) -> Dict[str, int]:
    """Count labeled bundles in a flat 2D mask."""
    _validate_2d(mask=mask)
    labels = np.unique(mask)
    return {"n_bundles_total": int(np.sum(labels > 0))}


def extract_morphology_2d(
    mask: np.ndarray,
    pixel_size: Tuple[float, float] = PIXEL_SIZE_UM,
    bundle_label: Optional[int] = None,
) -> Dict[str, float]:
    """Extract 2D shape features for one labeled bundle."""
    _validate_2d(mask=mask)
    region_mask = (mask == bundle_label) if bundle_label is not None else (mask > 0)
    props = measure.regionprops(region_mask.astype(np.uint8))
    if not props:
        return {}

    dy, dx = pixel_size
    region = props[0]
    minr, minc, maxr, maxc = region.bbox
    width_um = (maxc - minc) * dx
    height_um = (maxr - minr) * dy
    area_px = float(region.area)
    perimeter_px = float(region.perimeter)
    equiv_diameter_px = (
        region.equivalent_diameter_area
        if hasattr(region, "equivalent_diameter_area")
        else region.equivalent_diameter
    )

    return {
        "area_px": area_px,
        "area_um2": float(area_px * dy * dx),
        "perimeter_px": perimeter_px,
        "perimeter_um": float(perimeter_px * ((dy + dx) / 2.0)),
        "extent_x_um": float(width_um),
        "extent_y_um": float(height_um),
        "aspect_ratio": float(width_um / (height_um + 1e-8)),
        "solidity": float(region.solidity),
        "eccentricity": float(region.eccentricity),
        "extent": float(region.extent),
        "major_axis_um": float(region.axis_major_length * dx),
        "minor_axis_um": float(region.axis_minor_length * dy),
        "equiv_diameter_um": float(equiv_diameter_px * np.sqrt(dx * dy)),
        "centroid_y_um": float(region.centroid[0] * dy),
        "centroid_x_um": float(region.centroid[1] * dx),
    }


def extract_intensity_features(
    image: np.ndarray,
    mask: np.ndarray,
    bundle_label: Optional[int] = None,
) -> Dict[str, float]:
    """Extract intensity statistics from a flat image inside a label mask."""
    _validate_2d(image=image, mask=mask)
    region_mask = (mask == bundle_label) if bundle_label is not None else (mask > 0)
    intensities = image[region_mask].astype(np.float32)
    if intensities.size == 0:
        return {}

    mean = float(np.mean(intensities))
    std = float(np.std(intensities))
    return {
        "intensity_mean": mean,
        "intensity_max": float(np.max(intensities)),
        "intensity_min": float(np.min(intensities)),
        "intensity_std": std,
        "intensity_cv": float(std / (mean + 1e-8)),
        "intensity_total": float(np.sum(intensities)),
    }


def extract_glcm_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    distances: List[int] = [1, 2, 5],
    angles: List[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
) -> Dict[str, float]:
    """Extract GLCM texture features directly from a 2D image."""
    _validate_2d(image=image, mask=mask)
    roi = image.astype(np.float32)
    if mask is not None:
        roi = np.where(mask > 0, roi, 0)

    mn, mx = float(roi.min()), float(roi.max())
    if mx - mn < 1e-8:
        return {}
    image_uint8 = ((roi - mn) / (mx - mn) * 255).astype(np.uint8)

    glcm = feature.graycomatrix(
        image_uint8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    return {
        "glcm_contrast": float(np.mean(feature.graycoprops(glcm, "contrast"))),
        "glcm_energy": float(np.mean(feature.graycoprops(glcm, "energy"))),
        "glcm_homogeneity": float(np.mean(feature.graycoprops(glcm, "homogeneity"))),
        "glcm_correlation": float(np.mean(feature.graycoprops(glcm, "correlation"))),
    }


def extract_lbp_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    radius: int = 3,
    n_points: int = 24,
) -> Dict[str, float]:
    """Extract LBP texture features directly from a 2D image."""
    _validate_2d(image=image, mask=mask)
    img = image.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return {}
    image_u8 = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(image_u8, n_points, radius, method="uniform")
    values = lbp[mask > 0] if mask is not None else lbp.ravel()
    if values.size == 0:
        return {}

    n_bins = n_points + 2
    hist, _ = np.histogram(values, bins=n_bins, range=(0, n_bins), density=True)
    return {
        "lbp_mean": float(np.mean(values)),
        "lbp_std": float(np.std(values)),
        "lbp_entropy": float(stats.entropy(hist + 1e-10)),
    }


def extract_pcp_orientation(
    mask: np.ndarray,
    bundle_label: Optional[int] = None,
) -> Dict[str, float]:
    """Extract an in-plane XY bundle heading in the 0-180 degree range."""
    _validate_2d(mask=mask)
    region_mask = (mask == bundle_label) if bundle_label is not None else (mask > 0)
    props = measure.regionprops(region_mask.astype(np.uint8))
    if not props:
        return {}

    region = props[0]
    angle_deg = (90.0 - np.degrees(region.orientation)) % 180.0
    return {
        "pcp_angle_deg": float(angle_deg),
        "pcp_eccentricity": float(region.eccentricity),
    }


def extract_all_bundle_features(
    image: np.ndarray,
    mask: np.ndarray,
    pixel_size: Tuple[float, float] = PIXEL_SIZE_UM,
) -> pd.DataFrame:
    """Extract 2D morphology, intensity, and PCP features for all bundles."""
    _validate_2d(image=image, mask=mask)
    bundle_labels = np.unique(mask)
    bundle_labels = bundle_labels[bundle_labels > 0]

    rows = []
    for label in bundle_labels:
        row = {"bundle_id": int(label)}
        row.update(extract_morphology_2d(mask, pixel_size, bundle_label=int(label)))
        row.update(extract_intensity_features(image, mask, bundle_label=int(label)))
        row.update(extract_pcp_orientation(mask, bundle_label=int(label)))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_stack_level_texture(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extract whole-image 2D texture features inside the nonzero mask region."""
    features = {}
    features.update(extract_glcm_features(image, mask))
    features.update(extract_lbp_features(image, mask))
    return features


if __name__ == "__main__":
    print("Testing feature_extractor.py on synthetic 2D data...")
    test_image = np.random.rand(64, 64).astype(np.float32)
    test_mask = np.zeros((64, 64), dtype=np.int32)
    test_mask[10:20, 10:20] = 1
    test_mask[30:45, 30:45] = 2
    df_bundles = extract_all_bundle_features(test_image, test_mask)
    print(df_bundles.head())
    print("feature_extractor.py tests passed")
