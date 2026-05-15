"""
Functional 2D preprocessing helpers for VASCilia flat images.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import exposure, restoration


def normalize_image_2d(
    image: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Normalize a 2D image to [0, 1] using percentile clipping."""
    if image.ndim != 2:
        raise ValueError(f"normalize_image_2d expects a 2D image, got {image.shape}")
    p_low = np.percentile(image, lower_percentile)
    p_high = np.percentile(image, upper_percentile)
    clipped = np.clip(image.astype(np.float32), p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low + 1e-8)).astype(np.float32)


def smooth_2d(
    image: np.ndarray,
    method: str = "gaussian",
    sigma_xy: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian, median, or bilateral filtering to a 2D image."""
    if image.ndim != 2:
        raise ValueError(f"smooth_2d expects a 2D image, got {image.shape}")
    if method == "gaussian":
        return ndimage.gaussian_filter(image, sigma=sigma_xy)
    if method == "median":
        return ndimage.median_filter(image, size=3)
    if method == "bilateral":
        return restoration.denoise_bilateral(
            image.astype(np.float32),
            sigma_color=0.05,
            sigma_spatial=sigma_xy,
        )
    raise ValueError("method must be one of: gaussian, median, bilateral")


def equalize_image_2d(image: np.ndarray, clip_limit: float = 0.02) -> np.ndarray:
    """Apply CLAHE contrast enhancement to a normalized 2D image."""
    if image.ndim != 2:
        raise ValueError(f"equalize_image_2d expects a 2D image, got {image.shape}")
    img = image.astype(np.float32)
    if img.max() > 1.0 or img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return exposure.equalize_adapthist(img, clip_limit=clip_limit)


def preprocess_image_2d(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalize: bool = True,
    use_clahe: bool = False,
    denoise_method: Optional[str] = "gaussian",
    sigma_xy: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    """Run the complete 2D preprocessing path and return image, mask, metadata."""
    if image.ndim != 2:
        raise ValueError(f"preprocess_image_2d expects a 2D image, got {image.shape}")
    if mask is not None and mask.shape != image.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}")

    metadata: Dict[str, object] = {"original_shape": image.shape}
    out = normalize_image_2d(image) if normalize else image.astype(np.float32)

    if use_clahe:
        out = equalize_image_2d(out)
        metadata["clahe_applied"] = True

    if denoise_method:
        out = smooth_2d(out, method=denoise_method, sigma_xy=sigma_xy)
        metadata["denoise_method"] = denoise_method

    metadata["processed_shape"] = out.shape
    metadata["intensity_min"] = float(out.min())
    metadata["intensity_max"] = float(out.max())
    metadata["intensity_mean"] = float(out.mean())
    return out.astype(np.float32, copy=False), mask, metadata


if __name__ == "__main__":
    print("Testing preprocess.py on synthetic 2D data...")
    test_image = np.random.rand(128, 128).astype(np.float32) * 1000
    processed, _, metadata = preprocess_image_2d(test_image)
    print(f"Original shape: {metadata['original_shape']}")
    print(f"Processed shape: {metadata['processed_shape']}")
    print("preprocess.py tests passed")
