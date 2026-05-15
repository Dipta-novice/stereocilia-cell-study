
"""
================================================================================
VASCilia Preprocessing Classes
================================================================================
Image enhancement and denoising for flat 2D TIFF images.

Architecture:
    IntensityNormalizer  - Percentile clipping → [0, 1]
    HistogramEqualizer   - Plain HE and CLAHE
    Denoiser             - Gaussian / Median / Bilateral
    PreprocessingPipeline - Orchestrator
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import ndimage
from skimage import exposure, restoration


class IntensityNormalizer:
    """
    Percentile-based clip + linear scale to [0, 1] float32.
    """

    def __init__(self, p_lo: float = 2.0, p_hi: float = 98.0):
        self.p_lo = p_lo
        self.p_hi = p_hi

    def apply(self, image: np.ndarray) -> np.ndarray:
        lo = np.percentile(image, self.p_lo)
        hi = np.percentile(image, self.p_hi)
        out = np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1e-8), 0, 1)
        return out


class HistogramEqualizer:
    """
    Histogram equalization for contrast enhancement.
    """

    def __init__(self, method: str = "clahe", clip_limit: float = 0.02,
                 kernel_size: int = 64):
        assert method in ("clahe", "global"), "method must be clahe or global"
        self.method = method
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if img.max() > 1.0 or img.min() < 0.0:
            img = (img - img.min()) / max(img.max() - img.min(), 1e-8)

        if self.method == "clahe":
            return exposure.equalize_adapthist(
                img, clip_limit=self.clip_limit, kernel_size=self.kernel_size
            )
        return exposure.equalize_hist(img)


class Denoiser:
    """
    Multi-method denoiser for 2D images.
    """

    SUPPORTED = ("gaussian", "median", "bilateral")

    def __init__(self, method: str = "gaussian",
                 sigma_xy: float = 1.0):
        assert method in self.SUPPORTED, f"method must be one of {self.SUPPORTED}"
        self.method = method
        self.sigma_xy = sigma_xy

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"Denoiser expects a 2D image, got shape {image.shape}")
        img = image.astype(np.float32)
        if self.method == "gaussian":
            return ndimage.gaussian_filter(img, sigma=self.sigma_xy)

        if self.method == "median":
            return ndimage.median_filter(img, size=3)

        if self.method == "bilateral":
            return restoration.denoise_bilateral(
                img, sigma_color=0.05, sigma_spatial=self.sigma_xy
            )

        return img


class PreprocessingPipeline:
    """
    Run normalization, optional CLAHE, and optional denoise on flat 2D images.
    """

    def __init__(self,
                 use_clahe: bool = False,
                 denoise_method: Optional[str] = "gaussian",
                 p_lo: float = 2.0, p_hi: float = 98.0,
                 sigma_xy: float = 1.0):
        self.normalizer = IntensityNormalizer(p_lo=p_lo, p_hi=p_hi)
        self.clahe = HistogramEqualizer("clahe") if use_clahe else None
        self.denoiser = (
            Denoiser(denoise_method, sigma_xy=sigma_xy)
            if denoise_method else None
        )
        self.last_metadata: Dict[str, Any] = {}

    def run(self, image: np.ndarray,
            mask: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        if image.ndim != 2:
            raise ValueError(f"PreprocessingPipeline expects a 2D image, got {image.shape}")
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}")

        md: Dict[str, Any] = {"original_shape": image.shape}

        normed = self.normalizer.apply(image)
        md["intensity_range"] = (float(normed.min()), float(normed.max()))

        if self.clahe is not None:
            normed = self.clahe.apply(normed)
            md["clahe_applied"] = True

        if self.denoiser is not None:
            normed = self.denoiser.apply(normed)
            md["denoise_method"] = self.denoiser.method

        md["output_shape"] = normed.shape
        self.last_metadata = md
        return normed, mask, md
