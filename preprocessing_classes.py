"""
================================================================================
VASCilia 2D Preprocessing Classes
================================================================================
All techniques sourced from the Python for Microscopists repository:
  - IntensityNormalizer   → percentile clipping           (files 022, 096)
  - HistogramEnhancer     → global HE and CLAHE            (files 027, 113)
  - ImageSharpener        → unsharp mask + Wiener          (files 020, 102)
  - Denoiser              → 5-method denoising suite       (files 096–100)
  - BackgroundSubtractor  → rolling ball / Gaussian approx (file 117)
  - FrequencyFilter       → FFT low/high/band-pass         (files 105, 106)
  - QualityAssessor       → PSNR, SSIM, Laplacian sharp.  (files 114, 123, 124)
  - PreprocessingPipeline → orchestrator
================================================================================
All operations are strictly 2D.  No Z-stack, no voxel Z-axis, no 3D parameters.
"""

from typing import Dict, Optional, Any, Tuple

import numpy as np
import scipy.stats as st
from scipy import ndimage, fftpack
from skimage import exposure, filters, restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ──────────────────────────────────────────────────────────────────────────────
# 1. Intensity Normalizer
# ──────────────────────────────────────────────────────────────────────────────
class IntensityNormalizer:
    """
    Percentile-clip → linear scale to [0, 1] float32.

    Formula (file 022, 096):
        I_norm = clip(I - p_lo, 0, p_hi - p_lo) / (p_hi - p_lo)
    """

    def __init__(self, p_lo: float = 2.0, p_hi: float = 98.0):
        self.p_lo = p_lo
        self.p_hi = p_hi

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")
        lo = np.percentile(image, self.p_lo)
        hi = np.percentile(image, self.p_hi)
        return np.clip(
            (image.astype(np.float32) - lo) / max(hi - lo, 1e-8), 0.0, 1.0
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Histogram Enhancer
# ──────────────────────────────────────────────────────────────────────────────
class HistogramEnhancer:
    """
    Contrast enhancement via histogram operations (files 027, 113).

    Methods:
        'clahe'     — Contrast-Limited Adaptive Histogram Equalization.
                      clip_limit clips over-amplification; kernel_size sets tile size.
        'global_he' — Global histogram equalization (stretches full histogram).
    """

    METHODS = ("clahe", "global_he")

    def __init__(self, method: str = "clahe", clip_limit: float = 0.02,
                 kernel_size: int = 64):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")
        img = image.astype(np.float32)
        if img.max() > 1.0 or img.min() < 0.0:
            img = (img - img.min()) / max(img.max() - img.min(), 1e-8)
        if self.method == "clahe":
            return exposure.equalize_adapthist(
                img, clip_limit=self.clip_limit, kernel_size=self.kernel_size
            ).astype(np.float32)
        return exposure.equalize_hist(img).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Image Sharpener
# ──────────────────────────────────────────────────────────────────────────────
class ImageSharpener:
    """
    Image sharpening (files 020, 102).

    Methods:
        'unsharp_mask' — I_sharp = I + amount * (I - G_sigma * I)
        'wiener'       — Unsupervised Wiener deconvolution with Gaussian PSF
    """

    METHODS = ("unsharp_mask", "wiener")

    def __init__(self, method: str = "unsharp_mask", radius: float = 2.0,
                 amount: float = 1.5, psf_ksize: int = 9, psf_sigma: float = 1.5):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self.radius = radius
        self.amount = amount
        self.psf_ksize = psf_ksize
        self.psf_sigma = psf_sigma

    def _gaussian_psf(self) -> np.ndarray:
        """Build a Gaussian PSF kernel (from file 020)."""
        lim = self.psf_ksize // 2 + (self.psf_ksize % 2) / 2
        x = np.linspace(-lim, lim, self.psf_ksize + 1)
        k1d = np.diff(st.norm.cdf(x, scale=self.psf_sigma))
        k2d = np.outer(k1d, k1d)
        return (k2d / k2d.sum()).astype(np.float64)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")
        img = image.astype(np.float32)
        if self.method == "unsharp_mask":
            return filters.unsharp_mask(img, radius=self.radius,
                                        amount=self.amount).astype(np.float32)
        psf = self._gaussian_psf()
        result, _ = restoration.unsupervised_wiener(img.astype(np.float64), psf)
        return np.clip(result, 0.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Denoiser — 5 CPU methods
# ──────────────────────────────────────────────────────────────────────────────
class Denoiser:
    """
    Five-method CPU denoiser (files 022, 096–100).

    Methods:
        'gaussian'     — scipy.ndimage.gaussian_filter         (file 096)
        'median'       — scipy.ndimage.median_filter           (file 097)
        'bilateral'    — skimage denoise_bilateral              (file 098)
        'nlm'          — skimage denoise_nl_means (fast)       (file 099)
        'tv_chambolle' — skimage denoise_tv_chambolle          (file 100)

    No GPU required. No BM3D (heavy dependency).
    """

    METHODS = ("gaussian", "median", "bilateral", "nlm", "tv_chambolle")

    def __init__(self,
                 method: str = "nlm",
                 sigma: float = 1.0,
                 nlm_h_factor: float = 1.15,
                 nlm_patch_size: int = 5,
                 nlm_patch_distance: int = 3,
                 bilateral_sigma_color: float = 0.05,
                 bilateral_sigma_spatial: float = 2.0,
                 tv_weight: float = 0.05):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self.sigma = sigma
        self.nlm_h_factor = nlm_h_factor
        self.nlm_patch_size = nlm_patch_size
        self.nlm_patch_distance = nlm_patch_distance
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_spatial = bilateral_sigma_spatial
        self.tv_weight = tv_weight

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"Denoiser expects 2D image, got {image.shape}")
        img = image.astype(np.float32)

        if self.method == "gaussian":
            return ndimage.gaussian_filter(img, sigma=self.sigma)

        if self.method == "median":
            ksize = max(3, int(self.sigma) * 2 + 1)
            return ndimage.median_filter(img, size=ksize).astype(np.float32)

        if self.method == "bilateral":
            return restoration.denoise_bilateral(
                img,
                sigma_color=self.bilateral_sigma_color,
                sigma_spatial=self.bilateral_sigma_spatial,
            ).astype(np.float32)

        if self.method == "nlm":
            sigma_est = float(restoration.estimate_sigma(img))
            h = max(sigma_est * self.nlm_h_factor, 1e-6)
            return restoration.denoise_nl_means(
                img, h=h, fast_mode=True,
                patch_size=self.nlm_patch_size,
                patch_distance=self.nlm_patch_distance,
            ).astype(np.float32)

        if self.method == "tv_chambolle":
            return restoration.denoise_tv_chambolle(
                img, weight=self.tv_weight
            ).astype(np.float32)

        return img


# ──────────────────────────────────────────────────────────────────────────────
# 5. Background Subtractor
# ──────────────────────────────────────────────────────────────────────────────
class BackgroundSubtractor:
    """
    Uneven illumination correction (file 117).

    Primary:  cv2_rolling_ball.subtract_background_rolling_ball
    Fallback: subtract large-sigma Gaussian blur as estimated background.
    """

    def __init__(self, radius: int = 100, gaussian_sigma: float = 50.0):
        self.radius = radius
        self.gaussian_sigma = gaussian_sigma

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (corrected, background), both float32 in [0, 1]."""
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")
        img = np.clip(image.astype(np.float32), 0.0, 1.0)

        try:
            from cv2_rolling_ball import subtract_background_rolling_ball
            img_u8 = (img * 255).astype(np.uint8)
            corr_u8, bg_u8 = subtract_background_rolling_ball(
                img_u8, radius=self.radius,
                light_background=False, use_paraboloid=False, do_presmooth=True,
            )
            return (corr_u8.astype(np.float32) / 255.0,
                    bg_u8.astype(np.float32) / 255.0)
        except ImportError:
            bg = ndimage.gaussian_filter(img, sigma=self.gaussian_sigma)
            corrected = np.clip(img - bg + 0.5, 0.0, 1.0)
            return corrected.astype(np.float32), bg.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Frequency Filter
# ──────────────────────────────────────────────────────────────────────────────
class FrequencyFilter:
    """
    FFT-based spatial frequency filtering (files 105, 106).

    Modes:
        'low_pass'  — pass frequencies inside circular radius (blurring)
        'high_pass' — suppress frequencies inside circular radius (edge enhance)
        'band_pass' — annular mask between radius_inner and radius_outer
    """

    MODES = ("low_pass", "high_pass", "band_pass")

    def __init__(self, mode: str = "low_pass",
                 radius: int = 30, radius_inner: int = 15):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.radius = radius
        self.radius_inner = radius_inner

    @staticmethod
    def _disk_mask(shape: tuple, radius: int) -> np.ndarray:
        h, w = shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        return (np.sqrt((Y - cy) ** 2 + (X - cx) ** 2) <= radius).astype(np.float32)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")
        img = image.astype(np.float32)
        F = np.fft.fftshift(np.fft.fft2(img))

        if self.mode == "low_pass":
            mask = self._disk_mask(img.shape, self.radius)
        elif self.mode == "high_pass":
            mask = 1.0 - self._disk_mask(img.shape, self.radius)
        else:  # band_pass
            mask = (self._disk_mask(img.shape, self.radius) -
                    self._disk_mask(img.shape, self.radius_inner))
            mask = np.clip(mask, 0.0, 1.0)

        result = np.abs(np.fft.ifft2(np.fft.ifftshift(F * mask))).astype(np.float32)
        lo, hi = result.min(), result.max()
        return (result - lo) / max(hi - lo, 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Quality Assessor
# ──────────────────────────────────────────────────────────────────────────────
class QualityAssessor:
    """
    Reference-based and no-reference image quality metrics (files 114, 123, 124).

    Metrics:
        psnr       — Peak Signal-to-Noise Ratio  (higher = less noise)
        ssim       — Structural Similarity Index  (1 = identical)
        sharpness  — Laplacian variance           (higher = sharper)
    """

    @staticmethod
    def psnr(reference: np.ndarray, img: np.ndarray) -> float:
        return float(peak_signal_noise_ratio(
            reference.astype(np.float32), img.astype(np.float32), data_range=1.0
        ))

    @staticmethod
    def ssim(reference: np.ndarray, img: np.ndarray) -> float:
        return float(structural_similarity(
            reference.astype(np.float32), img.astype(np.float32), data_range=1.0
        ))

    @staticmethod
    def sharpness(img: np.ndarray) -> float:
        return float(ndimage.laplace(img.astype(np.float64)).var())

    def evaluate(self, reference: np.ndarray,
                 variants: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compare multiple variants against a reference."""
        return {
            name: {
                "psnr": self.psnr(reference, v),
                "ssim": self.ssim(reference, v),
                "sharpness": self.sharpness(v),
            }
            for name, v in variants.items()
        }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────────────────────
class PreprocessingPipeline:
    """
    Orchestrates the full 2D preprocessing chain:
        normalize → [enhance] → [sharpen] → [denoise]

    All steps are optional except normalization.  Strictly 2D — no Z axis.

    Example:
        pipe = PreprocessingPipeline(
            enhance_method='clahe',
            denoise_method='nlm',
        )
        img_out, mask_out, meta = pipe.run(image_raw, mask)
    """

    def __init__(self,
                 p_lo: float = 2.0,
                 p_hi: float = 98.0,
                 enhance_method: Optional[str] = "clahe",
                 clahe_clip: float = 0.02,
                 clahe_tile: int = 64,
                 sharpen_method: Optional[str] = None,
                 sharpen_radius: float = 2.0,
                 sharpen_amount: float = 1.5,
                 denoise_method: Optional[str] = "gaussian",
                 denoise_sigma: float = 1.0,
                 nlm_h_factor: float = 1.15,
                 tv_weight: float = 0.05):

        self.normalizer = IntensityNormalizer(p_lo=p_lo, p_hi=p_hi)

        self.enhancer = (
            HistogramEnhancer(method=enhance_method,
                              clip_limit=clahe_clip, kernel_size=clahe_tile)
            if enhance_method else None
        )

        self.sharpener = (
            ImageSharpener(method=sharpen_method,
                           radius=sharpen_radius, amount=sharpen_amount)
            if sharpen_method else None
        )

        self.denoiser = (
            Denoiser(method=denoise_method,
                     sigma=denoise_sigma,
                     nlm_h_factor=nlm_h_factor,
                     tv_weight=tv_weight)
            if denoise_method else None
        )

    def run(self,
            image: np.ndarray,
            mask: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Run pipeline.  Returns (processed_image, mask, metadata_dict).
        mask is passed through unchanged.
        """
        if image.ndim != 2:
            raise ValueError(f"PreprocessingPipeline expects 2D image, got {image.shape}")
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"mask shape {mask.shape} != image shape {image.shape}")

        meta: Dict[str, Any] = {"input_shape": image.shape, "input_dtype": str(image.dtype)}

        img = self.normalizer.apply(image)
        meta["intensity_range_after_norm"] = (float(img.min()), float(img.max()))

        if self.enhancer is not None:
            img = self.enhancer.apply(img)
            meta["enhance_method"] = self.enhancer.method

        if self.sharpener is not None:
            img = self.sharpener.apply(img)
            meta["sharpen_method"] = self.sharpener.method

        if self.denoiser is not None:
            img = self.denoiser.apply(img)
            meta["denoise_method"] = self.denoiser.method

        meta["output_shape"] = img.shape
        return img.astype(np.float32), mask, meta
