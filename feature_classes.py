"""
================================================================================
VASCilia 2D Feature Extraction Classes
================================================================================
All techniques sourced directly from the Python for Microscopists repository
and the 3D Organoid Analysis capstone project:

  - MorphologyExtractor    → regionprops in 2D             (files 032, 116)
  - IntensityExtractor     → per-bundle pixel statistics
  - EntropyExtractor       → local entropy texture         (files 020, 021)
  - GLCMTextureExtractor   → Gray-Level Co-occurrence      (file 200a)
  - LBPTextureExtractor    → Local Binary Pattern
  - GaborFeatureExtractor  → multi-angle filter bank       (file 061)
  - HOGFeatureExtractor    → Histogram of Oriented Grads   (file 020)
  - KeypointExtractor      → Harris + Shi-Tomasi counts    (file 029)
  - EdgeRidgeExtractor     → Sobel/Canny/Frangi/Meijering  (files 103, 104)
  - PCPOrientationExtractor→ in-plane XY bundle heading
  - FeaturePipeline        → runs all extractors, merges results

Public API for every extractor:
    extractor = SomeExtractor(voxel_size=(0.043, 0.043))
    df = extractor.extract(image_2d, mask_2d, stack_id="demo")

Strictly 2D — no Z-axis, no voxel Z spacing, no 3D parameters.
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage import feature, filters, measure, morphology


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────
class BaseFeatureExtractor(ABC):
    """
    Abstract base: enforce 2D, iterate over bundle labels, return DataFrame.
    """

    DEFAULT_VOXEL_SIZE = (0.043, 0.043)   # (dy_um, dx_um)

    def __init__(self, voxel_size: Optional[Tuple[float, float]] = None):
        self.voxel_size = voxel_size or self.DEFAULT_VOXEL_SIZE

    @abstractmethod
    def _extract_one(self, image: np.ndarray, mask: np.ndarray,
                     label_id: int) -> Optional[Dict[str, float]]:
        """Extract features for a single bundle label. Return None to skip."""

    def extract(self, image: np.ndarray, mask: np.ndarray,
                stack_id: str = "stack_0") -> pd.DataFrame:
        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects 2D arrays; "
                f"got image={image.shape}, mask={mask.shape}"
            )
        if image.shape != mask.shape:
            raise ValueError(f"image {image.shape} != mask {mask.shape}")

        labels = np.unique(mask)
        labels = labels[labels > 0]

        rows = []
        for lab in labels:
            feats = self._extract_one(image, mask, int(lab))
            if feats:
                feats["stack_id"] = stack_id
                feats["bundle_id"] = int(lab)
                rows.append(feats)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        id_cols = ["stack_id", "bundle_id"]
        return df[id_cols + [c for c in df.columns if c not in id_cols]]

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _crop(image: np.ndarray, mask: np.ndarray, label_id: int):
        """Bounding-box crop of one bundle. Returns (roi_img, roi_mask_bool)."""
        bm = mask == label_id
        ys, xs = np.where(bm)
        if ys.size == 0:
            return None, None
        minr, maxr = int(ys.min()), int(ys.max()) + 1
        minc, maxc = int(xs.min()), int(xs.max()) + 1
        return (image[minr:maxr, minc:maxc].astype(np.float32),
                bm[minr:maxr, minc:maxc])

    @staticmethod
    def _quantize(crop: np.ndarray, levels: int = 32) -> Optional[np.ndarray]:
        """Scale crop to [0, levels-1] uint8. Returns None if flat."""
        mn, mx = float(crop.min()), float(crop.max())
        if mx - mn < 1e-6:
            return None
        q = np.clip(((crop - mn) / (mx - mn) * (levels - 1)), 0, levels - 1)
        return q.astype(np.uint8)

    @staticmethod
    def _to_u8(crop: np.ndarray) -> Optional[np.ndarray]:
        """Scale crop to [0, 255] uint8. Returns None if flat."""
        mn, mx = float(crop.min()), float(crop.max())
        if mx - mn < 1e-8:
            return None
        return ((crop - mn) / (mx - mn) * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Morphology (files 032, 116)
# ──────────────────────────────────────────────────────────────────────────────
class MorphologyExtractor(BaseFeatureExtractor):
    """
    2D regionprops-based shape features (files 032, 116).

    Features: area, perimeter, solidity, eccentricity, aspect_ratio,
              major/minor axis lengths, equivalent diameter, centroid (µm).
    """

    def _extract_one(self, image, mask, label_id):
        bm = (mask == label_id).astype(np.uint8)
        if bm.sum() == 0:
            return None
        props = measure.regionprops(bm)
        if not props:
            return None
        rp = props[0]
        dy, dx = self.voxel_size
        minr, minc, maxr, maxc = rp.bbox
        equiv_d = (rp.equivalent_diameter_area
                   if hasattr(rp, "equivalent_diameter_area")
                   else rp.equivalent_diameter)
        return {
            "area_px":           float(rp.area),
            "area_um2":          float(rp.area * dx * dy),
            "perimeter_px":      float(rp.perimeter),
            "perimeter_um":      float(rp.perimeter * (dx + dy) / 2.0),
            "extent_x_um":       float((maxc - minc) * dx),
            "extent_y_um":       float((maxr - minr) * dy),
            "aspect_ratio":      float((maxc - minc) / max(maxr - minr, 1)),
            "solidity":          float(rp.solidity) if hasattr(rp, "solidity") else np.nan,
            "eccentricity":      float(rp.eccentricity),
            "extent":            float(rp.extent),
            "major_axis_um":     float(rp.axis_major_length * dx),
            "minor_axis_um":     float(rp.axis_minor_length * dy),
            "equiv_diameter_um": float(equiv_d * np.sqrt(dx * dy)),
            "centroid_y_um":     float(rp.centroid[0] * dy),
            "centroid_x_um":     float(rp.centroid[1] * dx),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Intensity
# ──────────────────────────────────────────────────────────────────────────────
class IntensityExtractor(BaseFeatureExtractor):
    """
    Per-bundle pixel intensity statistics.

    Features: mean, std, CV, skew, kurtosis, min, max, median, p95, total.
    """

    def _extract_one(self, image, mask, label_id):
        pix = image[mask == label_id].astype(np.float32)
        if pix.size == 0:
            return None
        mean = float(np.mean(pix))
        std  = float(np.std(pix))
        return {
            "intensity_mean":     mean,
            "intensity_std":      std,
            "intensity_cv":       std / (mean + 1e-8),
            "intensity_min":      float(np.min(pix)),
            "intensity_max":      float(np.max(pix)),
            "intensity_median":   float(np.median(pix)),
            "intensity_p95":      float(np.percentile(pix, 95)),
            "intensity_total":    float(np.sum(pix)),
            "intensity_skew":     float(stats.skew(pix))     if pix.size > 2 else 0.0,
            "intensity_kurtosis": float(stats.kurtosis(pix)) if pix.size > 3 else 0.0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Entropy (files 020, 021)
# ──────────────────────────────────────────────────────────────────────────────
class EntropyExtractor(BaseFeatureExtractor):
    """
    Shannon entropy of local texture inside each bundle (files 020, 021).

    The entropy image uses a disk neighbourhood — high entropy = complex texture
    (many cells), low entropy = uniform region (scratch / empty area).

    Features: entropy_mean, entropy_std, entropy_max (within bundle ROI).
    """

    def __init__(self, voxel_size=None, disk_radius: int = 3):
        super().__init__(voxel_size)
        self.disk_radius = disk_radius

    def _extract_one(self, image, mask, label_id):
        crop, crop_mask = self._crop(image, mask, label_id)
        if crop is None or crop.size < 16:
            return None
        u8 = self._to_u8(crop)
        if u8 is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ent = filters.rank.entropy(u8, morphology.disk(self.disk_radius))
        ent_in = ent[crop_mask].astype(np.float32)
        if ent_in.size == 0:
            return None
        return {
            "entropy_mean": float(np.mean(ent_in)),
            "entropy_std":  float(np.std(ent_in)),
            "entropy_max":  float(np.max(ent_in)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. GLCM Texture (file 200a)
# ──────────────────────────────────────────────────────────────────────────────
class GLCMTextureExtractor(BaseFeatureExtractor):
    """
    Gray-Level Co-occurrence Matrix texture features (file 200a).

    Computed at distances=[1,3,5] and angles=[0°,45°,90°,135°], then averaged.

    Features: contrast, energy, homogeneity, correlation, dissimilarity, ASM.

    Math:
        Contrast    = sum_{i,j} (i-j)^2 * G(i,j)
        Energy      = sum_{i,j} G(i,j)^2
        Homogeneity = sum_{i,j} G(i,j) / (1 + |i-j|)
        Correlation = sum_{i,j} (i-mu_i)(j-mu_j) G(i,j) / (sigma_i * sigma_j)
    """

    def __init__(self, voxel_size=None, distances=(1, 3, 5),
                 angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), levels: int = 32):
        super().__init__(voxel_size)
        self.distances = list(distances)
        self.angles    = list(angles)
        self.levels    = levels

    def _extract_one(self, image, mask, label_id):
        crop, crop_mask = self._crop(image, mask, label_id)
        if crop is None or crop.size < 16:
            return None
        roi = np.where(crop_mask, crop, 0.0)
        q = self._quantize(roi, levels=self.levels)
        if q is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glcm = feature.graycomatrix(
                q, distances=self.distances, angles=self.angles,
                levels=self.levels, symmetric=True, normed=True,
            )
        feats = {}
        for prop in ("contrast", "energy", "homogeneity",
                     "correlation", "dissimilarity", "ASM"):
            feats[f"glcm_{prop.lower()}"] = float(
                np.mean(feature.graycoprops(glcm, prop))
            )
        return feats


# ──────────────────────────────────────────────────────────────────────────────
# 5. LBP Texture
# ──────────────────────────────────────────────────────────────────────────────
class LBPTextureExtractor(BaseFeatureExtractor):
    """
    Local Binary Pattern histogram statistics.

    LBP_{P,R}(p) = sum_{k=0}^{P-1} s(g_k - g_c) * 2^k

    'uniform' mode counts 0-2 bit transitions → robust, sparse histogram.

    Features: lbp_mean, lbp_std, lbp_entropy, lbp_uniform_fraction.
    """

    def __init__(self, voxel_size=None, radius: int = 3, n_points: int = 24):
        super().__init__(voxel_size)
        self.radius   = radius
        self.n_points = n_points

    def _extract_one(self, image, mask, label_id):
        crop, crop_mask = self._crop(image, mask, label_id)
        if crop is None or crop.size < 32:
            return None
        u8 = self._to_u8(crop)
        if u8 is None:
            return None
        lbp = feature.local_binary_pattern(u8, P=self.n_points,
                                            R=self.radius, method="uniform")
        vals = lbp[crop_mask]
        if vals.size == 0:
            return None
        n_bins = self.n_points + 2
        hist, _ = np.histogram(vals, bins=n_bins, range=(0, n_bins), density=True)
        return {
            "lbp_mean":             float(np.mean(vals)),
            "lbp_std":              float(np.std(vals)),
            "lbp_entropy":          float(stats.entropy(hist + 1e-10)),
            "lbp_uniform_fraction": float(hist[:-1].sum()),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Gabor Filter Bank (file 061)
# ──────────────────────────────────────────────────────────────────────────────
class GaborFeatureExtractor(BaseFeatureExtractor):
    """
    Multi-orientation, multi-frequency Gabor filter bank (file 061).

    G(x,y) = exp(-(x'^2 + gamma^2 * y'^2) / (2*sigma^2)) * cos(2*pi*x'/lambda)
    where x' = x*cos(theta) + y*sin(theta)

    Sweeps theta in {0, 45, 90, 135} and frequency in {0.1, 0.3, 0.5}.

    Features: mean + std response per (theta, freq) channel,
              gabor_max_response, gabor_anisotropy.
    """

    def __init__(self, voxel_size=None,
                 thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                 frequencies=(0.1, 0.3, 0.5)):
        super().__init__(voxel_size)
        self.thetas      = list(thetas)
        self.frequencies = list(frequencies)

    def _extract_one(self, image, mask, label_id):
        crop, crop_mask = self._crop(image, mask, label_id)
        if crop is None or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None
        feats = {}
        responses = []
        for theta in self.thetas:
            for freq in self.frequencies:
                real, _ = filters.gabor(crop, frequency=freq, theta=theta)
                vals = real[crop_mask]
                if vals.size == 0:
                    continue
                key = f"gabor_t{int(np.degrees(theta)):03d}_f{freq:.1f}"
                feats[f"{key}_mean"] = float(np.mean(np.abs(vals)))
                feats[f"{key}_std"]  = float(np.std(vals))
                responses.append(float(np.mean(np.abs(vals))))
        if responses:
            rmax, rmin = max(responses), min(responses)
            feats["gabor_max_response"] = rmax
            feats["gabor_anisotropy"]   = (rmax - rmin) / (rmax + rmin + 1e-8)
        return feats or None


# ──────────────────────────────────────────────────────────────────────────────
# 7. HOG (file 020)
# ──────────────────────────────────────────────────────────────────────────────
class HOGFeatureExtractor(BaseFeatureExtractor):
    """
    Histogram of Oriented Gradients summary statistics (file 020).

    1. Compute gradient (magnitude, angle) at every pixel
    2. Tile into cells; build weighted orientation histogram per cell
    3. L2-normalise across overlapping blocks
    4. Summarise vector → mean, std, max, energy

    Features: hog_mean, hog_std, hog_max, hog_energy, hog_n_features.
    """

    def __init__(self, voxel_size=None,
                 pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 orientations: int = 9):
        super().__init__(voxel_size)
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations    = orientations

    def _extract_one(self, image, mask, label_id):
        crop, _ = self._crop(image, mask, label_id)
        if crop is None:
            return None
        ppc = self.pixels_per_cell
        if crop.shape[0] < ppc[0] * 2 or crop.shape[1] < ppc[1] * 2:
            return None
        try:
            hog_vec = feature.hog(
                crop, orientations=self.orientations,
                pixels_per_cell=ppc, cells_per_block=self.cells_per_block,
                feature_vector=True,
            )
        except Exception:
            return None
        if hog_vec.size == 0:
            return None
        return {
            "hog_mean":       float(np.mean(hog_vec)),
            "hog_std":        float(np.std(hog_vec)),
            "hog_max":        float(np.max(hog_vec)),
            "hog_energy":     float(np.sum(hog_vec ** 2)),
            "hog_n_features": int(hog_vec.size),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Keypoint Extractor (file 029)
# ──────────────────────────────────────────────────────────────────────────────
class KeypointExtractor(BaseFeatureExtractor):
    """
    Harris and Shi-Tomasi corner counts inside each bundle (file 029).

    Harris response:
        R = det(M) - k * trace(M)^2
        M = [[sum Ix^2, sum Ix*Iy], [sum Ix*Iy, sum Iy^2]]

    Corner counts proxy for the number of resolvable stereocilia tips.

    Features: n_harris_corners, n_shitomasi_corners, corner_density_per_px.
    """

    def _extract_one(self, image, mask, label_id):
        crop, crop_mask = self._crop(image, mask, label_id)
        if crop is None or crop.size < 25:
            return None
        try:
            h_resp = feature.corner_harris(crop, k=0.04, sigma=1.0)
            h_pts  = feature.corner_peaks(h_resp, min_distance=3, threshold_rel=0.02)
            n_h    = int(sum(1 for r, c in h_pts if crop_mask[r, c]))
        except Exception:
            n_h = 0
        try:
            s_resp = feature.corner_shi_tomasi(crop, sigma=1.0)
            s_pts  = feature.corner_peaks(s_resp, min_distance=3, threshold_rel=0.02)
            n_s    = int(sum(1 for r, c in s_pts if crop_mask[r, c]))
        except Exception:
            n_s = 0
        area = int(crop_mask.sum())
        return {
            "n_harris_corners":     n_h,
            "n_shitomasi_corners":  n_s,
            "corner_density_per_px": float(n_h / max(area, 1)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 9. Edge + Ridge Extractor (files 103, 104)
# ──────────────────────────────────────────────────────────────────────────────
class EdgeRidgeExtractor(BaseFeatureExtractor):
    """
    Edge detection and ridge/vessel filter statistics (files 103, 104).

    Edge filters (file 103):
        Sobel   — gradient magnitude |∇I|
        Canny   — multi-step: noise reduction → gradient → NMS → hysteresis

    Ridge / tubular-structure filters (file 104):
        Frangi   — vesselness via Hessian eigenvalues (λ1≈0, |λ2|>>0)
        Meijering — neurite/filament enhancement (negative eigenvalue emphasis)
        Sato    — multi-scale tubeness response

    Features: sobel_mean, sobel_std, canny_edge_density,
              frangi_mean, frangi_max, meijering_mean, sato_mean.
    """

    def _extract_one(self, image, mask, label_id):
        crop, crop_mask = self._crop(image, mask, label_id)
        if crop is None or crop.size < 16:
            return None
        mn, mx = float(crop.min()), float(crop.max())
        if mx - mn < 1e-6:
            return None
        norm = (crop - mn) / (mx - mn)

        def _masked_mean(arr):
            v = arr[crop_mask]
            return float(np.mean(v)) if v.size else 0.0

        def _masked_max(arr):
            v = arr[crop_mask]
            return float(np.max(v)) if v.size else 0.0

        # Sobel
        try:
            sob = filters.sobel(norm)
            sobel_mean = _masked_mean(sob)
            sobel_std  = float(np.std(sob[crop_mask])) if crop_mask.any() else 0.0
        except Exception:
            sobel_mean = sobel_std = 0.0

        # Canny
        try:
            canny_edges   = feature.canny(norm, sigma=1.0)
            canny_density = float(canny_edges[crop_mask].mean()) if crop_mask.any() else 0.0
        except Exception:
            canny_density = 0.0

        # Frangi vesselness (file 104)
        try:
            frangi_resp = filters.frangi(norm)
            frangi_mean = _masked_mean(frangi_resp)
            frangi_max  = _masked_max(frangi_resp)
        except Exception:
            frangi_mean = frangi_max = 0.0

        # Meijering neurite filter (file 104)
        try:
            meij_mean = _masked_mean(filters.meijering(norm))
        except Exception:
            meij_mean = 0.0

        # Sato tubeness (file 104)
        try:
            sato_mean = _masked_mean(filters.sato(norm))
        except Exception:
            sato_mean = 0.0

        return {
            "sobel_mean":         sobel_mean,
            "sobel_std":          sobel_std,
            "canny_edge_density": canny_density,
            "frangi_mean":        frangi_mean,
            "frangi_max":         frangi_max,
            "meijering_mean":     meij_mean,
            "sato_mean":          sato_mean,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 10. PCP Orientation
# ──────────────────────────────────────────────────────────────────────────────
class PCPOrientationExtractor(BaseFeatureExtractor):
    """
    Planar Cell Polarity — in-plane XY bundle orientation (0–180°).

    Uses the major axis of the regionprops inertia ellipse:
        theta_PCP = (90° - theta_major) mod 180°

    Features: pcp_angle_deg, pcp_elongation, pcp_eccentricity.
    """

    def _extract_one(self, image, mask, label_id):
        bm = (mask == label_id).astype(np.uint8)
        if bm.sum() < 5:
            return None
        props = measure.regionprops(bm)
        if not props:
            return None
        rp = props[0]
        angle = (90.0 - np.degrees(rp.orientation)) % 180.0
        elongation = float(rp.axis_major_length / (rp.axis_minor_length + 1e-8))
        return {
            "pcp_angle_deg":   float(angle),
            "pcp_elongation":  elongation,
            "pcp_eccentricity": float(rp.eccentricity),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────
class FeaturePipeline:
    """
    Runs all extractors in sequence and left-joins results on (stack_id, bundle_id).

    Default extractor order mirrors the Streamlit app's feature tabs:
        Morphology → Intensity → Entropy → GLCM → LBP → Gabor → HOG →
        Keypoints → EdgeRidge → PCP

    Usage:
        pipe = FeaturePipeline(voxel_size=(0.043, 0.043), verbose=True)
        df   = pipe.run(image, mask, stack_id="sample_01")
    """

    DEFAULT_EXTRACTORS = [
        MorphologyExtractor,
        IntensityExtractor,
        EntropyExtractor,
        GLCMTextureExtractor,
        LBPTextureExtractor,
        GaborFeatureExtractor,
        HOGFeatureExtractor,
        KeypointExtractor,
        EdgeRidgeExtractor,
        PCPOrientationExtractor,
    ]

    def __init__(self, voxel_size: Tuple[float, float] = (0.043, 0.043),
                 extractors: Optional[List] = None, verbose: bool = False):
        cls_list = extractors or self.DEFAULT_EXTRACTORS
        self.extractors = [c(voxel_size=voxel_size) for c in cls_list]
        self.verbose    = verbose

    def run(self, image: np.ndarray, mask: np.ndarray,
            stack_id: str = "stack_0") -> pd.DataFrame:
        dfs = []
        for ext in self.extractors:
            name = ext.__class__.__name__
            if self.verbose:
                print(f"  → {name}", end=" ... ", flush=True)
            df = ext.extract(image, mask, stack_id=stack_id)
            n  = (df.shape[1] - 2) if not df.empty else 0
            if self.verbose:
                print(f"{n} features")
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on=["stack_id", "bundle_id"], how="outer")
        return merged.sort_values("bundle_id").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running self-test on synthetic 2D data...")
    rng = np.random.default_rng(0)
    img  = rng.random((128, 128)).astype(np.float32) * 0.3
    msk  = np.zeros((128, 128), dtype=np.int32)
    msk[30:80, 20:60] = 1
    msk[40:110, 70:110] = 2
    img[msk > 0] += 0.7

    pipe = FeaturePipeline(verbose=True)
    out  = pipe.run(img, msk, stack_id="test")
    print(f"\nResult: {out.shape[0]} bundles × {out.shape[1]} columns")
    print(f"Feature columns: {out.shape[1] - 2}")
