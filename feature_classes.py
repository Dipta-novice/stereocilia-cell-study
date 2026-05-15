
"""
================================================================================
VASCilia Feature Extraction Classes
================================================================================
Comprehensive feature extraction for 2D hair cell stereocilia bundles.

Architecture:
    BaseFeatureExtractor (abstract)
        ├── MorphologyExtractor2D    - regionprops in 2D (area, length, etc.)
        ├── IntensityExtractor       - mean/std/CV of pixel intensities
        ├── GLCMTextureExtractor     - Gray-Level Co-occurrence Matrix
        ├── LBPTextureExtractor      - Local Binary Patterns
        ├── GaborFeatureExtractor    - Multi-orientation Gabor filter bank
        ├── HOGFeatureExtractor      - Histogram of Oriented Gradients
        ├── KeypointExtractor        - Harris / Shi-Tomasi counts
        ├── EdgeFeatureExtractor     - Sobel / Canny / Frangi statistics
        └── PCPOrientationExtractor  - Bundle orientation in XY plane

    FeaturePipeline                   - Orchestrator that runs all extractors

Each extractor follows the same API:
    extractor = MorphologyExtractor2D(voxel_size=(0.043, 0.043))
    df = extractor.extract(image_2d, mask_2d)   # returns DataFrame
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from skimage import measure, feature, filters


# ==============================================================================
# Base class
# ==============================================================================
class BaseFeatureExtractor(ABC):
    """
    Abstract base for all feature extractors.
    """

    DEFAULT_VOXEL_SIZE = (0.043, 0.043)  # (y_um, x_um)

    def __init__(self, voxel_size: Tuple[float, float] = None, **kwargs):
        self.voxel_size = voxel_size or self.DEFAULT_VOXEL_SIZE
        self.params = kwargs
        self.name = self.__class__.__name__.replace("Extractor", "").lower()

    @abstractmethod
    def _extract_single_bundle(
        self, image: np.ndarray, mask: np.ndarray, label_id: int
    ) -> Dict[str, float]:
        pass

    def extract(
        self, image: np.ndarray, mask: np.ndarray, stack_id: str = "stack_0"
    ) -> pd.DataFrame:
        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects 2D image/mask arrays, "
                f"got image={image.shape}, mask={mask.shape}"
            )
        if image.shape != mask.shape:
            raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}")

        labels = np.unique(mask)
        labels = labels[labels > 0]

        rows = []
        for lab in labels:
            feats = self._extract_single_bundle(image, mask, int(lab))
            if feats is None or len(feats) == 0:
                continue
            feats["stack_id"] = stack_id
            feats["bundle_id"] = int(lab)
            rows.append(feats)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        id_cols = ["stack_id", "bundle_id"]
        feat_cols = [c for c in df.columns if c not in id_cols]
        return df[id_cols + feat_cols]


# ==============================================================================
# 1. Morphology (2D regionprops)
# ==============================================================================
class MorphologyExtractor2D(BaseFeatureExtractor):
    def _extract_single_bundle(self, image, mask, label_id):
        bundle_mask = (mask == label_id)
        if bundle_mask.sum() == 0:
            return None

        dy, dx = self.voxel_size
        props = measure.regionprops(bundle_mask.astype(np.uint8))
        if not props:
            return None
        rp = props[0]

        minr, minc, maxr, maxc = rp.bbox
        width_um = float((maxc - minc) * dx)
        height_um = float((maxr - minr) * dy)
        area_px = float(rp.area)
        perimeter_px = float(rp.perimeter)
        perimeter_um = perimeter_px * float((dy + dx) / 2.0)
        equiv_diameter_px = (
            rp.equivalent_diameter_area
            if hasattr(rp, "equivalent_diameter_area")
            else rp.equivalent_diameter
        )

        return {
            "area_px": area_px,
            "area_um2": float(area_px * dx * dy),
            "perimeter_px": perimeter_px,
            "perimeter_um": perimeter_um,
            "extent_x_um": width_um,
            "extent_y_um": height_um,
            "aspect_ratio": float(width_um / max(height_um, 1e-8)),
            "solidity": float(rp.solidity) if hasattr(rp, "solidity") else np.nan,
            "eccentricity": float(rp.eccentricity),
            "extent": float(rp.extent),
            "major_axis_um": float(rp.axis_major_length * dx),
            "minor_axis_um": float(rp.axis_minor_length * dy),
            "equiv_diameter_um": float(equiv_diameter_px * np.sqrt(dx * dy)),
            "centroid_y_um": float(rp.centroid[0] * dy),
            "centroid_x_um": float(rp.centroid[1] * dx),
        }


# ==============================================================================
# 2. Intensity statistics
# ==============================================================================
class IntensityExtractor(BaseFeatureExtractor):
    def _extract_single_bundle(self, image, mask, label_id):
        pix = image[mask == label_id]
        if pix.size == 0:
            return None
        pix = pix.astype(np.float32)
        mean = float(np.mean(pix))
        std = float(np.std(pix))
        return {
            "intensity_mean": mean,
            "intensity_std": std,
            "intensity_min": float(np.min(pix)),
            "intensity_max": float(np.max(pix)),
            "intensity_median": float(np.median(pix)),
            "intensity_cv": std / (mean + 1e-8),
            "intensity_skew": float(stats.skew(pix)) if pix.size > 2 else 0.0,
            "intensity_kurtosis": float(stats.kurtosis(pix)) if pix.size > 3 else 0.0,
            "intensity_total": float(np.sum(pix)),
            "intensity_p95": float(np.percentile(pix, 95)),
        }


# ==============================================================================
# 3. GLCM Texture
# ==============================================================================
class GLCMTextureExtractor(BaseFeatureExtractor):
    DEFAULTS = dict(distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32)

    def __init__(self, voxel_size=None, **kwargs):
        super().__init__(voxel_size, **kwargs)
        for k, v in self.DEFAULTS.items():
            self.params.setdefault(k, v)

    def _extract_single_bundle(self, image, mask, label_id):
        bundle_mask = (mask == label_id)
        coords = np.column_stack(np.nonzero(bundle_mask))
        if coords.size == 0:
            return None

        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0) + 1
        crop_img = image[minr:maxr, minc:maxc]
        crop_mask = bundle_mask[minr:maxr, minc:maxc]
        roi = np.where(crop_mask, crop_img, 0)

        if roi.size < 16:
            return None

        levels = int(self.params["levels"])
        roi_f = roi.astype(np.float32)
        mn, mx = roi_f.min(), roi_f.max()
        if mx - mn < 1e-6:
            return None
        roi_q = np.clip(((roi_f - mn) / (mx - mn) * (levels - 1)).astype(np.uint8), 0, levels - 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glcm = feature.graycomatrix(
                roi_q, distances=self.params["distances"], angles=self.params["angles"],
                levels=levels, symmetric=True, normed=True,
            )

        feats = {}
        for prop in ("contrast", "energy", "homogeneity", "correlation", "dissimilarity", "ASM"):
            vals = feature.graycoprops(glcm, prop)
            feats[f"glcm_{prop.lower()}"] = float(np.mean(vals))
        return feats


# ==============================================================================
# 4. LBP Texture
# ==============================================================================
class LBPTextureExtractor(BaseFeatureExtractor):
    def __init__(self, voxel_size=None, radius=3, n_points=24, **kwargs):
        super().__init__(voxel_size, **kwargs)
        self.params.update(dict(radius=radius, n_points=n_points))

    def _extract_single_bundle(self, image, mask, label_id):
        bundle_mask = (mask == label_id)
        coords = np.column_stack(np.nonzero(bundle_mask))
        if coords.size == 0:
            return None

        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0) + 1
        crop_img = image[minr:maxr, minc:maxc].astype(np.float32)
        crop_mask = bundle_mask[minr:maxr, minc:maxc]

        if crop_img.size < 32:
            return None

        mn, mx = float(crop_img.min()), float(crop_img.max())
        if mx - mn < 1e-8:
            return None
        crop_u8 = ((crop_img - mn) / (mx - mn) * 255).astype(np.uint8)

        P, R = self.params["n_points"], self.params["radius"]
        lbp = feature.local_binary_pattern(crop_u8, P=P, R=R, method="uniform")
        lbp_in_bundle = lbp[crop_mask]
        if lbp_in_bundle.size == 0:
            return None

        n_bins = P + 2
        hist, _ = np.histogram(lbp_in_bundle, bins=n_bins, range=(0, n_bins), density=True)
        return {
            "lbp_mean": float(np.mean(lbp_in_bundle)),
            "lbp_std": float(np.std(lbp_in_bundle)),
            "lbp_entropy": float(stats.entropy(hist + 1e-10)),
            "lbp_uniform_fraction": float(hist[:-1].sum()),
        }


# ==============================================================================
# 5. Gabor filter bank
# ==============================================================================
class GaborFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, voxel_size=None,
                 thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                 frequencies=(0.1, 0.3, 0.5),
                 **kwargs):
        super().__init__(voxel_size, **kwargs)
        self.params.update(dict(thetas=thetas, frequencies=frequencies))

    def _extract_single_bundle(self, image, mask, label_id):
        bundle_mask = (mask == label_id)
        coords = np.column_stack(np.nonzero(bundle_mask))
        if coords.size == 0:
            return None

        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0) + 1
        crop_img = image[minr:maxr, minc:maxc].astype(np.float32)
        crop_mask = bundle_mask[minr:maxr, minc:maxc]

        if crop_img.shape[0] < 5 or crop_img.shape[1] < 5:
            return None

        feats = {}
        responses = []
        for theta in self.params["thetas"]:
            for freq in self.params["frequencies"]:
                real, _ = filters.gabor(crop_img, frequency=freq, theta=theta)
                vals = real[crop_mask]
                if vals.size == 0:
                    continue
                key = f"gabor_t{int(np.degrees(theta))}_f{freq:.1f}"
                feats[f"{key}_mean"] = float(np.mean(np.abs(vals)))
                feats[f"{key}_std"] = float(np.std(vals))
                responses.append(np.abs(vals).mean())

        if responses:
            feats["gabor_max_response"] = float(np.max(responses))
            feats["gabor_anisotropy"] = float(
                (np.max(responses) - np.min(responses)) /
                (np.max(responses) + np.min(responses) + 1e-8)
            )
        return feats


# ==============================================================================
# 6. HOG
# ==============================================================================
class HOGFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, voxel_size=None,
                 pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9,
                 **kwargs):
        super().__init__(voxel_size, **kwargs)
        self.params.update(dict(
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            orientations=orientations,
        ))

    def _extract_single_bundle(self, image, mask, label_id):
        bundle_mask = (mask == label_id)
        coords = np.column_stack(np.nonzero(bundle_mask))
        if coords.size == 0:
            return None

        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0) + 1
        roi_img = image[minr:maxr, minc:maxc].astype(np.float32)

        ppc = self.params["pixels_per_cell"]
        if roi_img.shape[0] < ppc[0] * 2 or roi_img.shape[1] < ppc[1] * 2:
            return None

        try:
            hog_vec = feature.hog(
                roi_img, orientations=self.params["orientations"],
                pixels_per_cell=ppc,
                cells_per_block=self.params["cells_per_block"],
                feature_vector=True,
            )
        except Exception:
            return None

        if hog_vec.size == 0:
            return None
        return {
            "hog_mean": float(np.mean(hog_vec)),
            "hog_std": float(np.std(hog_vec)),
            "hog_max": float(np.max(hog_vec)),
            "hog_energy": float(np.sum(hog_vec ** 2)),
            "hog_n_features": int(hog_vec.size),
        }


# ==============================================================================
# 7. Keypoint counts
# ==============================================================================
class KeypointExtractor(BaseFeatureExtractor):
    def _extract_single_bundle(self, image, mask, label_id):
        bundle_mask = (mask == label_id)
        coords = np.column_stack(np.nonzero(bundle_mask))
        if coords.size == 0:
            return None

        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0) + 1
        crop_img = image[minr:maxr, minc:maxc].astype(np.float32)
        crop_mask_2d = bundle_mask[minr:maxr, minc:maxc].astype(bool)

        if crop_img.size < 25:
            return None

        try:
            harris = feature.corner_harris(crop_img, k=0.04, sigma=1.0)
            harris_peaks = feature.corner_peaks(harris, min_distance=3, threshold_rel=0.02)
            n_harris = sum(1 for r, c in harris_peaks if crop_mask_2d[r, c])
        except Exception:
            n_harris = 0

        try:
            shi = feature.corner_shi_tomasi(crop_img, sigma=1.0)
            shi_peaks = feature.corner_peaks(shi, min_distance=3, threshold_rel=0.02)
            n_shi = sum(1 for r, c in shi_peaks if crop_mask_2d[r, c])
        except Exception:
            n_shi = 0

        area_px = crop_mask_2d.sum()
        return {
            "n_harris_corners": int(n_harris),
            "n_shitomasi_corners": int(n_shi),
            "corner_density_per_px": float(n_harris / max(area_px, 1)),
        }


# ==============================================================================
# 8. Edge statistics
# ==============================================================================
class EdgeFeatureExtractor(BaseFeatureExtractor):
    def _extract_single_bundle(self, image, mask, label_id):
        bundle_mask = (mask == label_id)
        coords = np.column_stack(np.nonzero(bundle_mask))
        if coords.size == 0:
            return None

        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0) + 1
        crop_img = image[minr:maxr, minc:maxc].astype(np.float32)
        crop_mask_2d = bundle_mask[minr:maxr, minc:maxc].astype(bool)

        if crop_img.size < 16:
            return None

        mn, mx = crop_img.min(), crop_img.max()
        if mx - mn < 1e-6:
            return None
        mip_n = (crop_img - mn) / (mx - mn)

        try:
            sobel_mag = filters.sobel(mip_n)
            sobel_in = sobel_mag[crop_mask_2d]
            sobel_mean = float(np.mean(sobel_in)) if sobel_in.size else 0.0
            sobel_std = float(np.std(sobel_in)) if sobel_in.size else 0.0
        except Exception:
            sobel_mean = sobel_std = 0.0

        try:
            canny_edges = feature.canny(mip_n, sigma=1.0)
            edge_density = float(canny_edges[crop_mask_2d].mean()) if crop_mask_2d.any() else 0.0
        except Exception:
            edge_density = 0.0

        try:
            frangi_resp = filters.frangi(mip_n)
            frangi_in = frangi_resp[crop_mask_2d]
            frangi_mean = float(np.mean(frangi_in)) if frangi_in.size else 0.0
            frangi_max = float(np.max(frangi_in)) if frangi_in.size else 0.0
        except Exception:
            frangi_mean = frangi_max = 0.0

        return {
            "sobel_mean": sobel_mean,
            "sobel_std": sobel_std,
            "canny_edge_density": edge_density,
            "frangi_mean": frangi_mean,
            "frangi_max": frangi_max,
        }


# ==============================================================================
# 9. PCP Orientation
# ==============================================================================
class PCPOrientationExtractor(BaseFeatureExtractor):
    def _extract_single_bundle(self, image, mask, label_id):
        bundle_mask = (mask == label_id)
        if bundle_mask.sum() < 5:
            return None

        props = measure.regionprops(bundle_mask.astype(np.uint8))
        if not props:
            return None
        rp = props[0]

        # skimage reports orientation relative to the image row axis. Convert to
        # an in-plane XY heading where 0/180 degrees is horizontal.
        orient_deg = (90.0 - np.degrees(rp.orientation)) % 180.0

        elongation = float(rp.axis_major_length / (rp.axis_minor_length + 1e-8))
        return {
            "pcp_angle_deg": float(orient_deg),
            "pcp_elongation": elongation,
            "pcp_eccentricity": float(rp.eccentricity),
        }


# ==============================================================================
# Pipeline orchestrator
# ==============================================================================
class FeaturePipeline:
    DEFAULT_EXTRACTORS = [
        MorphologyExtractor2D,
        IntensityExtractor,
        GLCMTextureExtractor,
        LBPTextureExtractor,
        GaborFeatureExtractor,
        HOGFeatureExtractor,
        KeypointExtractor,
        EdgeFeatureExtractor,
        PCPOrientationExtractor,
    ]

    def __init__(self,
                 voxel_size: Tuple[float, float] = (0.043, 0.043),
                 extractors: Optional[List] = None,
                 verbose: bool = False):
        cls_list = extractors or self.DEFAULT_EXTRACTORS
        self.extractors = [c(voxel_size=voxel_size) for c in cls_list]
        self.verbose = verbose

    def run(self, image: np.ndarray, mask: np.ndarray,
            stack_id: str = "stack_0") -> pd.DataFrame:
        dfs = []
        for ext in self.extractors:
            if self.verbose:
                print(f"  → {ext.name}", end=" ", flush=True)
            df = ext.extract(image, mask, stack_id=stack_id)
            if self.verbose:
                print(f"({df.shape[1] - 2 if not df.empty else 0} features)")
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on=["stack_id", "bundle_id"], how="outer")
        return merged.sort_values("bundle_id").reset_index(drop=True)

    def list_features(self) -> Dict[str, List[str]]:
        info = {}
        for ext in self.extractors:
            info[ext.name] = list(getattr(ext, "_feature_names", []))
        return info


if __name__ == "__main__":
    print("Running self-test on synthetic 2D data...")
    rng = np.random.default_rng(0)
    img = (rng.random((128, 128)).astype(np.float32) * 0.3)
    msk = np.zeros((128, 128), dtype=np.int32)
    msk[30:80, 20:60] = 1
    msk[40:110, 70:110] = 2
    img[msk > 0] += 0.7

    pipe = FeaturePipeline(verbose=True)
    out = pipe.run(img, msk, stack_id="test_stack")
    print(f"\nResult shape: {out.shape}")
    print(f"Feature columns: {len(out.columns) - 2}")
    print(out.T.head(20))
