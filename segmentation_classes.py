"""
================================================================================
VASCilia 2D Segmentation Classes
================================================================================
Two CPU-friendly algorithms sourced from Python for Microscopists:

  - MultiOtsuSegmenter    → file 115  (threshold_multiotsu)
  - WatershedSegmenter    → files 033-035 (distance transform + watershed)
  - CellposeSegmenter     → file 305  (optional, gpu=False, cyto3)
  - SegmentationEvaluator → count/IoU comparison against ground truth

Usage:
    seg = WatershedSegmenter(min_distance=20)
    labels, meta = seg.segment(image_preprocessed)
================================================================================
"""

from typing import Dict, Any, Tuple, Optional

import numpy as np
from scipy import ndimage
from skimage import filters, feature, measure, morphology, segmentation


# ──────────────────────────────────────────────────────────────────────────────
# 1. Multi-Otsu Segmenter (file 115)
# ──────────────────────────────────────────────────────────────────────────────
class MultiOtsuSegmenter:
    """
    Multi-Otsu thresholding for N-class segmentation (file 115).

    Algorithm:
        1. threshold_multiotsu(image, classes=N) → N-1 thresholds
        2. np.digitize → class map
        3. Treat highest-intensity class as foreground (bundles)
        4. Remove small objects → label connected components

    Math:
        t* = argmin_t  sum_k  w_k * sigma_k^2(t)
    """

    def __init__(self, n_classes: int = 3, min_size: int = 200,
                 connectivity: int = 2):
        self.n_classes = n_classes
        self.min_size = min_size
        self.connectivity = connectivity

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Parameters
        ----------
        image : 2D float array, intensity in [0, 1]

        Returns
        -------
        labeled : 2D int array (0 = background, 1..N = objects)
        meta    : dict with thresholds and counts
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")

        thresholds = filters.threshold_multiotsu(image, classes=self.n_classes)
        regions = np.digitize(image, bins=thresholds)
        binary = regions == regions.max()
        binary = morphology.remove_small_objects(binary, min_size=self.min_size)
        labeled = measure.label(binary, connectivity=self.connectivity)

        meta: Dict[str, Any] = {
            "method": "multi_otsu",
            "n_classes": self.n_classes,
            "thresholds": [round(float(t), 4) for t in thresholds],
            "n_objects": int(labeled.max()),
        }
        return labeled, meta


# ──────────────────────────────────────────────────────────────────────────────
# 2. Watershed Segmenter (files 033-035)
# ──────────────────────────────────────────────────────────────────────────────
class WatershedSegmenter:
    """
    Distance-transform seeded watershed segmentation (files 033-035).

    Algorithm:
        1. Otsu threshold → binary foreground
        2. Morphological opening (noise removal)
        3. Distance transform: D(p) = min_{q in BG} ||p-q||
        4. Peak local maxima of D → seeds
        5. watershed(-D, seeds, mask=binary) → labeled regions

    Math (file 033):
        D(p) = euclidean distance to nearest background pixel
        watershed fills from seed maxima toward region boundaries
    """

    def __init__(self, min_distance: int = 20, min_size: int = 200,
                 opening_radius: int = 3):
        self.min_distance = min_distance
        self.min_size = min_size
        self.opening_radius = opening_radius

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Parameters
        ----------
        image : 2D float array, intensity in [0, 1]

        Returns
        -------
        labeled : 2D int array (0 = background, 1..N = objects)
        meta    : dict with otsu threshold, seed count, region count
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")

        thresh = filters.threshold_otsu(image)
        binary = image > thresh
        binary = morphology.binary_opening(binary, morphology.disk(self.opening_radius))
        binary = morphology.remove_small_objects(binary, min_size=self.min_size)

        dist = ndimage.distance_transform_edt(binary)

        coords = feature.peak_local_max(dist, min_distance=self.min_distance,
                                         labels=binary)
        seed_mask = np.zeros(dist.shape, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers = measure.label(seed_mask)

        labeled = segmentation.watershed(-dist, markers, mask=binary)

        meta: Dict[str, Any] = {
            "method": "watershed",
            "otsu_threshold": round(float(thresh), 4),
            "n_seeds": int(markers.max()),
            "n_regions": int(labeled.max()),
        }
        return labeled, meta


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cellpose Segmenter (file 305) — optional dependency
# ──────────────────────────────────────────────────────────────────────────────
class CellposeSegmenter:
    """
    Cellpose flow-field segmentation, CPU-only (file 305).

    Model: cyto3 (latest cytoplasm model, handles hair cell bundles well).
    Parameters from SOP:
        diameter=50px, flow_threshold=0.6, cellprob_threshold=0

    Requires: pip install cellpose
    """

    def __init__(self, model_type: str = "cyto3", diameter: int = 50,
                 flow_threshold: float = 0.6, cellprob_threshold: float = 0.0):
        self.model_type = model_type
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from cellpose import models
                self._model = models.CellposeModel(gpu=False,
                                                   model_type=self.model_type)
            except ImportError:
                raise ImportError("Cellpose not installed. Run: pip install cellpose")
        return self._model

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")
        model = self._load()
        masks, flows, styles = model.eval(
            image.astype(np.float32),
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[0, 0],
        )
        meta: Dict[str, Any] = {
            "method": "cellpose",
            "model_type": self.model_type,
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "n_cells": int(masks.max()),
        }
        return masks.astype(np.int32), meta


# ──────────────────────────────────────────────────────────────────────────────
# 4. Segmentation Evaluator
# ──────────────────────────────────────────────────────────────────────────────
class SegmentationEvaluator:
    """
    Compare a predicted segmentation mask against the ground-truth VASCilia mask.

    Metrics:
        count_comparison — predicted vs GT object count, difference
        pixel_iou        — foreground intersection-over-union
        dice             — Dice coefficient of foreground pixels
    """

    @staticmethod
    def count_comparison(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
        n_pred = int(pred.max())
        n_gt = int(len(np.unique(gt)) - 1)
        return {
            "n_predicted": n_pred,
            "n_ground_truth": n_gt,
            "difference": n_pred - n_gt,
        }

    @staticmethod
    def pixel_iou(pred: np.ndarray, gt: np.ndarray) -> float:
        p = pred > 0
        g = gt > 0
        inter = float((p & g).sum())
        union = float((p | g).sum())
        return inter / max(union, 1)

    @staticmethod
    def dice(pred: np.ndarray, gt: np.ndarray) -> float:
        p = pred > 0
        g = gt > 0
        inter = float((p & g).sum())
        return (2.0 * inter) / max(float(p.sum() + g.sum()), 1)

    def evaluate(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
        return {
            **self.count_comparison(pred, gt),
            "pixel_iou": round(self.pixel_iou(pred, gt), 4),
            "dice": round(self.dice(pred, gt), 4),
        }
