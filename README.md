# VASCilia Hair Cell 2D Analysis

Pipeline for flat RGB TIFF images and flat PNG label masks. The active data
path has no Z dimension; each TIFF/PNG pair is treated as an independent 2D
image/mask pair.

## Quick Navigation

| File | What it does |
|------|--------------|
| `src/data_loader.py` | Loads individual green-channel TIFFs and PNG masks; parses VASCilia folder names |
| `src/preprocessing_classes.py` | 2D normalization, CLAHE, Gaussian/median/bilateral denoising |
| `src/feature_classes.py` | 2D morphology, intensity, texture, edge, keypoint, and PCP extractors |
| `src/row_classifier.py` | IHC / OHC1 / OHC2 / OHC3 via 2D centroids, PCA, and 4-cluster K-Means |
| `src/visualization_2d.py` | Raw image display with colored bundle contour overlays |
| `src/app.py` | Streamlit UI with 2D view, counts, features, PCP, and UMAP |

## Streamlit UI Tabs

1. **2D View** - raw grayscale image with colored mask contours and bundle ids
2. **Cell Count** - bar chart for IHC vs OHC1/OHC2/OHC3
3. **Morphology** - table and histograms for area, perimeter, aspect ratio, solidity, eccentricity
4. **GLCM Texture** - heatmap for contrast, energy, homogeneity, correlation, and related metrics
5. **PCP Orientation** - XY in-plane 0-180 degree polar plot with circular mean and std
6. **UMAP** - feature-space embedding when precomputed feature tables are available

## Run

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Set the dashboard data root to the folder that contains `P21/`.

## Feature Extractors

| Class | Features |
|-------|----------|
| `MorphologyExtractor2D` | area, perimeter, solidity, eccentricity, axis lengths, centroid |
| `IntensityExtractor` | mean, std, min, max, median, CV, skew, kurtosis, total, p95 |
| `GLCMTextureExtractor` | contrast, energy, homogeneity, correlation, dissimilarity, ASM |
| `LBPTextureExtractor` | local binary pattern histogram statistics |
| `GaborFeatureExtractor` | multi-angle, multi-frequency 2D filter responses |
| `HOGFeatureExtractor` | oriented-gradient summary |
| `KeypointExtractor` | Harris and Shi-Tomasi corner counts |
| `EdgeFeatureExtractor` | Sobel, Canny, and Frangi response statistics |
| `PCPOrientationExtractor` | XY heading, elongation, eccentricity |
