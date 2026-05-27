"""
VASCilia Streamlit UI for flat 2D hair-cell bundle analysis.

Run with:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data_loader import VOXEL_SIZE, list_stack_pairs, load_stack_pair, ensure_metadata_csv
from feature_classes import FeaturePipeline
from preprocessing_classes import PreprocessingPipeline
from row_classifier import classify_rows, counts_per_row
from visualization_2d import plot_bundle_image_overlay


st.set_page_config(
    page_title="VASCilia Hair Cell Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("VASCilia P21")
st.sidebar.caption("2D hair cell bundle analysis")

# Discover paths dynamically
repo_root = Path(__file__).resolve().parents[1]

# Find Excel file to determine data_root
excel_candidates = [
    repo_root / "VASCilia_2D_Dataset.xlsx",
    repo_root / "VASCilia_3D_Dataset.xlsx",
    repo_root.parents[1] / "VASCilia_2D_Dataset.xlsx",
    repo_root.parents[1] / "VASCilia_3D_Dataset.xlsx",
    repo_root.parents[2] / "VASCilia_2D_Dataset.xlsx",
    repo_root.parents[2] / "VASCilia_3D_Dataset.xlsx",
]

excel_path = None
for candidate in excel_candidates:
    if candidate.exists():
        excel_path = candidate
        break

if excel_path is None:
    st.error("Excel file not found. Cannot determine dataset root.")
    st.stop()

data_root = excel_path.parent
md_path = repo_root / "results" / "master_metadata_P21.csv"

# Auto-generate metadata CSV if missing
with st.spinner("Loading metadata..."):
    try:
        ensure_metadata_csv(repo_root)
    except FileNotFoundError as e:
        st.error(f"Error generating metadata: {e}")
        st.stop()

if not md_path.exists():
    st.error(f"Metadata CSV not found at {md_path}.")
    st.stop()

md = pd.read_csv(md_path)

# Display dataset path info
st.sidebar.info(f"📂 Dataset: {data_root}")

stack_id = st.sidebar.selectbox(
    "Pick a subject",
    md["stack_id"].tolist(),
    format_func=lambda s: f"{md.loc[md['stack_id'] == s, 'region'].iloc[0]} | {s[:32]}",
)


@st.cache_data(show_spinner="Scanning 2D image pairs...")
def cached_pair_options(data_root_str: str, sid: str):
    return [
        {"image_index": p["image_index"], "image_id": p["image_id"]}
        for p in list_stack_pairs(Path(data_root_str), sid)
    ]


pair_options = cached_pair_options(str(data_root), stack_id)
if pair_options:
    image_index = st.sidebar.selectbox(
        "2D image",
        [p["image_index"] for p in pair_options],
        format_func=lambda i: pair_options[i]["image_id"].split("_")[-1],
    )
else:
    image_index = 0

features_csv = Path("results/tables/features_all_bundles_P21.csv")
features_all = pd.read_csv(features_csv) if features_csv.exists() else None
if features_all is None:
    st.sidebar.warning("Run feature extraction to enable UMAP and cross-subject views.")

st.sidebar.divider()
pixel_xy = st.sidebar.number_input(
    "Pixel XY (um)",
    value=float(VOXEL_SIZE[0]),
    step=0.001,
    format="%.3f",
)
pixel_size = (pixel_xy, pixel_xy)


@st.cache_data(show_spinner="Loading 2D image...")
def cached_load(data_root_str: str, sid: str, img_idx: int):
    img, msk, folder = load_stack_pair(
        Path(data_root_str),
        sid,
        verbose=False,
        image_index=img_idx,
    )
    return img, msk, str(folder) if folder else None


@st.cache_data(show_spinner="Preprocessing...")
def cached_preprocess(data_root_str: str, sid: str, img_idx: int, px_size):
    img, msk, _ = cached_load(data_root_str, sid, img_idx)
    if img is None or msk is None:
        return None, None
    pipe = PreprocessingPipeline(denoise_method="gaussian")
    img_pp, mask_pp, _ = pipe.run(img, msk)
    return img_pp, mask_pp


@st.cache_data(show_spinner="Extracting 2D features...")
def cached_features(data_root_str: str, sid: str, img_idx: int, px_size):
    img_pp, mask_pp = cached_preprocess(data_root_str, sid, img_idx, px_size)
    if img_pp is None or mask_pp is None:
        return None, None
    rows = classify_rows(mask_pp, voxel_size=px_size)
    feats = FeaturePipeline(voxel_size=px_size).run(img_pp, mask_pp, stack_id=sid)
    if not rows.empty and not feats.empty:
        feats = feats.merge(rows[["bundle_id", "row_label"]], on="bundle_id", how="left")
    return feats, rows


st.title("VASCilia Hair Cell Bundle Explorer")
sel = md[md["stack_id"] == stack_id].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Region", sel["region"])
c2.metric("Litter / Mouse", f"{sel['litter_id']} / {sel['mouse_id']}")
c3.metric("Reported bundles", int(sel["n_bundles"]))
c4.metric("Reported IHC", int(sel["n_ihc"]))
c5.metric("Annotator", sel["annotator"])

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "2D View",
    "Cell Count",
    "Morphology",
    "GLCM Texture",
    "PCP Orientation",
    "UMAP",
])

with tab1:
    st.subheader("2D image + bundle contour overlay")
    image, mask, folder = cached_load(str(data_root), stack_id, image_index)
    if image is None:
        st.error("Could not load this 2D image. Check the data root.")
    else:
        c1, c2 = st.columns(2)
        c1.write(f"**Image shape:** {image.shape} | **dtype:** {image.dtype}")
        c2.write(f"**Mask labels:** {len(np.unique(mask)) - 1 if mask is not None else 0}")

        view_mode = st.radio("View", ["Overlay", "Raw image"], horizontal=True)
        fig = plot_bundle_image_overlay(
            image,
            mask if view_mode == "Overlay" else None,
        )
        st.pyplot(fig)

with tab2:
    st.subheader("Hair cell row counts")
    feats, rows = cached_features(str(data_root), stack_id, image_index, pixel_size)
    if rows is None or rows.empty:
        st.warning("No bundles found.")
    else:
        cnt = counts_per_row(rows)
        cnt["color"] = ["IHC" if r == "IHC" else "OHC" for r in cnt["row"]]
        fig = px.bar(
            cnt,
            x="row",
            y="count",
            color="color",
            color_discrete_map={"IHC": "#d84a3a", "OHC": "#2f7fbf"},
            text="count",
            category_orders={"row": ["IHC", "OHC1", "OHC2", "OHC3"]},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, showlegend=False,
                          yaxis_title="Count", xaxis_title="Row")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.write("**Measured rows**")
        c1.dataframe(cnt[["row", "count"]], hide_index=True)
        c2.write("**Reported totals**")
        c2.dataframe(pd.DataFrame({
            "row": ["IHC", "OHC total"],
            "count": [int(sel["n_ihc"]), int(sel["n_ohc"])],
        }), hide_index=True)

with tab3:
    st.subheader("2D morphological features per bundle")
    feats, _ = cached_features(str(data_root), stack_id, image_index, pixel_size)
    if feats is None or feats.empty:
        st.warning("No features available.")
    else:
        morph_cols = [
            "bundle_id", "row_label", "area_um2", "perimeter_um",
            "aspect_ratio", "solidity", "eccentricity", "equiv_diameter_um",
        ]
        morph_cols = [c for c in morph_cols if c in feats.columns]
        st.dataframe(feats[morph_cols].round(3), use_container_width=True, height=300)

        c1, c2, c3 = st.columns(3)
        for col, (label, key) in zip(
            [c1, c2, c3],
            [("Area (um^2)", "area_um2"),
             ("Perimeter (um)", "perimeter_um"),
             ("Aspect ratio", "aspect_ratio")],
        ):
            if key not in feats.columns:
                continue
            fig = px.histogram(
                feats,
                x=key,
                color="row_label" if "row_label" in feats.columns else None,
                nbins=20,
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(height=300, showlegend=True,
                              margin=dict(t=30, b=10, l=5, r=5),
                              title=label, title_x=0.5)
            col.plotly_chart(fig, use_container_width=True)

        st.write("**Summary statistics by row**")
        summary_cols = [c for c in ["area_um2", "perimeter_um", "aspect_ratio"]
                        if c in feats.columns]
        if "row_label" in feats.columns and summary_cols:
            summary = feats.groupby("row_label")[summary_cols].agg(["mean", "std"]).round(3)
            st.dataframe(summary, use_container_width=True)

with tab4:
    st.subheader("GLCM texture features per bundle")
    feats, _ = cached_features(str(data_root), stack_id, image_index, pixel_size)
    if feats is None or feats.empty:
        st.warning("No features.")
    else:
        glcm_cols = [c for c in feats.columns if c.startswith("glcm_")]
        if not glcm_cols:
            st.warning("No GLCM columns found.")
        else:
            heat_df = feats[["bundle_id"] + glcm_cols].set_index("bundle_id")
            heat_norm = (heat_df - heat_df.mean()) / (heat_df.std() + 1e-8)
            fig = go.Figure(data=go.Heatmap(
                z=heat_norm.T.values,
                x=heat_norm.index,
                y=heat_norm.columns,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(title="z-score"),
            ))
            fig.update_layout(height=400, xaxis_title="bundle_id",
                              yaxis_title="GLCM feature")
            st.plotly_chart(fig, use_container_width=True)
            st.write("**Raw values**")
            st.dataframe(heat_df.round(3), use_container_width=True, height=200)

with tab5:
    st.subheader("Planar Cell Polarity orientations")
    feats, _ = cached_features(str(data_root), stack_id, image_index, pixel_size)
    if feats is None or "pcp_angle_deg" not in feats.columns or feats.empty:
        st.warning("No PCP angles.")
    else:
        angles = feats["pcp_angle_deg"].dropna().values
        rad = np.deg2rad(angles * 2.0)
        mean_rad = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
        mean_deg = (np.degrees(mean_rad) / 2.0) % 180.0
        R = np.abs(np.mean(np.exp(1j * rad)))
        circ_std_deg = np.degrees(np.sqrt(-2 * np.log(R + 1e-8))) / 2.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Circular mean (deg)", f"{mean_deg:.1f}")
        c2.metric("Circular std (deg)", f"{circ_std_deg:.1f}")
        c3.metric("N bundles", len(angles))

        hist, edges = np.histogram(angles, bins=18, range=(0, 180))
        theta = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=hist,
            theta=theta,
            width=10,
            marker=dict(color=hist, colorscale="Viridis"),
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0, max(hist) * 1.1 if len(hist) else 1],
            theta=[mean_deg, mean_deg],
            mode="lines",
            line=dict(color="red", width=4),
            name=f"mean ({mean_deg:.0f} deg)",
        ))
        fig.update_layout(
            height=500,
            showlegend=True,
            polar=dict(
                angularaxis=dict(direction="clockwise", thetaunit="degrees"),
                radialaxis=dict(title="Bundle count"),
            ),
            title=f"XY PCP angle distribution - {stack_id[:30]}",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("UMAP feature-space embedding")
    if features_all is None:
        st.info("Run feature extraction first to populate features for all subjects.")
    else:
        try:
            import umap
        except ImportError:
            st.error("Install umap-learn: pip install umap-learn")
            st.stop()

        excl = {"bundle_id", "stack_id", "row_label", "region",
                "annotator", "litter_id", "mouse_id"}
        feat_cols = [
            c for c in features_all.columns
            if c not in excl and pd.api.types.is_numeric_dtype(features_all[c])
        ]
        X = features_all[feat_cols].fillna(0).values
        X_n = (X - X.mean(0)) / (X.std(0) + 1e-8)

        n_neighbors = st.slider("n_neighbors", 5, 30, 10)
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, 0.05)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
        )
        emb = reducer.fit_transform(X_n)

        df_plot = features_all.copy()
        df_plot["UMAP1"] = emb[:, 0]
        df_plot["UMAP2"] = emb[:, 1]
        df_plot["is_current"] = df_plot["stack_id"] == stack_id

        fig = px.scatter(
            df_plot,
            x="UMAP1",
            y="UMAP2",
            color="region",
            symbol="is_current",
            hover_data=["stack_id", "bundle_id", "row_label"],
            symbol_map={True: "star", False: "circle"},
        )
        fig.update_traces(selector=dict(name=str(True)),
                          marker=dict(size=14, line=dict(color="black", width=1)))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Star markers are bundles from the currently selected subject.")
