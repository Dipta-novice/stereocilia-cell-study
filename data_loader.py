"""
VASCilia 2D data loader.

The active dataset is made of flat RGB TIFF images and flat PNG label masks.
There is no Z dimension, so this module loads individual 2D image/mask pairs
from each subject folder instead of stacking files into a volume.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import re

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import io


PIXEL_SIZE_UM = (0.043, 0.043)  # (dy, dx) in microns
VOXEL_SIZE = PIXEL_SIZE_UM       # Backward-compatible alias; no Z step exists.


def parse_subject_name(name: str) -> dict:
    """Parse structured metadata from a VASCilia subject folder name."""
    info = {"stack_id": name, "age": "P21"}

    m = re.search(r"Litter(\d+)", name)
    info["litter_id"] = m.group(1) if m else "Unknown"

    m = re.search(r"Mouse(\d+)", name)
    info["mouse_id"] = m.group(1) if m else "Unknown"

    n = name.upper()
    if "APEX" in n:
        info["region"] = "Apex"
    elif "BASE" in n:
        info["region"] = "Base"
    elif "MIDDLE" in n or "MID" in n:
        info["region"] = "Middle"
    else:
        info["region"] = "Unknown"

    info["is_airyscan"] = bool(re.search(r"AIRY", n))
    return info


def _natural_key(path: Path) -> List[object]:
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", path.stem)]


def _list_tiffs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")],
        key=_natural_key,
    )


def _list_pngs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".png"],
                  key=_natural_key)


def _extract_green_channel(arr: np.ndarray) -> np.ndarray:
    """Return the F-actin green channel from RGB/RGBA TIFF data."""
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., 1]
        if arr.shape[0] in (3, 4):
            return arr[1, ...]
    return arr


def _load_one_tiff(path: Path) -> np.ndarray:
    """Load a single TIFF file as a 2D grayscale green-channel array."""
    arr = _extract_green_channel(tiff.imread(str(path)))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported TIFF shape for 2D loader: {arr.shape} at {path}")
    return arr


def _load_one_png(path: Path) -> np.ndarray:
    """Load a single PNG label mask as a 2D integer array."""
    arr = io.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported PNG mask shape for 2D loader: {arr.shape} at {path}")
    return arr.astype(np.int32, copy=False)


def _load_raw_folder(folder: Path, image_index: int = 0) -> Optional[np.ndarray]:
    """Load one flat raw TIFF from a folder by index; never stack along Z."""
    files = _list_tiffs(folder)
    if not files:
        return None
    if image_index < 0 or image_index >= len(files):
        raise IndexError(f"image_index {image_index} outside 0..{len(files) - 1}")
    return _load_one_tiff(files[image_index]).astype(np.float32)


def _load_png_folder(folder: Path, image_index: int = 0) -> Optional[np.ndarray]:
    """Load one flat PNG mask from a folder by index."""
    files = _list_pngs(folder)
    if not files:
        return None
    if image_index < 0 or image_index >= len(files):
        raise IndexError(f"image_index {image_index} outside 0..{len(files) - 1}")
    return _load_one_png(files[image_index])


def find_subject_folder(data_root: Path, stack_id: str,
                        age: str = "P21") -> Optional[Path]:
    """Find the folder for a subject id under a few common VASCilia layouts."""
    data_root = Path(data_root)
    candidates = [
        data_root / "Vascilia" / age / stack_id,
        data_root / age / stack_id,
        data_root / stack_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    if data_root.exists():
        for path in data_root.rglob("Litter*"):
            if path.is_dir() and path.name.startswith(stack_id):
                return path
    return None


def list_stack_pairs(data_root: Path, stack_id: str,
                     age: str = "P21") -> List[Dict[str, object]]:
    """Return metadata for each flat 2D TIFF/PNG pair in a subject folder."""
    folder = find_subject_folder(data_root, stack_id, age)
    if folder is None:
        return []

    raw_files = _list_tiffs(folder / "raw_images")
    mask_files = _list_pngs(folder / "new_assignment_obj")
    n_pairs = max(len(raw_files), len(mask_files))

    pairs = []
    for i in range(n_pairs):
        raw_path = raw_files[i] if i < len(raw_files) else None
        mask_path = mask_files[i] if i < len(mask_files) else None
        image_id = raw_path.stem if raw_path is not None else mask_path.stem
        pairs.append({
            "image_index": i,
            "image_id": image_id,
            "raw_path": raw_path,
            "mask_path": mask_path,
            "folder": folder,
        })
    return pairs


def load_image_mask_pairs(data_root: Path, stack_id: str,
                          age: str = "P21") -> List[Dict[str, object]]:
    """Load every flat 2D image/mask pair for a subject without Z stacking."""
    records = []
    for pair in list_stack_pairs(data_root, stack_id, age):
        image = _load_one_tiff(pair["raw_path"]).astype(np.float32) if pair["raw_path"] else None
        mask = _load_one_png(pair["mask_path"]) if pair["mask_path"] else None
        records.append({**pair, "image": image, "mask": mask})
    return records


def load_stack_pair(data_root: Path, stack_id: str,
                    age: str = "P21",
                    verbose: bool = False,
                    image_index: int = 0
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Path]]:
    """
    Load one flat (image, mask, found_path) pair for a subject.

    `image_index` selects the TIFF/PNG pair within the subject folder. The files
    are treated as independent 2D planes, not as a Z stack.
    """
    folder = find_subject_folder(data_root, stack_id, age)
    if folder is None:
        if verbose:
            print(f"  folder not found: {stack_id}")
        return None, None, None

    img = _load_raw_folder(folder / "raw_images", image_index=image_index)
    msk = _load_png_folder(folder / "new_assignment_obj", image_index=image_index)

    if verbose:
        print(f"  loaded 2D pair {image_index} from {folder.name}")
        print(f"      image: {None if img is None else img.shape} "
              f"dtype={None if img is None else img.dtype}")
        print(f"      mask : {None if msk is None else msk.shape} "
              f"n_labels={None if msk is None else len(np.unique(msk)) - 1}")

    return img, msk, folder


def build_metadata_from_excel(excel_path: Path,
                              sheet_name: str = "P21",
                              save_csv: Optional[Path] = None
                              ) -> pd.DataFrame:
    """Build the master metadata DataFrame from the VASCilia workbook."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.loc[:, [c for c in df.columns
                    if not str(c).startswith("Unnamed")]].copy()

    rename = {
        "Subject Name": "Subject_Name",
        "Cell count": "Cell_count",
        "Inner cell count": "Inner_cell_count",
        "Outer cell count": "Outer_cell_count",
    }
    df = df.rename(columns=rename)
    df = df.dropna(subset=["Subject_Name"])

    rows = []
    for _, row in df.iterrows():
        info = parse_subject_name(str(row["Subject_Name"]))
        info["n_bundles"] = int(row["Cell_count"]) if pd.notna(row.get("Cell_count")) else 0
        info["n_ihc"] = int(row["Inner_cell_count"]) if pd.notna(row.get("Inner_cell_count")) else 0
        info["n_ohc"] = int(row["Outer_cell_count"]) if pd.notna(row.get("Outer_cell_count")) else 0
        info["annotator"] = row.get("Annotator", "Unknown")
        rows.append(info)

    md = pd.DataFrame(rows)
    cols = ["stack_id", "age", "region", "litter_id", "mouse_id",
            "n_bundles", "n_ihc", "n_ohc", "annotator", "is_airyscan"]
    md = md[cols]

    if save_csv is not None:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        md.to_csv(save_csv, index=False)
    return md


def ensure_metadata_csv(repo_root: Optional[Path] = None,
                        sheet_name: str = "P21") -> Path:
    """
    Ensure master metadata CSV exists. Generate if missing.
    Returns the path to the metadata CSV.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    
    output_path = repo_root / "results" / "master_metadata_P21.csv"
    
    if output_path.exists():
        return output_path
    
    # Look for Excel file in multiple locations
    excel_candidates = [
        repo_root / "VASCilia_2D_Dataset.xlsx",
        repo_root / "VASCilia_3D_Dataset.xlsx",
        repo_root.parents[1] / "VASCilia_3D_Dataset.xlsx",
        repo_root.parents[2] / "VASCilia_3D_Dataset.xlsx",
    ]
    
    excel_path = None
    for candidate in excel_candidates:
        if candidate.exists():
            excel_path = candidate
            break
    
    if excel_path is None:
        raise FileNotFoundError(
            f"Excel file not found in any of these locations:\n" +
            "\n".join(str(c) for c in excel_candidates)
        )
    
    build_metadata_from_excel(
        excel_path,
        sheet_name=sheet_name,
        save_csv=output_path,
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build master metadata CSV from the VASCilia Excel workbook"
    )
    repo_root = Path(__file__).resolve().parents[1]
    
    # Find Excel file in multiple locations
    excel_candidates = [
        repo_root / "VASCilia_2D_Dataset.xlsx",
        repo_root / "VASCilia_3D_Dataset.xlsx",
        repo_root.parents[1] / "VASCilia_3D_Dataset.xlsx",
        repo_root.parents[2] / "VASCilia_3D_Dataset.xlsx",
    ]
    default_excel = None
    for candidate in excel_candidates:
        if candidate.exists():
            default_excel = candidate
            break
    
    if default_excel is None:
        default_excel = repo_root / "VASCilia_2D_Dataset.xlsx"  # fallback
    
    parser.add_argument(
        "--excel-path", "-e",
        type=Path,
        default=default_excel,
        help="Path to VASCilia_2D_Dataset.xlsx workbook",
    )
    parser.add_argument(
        "--sheet-name", "-s",
        default="P21",
        help="Excel sheet name to load",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=repo_root / "results" / "master_metadata_P21.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if not args.excel_path.exists():
        parser.error(f"Excel file not found: {args.excel_path}")

    metadata = build_metadata_from_excel(
        args.excel_path,
        sheet_name=args.sheet_name,
        save_csv=args.output,
    )
    print(f"Saved metadata CSV to {args.output} ({len(metadata)} rows)")
