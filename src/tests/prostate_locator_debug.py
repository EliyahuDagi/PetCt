"""Debug script to visualize prostate locator steps on a fixed case.

Usage:
    python -m src.tests.prostate_locator_debug

This script loads patient 3129058 from the local dataset, runs the prostate
locator, and writes step-by-step coronal debug images into debug/prostate_locator.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import numpy as np

# Make project imports work when executed as a module
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.DataViewer.model import DicomModel
from src.utils.prostate_locator import locate_prostate_bbox, _pet_bladder_guess

# Fixed patient location for the requested debug run
PATIENT_ID_DEFAULT = "3129058"
PATIENT_DIR_ROOT = ROOT / "data" / "gdrive_downloads" / "1T4LsR5QOGwwFyGnoQFCXtGBJq4qf8HO2"


def _ensure_debug_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _overlay_mask(base: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0), alpha=0.4) -> np.ndarray:
    overlay = np.stack([base, base, base], axis=-1)
    mask_bool = mask > 0
    overlay[mask_bool, 0] = base[mask_bool] * (1 - alpha) + color[0] * alpha * np.max(base)
    overlay[mask_bool, 1] = base[mask_bool] * (1 - alpha) + color[1] * alpha * np.max(base)
    overlay[mask_bool, 2] = base[mask_bool] * (1 - alpha) + color[2] * alpha * np.max(base)
    return overlay


def _draw_bbox(ax, bbox_mm: Dict[str, Tuple[float, float]], color="lime", label="bbox") -> None:
    x1, x2 = bbox_mm["x"]
    z1, z2 = bbox_mm["z"]
    width = x2 - x1
    height = z2 - z1
    rect = plt.Rectangle((x1, z1), width, height, linewidth=1.5, edgecolor=color, facecolor="none", label=label)
    ax.add_patch(rect)


def _bbox_vox_to_mm(bbox: Tuple[np.ndarray, np.ndarray], spacing: Tuple[float, float, float], origin: Tuple[float, float, float]) -> Dict[str, Tuple[float, float]]:
    z0, y0, x0 = bbox[0]
    z1, y1, x1 = bbox[1]
    return {
        "x": (float(origin[2] + x0 * spacing[2]), float(origin[2] + x1 * spacing[2])),
        "z": (float(origin[0] + z0 * spacing[0]), float(origin[0] + z1 * spacing[0])),
    }


def _extent_to_axes(extent: Tuple[float, float, float, float]) -> Dict[str, float]:
    left, right, bottom, top = extent
    return {"x_left": float(left), "x_right": float(right), "z_bottom": float(bottom), "z_top": float(top)}


def _build_segmentation_overlay(seg_slice: np.ndarray) -> np.ndarray:
    """Create RGBA overlay for all labels in a 2D slice."""
    if seg_slice is None or seg_slice.size == 0:
        return None
    labels = [int(v) for v in np.unique(seg_slice) if v != 0]
    if not labels:
        return None
    overlay = np.zeros(seg_slice.shape + (4,), dtype=float)
    for label_id in labels:
        hue = ((label_id * 37) % 360) / 360.0
        r, g, b, _ = plt.cm.hsv(hue)
        color = (float(r), float(g), float(b), 0.45)
        overlay[seg_slice == label_id] = color
    return overlay


def main(patient_id: str, coronal_mode: str = "middle"):
    patient_dir = PATIENT_DIR_ROOT / patient_id
    debug_dir = ROOT / "debug" / "prostate_locator" / patient_id

    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    _ensure_debug_dir(debug_dir)

    model = DicomModel()
    model.load_patient_data(str(patient_dir))

    labels = model.get_segmentation_label_map()
    spacing = model.get_voxel_spacing()
    origin = model.get_origin()

    # Prefer model-driven ROI (supports PET_BOX bbox-only), fallback to legacy locator if we have a mask
    roi = model.locate_prostate_roi(orientation="CORONAL")
    bbox = None
    if roi and roi.get("bounds"):
        # reconstruct bbox_mm for coronal: bounds = [x1, x2, z2, z1]
        b = roi["bounds"]
        bbox = {
            "method": roi.get("method", "model_roi"),
            "bbox_mm": {"x": [float(b[0]), float(b[1])], "z": [float(b[3]), float(b[2])]},
            "bbox_vox": None,
            "center_vox": None,
            "center_mm": None,
        }
    elif model.segmentation_mask is not None:
        bbox = locate_prostate_bbox(
            mask=model.segmentation_mask,
            labels=labels,
            spacing=spacing,
            origin=origin,
            pet_volume=model.pet_volume,
            ct_volume=model.ct_volume,
            debug=True,
        )
    elif model.pet_volume is not None:
        # PET-only fallback: approximate bbox via PET glow
        hot = _pet_bladder_guess(model.pet_volume)
        if hot:
            (z0, y0, x0), (z1, y1, x1) = hot
            bbox = {
                "method": "pet_hotspot_only",
                "bbox_mm": {
                    "z": [float(origin[0] + z0 * spacing[0]), float(origin[0] + z1 * spacing[0])],
                    "y": [float(origin[1] + y0 * spacing[1]), float(origin[1] + y1 * spacing[1])],
                    "x": [float(origin[2] + x0 * spacing[2]), float(origin[2] + x1 * spacing[2])],
                },
                "bbox_vox": {"z": [z0, z1], "y": [y0, y1], "x": [x0, x1]},
            }

    if not bbox:
        raise RuntimeError("Prostate locator failed to produce a bbox (no mask or PET hotspot).")

    bladder_id = next((k for k, v in labels.items() if "bladder" in v.lower()), None) if model.segmentation_mask is not None else None
    bladder_bbox = None
    if bladder_id is not None and model.segmentation_mask is not None:
        bladder_mask_full = (model.segmentation_mask == bladder_id).astype(np.uint8)
        coords = np.argwhere(bladder_mask_full > 0)
        if coords.size > 0:
            bladder_bbox = (coords.min(axis=0), coords.max(axis=0))

    bladder_bbox_mm = _bbox_vox_to_mm(bladder_bbox, spacing, origin) if bladder_bbox is not None else None

    # Choose coronal slice: viewer-like middle by default; optional bbox center if requested
    coronal_middle = model.get_slice_count("CORONAL") // 2
    coronal_idx = coronal_middle
    if coronal_mode == "bbox" and bladder_bbox is not None:
        coronal_idx = int((bladder_bbox[0][1] + bladder_bbox[1][1]) // 2)
    if roi and roi.get("center_slice") is not None:
        coronal_idx = int(roi["center_slice"])

    ct_slice_bladder, _, seg_slice_bladder = model.get_images(coronal_idx, orientation="CORONAL")
    ct_extent = model.get_bounds("CORONAL")
    pet_extent = model.get_pet_bounds("CORONAL") or ct_extent

    bladder_mask_slice = (seg_slice_bladder == bladder_id).astype(np.uint8) if (bladder_id is not None and seg_slice_bladder is not None) else None

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.imshow(ct_slice_bladder, cmap="gray", extent=ct_extent, origin="upper")
    if bladder_mask_slice is not None:
        ax1.imshow(bladder_mask_slice, cmap="autumn", alpha=0.45, extent=ct_extent, origin="upper")
    if bladder_bbox_mm is not None:
        _draw_bbox(ax1, bladder_bbox_mm, color="cyan", label="bbox_seg")
    ax1.set_title(f"Coronal CT + bladder mask (slice {coronal_idx})")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("z (mm)")
    ax1.set_aspect("equal")
    fig1.savefig(debug_dir / "step1_bladder_mask.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Prostate-centric coronal slice (viewer-style selection)
    ct_slice, pet_slice, seg_slice = model.get_images(coronal_idx, orientation="CORONAL")

    # Step 2a: Save raw coronal CT/PET for this slice
    fig2a, ax2a = plt.subplots(figsize=(8, 6))
    ax2a.imshow(ct_slice, cmap="gray", extent=ct_extent, origin="upper")
    if bbox.get("bbox_mm"):
        _draw_bbox(ax2a, bbox["bbox_mm"], color="lime", label=bbox.get("method", "roi"))
        ax2a.legend(loc="lower right")
    ax2a.set_title(f"Coronal CT (slice {coronal_idx})")
    ax2a.set_xlabel("x (mm)")
    ax2a.set_ylabel("z (mm)")
    ax2a.set_aspect("equal")
    fig2a.savefig(debug_dir / "step2a_coronal_ct.png", dpi=200, bbox_inches="tight")
    plt.close(fig2a)

    fig2b, ax2b = plt.subplots(figsize=(8, 6))
    if pet_slice is not None:
        ax2b.imshow(pet_slice, cmap="hot", extent=pet_extent, origin="upper")
    if bbox.get("bbox_mm"):
        _draw_bbox(ax2b, bbox["bbox_mm"], color="lime", label=bbox.get("method", "roi"))
        ax2b.legend(loc="lower right")
    ax2b.set_title(f"Coronal PET (slice {coronal_idx})")
    ax2b.set_xlabel("x (mm)")
    ax2b.set_ylabel("z (mm)")
    ax2b.set_aspect("equal")
    fig2b.savefig(debug_dir / "step2b_coronal_pet.png", dpi=200, bbox_inches="tight")
    plt.close(fig2b)

    # Step 2c: Fusion-style overlay (CT base with PET overlay) mimicking viewer
    fig2c, ax2c = plt.subplots(figsize=(8, 6))
    ax2c.imshow(ct_slice, cmap="gray", extent=ct_extent, origin="upper")
    if pet_slice is not None:
        ax2c.imshow(pet_slice, cmap="hot", alpha=0.35, extent=pet_extent, origin="upper")
    if bbox.get("bbox_mm"):
        _draw_bbox(ax2c, bbox["bbox_mm"], color="lime", label=bbox.get("method", "roi"))
        ax2c.legend(loc="lower right")
    ax2c.set_title(f"Coronal Fusion (CT+PET) slice {coronal_idx}")
    ax2c.set_xlabel("x (mm)")
    ax2c.set_ylabel("z (mm)")
    ax2c.set_aspect("equal")
    fig2c.savefig(debug_dir / "step2c_coronal_fusion.png", dpi=200, bbox_inches="tight")
    plt.close(fig2c)

    # Step 2: PET hot spot threshold (percentile) and component mask
    hot_bbox = _pet_bladder_guess(model.pet_volume) if model.pet_volume is not None else None
    hot_mask_slice = np.zeros_like(pet_slice, dtype=np.uint8) if pet_slice is not None else None
    if hot_bbox is not None and model.pet_volume is not None:
        (z_min, y_min, x_min), (z_max, y_max, x_max) = hot_bbox
        # Build 3D hot mask to extract coronal slice
        hot_mask = np.zeros_like(model.pet_volume, dtype=np.uint8)
        hot_mask[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = 1
        pet_coronal_idx = min(coronal_idx, hot_mask.shape[1] - 1)
        hot_mask_slice = hot_mask[:, pet_coronal_idx, :]
        hot_mask_slice = np.flipud(hot_mask_slice)  # match get_images coronal flip

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    if pet_slice is not None:
        ax2.imshow(pet_slice, cmap="hot", extent=pet_extent, origin="upper")
    if hot_mask_slice is not None:
        ax2.imshow(hot_mask_slice, cmap="winter", alpha=0.35, extent=pet_extent, origin="upper")
    if hot_bbox is not None:
        hot_bbox_mm = {
            "x": [float(origin[2] + hot_bbox[0][2] * spacing[2]), float(origin[2] + hot_bbox[1][2] * spacing[2])],
            "z": [float(origin[0] + hot_bbox[0][0] * spacing[0]), float(origin[0] + hot_bbox[1][0] * spacing[0])],
        }
        _draw_bbox(ax2, hot_bbox_mm, color="cyan", label="pet_hot")
    ax2.set_title(f"Coronal PET + hot mask (slice {coronal_idx})")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("z (mm)")
    ax2.set_aspect("equal")
    ax2.legend(loc="lower right")
    fig2.savefig(debug_dir / "step2_pet_hot.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Step 3: CT with prostate ROI bbox only
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.imshow(ct_slice, cmap="gray", extent=ct_extent, origin="upper")
    # Also overlay bladder mask on the prostate ROI slice for reference
    if bladder_id is not None and seg_slice is not None:
        bladder_mask_slice_roi = (seg_slice == bladder_id).astype(np.uint8)
        ax3.imshow(bladder_mask_slice_roi, cmap="autumn", alpha=0.35, extent=ct_extent, origin="upper")
    if bbox.get("bbox_mm"):
        _draw_bbox(ax3, bbox["bbox_mm"], color="lime", label=bbox.get("method", "roi"))
    ax3.set_title(f"Prostate ROI (method={bbox.get('method')})")
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("z (mm)")
    ax3.set_aspect("equal")
    ax3.legend(loc="lower right")
    fig3.savefig(debug_dir / "step3_prostate_roi.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # Step 4: Full segmentation overlay on the coronal slice to check alignment
    seg_overlay = _build_segmentation_overlay(seg_slice) if seg_slice is not None else None
    if seg_overlay is not None:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.imshow(ct_slice, cmap="gray", extent=ct_extent, origin="upper")
        ax4.imshow(seg_overlay, extent=ct_extent, origin="upper")
        ax4.set_title(f"Coronal CT + all seg labels (slice {coronal_idx})")
        ax4.set_xlabel("x (mm)")
        ax4.set_ylabel("z (mm)")
        ax4.set_aspect("equal")
        fig4.savefig(debug_dir / "step4_seg_overlay.png", dpi=200, bbox_inches="tight")
        plt.close(fig4)

    # Persist numeric debug metadata for quick inspection and measurement
    summary = {
        "patient_id": patient_id,
        "coronal_slice_index_vox": int(coronal_idx),
        "method": bbox.get("method"),
        "spacing_mm": {
            "z": float(spacing[0]),
            "y": float(spacing[1]),
            "x": float(spacing[2]),
        },
        "origin_mm": {
            "z": float(origin[0]),
            "y": float(origin[1]),
            "x": float(origin[2]),
        },
        "shapes_vox": {
            "ct": list(model.ct_volume.shape) if model.ct_volume is not None else None,
            "pet": list(model.pet_volume.shape) if model.pet_volume is not None else None,
            "seg": list(model.segmentation_mask.shape) if model.segmentation_mask is not None else None,
        },
        "coronal_extent_ct_mm": _extent_to_axes(ct_extent),
        "coronal_extent_pet_mm": _extent_to_axes(pet_extent) if pet_extent is not None else None,
        "coronal_extent_ct_vox": {"x": [0, int(model.ct_volume.shape[2] - 1)], "z": [0, int(model.ct_volume.shape[0] - 1)]} if model.ct_volume is not None else None,
        "coronal_extent_pet_vox": {"x": [0, int(model.pet_volume.shape[2] - 1)], "z": [0, int(model.pet_volume.shape[0] - 1)]} if model.pet_volume is not None else None,
        "bbox_prostate_vox": bbox.get("bbox_vox"),
        "bbox_prostate_mm": bbox.get("bbox_mm"),
        "center_vox": bbox.get("center_vox"),
        "center_mm": bbox.get("center_mm"),
        "roi_method": bbox.get("method"),
        "bbox_bladder_vox": {
            "z": [int(bladder_bbox[0][0]), int(bladder_bbox[1][0])],
            "y": [int(bladder_bbox[0][1]), int(bladder_bbox[1][1])],
            "x": [int(bladder_bbox[0][2]), int(bladder_bbox[1][2])],
        } if bladder_bbox is not None else None,
        "bbox_bladder_mm": bladder_bbox_mm,
        "bbox_pet_hot_vox": {
            "z": [int(hot_bbox[0][0]), int(hot_bbox[1][0])],
            "y": [int(hot_bbox[0][1]), int(hot_bbox[1][1])],
            "x": [int(hot_bbox[0][2]), int(hot_bbox[1][2])],
        } if hot_bbox is not None else None,
        "bbox_pet_hot_mm": {
            "x": [float(origin[2] + hot_bbox[0][2] * spacing[2]), float(origin[2] + hot_bbox[1][2] * spacing[2])],
            "z": [float(origin[0] + hot_bbox[0][0] * spacing[0]), float(origin[0] + hot_bbox[1][0] * spacing[0])],
        } if hot_bbox is not None else None,
        "locator_debug": bbox.get("debug"),
    }

    with open(debug_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Debug images and summary saved to {debug_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default=PATIENT_ID_DEFAULT, help="Patient folder name under gdrive_downloads")
    parser.add_argument("--coronal_mode", choices=["middle", "bbox"], default="middle", help="Coronal slice selection mode: middle (viewer-like) or bbox center")
    args = parser.parse_args()
    main(patient_id=args.patient, coronal_mode=args.coronal_mode)
