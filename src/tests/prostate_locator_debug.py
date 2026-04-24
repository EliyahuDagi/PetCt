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


def _draw_band(ax, z_band_mm: Tuple[float, float], x_extent: Tuple[float, float], color="orange", label="band") -> None:
    x1, x2 = x_extent
    z1, z2 = z_band_mm
    width = x2 - x1
    height = z2 - z1
    rect = plt.Rectangle((x1, z1), width, height, linewidth=1.0, linestyle="--", edgecolor=color, facecolor="none", label=label)
    ax.add_patch(rect)


def _pick_urinary_bladder_id(labels: Dict[int, str]) -> Optional[int]:
    if not labels:
        return None

    def _find_by_needles(needles, exclude=None):
        for needle in needles:
            needle_l = str(needle).lower()
            for idx, name in labels.items():
                name_l = str(name).lower()
                if needle_l in name_l:
                    if exclude and any(ex in name_l for ex in exclude):
                        continue
                    try:
                        return int(idx)
                    except Exception:
                        return idx
        return None

    # Prefer urinary bladder first; only then fallback to a generic bladder (excluding gallbladder).
    hit = _find_by_needles(["urinary_bladder", "urinary bladder"])
    if hit is not None:
        return hit
    return _find_by_needles(["bladder"], exclude=["gallbladder", "gall bladder"])


def _flip_z_mm_for_coronal_display(z_mm: float, extent: Optional[Tuple[float, float, float, float]]) -> float:
    if extent is None:
        return float(z_mm)
    z_bottom = float(extent[2])
    z_top = float(extent[3])
    return float(z_top + z_bottom - float(z_mm))


def _flip_bbox_z_for_coronal_display(
    bbox_mm: Optional[Dict[str, Tuple[float, float]]],
    extent: Optional[Tuple[float, float, float, float]],
) -> Optional[Dict[str, Tuple[float, float]]]:
    if not bbox_mm:
        return None
    try:
        x1, x2 = bbox_mm["x"]
        z1, z2 = bbox_mm["z"]
    except Exception:
        return bbox_mm

    z1_f = _flip_z_mm_for_coronal_display(float(z1), extent)
    z2_f = _flip_z_mm_for_coronal_display(float(z2), extent)
    return {
        "x": (float(x1), float(x2)),
        "z": (float(min(z1_f, z2_f)), float(max(z1_f, z2_f))),
    }


def _flip_band_z_for_coronal_display(
    z_band_mm: Tuple[float, float],
    extent: Optional[Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    z0, z1 = float(z_band_mm[0]), float(z_band_mm[1])
    z0_f = _flip_z_mm_for_coronal_display(z0, extent)
    z1_f = _flip_z_mm_for_coronal_display(z1, extent)
    return float(min(z0_f, z1_f)), float(max(z0_f, z1_f))


def _bbox_vox_to_mm(bbox: Tuple[np.ndarray, np.ndarray], spacing: Tuple[float, float, float], origin: Tuple[float, float, float]) -> Dict[str, Tuple[float, float]]:
    z0, y0, x0 = bbox[0]
    z1, y1, x1 = bbox[1]
    return {
        "x": (float(origin[2] + x0 * spacing[2]), float(origin[2] + x1 * spacing[2])),
        "z": (float(origin[0] + z0 * spacing[0]), float(origin[0] + z1 * spacing[0])),
    }


def _bbox_dict_vox_to_mm(bbox: Dict[str, Tuple[int, int]], spacing: Tuple[float, float, float], origin: Tuple[float, float, float]) -> Optional[Dict[str, Tuple[float, float]]]:
    if not bbox:
        return None
    try:
        z0, z1 = bbox["z"]
        x0, x1 = bbox["x"]
    except Exception:
        return None
    return {
        "x": (float(origin[2] + x0 * spacing[2]), float(origin[2] + x1 * spacing[2])),
        "z": (float(origin[0] + z0 * spacing[0]), float(origin[0] + z1 * spacing[0])),
    }


def _extract_landmarks_mm(seg_debug: object, spacing: Tuple[float, float, float], origin: Tuple[float, float, float]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    if not isinstance(seg_debug, dict):
        return {}
    if "landmarks_mm" in seg_debug and isinstance(seg_debug.get("landmarks_mm"), dict):
        return seg_debug.get("landmarks_mm") or {}
    vox = seg_debug.get("landmarks_vox") if isinstance(seg_debug, dict) else None
    if not isinstance(vox, dict):
        return {}
    result = {}
    for name, bbox in vox.items():
        mm = _bbox_dict_vox_to_mm(bbox, spacing, origin)
        if mm:
            result[name] = mm
    return result


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


def _legend_if_any(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8)


def _bbox_mm_to_coronal_xz(bbox_mm: Optional[Dict[str, Tuple[float, float]]]) -> Optional[Dict[str, Tuple[float, float]]]:
    if not isinstance(bbox_mm, dict):
        return None
    if "x" not in bbox_mm or "z" not in bbox_mm:
        return None
    try:
        x0, x1 = bbox_mm["x"]
        z0, z1 = bbox_mm["z"]
    except Exception:
        return None
    return {
        "x": (float(min(x0, x1)), float(max(x0, x1))),
        "z": (float(min(z0, z1)), float(max(z0, z1))),
    }


def _vox_half_open_box_to_mm_xz(
    z_range: Tuple[int, int],
    x_range: Tuple[int, int],
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
) -> Dict[str, Tuple[float, float]]:
    z0, z1_ex = int(z_range[0]), int(z_range[1])
    x0, x1_ex = int(x_range[0]), int(x_range[1])
    z1_in = max(z0, z1_ex - 1)
    x1_in = max(x0, x1_ex - 1)
    return {
        "x": (
            float(origin[2] + x0 * spacing[2]),
            float(origin[2] + x1_in * spacing[2]),
        ),
        "z": (
            float(origin[0] + z0 * spacing[0]),
            float(origin[0] + z1_in * spacing[0]),
        ),
    }


def main(patient_id: str, coronal_mode: str = "middle"):
    patient_dir = PATIENT_DIR_ROOT / patient_id
    debug_dir = ROOT / "debug" / "prostate_locator" / patient_id

    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    _ensure_debug_dir(debug_dir)

    model = DicomModel()
    model.load_patient_data(str(patient_dir), segmentation_name='TotalSegmentor')

    labels = model.get_segmentation_label_map()
    spacing = model.get_voxel_spacing()
    origin = model.get_origin()
    pet_spacing, pet_origin = model._get_pet_spacing_origin()

    seg_debug = getattr(model, "segmentation_debug", None)

    ct_aligned = (
        model.ct_volume is not None
        and model.segmentation_mask is not None
        and model.ct_volume.shape == model.segmentation_mask.shape
    )
    pet_aligned = (
        model.pet_volume is not None
        and model.segmentation_mask is not None
        and model.pet_volume.shape == model.segmentation_mask.shape
    )

    # Always compute locator debug payload when mask exists so debug images reflect
    # the exact landmark/pelvic-band/CT-refinement internals from locate_prostate_bbox.
    locator_bbox_debug = None
    if model.segmentation_mask is not None:
        locator_bbox_debug = locate_prostate_bbox(
            mask=model.segmentation_mask,
            labels=labels,
            spacing=spacing,
            origin=origin,
            pet_volume=model.pet_volume,
            pet_spacing=pet_spacing,
            pet_origin=pet_origin,
            ct_volume=model.ct_volume,
            ct_spacing=model.get_voxel_spacing(),
            ct_origin=model.get_origin(),
            debug=True,
        )

    # Prefer model-driven ROI (supports PET_BOX bbox-only), fallback to locator when needed
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
    elif locator_bbox_debug is not None:
        bbox = locator_bbox_debug
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

    if bbox.get("debug"):
        seg_debug = bbox.get("debug")
    elif isinstance(locator_bbox_debug, dict) and locator_bbox_debug.get("debug"):
        seg_debug = locator_bbox_debug.get("debug")

    bladder_id = _pick_urinary_bladder_id(labels) if model.segmentation_mask is not None else None
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

    # Prostate-centric coronal slice (viewer-style selection)
    ct_slice, pet_slice, seg_slice = model.get_images(coronal_idx, orientation="CORONAL")

    locator_debug = None
    if isinstance(locator_bbox_debug, dict) and isinstance(locator_bbox_debug.get("debug"), dict):
        locator_debug = locator_bbox_debug.get("debug")
    elif isinstance(bbox, dict):
        locator_debug = bbox.get("debug")
    landmarks_mm = _extract_landmarks_mm(locator_debug, spacing, origin) if isinstance(locator_debug, dict) else {}

    # Step 1: Segmentation landmarks and organs used
    fig1, ax1 = plt.subplots(figsize=(8, 10))
    ax1.imshow(ct_slice, cmap="gray", extent=ct_extent, origin="upper")

    landmark_colors = {
        "bladder": "cyan",
        "vesicles": "yellow",
        "rectum": "deepskyblue",
        "colon": "orange",
        "femoral_head_l": "magenta",
        "femoral_head_r": "magenta",
    }
    organ_lines = []
    label_ids = locator_debug.get("label_ids") if isinstance(locator_debug, dict) else None
    for organ_name in ["bladder", "vesicles", "rectum", "colon", "femoral_head_l", "femoral_head_r"]:
        mm_box = _bbox_mm_to_coronal_xz(landmarks_mm.get(organ_name))
        if mm_box is not None:
            mm_box_plot = _flip_bbox_z_for_coronal_display(mm_box, ct_extent)
            _draw_bbox(
                ax1,
                mm_box_plot,
                color=landmark_colors.get(organ_name, "white"),
                label=organ_name,
            )
            organ_lines.append(f"{organ_name}: used")
        else:
            organ_lines.append(f"{organ_name}: missing")

    if isinstance(label_ids, dict):
        organ_lines.append(f"ids bladder={label_ids.get('bladder')} rectum={label_ids.get('rectum')}")
        colon_ids = label_ids.get("colon")
        if isinstance(colon_ids, list):
            organ_lines.append(f"ids colon={colon_ids}")

    ax1.text(
        0.02,
        0.98,
        "organs used for segmentation landmarks\n" + "\n".join(organ_lines),
        transform=ax1.transAxes,
        color="white",
        fontsize=8,
        ha="left",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
    )
    ax1.set_title(f"Step 1 - Segmentation landmarks (slice {coronal_idx})")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("z (mm)")
    ax1.set_aspect("equal")
    _legend_if_any(ax1)
    fig1.savefig(debug_dir / "step1_segmentation_landmarks.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Step 2: Added pelvic z-band
    fig2, ax2 = plt.subplots(figsize=(8, 10))
    ax2.imshow(ct_slice, cmap="gray", extent=ct_extent, origin="upper")
    if pet_slice is not None:
        ax2.imshow(pet_slice, cmap="hot", alpha=0.25, extent=pet_extent, origin="upper")

    band_mm = None
    band_source = None
    if isinstance(locator_debug, dict):
        band_source = locator_debug.get("pelvic_band_source")
        if isinstance(locator_debug.get("pelvic_band_mm"), list) and len(locator_debug["pelvic_band_mm"]) == 2:
            band_mm = (float(locator_debug["pelvic_band_mm"][0]), float(locator_debug["pelvic_band_mm"][1]))
    if band_mm is not None:
        band_mm_plot = _flip_band_z_for_coronal_display(band_mm, ct_extent)
        _draw_band(ax2, band_mm_plot, (ct_extent[0], ct_extent[1]), color="orange", label="pelvic_band")

    if bbox.get("bbox_mm"):
        prostate_plot = _flip_bbox_z_for_coronal_display(bbox["bbox_mm"], ct_extent)
        _draw_bbox(ax2, prostate_plot, color="lime", label="final_prostate_bbox")

    text_band = f"z-band source: {band_source}" if band_source else "z-band source: n/a"
    ax2.text(
        0.02,
        0.98,
        text_band,
        transform=ax2.transAxes,
        color="orange",
        fontsize=9,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.45, "edgecolor": "none"},
    )
    ax2.set_title(f"Step 2 - Added pelvic z-band (slice {coronal_idx})")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("z (mm)")
    ax2.set_aspect("equal")
    _legend_if_any(ax2)
    fig2.savefig(debug_dir / "step2_pelvic_z_band.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Step 3: CT refinement search window and selected CT result
    if isinstance(locator_debug, dict) and isinstance(locator_debug.get("pelvic_refinement"), dict):
        pelvic_ref = locator_debug["pelvic_refinement"]
        trials = pelvic_ref.get("trials") if isinstance(pelvic_ref.get("trials"), list) else []
        chosen_trial = None
        for trial in trials:
            if isinstance(trial, dict) and trial.get("success"):
                chosen_trial = trial
                break
        if chosen_trial is None and trials:
            chosen_trial = trials[0] if isinstance(trials[0], dict) else None

        if isinstance(chosen_trial, dict):
            ranges = chosen_trial.get("ranges_clamped")
            ranges_source = "clamped"
            if not (isinstance(ranges, dict) and all(k in ranges for k in ("z", "y", "x"))):
                z_req = chosen_trial.get("z_range_requested")
                y_req = pelvic_ref.get("y_range_requested")
                x_req = pelvic_ref.get("x_range_requested")
                if (
                    isinstance(z_req, list)
                    and len(z_req) == 2
                    and isinstance(y_req, list)
                    and len(y_req) == 2
                    and isinstance(x_req, list)
                    and len(x_req) == 2
                ):
                    ranges = {
                        "z": [int(z_req[0]), int(z_req[1])],
                        "y": [int(y_req[0]), int(y_req[1])],
                        "x": [int(x_req[0]), int(x_req[1])],
                    }
                    ranges_source = "requested"

            if isinstance(ranges, dict) and all(k in ranges for k in ("z", "y", "x")):
                y0_raw, y1_raw = int(ranges["y"][0]), int(ranges["y"][1])
                y0, y1 = (min(y0_raw, y1_raw), max(y0_raw, y1_raw))
                ct_ref_idx = int(max(0, min(model.get_slice_count("CORONAL") - 1, (y0 + max(y0 + 1, y1) - 1) // 2)))
                ct_slice_ref, pet_slice_ref, _ = model.get_images(ct_ref_idx, orientation="CORONAL")

                fig3, ax3 = plt.subplots(figsize=(8, 10))
                ax3.imshow(ct_slice_ref, cmap="gray", extent=ct_extent, origin="upper")
                if pet_slice_ref is not None:
                    ax3.imshow(pet_slice_ref, cmap="hot", alpha=0.18, extent=pet_extent, origin="upper")

                search_mm = _vox_half_open_box_to_mm_xz(
                    (int(ranges["z"][0]), int(ranges["z"][1])),
                    (int(ranges["x"][0]), int(ranges["x"][1])),
                    spacing,
                    origin,
                )
                search_mm_plot = _flip_bbox_z_for_coronal_display(search_mm, ct_extent)
                _draw_bbox(ax3, search_mm_plot, color="yellow", label=f"ct_search_{chosen_trial.get('tag', 'trial')}")

                com_global = chosen_trial.get("com_global")
                if isinstance(com_global, list) and len(com_global) == 3:
                    z_mm = float(origin[0] + float(com_global[0]) * spacing[0])
                    x_mm = float(origin[2] + float(com_global[2]) * spacing[2])
                    z_mm_plot = _flip_z_mm_for_coronal_display(z_mm, ct_extent)
                    ax3.scatter([x_mm], [z_mm_plot], c="red", s=36, marker="x", label="ct_component_com")

                final_centroid = locator_debug.get("final_centroid_vox")
                if isinstance(final_centroid, list) and len(final_centroid) == 3:
                    z_mm = float(origin[0] + float(final_centroid[0]) * spacing[0])
                    x_mm = float(origin[2] + float(final_centroid[2]) * spacing[2])
                    z_mm_plot = _flip_z_mm_for_coronal_display(z_mm, ct_extent)
                    ax3.scatter([x_mm], [z_mm_plot], c="lime", s=28, marker="o", label="final_centroid")

                method_text = str(locator_debug.get("final_method", bbox.get("method", "unknown")))
                ax3.text(
                    0.02,
                    0.98,
                    f"trial={chosen_trial.get('tag', 'n/a')} success={bool(chosen_trial.get('success', False))} ({ranges_source})\n"
                    f"method={method_text}\n"
                    f"fail_reason={chosen_trial.get('fail_reason', 'n/a')}",
                    transform=ax3.transAxes,
                    color="white",
                    fontsize=8,
                    ha="left",
                    va="top",
                    bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
                )
                ax3.set_title(f"Step 3 - CT refinement result (slice {ct_ref_idx})")
                ax3.set_xlabel("x (mm)")
                ax3.set_ylabel("z (mm)")
                ax3.set_aspect("equal")
                _legend_if_any(ax3)
                fig3.savefig(debug_dir / "step3_ct_refinement.png", dpi=200, bbox_inches="tight")
                plt.close(fig3)

                # Step 3b: Detailed thresholding/mask view for CT refinement on the selected coronal slice
                if (
                    model.ct_volume is not None
                    and model.segmentation_mask is not None
                    and bool(ct_aligned)
                    and isinstance(pelvic_ref.get("ct_soft_tissue_hu"), list)
                    and len(pelvic_ref.get("ct_soft_tissue_hu")) == 2
                ):
                    hu_lo = float(pelvic_ref["ct_soft_tissue_hu"][0])
                    hu_hi = float(pelvic_ref["ct_soft_tissue_hu"][1])

                    z0, z1 = int(ranges["z"][0]), int(ranges["z"][1])
                    x0, x1 = int(ranges["x"][0]), int(ranges["x"][1])
                    y_idx = int(ct_ref_idx)

                    z0 = max(0, min(z0, model.ct_volume.shape[0]))
                    z1 = max(z0, min(z1, model.ct_volume.shape[0]))
                    x0 = max(0, min(x0, model.ct_volume.shape[2]))
                    x1 = max(x0, min(x1, model.ct_volume.shape[2]))

                    # Build masks in raw coronal plane (Z,X), then flip vertically to match
                    # DicomModel.get_images(..., orientation="CORONAL") display orientation.
                    ct_plane_raw = np.asarray(model.ct_volume[:, y_idx, :], dtype=float)
                    seg_plane_raw = np.asarray(model.segmentation_mask[:, y_idx, :], dtype=int)

                    roi_plane = np.zeros_like(ct_plane_raw, dtype=bool)
                    if z1 > z0 and x1 > x0:
                        roi_plane[z0:z1, x0:x1] = True

                    soft_plane = (ct_plane_raw >= hu_lo) & (ct_plane_raw <= hu_hi) & roi_plane

                    exclude_ids = pelvic_ref.get("exclude_ids") if isinstance(pelvic_ref.get("exclude_ids"), list) else []
                    exclude_plane = np.zeros_like(ct_plane_raw, dtype=bool)
                    for lab_id in exclude_ids:
                        try:
                            exclude_plane |= (seg_plane_raw == int(lab_id))
                        except Exception:
                            pass
                    exclude_plane &= roi_plane

                    candidate_plane = soft_plane & (~exclude_plane)

                    pet_thresh = chosen_trial.get("pet_thresh") if isinstance(chosen_trial, dict) else None
                    hot_plane = None
                    refined_plane = candidate_plane.copy()
                    if pet_thresh is not None and model.pet_volume is not None and bool(pet_aligned):
                        pet_plane = np.asarray(model.pet_volume[:, y_idx, :], dtype=float)
                        hot_plane = (pet_plane >= float(pet_thresh)) & roi_plane
                        refined_plane = candidate_plane & hot_plane

                    # Viewer-aligned display planes (CORONAL uses np.flipud)
                    ct_plane_disp = (
                        np.asarray(ct_slice_ref, dtype=float)
                        if ct_slice_ref is not None
                        else np.flipud(ct_plane_raw)
                    )
                    soft_plane_disp = np.flipud(soft_plane)
                    exclude_plane_disp = np.flipud(exclude_plane)
                    candidate_plane_disp = np.flipud(candidate_plane)
                    refined_plane_disp = np.flipud(refined_plane)
                    hot_plane_disp = np.flipud(hot_plane) if hot_plane is not None else None

                    fig3b, axes = plt.subplots(2, 2, figsize=(12, 12))
                    ax_a, ax_b = axes[0]
                    ax_c, ax_d = axes[1]

                    # A) CT + search ROI
                    ax_a.imshow(ct_plane_disp, cmap="gray", extent=ct_extent, origin="upper")
                    search_mm_plot = _flip_bbox_z_for_coronal_display(search_mm, ct_extent)
                    _draw_bbox(ax_a, search_mm_plot, color="yellow", label="ct_search_window")
                    ax_a.set_title("CT with refinement search window")
                    ax_a.set_xlabel("x (mm)")
                    ax_a.set_ylabel("z (mm)")
                    ax_a.set_aspect("equal")
                    _legend_if_any(ax_a)

                    # B) Soft-tissue HU mask
                    ax_b.imshow(ct_plane_disp, cmap="gray", extent=ct_extent, origin="upper")
                    ax_b.imshow(np.ma.masked_where(~soft_plane_disp, soft_plane_disp), cmap="Blues", alpha=0.55, extent=ct_extent, origin="upper")
                    ax_b.set_title(f"Soft tissue mask: {hu_lo:.1f} <= HU <= {hu_hi:.1f}")
                    ax_b.set_xlabel("x (mm)")
                    ax_b.set_ylabel("z (mm)")
                    ax_b.set_aspect("equal")

                    # C) Candidate after exclusion labels
                    ax_c.imshow(ct_plane_disp, cmap="gray", extent=ct_extent, origin="upper")
                    ax_c.imshow(np.ma.masked_where(~exclude_plane_disp, exclude_plane_disp), cmap="Reds", alpha=0.45, extent=ct_extent, origin="upper")
                    ax_c.imshow(np.ma.masked_where(~candidate_plane_disp, candidate_plane_disp), cmap="Greens", alpha=0.45, extent=ct_extent, origin="upper")
                    ax_c.set_title("Candidate mask (green) after organ exclusion (red)")
                    ax_c.set_xlabel("x (mm)")
                    ax_c.set_ylabel("z (mm)")
                    ax_c.set_aspect("equal")

                    # D) PET-hot refinement (if used) or final refined mask
                    ax_d.imshow(ct_plane_disp, cmap="gray", extent=ct_extent, origin="upper")
                    if hot_plane_disp is not None:
                        ax_d.imshow(np.ma.masked_where(~hot_plane_disp, hot_plane_disp), cmap="magma", alpha=0.40, extent=ct_extent, origin="upper")
                        ax_d.imshow(np.ma.masked_where(~refined_plane_disp, refined_plane_disp), cmap="spring", alpha=0.55, extent=ct_extent, origin="upper")
                        ax_d.set_title(f"PET hot (magma) + refined mask (spring), PET thresh={float(pet_thresh):.2f}")
                    else:
                        ax_d.imshow(np.ma.masked_where(~refined_plane_disp, refined_plane_disp), cmap="spring", alpha=0.55, extent=ct_extent, origin="upper")
                        ax_d.set_title("Refined mask (no PET threshold used)")
                    ax_d.set_xlabel("x (mm)")
                    ax_d.set_ylabel("z (mm)")
                    ax_d.set_aspect("equal")

                    details_text = (
                        f"trial={chosen_trial.get('tag', 'n/a')} success={bool(chosen_trial.get('success', False))}\n"
                        f"fail_reason={chosen_trial.get('fail_reason', 'n/a')}  ranges_source={ranges_source}\n"
                        f"ranges_y_requested={ranges.get('y', 'n/a')}\n"
                        f"ct_hu=[{hu_lo:.1f}, {hu_hi:.1f}]  exclude_ids={exclude_ids}\n"
                        f"soft_vox={chosen_trial.get('soft_voxels', 'n/a')}  excluded={chosen_trial.get('exclude_voxels', 'n/a')}\n"
                        f"candidate={chosen_trial.get('candidate_voxels', 'n/a')}  refined={chosen_trial.get('refined_voxels', 'n/a')}\n"
                        f"largest_component={chosen_trial.get('largest_component_voxels', 'n/a')}"
                    )
                    fig3b.suptitle("Step 3b - CT prostate refinement details", fontsize=12)
                    fig3b.text(
                        0.02,
                        0.01,
                        details_text,
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
                    fig3b.savefig(debug_dir / "step3b_ct_refinement_details.png", dpi=200, bbox_inches="tight")
                    plt.close(fig3b)

    # Single fusion view with all bboxes (ROI, bladder, pelvic band, PET hot)
    fig_bbox, ax_bbox = plt.subplots(figsize=(8, 10))
    ax_bbox.imshow(ct_slice, cmap="gray", extent=ct_extent, origin="upper")
    if pet_slice is not None:
        ax_bbox.imshow(pet_slice, cmap="hot", alpha=0.35, extent=pet_extent, origin="upper")
    legends = []

    if bbox.get("bbox_mm"):
        bbox_mm_plot = _flip_bbox_z_for_coronal_display(bbox["bbox_mm"], ct_extent)
        _draw_bbox(ax_bbox, bbox_mm_plot, color="lime", label=bbox.get("method", "roi"))
        legends.append(bbox.get("method", "roi"))

    if bladder_bbox_mm is not None:
        bladder_bbox_mm_plot = _flip_bbox_z_for_coronal_display(bladder_bbox_mm, ct_extent)
        _draw_bbox(ax_bbox, bladder_bbox_mm_plot, color="cyan", label="bladder_bbox")
        legends.append("bladder_bbox")

    if seg_debug:
        band_mm = None
        band_x_extent = None
        band_extent_ref = None
        if isinstance(seg_debug, dict):
            if seg_debug.get("pelvic_band_mm"):
                band_mm = tuple(seg_debug["pelvic_band_mm"])
                band_x_extent = (ct_extent[0], ct_extent[1]) if ct_extent else None
                band_extent_ref = ct_extent
            elif seg_debug.get("pelvic_band_pet_mm") and pet_extent:
                band_mm = tuple(seg_debug["pelvic_band_pet_mm"])
                band_x_extent = (pet_extent[0], pet_extent[1])
                band_extent_ref = pet_extent
        if band_mm and band_x_extent and band_extent_ref is not None:
            band_mm_plot = _flip_band_z_for_coronal_display((float(band_mm[0]), float(band_mm[1])), band_extent_ref)
            _draw_band(ax_bbox, band_mm_plot, band_x_extent, color="orange", label="pelvic_band")
            legends.append("pelvic_band")

        hot_mm = seg_debug.get("pet_hot_bbox_mm") if isinstance(seg_debug, dict) else None
        if hot_mm:
            hot_mm_plot = _flip_bbox_z_for_coronal_display(hot_mm, pet_extent if pet_extent is not None else ct_extent)
            _draw_bbox(ax_bbox, hot_mm_plot, color="magenta", label="pet_hot")
            legends.append("pet_hot")

        band_source = None
        if isinstance(seg_debug, dict):
            band_source = seg_debug.get("pelvic_band_source")
            if not band_source:
                band_anchor = seg_debug.get("pelvic_band_anchor") if isinstance(seg_debug, dict) else None
                if isinstance(band_anchor, dict):
                    band_source = band_anchor.get("source")
        if band_source:
            ax_bbox.text(
                0.02,
                0.96,
                f"band: {band_source}",
                transform=ax_bbox.transAxes,
                color="orange",
                fontsize=8,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.4, "edgecolor": "none"},
            )

    if legends:
        ax_bbox.legend(loc="lower right")
    ax_bbox.set_title(f"Fusion with bboxes (slice {coronal_idx})")
    ax_bbox.set_xlabel("x (mm)")
    ax_bbox.set_ylabel("z (mm)")
    ax_bbox.set_aspect("equal")
    fig_bbox.savefig(debug_dir / "step_all_bboxes.png", dpi=200, bbox_inches="tight")
    plt.close(fig_bbox)

    # PET hot bbox (for summary/metadata)
    hot_bbox = _pet_bladder_guess(model.pet_volume) if model.pet_volume is not None else None

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
        "ct_aligned": bool(ct_aligned),
        "pet_aligned": bool(pet_aligned),
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
            "x": [float(pet_origin[2] + hot_bbox[0][2] * pet_spacing[2]), float(pet_origin[2] + hot_bbox[1][2] * pet_spacing[2])],
            "z": [float(pet_origin[0] + hot_bbox[0][0] * pet_spacing[0]), float(pet_origin[0] + hot_bbox[1][0] * pet_spacing[0])],
        } if hot_bbox is not None else None,
        "locator_debug": bbox.get("debug"),
        "segmentation_debug": seg_debug,
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
