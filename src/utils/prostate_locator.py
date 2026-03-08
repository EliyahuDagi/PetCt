import json
import math
from typing import Dict, Optional, Tuple

import numpy as np


def _find_label_id(labels: Dict[int, str], needle: str) -> Optional[int]:
    for idx, name in labels.items():
        if needle.lower() in str(name).lower():
            return int(idx)
    return None


def _find_first_match(labels: Dict[int, str], needles) -> Optional[int]:
    """Return the first label id whose name contains any of the provided substrings."""
    for needle in needles:
        hit = _find_label_id(labels, needle)
        if hit is not None:
            return hit
    return None


def _label_bbox(mask: np.ndarray, labels: Dict[int, str], names) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find bbox for the first matching label name using largest component."""
    label_id = _find_first_match(labels, names)
    if label_id is None:
        return None
    comp = (mask == label_id).astype(np.uint8)
    comp = _largest_component(comp)
    return _bbox_from_mask(comp)


def _centroid_from_bbox(bbox: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return np.array([(bbox[0][i] + bbox[1][i]) / 2.0 for i in range(3)], dtype=float)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    return coords.min(axis=0), coords.max(axis=0)


def _vox_to_mm(indexes: np.ndarray, spacing: Tuple[float, float, float], origin: Tuple[float, float, float]) -> np.ndarray:
    dz, dy, dx = spacing
    oz, oy, ox = origin
    zyx_mm = np.zeros(3, dtype=float)
    zyx_mm[0] = oz + indexes[0] * dz
    zyx_mm[1] = oy + indexes[1] * dy
    zyx_mm[2] = ox + indexes[2] * dx
    return zyx_mm


def _clamp(min_idx: int, max_idx: int, limit: int) -> Tuple[int, int]:
    lo = max(0, min_idx)
    hi = min(limit - 1, max_idx)
    return lo, hi


def _clamp_exclusive(start: int, end: int, limit: int) -> Tuple[int, int]:
    """Clamp a half-open range [start, end) to [0, limit]."""
    lo = max(0, start)
    hi = min(limit, end)
    if hi < lo:
        hi = lo
    return lo, hi


def _find_all_label_ids(labels: Dict[int, str], needles) -> list[int]:
    """Return all label ids whose name contains any of the provided substrings."""
    hits: set[int] = set()
    for idx, name in labels.items():
        name_l = str(name).lower()
        for needle in needles:
            if str(needle).lower() in name_l:
                try:
                    hits.add(int(idx))
                except Exception:
                    pass
                break
    return sorted(hits)


def _safe_center_of_mass(binary_mask: np.ndarray, weights: Optional[np.ndarray] = None) -> Optional[Tuple[float, float, float]]:
    """Compute center-of-mass without requiring SciPy."""
    if binary_mask is None:
        return None
    coords = np.argwhere(binary_mask > 0)
    if coords.size == 0:
        return None
    if weights is None:
        com = coords.mean(axis=0)
        return float(com[0]), float(com[1]), float(com[2])

    w = np.asarray(weights, dtype=float)
    wv = w[binary_mask > 0]
    w_sum = float(np.sum(wv))
    if w_sum <= 0:
        com = coords.mean(axis=0)
        return float(com[0]), float(com[1]), float(com[2])
    com = (coords * wv[:, None]).sum(axis=0) / w_sum
    return float(com[0]), float(com[1]), float(com[2])


def _clamp_bbox_to_shape(
    bbox: Tuple[np.ndarray, np.ndarray], shape: Tuple[int, int, int]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    z0, y0, x0 = bbox[0].astype(int)
    z1, y1, x1 = bbox[1].astype(int)
    z0, z1 = _clamp(z0, z1, shape[0])
    y0, y1 = _clamp(y0, y1, shape[1])
    x0, x1 = _clamp(x0, x1, shape[2])
    if z0 > z1 or y0 > y1 or x0 > x1:
        return None
    return np.array([z0, y0, x0]), np.array([z1, y1, x1])


def _scale_bbox_between_grids(
    bbox: Tuple[np.ndarray, np.ndarray],
    src_shape: Tuple[int, int, int],
    dst_shape: Tuple[int, int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Map an inclusive bbox from a source grid into a destination grid by shape ratios."""
    src = np.array(src_shape, dtype=float)
    dst = np.array(dst_shape, dtype=float)
    scale = dst / src
    z0, y0, x0 = bbox[0]
    z1, y1, x1 = bbox[1]
    min_dst = np.floor(np.array([z0, y0, x0], dtype=float) * scale).astype(int)
    max_dst = np.ceil((np.array([z1, y1, x1], dtype=float) + 1.0) * scale).astype(int) - 1
    return _clamp_bbox_to_shape((min_dst, max_dst), dst_shape)


def _extend_inferior(
    bbox: Tuple[np.ndarray, np.ndarray],
    shape: Tuple[int, int, int],
    inferior_frac: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pad inferiorly (positive z) by a fraction of bbox height."""
    if inferior_frac <= 0.0:
        return _clamp_bbox_to_shape(bbox, shape)
    z0, y0, x0 = bbox[0]
    z1, y1, x1 = bbox[1]
    height = max(1, int(z1 - z0 + 1))
    extra = max(1, int(round(height * inferior_frac)))
    z1 = z1 + extra
    return _clamp_bbox_to_shape((np.array([z0, y0, x0]), np.array([z1, y1, x1])), shape)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label  # type: ignore
    except Exception:
        return mask  # fallback: return raw mask
    labeled, n = label(mask)
    if n <= 1:
        return (labeled > 0).astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    max_label = sizes.argmax()
    return (labeled == max_label).astype(np.uint8)


def _pet_bladder_guess(
    pet_volume: np.ndarray,
    crop_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if pet_volume is None:
        return None
    pet = np.asarray(pet_volume, dtype=float)

    if crop_bbox is not None:
        roi = _clamp_bbox_to_shape(crop_bbox, pet.shape)
        if roi is None:
            return None
        (z0, y0, x0), (z1, y1, x1) = roi
        pet_view = pet[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        offset = np.array([z0, y0, x0], dtype=int)
    else:
        pet_view = pet
        offset = np.zeros(3, dtype=int)

    positive = pet_view[pet_view > 0]
    if positive.size == 0:
        return None
    # Focus on lower torso to avoid brain/arms hot spots
    z_lo = int(pet_view.shape[0] * 0.35)
    z_hi = int(pet_view.shape[0] * 0.85)
    y_lo = int(pet_view.shape[1] * 0.15)
    y_hi = int(pet_view.shape[1] * 0.85)
    x_lo = int(pet_view.shape[2] * 0.15)
    x_hi = int(pet_view.shape[2] * 0.85)
    crop = pet_view[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    if crop.size == 0:
        return None
    # Adaptive high percentile threshold
    thresh = np.percentile(crop[crop > 0], 99)
    if math.isclose(thresh, 0.0):
        return None
    hot = (crop >= thresh).astype(np.uint8)

    # Choose largest component; if multiple, bias to lower (feet-ward) centroid
    try:
        from scipy.ndimage import label, center_of_mass  # type: ignore
        labeled, n = label(hot)
        if n == 0:
            return None
        best_label = None
        best_score = -1.0
        for lab in range(1, n + 1):
            vol = np.sum(labeled == lab)
            cz, cy, cx = center_of_mass(hot, labeled, lab)
            # Prefer larger volume and lower z (smaller cz is closer to z_lo)
            score = vol - cz * 0.1
            if score > best_score:
                best_score = score
                best_label = lab
        hot = (labeled == best_label).astype(np.uint8)
    except Exception:
        hot = _largest_component(hot)

    if hot.sum() == 0:
        return None
    # Map back to full-volume indices
    coords = np.argwhere(hot > 0)
    coords[:, 0] += z_lo
    coords[:, 1] += y_lo
    coords[:, 2] += x_lo
    coords = coords + offset
    return coords.min(axis=0), coords.max(axis=0)


def locate_prostate_bbox(
    mask: np.ndarray,
    labels: Dict[int, str],
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    pet_volume: Optional[np.ndarray] = None,
    ct_volume: Optional[np.ndarray] = None,
    inferior_pad_mm: float = 28.0,
    superior_pad_mm: float = 14.0,
    lateral_pad_mm: float = 18.0,
    ap_pad_mm: float = 14.0,
    inferior_shift_mm: float = 32.0,
    pet_inferior_extension_frac: float = 0.35,
    lateral_from_heads_pad_mm: float = 6.0,
    debug: bool = False,
) -> Optional[Dict]:
    """
    Estimate prostate ROI using TotalSegmentor-like landmarks:
    - Anchor on bladder; nudge inferior.
    - If ct_volume provided, refine by searching soft tissue below bladder.
    - Refine with PET bladder glow if available.
    - Constrain lateral span using femoral heads.
    - Refine superior plane using seminal vesicles if present.
    - Refine AP center using rectum vs bladder mid-plane.

    Returns bbox in vox/mm plus provenance metadata.
    """
    if mask is None or mask.size == 0:
        return None

    debug_info: Optional[dict] = {} if debug else None

    dz, dy, dx = spacing
    oz, oy, ox = origin

    # If CT/PET are not on the same grid as the segmentation, we can still use PET
    # for bladder glow (via bbox scaling), but we should avoid voxel-wise CT/PET fusion.
    ct_aligned = ct_volume is not None and getattr(ct_volume, "shape", None) == mask.shape
    pet_aligned = pet_volume is not None and getattr(pet_volume, "shape", None) == mask.shape

    if debug_info is not None:
        debug_info["ct_aligned"] = bool(ct_aligned)
        debug_info["pet_aligned"] = bool(pet_aligned)
        debug_info["spacing_mm"] = [float(dz), float(dy), float(dx)]
        debug_info["origin_mm"] = [float(oz), float(oy), float(ox)]
        debug_info["shapes_vox"] = {
            "mask": [int(v) for v in mask.shape],
            "ct": [int(v) for v in ct_volume.shape] if ct_volume is not None else None,
            "pet": [int(v) for v in pet_volume.shape] if pet_volume is not None else None,
        }

    # 0) Direct detection: use prostate label if present
    prostate_id = _find_first_match(labels, ["prostate", "prostate_gland", "prostate gland"])
    if prostate_id is not None:
        prostate_bbox = _label_bbox(mask, labels, ["prostate", "prostate_gland", "prostate gland"])
        if prostate_bbox is not None:
            (z0, y0, x0), (z1, y1, x1) = prostate_bbox
            # Small padding around the exact segmentation (in mm)
            pad_mm = 6.0
            z0 = int(round(z0 - pad_mm / max(dz, 1e-6)))
            z1 = int(round(z1 + pad_mm / max(dz, 1e-6)))
            y0 = int(round(y0 - pad_mm / max(dy, 1e-6)))
            y1 = int(round(y1 + pad_mm / max(dy, 1e-6)))
            x0 = int(round(x0 - pad_mm / max(dx, 1e-6)))
            x1 = int(round(x1 + pad_mm / max(dx, 1e-6)))

            z0, z1 = _clamp(z0, z1, mask.shape[0])
            y0, y1 = _clamp(y0, y1, mask.shape[1])
            x0, x1 = _clamp(x0, x1, mask.shape[2])

            center_vox = np.array([
                (z0 + z1) // 2,
                (y0 + y1) // 2,
                (x0 + x1) // 2,
            ], dtype=int)

            bbox_mm = {
                "z": [
                    float(_vox_to_mm(np.array([z0, 0, 0]), spacing, origin)[0]),
                    float(_vox_to_mm(np.array([z1, 0, 0]), spacing, origin)[0]),
                ],
                "y": [
                    float(_vox_to_mm(np.array([0, y0, 0]), spacing, origin)[1]),
                    float(_vox_to_mm(np.array([0, y1, 0]), spacing, origin)[1]),
                ],
                "x": [
                    float(_vox_to_mm(np.array([0, 0, x0]), spacing, origin)[2]),
                    float(_vox_to_mm(np.array([0, 0, x1]), spacing, origin)[2]),
                ],
            }
            center_mm = _vox_to_mm(center_vox, spacing, origin).tolist()
            payload = {
                "method": "prostate_label",
                "source_label_id": int(prostate_id),
                "bbox_vox": {"z": [int(z0), int(z1)], "y": [int(y0), int(y1)], "x": [int(x0), int(x1)]},
                "bbox_mm": bbox_mm,
                "center_vox": center_vox.tolist(),
                "center_mm": [float(v) for v in center_mm],
            }

            if debug_info is not None:
                debug_info["path"] = "direct_prostate_label"
                debug_info["direct_prostate_label"] = {
                    "label_id": int(prostate_id),
                    "pad_mm": float(pad_mm),
                    "raw_bbox_vox": {
                        "z": [int(prostate_bbox[0][0]), int(prostate_bbox[1][0])],
                        "y": [int(prostate_bbox[0][1]), int(prostate_bbox[1][1])],
                        "x": [int(prostate_bbox[0][2]), int(prostate_bbox[1][2])],
                    },
                }
                payload["debug"] = debug_info

            return payload

    # Landmarks (TotalSegmentor class names may vary slightly)
    bladder_bbox = _label_bbox(mask, labels, ["urinary_bladder", "bladder"])
    ves_bbox = _label_bbox(mask, labels, ["seminal_vesicle", "seminal vesicle", "seminal_vesicles", "seminalvesicle"])
    rectum_bbox = _label_bbox(mask, labels, ["rectum"])
    colon_bbox = _label_bbox(mask, labels, ["colon", "large_bowel", "large bowel", "bowel", "sigmoid", "rectosigmoid"])
    femoral_head_l = _label_bbox(mask, labels, [
        "femur_head_left",
        "femoral_head_left",
        "femur head left",
        "femoral head left",
    ])
    femoral_head_r = _label_bbox(mask, labels, [
        "femur_head_right",
        "femoral_head_right",
        "femur head right",
        "femoral head right",
    ])

    def _bbox_to_dict(bbox: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[dict]:
        if bbox is None:
            return None
        (z0_, y0_, x0_), (z1_, y1_, x1_) = bbox
        return {
            "z": [int(z0_), int(z1_)],
            "y": [int(y0_), int(y1_)],
            "x": [int(x0_), int(x1_)],
        }

    bladder_id = _find_first_match(labels, ["urinary_bladder", "bladder"])
    rectum_id = _find_first_match(labels, ["rectum"])
    colon_ids = _find_all_label_ids(labels, ["colon", "large_bowel", "large bowel", "bowel", "sigmoid", "rectosigmoid"])

    if debug_info is not None:
        debug_info["path"] = "heuristic"
        debug_info["landmarks_vox"] = {
            "bladder": _bbox_to_dict(bladder_bbox),
            "vesicles": _bbox_to_dict(ves_bbox),
            "rectum": _bbox_to_dict(rectum_bbox),
            "colon": _bbox_to_dict(colon_bbox),
            "femoral_head_l": _bbox_to_dict(femoral_head_l),
            "femoral_head_r": _bbox_to_dict(femoral_head_r),
        }
        debug_info["label_ids"] = {
            "bladder": int(bladder_id) if bladder_id is not None else None,
            "rectum": int(rectum_id) if rectum_id is not None else None,
            "colon": [int(v) for v in colon_ids],
        }

    # PET glow near bladder
    pet_crop_bbox = None
    if bladder_bbox and pet_volume is not None:
        pet_crop_bbox = _scale_bbox_between_grids(bladder_bbox, mask.shape, pet_volume.shape)
        if pet_crop_bbox is not None:
            pet_crop_bbox = _extend_inferior(pet_crop_bbox, pet_volume.shape, pet_inferior_extension_frac)
    bbox_pet = _pet_bladder_guess(pet_volume, crop_bbox=pet_crop_bbox) if pet_volume is not None else None

    if debug_info is not None:
        debug_info["pet_bladder_guess"] = {
            "crop_bbox_vox": _bbox_to_dict(pet_crop_bbox),
            "bbox_vox": _bbox_to_dict(bbox_pet),
        }

    # Choose centroid
    centroid = None
    method = None
    if bladder_bbox is not None:
        centroid = _centroid_from_bbox(bladder_bbox)
        method = "bladder_anchor"
        if bbox_pet is not None:
            pet_centroid = _centroid_from_bbox(bbox_pet)
            centroid = 0.25 * pet_centroid + 0.75 * centroid
            method = "bladder_pet_fusion"
    elif bbox_pet is not None:
        centroid = _centroid_from_bbox(bbox_pet)
        method = "pet_only"
    else:
        return None

    # In this project, CT slices are sorted by ImagePositionPatient[2] (ascending),
    # so z-index 0 is typically inferior (feet) and increasing z moves superior (head).
    # Prostate is inferior to the bladder, so the first search direction is towards LOWER z indices.
    inferior_step = -1

    if debug_info is not None:
        debug_info["initial_centroid_vox"] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
        debug_info["initial_method"] = str(method)
        debug_info["inferior_step"] = int(inferior_step)
    
    # Let's apply valid lateral constraints
    lateral_lims = None
    if femoral_head_l is not None and femoral_head_r is not None:
        x_vals = [
            femoral_head_l[0][2], femoral_head_l[1][2],
            femoral_head_r[0][2], femoral_head_r[1][2],
        ]
        lateral_lims = (int(min(x_vals)), int(max(x_vals)))

    if debug_info is not None:
        debug_info["lateral_lims_from_heads_vox"] = [int(v) for v in lateral_lims] if lateral_lims is not None else None

    # REFINEMENT WITH CT (+ PET): pelvic search below bladder between hips
    if ct_aligned and bladder_bbox is not None:
        b_z_min, b_z_max = int(bladder_bbox[0][0]), int(bladder_bbox[1][0])
        b_y_min, b_y_max = int(bladder_bbox[0][1]), int(bladder_bbox[1][1])
        b_x_min, b_x_max = int(bladder_bbox[0][2]), int(bladder_bbox[1][2])

        # Search zone definition (mm)
        search_mm = 60.0
        neck_mm = 12.0
        search_vox = int(round(search_mm / max(dz, 1e-6)))
        neck_vox = int(round(neck_mm / max(dz, 1e-6)))

        hip_pad_vox = int(round(lateral_from_heads_pad_mm / max(dx, 1e-6)))

        # Build X bounds from hips when possible
        if lateral_lims is not None:
            x0 = int(lateral_lims[0] - hip_pad_vox)
            x1 = int(lateral_lims[1] + hip_pad_vox + 1)  # exclusive
        else:
            x_pad = int(round(60.0 / max(dx, 1e-6)))
            b_cx = int(round((b_x_min + b_x_max) / 2.0))
            x0 = b_cx - x_pad
            x1 = b_cx + x_pad + 1

        # Build Y bounds: posterior bound from rectum, fallback to colon
        y0 = int(round(b_y_min + (b_y_max - b_y_min) * 0.20))
        if rectum_bbox is not None:
            y1 = int(rectum_bbox[0][1])  # exclusive: stop before rectum
        elif colon_bbox is not None:
            y1 = int(colon_bbox[0][1])  # exclusive: stop before bowel if present
        else:
            y1 = int(b_y_max + int(round(20.0 / max(dy, 1e-6))) + 1)

        # Exclusion labels: bladder, rectum, colon/bowel
        exclude_ids: list[int] = []
        for v in [bladder_id, rectum_id]:
            if v is not None:
                exclude_ids.append(int(v))
        exclude_ids.extend([int(v) for v in colon_ids])

        pelvic_debug: Optional[dict] = None
        if debug_info is not None:
            pelvic_debug = {
                "search_mm": float(search_mm),
                "neck_mm": float(neck_mm),
                "search_vox": int(search_vox),
                "neck_vox": int(neck_vox),
                "x_range_requested": [int(x0), int(x1)],  # half-open
                "y_range_requested": [int(y0), int(y1)],  # half-open
                "exclude_ids": [int(v) for v in exclude_ids],
                "ct_soft_tissue_hu": [20.0, 110.0],
                "pet_hot_percentile": 99.0,
                "pet_min_vals": 50,
                "trials": [],
            }
            debug_info["pelvic_refinement"] = pelvic_debug

        def _try_refine(z0_cand: int, z1_cand: int, tag: str) -> bool:
            nonlocal centroid, method

            trial: Optional[dict] = None
            if pelvic_debug is not None:
                trial = {
                    "tag": str(tag),
                    "z_range_requested": [int(z0_cand), int(z1_cand)],  # half-open
                    "success": False,
                }
                pelvic_debug["trials"].append(trial)

            # Clamp to volumes (half-open ranges)
            z0_cl, z1_cl = _clamp_exclusive(z0_cand, z1_cand, mask.shape[0])
            y0_cl, y1_cl = _clamp_exclusive(y0, y1, mask.shape[1])
            x0_cl, x1_cl = _clamp_exclusive(x0, x1, mask.shape[2])
            if z1_cl <= z0_cl or y1_cl <= y0_cl or x1_cl <= x0_cl:
                if trial is not None:
                    trial["fail_reason"] = "empty_range_after_clamp"
                return False

            if trial is not None:
                trial["ranges_clamped"] = {
                    "z": [int(z0_cl), int(z1_cl)],
                    "y": [int(y0_cl), int(y1_cl)],
                    "x": [int(x0_cl), int(x1_cl)],
                }
                trial["crop_shape"] = [int(z1_cl - z0_cl), int(y1_cl - y0_cl), int(x1_cl - x0_cl)]

            ct_crop = ct_volume[z0_cl:z1_cl, y0_cl:y1_cl, x0_cl:x1_cl]
            seg_crop = mask[z0_cl:z1_cl, y0_cl:y1_cl, x0_cl:x1_cl]

            exclude_mask = np.zeros(seg_crop.shape, dtype=bool)
            for lab_id in exclude_ids:
                exclude_mask |= (seg_crop == int(lab_id))

            # Soft tissue range (reject air/bone)
            soft = (ct_crop >= 20.0) & (ct_crop <= 110.0)
            candidate_base = soft & (~exclude_mask)
            if not np.any(candidate_base):
                if trial is not None:
                    trial["soft_voxels"] = int(np.count_nonzero(soft))
                    trial["exclude_voxels"] = int(np.count_nonzero(exclude_mask))
                    trial["candidate_voxels"] = 0
                    trial["fail_reason"] = "no_candidate_voxels"
                return False

            if trial is not None:
                trial["soft_voxels"] = int(np.count_nonzero(soft))
                trial["exclude_voxels"] = int(np.count_nonzero(exclude_mask))
                trial["candidate_voxels"] = int(np.count_nonzero(candidate_base))

            # If PET is aligned, use it to select a hotspot component instead of raw argmax
            refined_mask = None
            weights = None
            if pet_aligned:
                pet_crop = np.asarray(pet_volume[z0_cl:z1_cl, y0_cl:y1_cl, x0_cl:x1_cl], dtype=float)
                weights = np.maximum(pet_crop, 0.0)
                vals = weights[(weights > 0) & candidate_base]
                if trial is not None:
                    trial["pet_candidate_vals"] = int(vals.size)
                if vals.size >= 50:
                    thresh = float(np.percentile(vals, 99.0))
                    if thresh > 0 and not math.isclose(thresh, 0.0):
                        hot = weights >= thresh
                        refined_mask = (candidate_base & hot).astype(np.uint8)
                        if trial is not None:
                            trial["pet_thresh"] = float(thresh)
                            trial["pet_hot_voxels"] = int(np.count_nonzero(candidate_base & hot))
            if refined_mask is None:
                refined_mask = candidate_base.astype(np.uint8)

            if trial is not None:
                trial["refined_voxels"] = int(np.count_nonzero(refined_mask))

            comp = _largest_component(refined_mask)
            if comp is None or int(np.sum(comp)) == 0:
                if trial is not None:
                    trial["fail_reason"] = "empty_component"
                return False

            if trial is not None:
                trial["largest_component_voxels"] = int(np.count_nonzero(comp))

            com = _safe_center_of_mass(comp, weights=weights)
            if com is None:
                if trial is not None:
                    trial["fail_reason"] = "com_failed"
                return False

            cz, cy, cx = com
            centroid = np.array([z0_cl + cz, y0_cl + cy, x0_cl + cx], dtype=float)
            method = f"ct{'_pet' if pet_aligned else ''}_pelvic_refinement_{tag}"
            if trial is not None:
                trial["com_local"] = [float(cz), float(cy), float(cx)]
                trial["com_global"] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
                trial["success"] = True
            return True

        # Preferred: search inferior (lower z indices) below the bladder's inferior surface
        z0_a = b_z_min + inferior_step * search_vox
        z1_a = b_z_min + max(1, neck_vox + 1)
        ok = _try_refine(z0_a, z1_a, "below")

        # Fallback: if direction is flipped in the input arrays, try the opposite side
        if not ok:
            z0_b = b_z_max - max(1, neck_vox)
            z1_b = b_z_max + search_vox + 1
            _try_refine(z0_b, z1_b, "above")
             
    if method is None or not method.startswith("ct"):
        # Fallback to fixed shift from bladder anchor when refinement failed
        if bladder_bbox is not None:
            shift_vox = (inferior_shift_mm / max(dz, 1e-6)) * float(inferior_step)
            centroid[0] = centroid[0] + shift_vox
            if debug_info is not None:
                debug_info["fallback_shift_z_vox"] = float(shift_vox)

    # AP refinement: pull toward midpoint of bladder and rectum if both present
    if bladder_bbox is not None and rectum_bbox is not None:
        bladder_cy = _centroid_from_bbox(bladder_bbox)[1]
        rectum_cy = _centroid_from_bbox(rectum_bbox)[1]
        centroid[1] = 0.55 * bladder_cy + 0.45 * rectum_cy
        method += "_ap_rectum_mix"

        if debug_info is not None:
            debug_info["ap_refinement"] = {
                "bladder_cy": float(bladder_cy),
                "rectum_cy": float(rectum_cy),
                "mix": [0.55, 0.45],
                "result_cy": float(centroid[1]),
            }

    # Superior limit refinement using seminal vesicles (prostate sits inferior to them)
    # With z-index increasing superior (typical for this project), the *inferior* edge of the vesicles
    # is ves_bbox[0][0] (min z). Keep prostate bbox below that plane.
    ves_z_min = None
    if ves_bbox is not None:
        ves_z_min = int(ves_bbox[0][0])

    # Lateral constraints from femoral heads
    lateral_min = None
    lateral_max = None
    if femoral_head_l is not None and femoral_head_r is not None:
        x_vals = [femoral_head_l[0][2], femoral_head_l[1][2], femoral_head_r[0][2], femoral_head_r[1][2]]
        lateral_min = min(x_vals) - lateral_from_heads_pad_mm / max(dx, 1e-6)
        lateral_max = max(x_vals) + lateral_from_heads_pad_mm / max(dx, 1e-6)
        method += "_lat_heads"

        if debug_info is not None:
            debug_info["lateral_mm_from_heads"] = [float(lateral_min), float(lateral_max)]

    # Build pads
    z_min_pad = int(round(centroid[0] - inferior_pad_mm / max(dz, 1e-6)))
    z_max_pad = int(round(centroid[0] + superior_pad_mm / max(dz, 1e-6)))
    if ves_z_min is not None:
        z_max_pad = min(z_max_pad, int(ves_z_min))

    y_min_pad = int(round(centroid[1] - ap_pad_mm / max(dy, 1e-6)))
    y_max_pad = int(round(centroid[1] + ap_pad_mm / max(dy, 1e-6)))

    if lateral_min is not None and lateral_max is not None:
        x_min_pad = int(math.floor(lateral_min))
        x_max_pad = int(math.ceil(lateral_max))
    else:
        x_min_pad = int(round(centroid[2] - lateral_pad_mm / max(dx, 1e-6)))
        x_max_pad = int(round(centroid[2] + lateral_pad_mm / max(dx, 1e-6)))

    z_min_pad, z_max_pad = _clamp(z_min_pad, z_max_pad, mask.shape[0])
    y_min_pad, y_max_pad = _clamp(y_min_pad, y_max_pad, mask.shape[1])
    x_min_pad, x_max_pad = _clamp(x_min_pad, x_max_pad, mask.shape[2])

    center_vox = np.array([
        (z_min_pad + z_max_pad) // 2,
        (y_min_pad + y_max_pad) // 2,
        (x_min_pad + x_max_pad) // 2,
    ], dtype=int)

    bbox_mm = {
        "z": [float(_vox_to_mm(np.array([z_min_pad, 0, 0]), spacing, origin)[0]),
               float(_vox_to_mm(np.array([z_max_pad, 0, 0]), spacing, origin)[0])],
        "y": [float(_vox_to_mm(np.array([0, y_min_pad, 0]), spacing, origin)[1]),
               float(_vox_to_mm(np.array([0, y_max_pad, 0]), spacing, origin)[1])],
        "x": [float(_vox_to_mm(np.array([0, 0, x_min_pad]), spacing, origin)[2]),
               float(_vox_to_mm(np.array([0, 0, x_max_pad]), spacing, origin)[2])],
    }

    center_mm = _vox_to_mm(center_vox, spacing, origin).tolist()

    payload = {
        "method": method,
        "source_label_id": _find_first_match(labels, ["urinary_bladder", "bladder"]),
        "bbox_vox": {
            "z": [int(z_min_pad), int(z_max_pad)],
            "y": [int(y_min_pad), int(y_max_pad)],
            "x": [int(x_min_pad), int(x_max_pad)],
        },
        "bbox_mm": bbox_mm,
        "center_vox": center_vox.tolist(),
        "center_mm": [float(x) for x in center_mm],
    }

    if debug_info is not None:
        debug_info["final_centroid_vox"] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
        debug_info["final_method"] = str(method)
        payload["debug"] = debug_info

    return payload


def save_prostate_bbox_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
