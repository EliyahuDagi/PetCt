import json
import math
from typing import Dict, Optional, Tuple

import numpy as np

from src.utils.geometry import VolumeGeometry


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


def _bbox_to_dict(bbox: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[dict]:
    if bbox is None:
        return None
    (z0_, y0_, x0_), (z1_, y1_, x1_) = bbox
    return {
        "z": [int(z0_), int(z1_)],
        "y": [int(y0_), int(y1_)],
        "x": [int(x0_), int(x1_)],
    }


def _intersect_bbox(
    b1: Optional[Tuple[np.ndarray, np.ndarray]],
    b2: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    (a0, a1) = b1
    (b0, b1_) = b2
    lo = np.maximum(a0, b0)
    hi = np.minimum(a1, b1_)
    if np.any(hi < lo):
        return None
    return lo, hi


def _direct_prostate_label_payload(
    mask: np.ndarray,
    labels: Dict[int, str],
    geom_seg: VolumeGeometry,
    spacing: Tuple[float, float, float],
    debug_info: Optional[dict],
) -> Optional[Dict]:
    dz, dy, dx = spacing
    prostate_id = _find_first_match(labels, ["prostate", "prostate_gland", "prostate gland"])
    if prostate_id is None:
        return None

    prostate_bbox = _label_bbox(mask, labels, ["prostate", "prostate_gland", "prostate gland"])
    if prostate_bbox is None:
        return None

    (z0, y0, x0), (z1, y1, x1) = prostate_bbox
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
    bbox_mm = geom_seg.bbox_vox_to_mm((np.array([z0, y0, x0]), np.array([z1, y1, x1])))
    center_mm = geom_seg.vox_to_mm(center_vox).tolist()
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


def _collect_landmarks(mask: np.ndarray, labels: Dict[int, str]) -> tuple[dict, dict]:
    landmarks = {
        "bladder": _label_bbox(mask, labels, ["urinary_bladder", "bladder"]),
        "vesicles": _label_bbox(mask, labels, ["seminal_vesicle", "seminal vesicle", "seminal_vesicles", "seminalvesicle"]),
        "rectum": _label_bbox(mask, labels, ["rectum"]),
        "colon": _label_bbox(mask, labels, ["colon", "large_bowel", "large bowel", "bowel", "sigmoid", "rectosigmoid"]),
        "femoral_head_l": _label_bbox(
            mask,
            labels,
            ["femur_head_left", "femoral_head_left", "femur head left", "femoral head left"],
        ),
        "femoral_head_r": _label_bbox(
            mask,
            labels,
            ["femur_head_right", "femoral_head_right", "femur head right", "femoral head right"],
        ),
    }
    label_ids = {
        "bladder": _find_first_match(labels, ["urinary_bladder", "bladder"]),
        "rectum": _find_first_match(labels, ["rectum"]),
        "colon": _find_all_label_ids(labels, ["colon", "large_bowel", "large bowel", "bowel", "sigmoid", "rectosigmoid"]),
    }
    return landmarks, label_ids


def _derive_pelvic_band_from_landmarks(
    mask: np.ndarray,
    dz: float,
    bladder_bbox: Optional[Tuple[np.ndarray, np.ndarray]],
    rectum_bbox: Optional[Tuple[np.ndarray, np.ndarray]],
    colon_bbox: Optional[Tuple[np.ndarray, np.ndarray]],
    femoral_head_l: Optional[Tuple[np.ndarray, np.ndarray]],
    femoral_head_r: Optional[Tuple[np.ndarray, np.ndarray]],
) -> tuple[Optional[Tuple[int, int]], Optional[str]]:
    z_len = mask.shape[0]
    if z_len <= 0:
        return None, None

    def pad_vox(mm: float) -> int:
        return int(round(mm / max(dz, 1e-6)))

    if bladder_bbox is not None:
        z0, z1 = int(bladder_bbox[0][0]), int(bladder_bbox[1][0])
        lo = max(0, z0 - pad_vox(80.0))
        hi = min(z_len - 1, z1 + pad_vox(60.0))
        return (lo, hi), "bladder"

    if rectum_bbox is not None:
        z0, z1 = int(rectum_bbox[0][0]), int(rectum_bbox[1][0])
        lo = max(0, z0 - pad_vox(60.0))
        hi = min(z_len - 1, z1 + pad_vox(40.0))
        return (lo, hi), "rectum"

    if colon_bbox is not None:
        z0, z1 = int(colon_bbox[0][0]), int(colon_bbox[1][0])
        height = max(1, z1 - z0 + 1)
        lower_hi = z0 + int(math.ceil(height * 0.4))
        lo = max(0, min(z0 - pad_vox(30.0), int(z_len * 0.35)))
        hi = min(z_len - 1, lower_hi + pad_vox(15.0), int(z_len * 0.65))
        return (lo, hi), "colon_lower"

    if femoral_head_l is not None and femoral_head_r is not None:
        z_vals = [femoral_head_l[0][0], femoral_head_l[1][0], femoral_head_r[0][0], femoral_head_r[1][0]]
        lo = max(0, int(min(z_vals) - pad_vox(40.0)))
        hi = min(z_len - 1, int(max(z_vals) + pad_vox(80.0)))
        return (lo, hi), "femoral_heads"

    hi = z_len - 1
    mid = int(round(z_len * 0.45))
    return (mid, hi), "inferior_half"


def _build_pet_crop_bbox(
    mask: np.ndarray,
    pet_volume: Optional[np.ndarray],
    geom_seg: VolumeGeometry,
    pet_geom: Optional[VolumeGeometry],
    bladder_bbox: Optional[Tuple[np.ndarray, np.ndarray]],
    pelvic_band_vox: Optional[Tuple[int, int]],
    pet_inferior_extension_frac: float,
    debug_info: Optional[dict],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if pet_volume is None:
        return None

    pet_crop_bbox = None

    if bladder_bbox is not None:
        if pet_geom is not None:
            bbox_mm = geom_seg.bbox_vox_to_mm(bladder_bbox)
            pet_crop_bbox = pet_geom.bbox_mm_to_vox(bbox_mm)
        else:
            pet_crop_bbox = _scale_bbox_between_grids(bladder_bbox, mask.shape, pet_volume.shape)
        if pet_crop_bbox is not None:
            pet_crop_bbox = _extend_inferior(pet_crop_bbox, pet_volume.shape, pet_inferior_extension_frac)

    pet_band_bbox = None
    if pelvic_band_vox is not None:
        z0_band, z1_band = pelvic_band_vox
        if pet_geom is not None:
            band_mm = (
                geom_seg.vox_to_mm(np.array([z0_band, 0, 0]))[0],
                geom_seg.vox_to_mm(np.array([z1_band, 0, 0]))[0],
            )
            z_band_pet = pet_geom.z_band_mm_to_vox(band_mm)
            if z_band_pet is not None:
                z_lo, z_hi = z_band_pet
                pet_band_bbox = (
                    np.array([z_lo, 0, 0]),
                    np.array([z_hi, pet_volume.shape[1] - 1, pet_volume.shape[2] - 1]),
                )
                if debug_info is not None:
                    debug_info["pelvic_band_pet_vox"] = [int(z_lo), int(z_hi)]
                    debug_info["pelvic_band_pet_mm"] = [float(band_mm[0]), float(band_mm[1])]
        else:
            band_bbox = (np.array([z0_band, 0, 0]), np.array([z1_band, mask.shape[1] - 1, mask.shape[2] - 1]))
            pet_band_bbox = _scale_bbox_between_grids(band_bbox, mask.shape, pet_volume.shape)

    return _intersect_bbox(pet_crop_bbox, pet_band_bbox)


def _sample_src_to_dst_roi_nearest(
    src_vol: np.ndarray,
    src_geom: VolumeGeometry,
    dst_geom: VolumeGeometry,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    fill_value: float = -1e9,
) -> np.ndarray:
    """Nearest-neighbor sample src volume onto destination voxel grid for half-open ROI."""
    dz_d, dy_d, dx_d = dst_geom.spacing
    oz_d, oy_d, ox_d = dst_geom.origin
    dz_s, dy_s, dx_s = src_geom.spacing
    oz_s, oy_s, ox_s = src_geom.origin

    z_idx = np.arange(z0, z1, dtype=float)
    y_idx = np.arange(y0, y1, dtype=float)
    x_idx = np.arange(x0, x1, dtype=float)

    z_mm = oz_d + z_idx * dz_d
    y_mm = oy_d + y_idx * dy_d
    x_mm = ox_d + x_idx * dx_d

    z_src = np.round((z_mm - oz_s) / max(dz_s, 1e-6)).astype(int)
    y_src = np.round((y_mm - oy_s) / max(dy_s, 1e-6)).astype(int)
    x_src = np.round((x_mm - ox_s) / max(dx_s, 1e-6)).astype(int)

    valid_z = (z_src >= 0) & (z_src < src_vol.shape[0])
    valid_y = (y_src >= 0) & (y_src < src_vol.shape[1])
    valid_x = (x_src >= 0) & (x_src < src_vol.shape[2])

    z_src = np.clip(z_src, 0, src_vol.shape[0] - 1)
    y_src = np.clip(y_src, 0, src_vol.shape[1] - 1)
    x_src = np.clip(x_src, 0, src_vol.shape[2] - 1)

    out = np.asarray(src_vol[z_src[:, None, None], y_src[None, :, None], x_src[None, None, :]], dtype=float)
    valid = valid_z[:, None, None] & valid_y[None, :, None] & valid_x[None, None, :]
    if not bool(np.all(valid)):
        out = out.copy()
        out[~valid] = float(fill_value)
    return out


def _ct_crop_on_seg_grid(
    ct_volume: np.ndarray,
    ct_geom: VolumeGeometry,
    geom_seg: VolumeGeometry,
    ct_index_aligned: bool,
    ct_same_geom: bool,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Return CT crop aligned to segmentation voxel grid for half-open ROI."""
    if ct_index_aligned and ct_same_geom:
        return np.asarray(ct_volume[z0:z1, y0:y1, x0:x1], dtype=float)
    return _sample_src_to_dst_roi_nearest(ct_volume, ct_geom, geom_seg, z0, z1, y0, y1, x0, x1)


def _ct_try_refine_trial(
    tag: str,
    z0_cand: int,
    z1_cand: int,
    mask: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    exclude_ids: list[int],
    ct_volume: np.ndarray,
    ct_geom: VolumeGeometry,
    geom_seg: VolumeGeometry,
    ct_index_aligned: bool,
    ct_same_geom: bool,
    pet_aligned: bool,
    pet_volume: Optional[np.ndarray],
    pelvic_debug: Optional[dict],
) -> tuple[bool, Optional[np.ndarray], Optional[str]]:
    trial: Optional[dict] = None
    if pelvic_debug is not None:
        trial = {
            "tag": str(tag),
            "z_range_requested": [int(z0_cand), int(z1_cand)],
            "success": False,
        }
        pelvic_debug["trials"].append(trial)

    z0_cl, z1_cl = _clamp_exclusive(z0_cand, z1_cand, mask.shape[0])
    y0_cl, y1_cl = _clamp_exclusive(y0, y1, mask.shape[1])
    x0_cl, x1_cl = _clamp_exclusive(x0, x1, mask.shape[2])
    if z1_cl <= z0_cl or y1_cl <= y0_cl or x1_cl <= x0_cl:
        if trial is not None:
            trial["fail_reason"] = "empty_range_after_clamp"
        return False, None, None

    if trial is not None:
        trial["ranges_clamped"] = {
            "z": [int(z0_cl), int(z1_cl)],
            "y": [int(y0_cl), int(y1_cl)],
            "x": [int(x0_cl), int(x1_cl)],
        }
        trial["crop_shape"] = [int(z1_cl - z0_cl), int(y1_cl - y0_cl), int(x1_cl - x0_cl)]

    ct_crop = _ct_crop_on_seg_grid(
        ct_volume,
        ct_geom,
        geom_seg,
        ct_index_aligned,
        ct_same_geom,
        z0_cl,
        z1_cl,
        y0_cl,
        y1_cl,
        x0_cl,
        x1_cl,
    )
    seg_crop = mask[z0_cl:z1_cl, y0_cl:y1_cl, x0_cl:x1_cl]

    exclude_mask = np.zeros(seg_crop.shape, dtype=bool)
    for lab_id in exclude_ids:
        exclude_mask |= (seg_crop == int(lab_id))

    soft = (ct_crop >= 20.0) & (ct_crop <= 110.0)
    candidate_base = soft & (~exclude_mask)
    if not np.any(candidate_base):
        if trial is not None:
            trial["soft_voxels"] = int(np.count_nonzero(soft))
            trial["exclude_voxels"] = int(np.count_nonzero(exclude_mask))
            trial["candidate_voxels"] = 0
            trial["fail_reason"] = "no_candidate_voxels"
        return False, None, None

    if trial is not None:
        trial["soft_voxels"] = int(np.count_nonzero(soft))
        trial["exclude_voxels"] = int(np.count_nonzero(exclude_mask))
        trial["candidate_voxels"] = int(np.count_nonzero(candidate_base))

    refined_mask = None
    weights = None
    if pet_aligned and pet_volume is not None:
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
        return False, None, None

    if trial is not None:
        trial["largest_component_voxels"] = int(np.count_nonzero(comp))

    com = _safe_center_of_mass(comp, weights=weights)
    if com is None:
        if trial is not None:
            trial["fail_reason"] = "com_failed"
        return False, None, None

    cz, cy, cx = com
    centroid = np.array([z0_cl + cz, y0_cl + cy, x0_cl + cx], dtype=float)
    method = f"ct{'_pet' if pet_aligned else ''}_pelvic_refinement_{tag}"
    if trial is not None:
        trial["com_local"] = [float(cz), float(cy), float(cx)]
        trial["com_global"] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
        trial["success"] = True
    return True, centroid, method


def _run_ct_pelvic_refinement(
    centroid: np.ndarray,
    method: str,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    geom_seg: VolumeGeometry,
    ct_volume: Optional[np.ndarray],
    ct_geom: Optional[VolumeGeometry],
    ct_index_aligned: bool,
    ct_spacing_eff: Optional[Tuple[float, float, float]],
    ct_origin_eff: Optional[Tuple[float, float, float]],
    pet_aligned: bool,
    pet_volume: Optional[np.ndarray],
    bladder_bbox: Optional[Tuple[np.ndarray, np.ndarray]],
    rectum_bbox: Optional[Tuple[np.ndarray, np.ndarray]],
    colon_bbox: Optional[Tuple[np.ndarray, np.ndarray]],
    lateral_lims: Optional[Tuple[int, int]],
    lateral_from_heads_pad_mm: float,
    bladder_id: Optional[int],
    rectum_id: Optional[int],
    colon_ids: list[int],
    inferior_step: int,
    debug_info: Optional[dict],
) -> tuple[np.ndarray, str]:
    if ct_volume is None or ct_geom is None or bladder_bbox is None:
        return centroid, method

    dz, dy, dx = spacing
    b_z_min, b_z_max = int(bladder_bbox[0][0]), int(bladder_bbox[1][0])
    b_y_min, b_y_max = int(bladder_bbox[0][1]), int(bladder_bbox[1][1])
    b_x_min, b_x_max = int(bladder_bbox[0][2]), int(bladder_bbox[1][2])

    search_mm = 60.0
    neck_mm = 12.0
    search_vox = int(round(search_mm / max(dz, 1e-6)))
    neck_vox = int(round(neck_mm / max(dz, 1e-6)))
    hip_pad_vox = int(round(lateral_from_heads_pad_mm / max(dx, 1e-6)))

    if lateral_lims is not None:
        x0 = int(lateral_lims[0] - hip_pad_vox)
        x1 = int(lateral_lims[1] + hip_pad_vox + 1)
    else:
        x_pad = int(round(60.0 / max(dx, 1e-6)))
        b_cx = int(round((b_x_min + b_x_max) / 2.0))
        x0 = b_cx - x_pad
        x1 = b_cx + x_pad + 1

    y0 = int(round(b_y_min + (b_y_max - b_y_min) * 0.20))
    if rectum_bbox is not None:
        y1 = int(rectum_bbox[0][1])
    elif colon_bbox is not None:
        y1 = int(colon_bbox[0][1])
    else:
        y1 = int(b_y_max + int(round(20.0 / max(dy, 1e-6))) + 1)

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
            "x_range_requested": [int(x0), int(x1)],
            "y_range_requested": [int(y0), int(y1)],
            "exclude_ids": [int(v) for v in exclude_ids],
            "ct_index_aligned": bool(ct_index_aligned),
            "ct_soft_tissue_hu": [20.0, 110.0],
            "pet_hot_percentile": 99.0,
            "pet_min_vals": 50,
            "trials": [],
        }
        debug_info["pelvic_refinement"] = pelvic_debug

    ct_same_geom = False
    if ct_spacing_eff is not None and ct_origin_eff is not None:
        ct_same_geom = bool(np.allclose(np.array(ct_spacing_eff), np.array(spacing))) and bool(
            np.allclose(np.array(ct_origin_eff), np.array(origin))
        )

    z0_a = b_z_min + inferior_step * search_vox
    z1_a = b_z_min + max(1, neck_vox + 1)
    ok, refined_centroid, refined_method = _ct_try_refine_trial(
        "below",
        z0_a,
        z1_a,
        mask,
        y0,
        y1,
        x0,
        x1,
        exclude_ids,
        ct_volume,
        ct_geom,
        geom_seg,
        ct_index_aligned,
        ct_same_geom,
        pet_aligned,
        pet_volume,
        pelvic_debug,
    )
    if ok and refined_centroid is not None and refined_method is not None:
        return refined_centroid, refined_method

    z0_b = b_z_max - max(1, neck_vox)
    z1_b = b_z_max + search_vox + 1
    ok, refined_centroid, refined_method = _ct_try_refine_trial(
        "above",
        z0_b,
        z1_b,
        mask,
        y0,
        y1,
        x0,
        x1,
        exclude_ids,
        ct_volume,
        ct_geom,
        geom_seg,
        ct_index_aligned,
        ct_same_geom,
        pet_aligned,
        pet_volume,
        pelvic_debug,
    )
    if ok and refined_centroid is not None and refined_method is not None:
        return refined_centroid, refined_method

    return centroid, method


def locate_prostate_bbox(
    mask: np.ndarray,
    labels: Dict[int, str],
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    pet_volume: Optional[np.ndarray] = None,
    pet_spacing: Optional[Tuple[float, float, float]] = None,
    pet_origin: Optional[Tuple[float, float, float]] = None,
    ct_volume: Optional[np.ndarray] = None,
    ct_spacing: Optional[Tuple[float, float, float]] = None,
    ct_origin: Optional[Tuple[float, float, float]] = None,
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
            If CT is not voxel-grid aligned with the segmentation mask, provide ct_spacing/ct_origin
            so CT can be sampled into mask space for this refinement.
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

    geom_seg = VolumeGeometry(spacing, origin, mask.shape)
    pet_geom: Optional[VolumeGeometry] = None
    if pet_volume is not None and pet_spacing is not None and pet_origin is not None:
        pet_geom = VolumeGeometry(pet_spacing, pet_origin, pet_volume.shape)

    ct_geom: Optional[VolumeGeometry] = None
    ct_spacing_eff: Optional[Tuple[float, float, float]] = None
    ct_origin_eff: Optional[Tuple[float, float, float]] = None
    if ct_volume is not None:
        ct_spacing_eff = ct_spacing if ct_spacing is not None else spacing
        ct_origin_eff = ct_origin if ct_origin is not None else origin
        ct_geom = VolumeGeometry(ct_spacing_eff, ct_origin_eff, ct_volume.shape)

    ct_index_aligned = ct_volume is not None and getattr(ct_volume, "shape", None) == mask.shape
    ct_aligned = bool(ct_index_aligned)
    pet_aligned = pet_volume is not None and getattr(pet_volume, "shape", None) == mask.shape
    ct_mappable = bool(ct_volume is not None and ct_geom is not None)

    if debug_info is not None:
        debug_info["ct_aligned"] = bool(ct_aligned)
        debug_info["ct_index_aligned"] = bool(ct_index_aligned)
        debug_info["ct_mappable"] = bool(ct_mappable)
        debug_info["pet_aligned"] = bool(pet_aligned)
        debug_info["spacing_mm"] = [float(dz), float(dy), float(dx)]
        debug_info["origin_mm"] = [float(oz), float(oy), float(ox)]
        if ct_spacing_eff is not None and ct_origin_eff is not None:
            debug_info["ct_spacing_mm"] = [float(ct_spacing_eff[0]), float(ct_spacing_eff[1]), float(ct_spacing_eff[2])]
            debug_info["ct_origin_mm"] = [float(ct_origin_eff[0]), float(ct_origin_eff[1]), float(ct_origin_eff[2])]
        if pet_spacing is not None and pet_origin is not None:
            debug_info["pet_spacing_mm"] = [float(pet_spacing[0]), float(pet_spacing[1]), float(pet_spacing[2])]
            debug_info["pet_origin_mm"] = [float(pet_origin[0]), float(pet_origin[1]), float(pet_origin[2])]
        debug_info["shapes_vox"] = {
            "mask": [int(v) for v in mask.shape],
            "ct": [int(v) for v in ct_volume.shape] if ct_volume is not None else None,
            "pet": [int(v) for v in pet_volume.shape] if pet_volume is not None else None,
        }

    direct_payload = _direct_prostate_label_payload(mask, labels, geom_seg, spacing, debug_info)
    if direct_payload is not None:
        return direct_payload

    landmarks, landmark_label_ids = _collect_landmarks(mask, labels)
    bladder_bbox = landmarks["bladder"]
    ves_bbox = landmarks["vesicles"]
    rectum_bbox = landmarks["rectum"]
    colon_bbox = landmarks["colon"]
    femoral_head_l = landmarks["femoral_head_l"]
    femoral_head_r = landmarks["femoral_head_r"]

    bladder_id = landmark_label_ids["bladder"]
    rectum_id = landmark_label_ids["rectum"]
    colon_ids = landmark_label_ids["colon"]

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

        def _bbox_to_mm_dict(bbox: Optional[Tuple[np.ndarray, np.ndarray]]):
            if bbox is None:
                return None
            return geom_seg.bbox_vox_to_mm(bbox)

        debug_info["landmarks_mm"] = {
            "bladder": _bbox_to_mm_dict(bladder_bbox),
            "vesicles": _bbox_to_mm_dict(ves_bbox),
            "rectum": _bbox_to_mm_dict(rectum_bbox),
            "colon": _bbox_to_mm_dict(colon_bbox),
            "femoral_head_l": _bbox_to_mm_dict(femoral_head_l),
            "femoral_head_r": _bbox_to_mm_dict(femoral_head_r),
        }
        debug_info["label_ids"] = {
            "bladder": int(bladder_id) if bladder_id is not None else None,
            "rectum": int(rectum_id) if rectum_id is not None else None,
            "colon": [int(v) for v in colon_ids],
        }

    pelvic_band_vox, pelvic_band_source = _derive_pelvic_band_from_landmarks(
        mask,
        dz,
        bladder_bbox,
        rectum_bbox,
        colon_bbox,
        femoral_head_l,
        femoral_head_r,
    )
    pelvic_band_mm = None
    if pelvic_band_vox is not None:
        z0_band, z1_band = pelvic_band_vox
        pelvic_band_mm = (
            geom_seg.vox_to_mm(np.array([z0_band, 0, 0]))[0],
            geom_seg.vox_to_mm(np.array([z1_band, 0, 0]))[0],
        )
    if debug_info is not None:
        debug_info["pelvic_band_source"] = pelvic_band_source
        debug_info["pelvic_band_vox"] = [int(pelvic_band_vox[0]), int(pelvic_band_vox[1])] if pelvic_band_vox else None
        debug_info["pelvic_band_mm"] = [float(pelvic_band_mm[0]), float(pelvic_band_mm[1])] if pelvic_band_mm else None

    pet_crop_bbox = _build_pet_crop_bbox(
        mask,
        pet_volume,
        geom_seg,
        pet_geom,
        bladder_bbox,
        pelvic_band_vox,
        pet_inferior_extension_frac,
        debug_info,
    )
    bbox_pet = _pet_bladder_guess(pet_volume, crop_bbox=pet_crop_bbox) if pet_volume is not None else None

    if debug_info is not None:
        debug_info["pet_bladder_guess"] = {
            "crop_bbox_vox": _bbox_to_dict(pet_crop_bbox),
            "bbox_vox": _bbox_to_dict(bbox_pet),
        }

    centroid = None
    method = None
    initial_anchor_source = None
    pet_fused_into_anchor = False
    if bladder_bbox is not None:
        centroid = _centroid_from_bbox(bladder_bbox)
        method = "bladder_anchor"
        initial_anchor_source = "bladder_segmentation"
    elif bbox_pet is not None:
        pet_centroid = _centroid_from_bbox(bbox_pet)
        if pet_geom is not None:
            centroid = pet_geom.map_point_to(geom_seg, np.array(pet_centroid))
        else:
            pet_shape = np.array(pet_volume.shape, dtype=float) if pet_volume is not None else np.array([1.0, 1.0, 1.0])
            seg_shape = np.array(mask.shape, dtype=float)
            centroid = np.array(pet_centroid, dtype=float) * (seg_shape / pet_shape)
        method = "pet_only"
        initial_anchor_source = "pet_hotspot"
        pet_fused_into_anchor = True
    else:
        return None

    inferior_step = -1

    if debug_info is not None:
        debug_info["initial_centroid_vox"] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
        debug_info["initial_method"] = str(method)
        debug_info["initial_anchor_source"] = str(initial_anchor_source)
        debug_info["pet_fused_into_initial_anchor"] = bool(pet_fused_into_anchor)
        debug_info["bladder_anchor_only"] = bool(bladder_bbox is not None)
        debug_info["inferior_step"] = int(inferior_step)
    
    lateral_lims = None
    if femoral_head_l is not None and femoral_head_r is not None:
        x_vals = [
            femoral_head_l[0][2], femoral_head_l[1][2],
            femoral_head_r[0][2], femoral_head_r[1][2],
        ]
        lateral_lims = (int(min(x_vals)), int(max(x_vals)))

    if debug_info is not None:
        debug_info["lateral_lims_from_heads_vox"] = [int(v) for v in lateral_lims] if lateral_lims is not None else None

    centroid, method = _run_ct_pelvic_refinement(
        centroid,
        method,
        mask,
        spacing,
        origin,
        geom_seg,
        ct_volume,
        ct_geom,
        ct_index_aligned,
        ct_spacing_eff,
        ct_origin_eff,
        pet_aligned,
        pet_volume,
        bladder_bbox,
        rectum_bbox,
        colon_bbox,
        lateral_lims,
        lateral_from_heads_pad_mm,
        bladder_id,
        rectum_id,
        colon_ids,
        inferior_step,
        debug_info,
    )
             
    if method is None or not method.startswith("ct"):
        if bladder_bbox is not None:
            shift_vox = (inferior_shift_mm / max(dz, 1e-6)) * float(inferior_step)
            centroid[0] = centroid[0] + shift_vox
            if debug_info is not None:
                debug_info["fallback_shift_z_vox"] = float(shift_vox)

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

    ves_z_min = None
    if ves_bbox is not None:
        ves_z_min = int(ves_bbox[0][0])

    lateral_min = None
    lateral_max = None
    if femoral_head_l is not None and femoral_head_r is not None:
        x_vals = [femoral_head_l[0][2], femoral_head_l[1][2], femoral_head_r[0][2], femoral_head_r[1][2]]
        lateral_min = min(x_vals) - lateral_from_heads_pad_mm / max(dx, 1e-6)
        lateral_max = max(x_vals) + lateral_from_heads_pad_mm / max(dx, 1e-6)
        method += "_lat_heads"

        if debug_info is not None:
            debug_info["lateral_mm_from_heads"] = [float(lateral_min), float(lateral_max)]

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

    bbox_mm = geom_seg.bbox_vox_to_mm((
        np.array([z_min_pad, y_min_pad, x_min_pad]),
        np.array([z_max_pad, y_max_pad, x_max_pad]),
    ))

    center_mm = geom_seg.vox_to_mm(center_vox).tolist()

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
