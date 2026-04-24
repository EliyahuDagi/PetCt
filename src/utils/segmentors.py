"""Segmentor abstractions and implementations (disk, PET-only box locator)."""
import os
import json
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import convolve, gaussian_filter, label  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dep
    convolve = None
    gaussian_filter = None
    label = None
    _HAVE_SCIPY = False


@dataclass
class SegmentationResult:
    mask: Optional[np.ndarray] = None
    labels: Optional[Dict[int, str]] = None
    bbox_vox: Optional[Dict[str, Tuple[int, int]]] = None
    bbox_mm: Optional[Dict[str, Tuple[float, float]]] = None
    center_vox: Optional[Tuple[int, int, int]] = None
    center_mm: Optional[Tuple[float, float, float]] = None
    method: str = ""
    debug: Optional[dict] = None


class BaseSegmentor:
    def segment(
        self,
        patient_path: str,
        ct_volume: Optional[np.ndarray],
        pet_volume: Optional[np.ndarray],
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ) -> Optional[SegmentationResult]:
        raise NotImplementedError


class DiskSegmentor(BaseSegmentor):
    """Load mask/labels from disk under Segmentation/<source>/mask.npy."""

    def __init__(self, source_name: str, seg_dir_name: str):
        self.source_name = source_name
        self.seg_dir_name = seg_dir_name

    def segment(
        self,
        patient_path: str,
        ct_volume: Optional[np.ndarray],
        pet_volume: Optional[np.ndarray],
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ) -> Optional[SegmentationResult]:
        base_dir = os.path.join(patient_path, self.seg_dir_name)
        candidates = [os.path.join(base_dir, self.source_name), base_dir]

        for seg_dir in candidates:
            seg_path = os.path.join(seg_dir, "mask.npy")
            if not os.path.exists(seg_path):
                continue
            try:
                mask = np.load(seg_path)
                labels_path = os.path.join(seg_dir, "labels.json")
                labels = None
                if os.path.exists(labels_path):
                    import json

                    with open(labels_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    labels = {int(k): v for k, v in data.items()}
                return SegmentationResult(
                    mask=mask,
                    labels=labels or {},
                    method=f"disk:{os.path.basename(seg_dir)}",
                )
            except Exception as e:  # pragma: no cover - defensive
                print(f"Error loading segmentation from {seg_path}: {e}")
                continue
        return None


class PetBoxSegmentor(BaseSegmentor):
    """PET-only locator: box filter to find hottest pelvic focus, returns bbox (no mask)."""

    def __init__(
        self,
        target_mm: Tuple[float, float, float] = (24.0, 24.0, 20.0),
        threshold_frac: float = 0.65,
    ):
        self.target_mm = target_mm
        self.threshold_frac = threshold_frac

    def _kernel(self, spacing: Tuple[float, float, float]) -> np.ndarray:
        dz, dy, dx = spacing
        kz = max(1, int(round(self.target_mm[2] / max(dz, 1e-6))))
        ky = max(1, int(round(self.target_mm[1] / max(dy, 1e-6))))
        kx = max(1, int(round(self.target_mm[0] / max(dx, 1e-6))))
        # force odd sizes to have a clear center
        if kz % 2 == 0: kz += 1
        if ky % 2 == 0: ky += 1
        if kx % 2 == 0: kx += 1
        return np.ones((kz, ky, kx), dtype=np.float32)

    def _convolve_sum(self, vol: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # Fast path: scipy uniform_filter (O(n)) then rescale to sum
        if _HAVE_SCIPY and gaussian_filter is not None:
            try:
                from scipy.ndimage import uniform_filter  # type: ignore

                return uniform_filter(vol, size=kernel.shape, mode="constant", cval=0.0) * float(kernel.size)
            except Exception:
                pass
        # NumPy fallback: integral image for box sum (O(n))
        kz, ky, kx = kernel.shape
        pad = ((kz // 2, kz - kz // 2 - 1), (ky // 2, ky - ky // 2 - 1), (kx // 2, kx - kx // 2 - 1))
        v = np.pad(vol, pad, mode="constant")
        # integral image with leading zero plane to simplify indexing
        ii = v.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

        # helper to slice inclusive cube sum using integral image
        def box_sum(z0, y0, x0, z1, y1, x1):
            return (
                ii[z1, y1, x1]
                - ii[z0, y1, x1]
                - ii[z1, y0, x1]
                - ii[z1, y1, x0]
                + ii[z0, y0, x1]
                + ii[z0, y1, x0]
                + ii[z1, y0, x0]
                - ii[z0, y0, x0]
            )

        z_len, y_len, x_len = vol.shape
        resp = np.empty_like(vol, dtype=np.float32)
        for z in range(z_len):
            z0 = z
            z1 = z + kz - 1
            for y in range(y_len):
                y0 = y
                y1 = y + ky - 1
                # vectorize x for speed using slicing
                x0 = np.arange(0, x_len)
                x1 = x0 + kx - 1
                # gather via broadcasting
                top = ii[z1, y1, x1]
                a = ii[z0, y1, x1]
                b = ii[z1, y0, x1]
                c = ii[z1, y1, x0]
                d = ii[z0, y0, x1]
                e = ii[z0, y1, x0]
                f = ii[z1, y0, x0]
                g = ii[z0, y0, x0]
                resp[z, y, :] = top - a - b - c + d + e + f - g
        return resp

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        if _HAVE_SCIPY and label is not None:
            lbl, n = label(mask)
            if n <= 1:
                return (lbl > 0).astype(np.uint8)
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0
            return (lbl == sizes.argmax()).astype(np.uint8)
        # fallback: return as-is
        return mask

    def segment(
        self,
        patient_path: str,
        ct_volume: Optional[np.ndarray],
        pet_volume: Optional[np.ndarray],
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ) -> Optional[SegmentationResult]:
        if pet_volume is None:
            return None
        pet = np.asarray(pet_volume, dtype=np.float32)

        debug_payload: Dict[str, object] = {
            "kernel_mm": list(self.target_mm),
            "threshold_frac": float(self.threshold_frac),
        }

        # Optional pelvic band derived from TotalSegmentor anchors (if present on disk)
        pelvic_band_pet = None
        band_source = None
        band_anchor_detail: Dict[str, object] = {}
        try:
            from src.utils.config import Config  # type: ignore

            ts_dir = os.path.join(
                patient_path,
                getattr(Config, "SEGMENTATION_DIR_NAME", "Segmentation"),
                "TotalSegmentor",
            )
            mask_path = os.path.join(ts_dir, "mask.npy")
            labels_path = os.path.join(ts_dir, "labels.json")
            if os.path.exists(mask_path) and os.path.exists(labels_path):
                ts_mask = np.load(mask_path)
                with open(labels_path, "r", encoding="utf-8") as f:
                    raw_labels = json.load(f)
                labels = {int(k): str(v).lower() for k, v in raw_labels.items()}

                def _find_id(names):
                    for k, name in labels.items():
                        for n in names:
                            if n in name:
                                return int(k)
                    return None

                def _bbox_z_for_id(lab_id):
                    if lab_id is None:
                        return None
                    coords = np.argwhere(ts_mask == int(lab_id))
                    if coords.size == 0:
                        return None
                    z0, z1 = coords[:, 0].min(), coords[:, 0].max()
                    return int(z0), int(z1)

                bladder_id = _find_id(["urinary_bladder", "bladder"])
                rectum_id = _find_id(["rectum"])
                colon_id = _find_id(["colon", "large_bowel", "large bowel", "bowel", "sigmoid", "rectosigmoid"])
                fem_l = _find_id(["femur_head_left", "femoral_head_left", "femur head left", "femoral head left"])
                fem_r = _find_id(["femur_head_right", "femoral_head_right", "femur head right", "femoral head right"])

                pet_z = pet.shape[0]

                def _scale_z(z_idx: int) -> int:
                    return int(round(z_idx * pet_z / max(1, ts_mask.shape[0] - 1)))

                def _pad_vox_mm(mm: float) -> int:
                    dz_pad, _, _ = spacing
                    return int(round(mm / max(dz_pad, 1e-6)))

                band = None
                b_bbox = _bbox_z_for_id(bladder_id)
                if b_bbox is not None:
                    z0, z1 = _scale_z(b_bbox[0]), _scale_z(b_bbox[1])
                    lo = max(0, z0 - _pad_vox_mm(80.0))
                    hi = min(pet_z - 1, z1 + _pad_vox_mm(60.0))
                    band_source = "bladder"
                    band_anchor_detail["ts_z"] = [int(b_bbox[0]), int(b_bbox[1])]
                    band = (lo, hi)

                if band is None:
                    r_bbox = _bbox_z_for_id(rectum_id)
                    if r_bbox is not None:
                        z0, z1 = _scale_z(r_bbox[0]), _scale_z(r_bbox[1])
                        lo = max(0, z0 - _pad_vox_mm(60.0))
                        hi = min(pet_z - 1, z1 + _pad_vox_mm(40.0))
                        band_source = "rectum"
                        band_anchor_detail["ts_z"] = [int(r_bbox[0]), int(r_bbox[1])]
                        band = (lo, hi)

                if band is None:
                    c_bbox = _bbox_z_for_id(colon_id)
                    if c_bbox is not None:
                        z0_ts, z1_ts = int(c_bbox[0]), int(c_bbox[1])
                        z0, z1 = _scale_z(z0_ts), _scale_z(z1_ts)
                        height = max(1, z1 - z0 + 1)
                        lower_hi = z0 + int(math.ceil(height * 0.4))  # inferior 40% of colon span
                        lo = max(0, min(z0 - _pad_vox_mm(30.0), int(pet_z * 0.35)))
                        hi = min(pet_z - 1, lower_hi + _pad_vox_mm(15.0), int(pet_z * 0.65))
                        band_source = "colon_lower"
                        band_anchor_detail["ts_z"] = [z0_ts, z1_ts]
                        band_anchor_detail["pet_z_full"] = [int(z0), int(z1)]
                        band = (lo, hi)

                if band is None and fem_l is not None and fem_r is not None:
                    fl = _bbox_z_for_id(fem_l)
                    fr = _bbox_z_for_id(fem_r)
                    if fl is not None and fr is not None:
                        z_vals = [_scale_z(fl[0]), _scale_z(fl[1]), _scale_z(fr[0]), _scale_z(fr[1])]
                        lo = max(0, int(min(z_vals) - _pad_vox_mm(40.0)))
                        hi = min(pet_z - 1, int(max(z_vals) + _pad_vox_mm(80.0)))
                        band_source = "femoral_heads"
                        band_anchor_detail["ts_z"] = [int(min(fl[0], fr[0])), int(max(fl[1], fr[1]))]
                        band_anchor_detail["femoral_heads_ts_z"] = [[int(fl[0]), int(fl[1])], [int(fr[0]), int(fr[1])]]
                        band = (lo, hi)

                if band is None:
                    hi = pet_z - 1
                    mid = int(round(pet_z * 0.45))
                    band_source = "inferior_half"
                    band_anchor_detail["ts_z"] = None
                    band = (mid, hi)

                if band is not None:
                    pelvic_band_pet = (int(band[0]), int(band[1]))
                    if pelvic_band_pet[0] > 0:
                        pet[:pelvic_band_pet[0], :, :] = 0
                    if pelvic_band_pet[1] < pet.shape[0] - 1:
                        pet[pelvic_band_pet[1] + 1 :, :, :] = 0
                    dz_curr, _, _ = spacing
                    oz_curr, _, _ = origin
                    debug_payload["pelvic_band_pet_vox"] = list(pelvic_band_pet)
                    debug_payload["pelvic_band_pet_mm"] = [
                        float(oz_curr + pelvic_band_pet[0] * dz_curr),
                        float(oz_curr + pelvic_band_pet[1] * dz_curr),
                    ]
                    debug_payload["pelvic_band_anchor"] = {
                        "source": band_source,
                        "ts_z": band_anchor_detail.get("ts_z"),
                        "femoral_heads_ts_z": band_anchor_detail.get("femoral_heads_ts_z"),
                        "pet_z": [int(pelvic_band_pet[0]), int(pelvic_band_pet[1])],
                        "pet_mm": [
                            float(oz_curr + pelvic_band_pet[0] * dz_curr),
                            float(oz_curr + pelvic_band_pet[1] * dz_curr),
                        ],
                    }
        except Exception:
            pelvic_band_pet = None

        # Suppress superior third (head/arms) using physical Z so ordering doesn't matter
        dz, dy, dx = spacing
        oz, _, _ = origin
        z_coords = oz + np.arange(pet.shape[0]) * dz
        top_cut = np.percentile(z_coords, 67) if pet.shape[0] > 0 else None
        if top_cut is not None:
            mask_head = z_coords >= top_cut
            if mask_head.any():
                pet[mask_head, :, :] = 0
                debug_payload["head_cut_mm"] = float(top_cut)
        positive = pet[pet > 0]
        if positive.size == 0:
            return None

        # Smooth to reduce noise
        if _HAVE_SCIPY and gaussian_filter is not None:
            pet_smooth = gaussian_filter(pet, sigma=1.0)
        else:
            pet_smooth = pet

        kernel = self._kernel(spacing)
        debug_payload["kernel_vox"] = list(kernel.shape)
        response = self._convolve_sum(pet_smooth, kernel)
        max_val = float(response.max()) if response.size else 0.0
        if max_val <= 0.0:
            return None

        peak = np.unravel_index(np.argmax(response), response.shape)

        # Threshold the response (windowed sum) not the raw PET
        thresh = max_val * self.threshold_frac
        hot_mask = (response >= thresh).astype(np.uint8)
        # Keep only component containing the peak
        if hot_mask[peak] == 0:
            hot_mask[peak] = 1
        hot_mask = self._largest_component(hot_mask)

        coords = np.argwhere(hot_mask > 0)
        if coords.size == 0:
            return None
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        center_vox = (
            int((z_min + z_max) // 2),
            int((y_min + y_max) // 2),
            int((x_min + x_max) // 2),
        )
        dz, dy, dx = spacing
        oz, oy, ox = origin
        bbox_mm = {
            "z": (float(oz + z_min * dz), float(oz + z_max * dz)),
            "y": (float(oy + y_min * dy), float(oy + y_max * dy)),
            "x": (float(ox + x_min * dx), float(ox + x_max * dx)),
        }
        debug_payload["pet_hot_bbox_vox"] = {"z": (int(z_min), int(z_max)), "y": (int(y_min), int(y_max)), "x": (int(x_min), int(x_max))}
        debug_payload["pet_hot_bbox_mm"] = bbox_mm
        center_mm = (
            float(oz + center_vox[0] * dz),
            float(oy + center_vox[1] * dy),
            float(ox + center_vox[2] * dx),
        )

        return SegmentationResult(
            mask=None,
            labels={},
            bbox_vox={"z": (z_min, z_max), "y": (y_min, y_max), "x": (x_min, x_max)},
            bbox_mm=bbox_mm,
            center_vox=center_vox,
            center_mm=center_mm,
            method="pet_box_filter",
            debug=debug_payload,
        )