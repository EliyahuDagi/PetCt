"""Segmentor abstractions and implementations (disk, PET-only box locator)."""
import os
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
        target_mm: Tuple[float, float, float] = (20.0, 20.0, 16.0),
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
        # Suppress superior third (head/arms) using physical Z so ordering doesn't matter
        dz, dy, dx = spacing
        oz, _, _ = origin
        z_coords = oz + np.arange(pet.shape[0]) * dz
        top_cut = np.percentile(z_coords, 67) if pet.shape[0] > 0 else None
        if top_cut is not None:
            mask_head = z_coords >= top_cut
            if mask_head.any():
                pet[mask_head, :, :] = 0
        positive = pet[pet > 0]
        if positive.size == 0:
            return None

        # Smooth to reduce noise
        if _HAVE_SCIPY and gaussian_filter is not None:
            pet_smooth = gaussian_filter(pet, sigma=1.0)
        else:
            pet_smooth = pet

        kernel = self._kernel(spacing)
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
        )