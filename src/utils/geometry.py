"""Geometry helpers for mapping between voxel and physical coordinates across volumes."""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VolumeGeometry:
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    shape: Tuple[int, int, int]

    def vox_to_mm(self, indexes: np.ndarray) -> np.ndarray:
        idx = np.asarray(indexes, dtype=float)
        dz, dy, dx = self.spacing
        oz, oy, ox = self.origin
        return np.array([
            oz + idx[0] * dz,
            oy + idx[1] * dy,
            ox + idx[2] * dx,
        ], dtype=float)

    def mm_to_vox(self, coords_mm: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords_mm, dtype=float)
        dz, dy, dx = self.spacing
        oz, oy, ox = self.origin
        return np.array([
            (coords[0] - oz) / max(dz, 1e-6),
            (coords[1] - oy) / max(dy, 1e-6),
            (coords[2] - ox) / max(dx, 1e-6),
        ], dtype=float)

    def _clamp_range(self, lo: int, hi: int, axis: int) -> Tuple[int, int]:
        limit = self.shape[axis]
        lo = max(0, int(lo))
        hi = min(limit - 1, int(hi))
        return lo, hi

    def clamp_bbox(self, bbox: Tuple[np.ndarray, np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        (z0, y0, x0), (z1, y1, x1) = bbox
        z0, z1 = self._clamp_range(z0, z1, 0)
        y0, y1 = self._clamp_range(y0, y1, 1)
        x0, x1 = self._clamp_range(x0, x1, 2)
        if z0 > z1 or y0 > y1 or x0 > x1:
            return None
        return np.array([z0, y0, x0]), np.array([z1, y1, x1])

    def bbox_vox_to_mm(self, bbox: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        lo_mm = self.vox_to_mm(bbox[0])
        hi_mm = self.vox_to_mm(bbox[1])
        return {
            "z": (float(min(lo_mm[0], hi_mm[0])), float(max(lo_mm[0], hi_mm[0]))),
            "y": (float(min(lo_mm[1], hi_mm[1])), float(max(lo_mm[1], hi_mm[1]))),
            "x": (float(min(lo_mm[2], hi_mm[2])), float(max(lo_mm[2], hi_mm[2]))),
        }

    def bbox_mm_to_vox(self, bbox_mm: Dict[str, Tuple[float, float]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            z0_mm, z1_mm = bbox_mm["z"]
            y0_mm, y1_mm = bbox_mm["y"]
            x0_mm, x1_mm = bbox_mm["x"]
        except Exception:
            return None

        dz, dy, dx = self.spacing
        oz, oy, ox = self.origin

        z0 = math.floor((min(z0_mm, z1_mm) - oz) / max(dz, 1e-6))
        z1 = math.ceil((max(z0_mm, z1_mm) - oz) / max(dz, 1e-6)) - 1
        y0 = math.floor((min(y0_mm, y1_mm) - oy) / max(dy, 1e-6))
        y1 = math.ceil((max(y0_mm, y1_mm) - oy) / max(dy, 1e-6)) - 1
        x0 = math.floor((min(x0_mm, x1_mm) - ox) / max(dx, 1e-6))
        x1 = math.ceil((max(x0_mm, x1_mm) - ox) / max(dx, 1e-6)) - 1

        return self.clamp_bbox((np.array([z0, y0, x0]), np.array([z1, y1, x1])))

    def z_band_mm_to_vox(self, z_band_mm: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        z0_mm, z1_mm = z_band_mm
        dz, _, _ = self.spacing
        oz, _, _ = self.origin
        z_lo = int(math.floor((min(z0_mm, z1_mm) - oz) / max(dz, 1e-6)))
        z_hi = int(math.ceil((max(z0_mm, z1_mm) - oz) / max(dz, 1e-6)) - 1)
        z_lo, z_hi = self._clamp_range(z_lo, z_hi, 0)
        if z_lo > z_hi:
            return None
        return z_lo, z_hi

    def map_bbox_to(self, target: "VolumeGeometry", bbox_vox: Tuple[np.ndarray, np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        bbox_mm = self.bbox_vox_to_mm(bbox_vox)
        return target.bbox_mm_to_vox(bbox_mm)

    def map_point_to(self, target: "VolumeGeometry", point_vox: np.ndarray) -> np.ndarray:
        point_mm = self.vox_to_mm(point_vox)
        return target.mm_to_vox(point_mm)
