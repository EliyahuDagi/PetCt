"""Tests for 3D volume augmentation (random crop / flip / rotation) used by the
multi-patient 3D stages (ae3d, ft3d), and for multi-plane 2D slicing (ae2d,
diff2d). These exercise src.training.data, which imports torch; the suite skips
cleanly when torch is unavailable.

Background: the 3D samplers previously resized each whole volume to a fixed cube,
so a small paired pool (e.g. ACRIN's ~20 patients) produced one identical example
per patient. Augmentation turns that into a diverse set; validation stays clean.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import numpy as np
    import torch
    from src.training import data
    HAVE_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestVolumeAugmentation(unittest.TestCase):
    SIZE = 8

    def _vol(self, shape, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.rand(shape, generator=g)  # values in [0,1)

    # --- helpers ---
    def test_draw_volume_aug_param_ranges(self):
        rng = np.random.RandomState(0)
        for _ in range(200):
            p = data.draw_volume_aug(rng, min_frac=0.7)
            self.assertEqual(len(p["windows"]), 3)
            for s, e in p["windows"]:
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(e, 1.0 + 1e-9)
                frac = e - s
                # frac must cover at least min_frac (minus rounding slack)
                self.assertGreaterEqual(frac, 0.7 - 1e-9)
            self.assertEqual(len(p["flips"]), 3)
            self.assertIn(p["rot_k"], (0, 1, 2, 3))

    def test_neutral_aug_equals_resize(self):
        # A no-op transform (full window, no flips, no rotation) must equal _resize_volume.
        vol = self._vol((10, 12, 9))
        neutral = {"windows": [(0.0, 1.0)] * 3, "flips": [False, False, False], "rot_k": 0}
        out = data.apply_volume_aug(vol, self.SIZE, neutral)
        ref = data._resize_volume(vol, self.SIZE)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_apply_volume_aug_shape_invariant(self):
        vol = self._vol((11, 13, 7))
        rng = np.random.RandomState(1)
        for _ in range(20):
            out = data.apply_volume_aug(vol, self.SIZE, data.draw_volume_aug(rng))
            self.assertEqual(tuple(out.shape), (1, self.SIZE, self.SIZE, self.SIZE))
            self.assertTrue(torch.isfinite(out).all())
            # trilinear interpolation of [0,1) values stays in range; flips/rot preserve it
            self.assertGreaterEqual(float(out.min()), -1e-5)
            self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    # --- sample_pair_volumes_full ---
    def test_pair_no_augment_matches_resize(self):
        # Default (augment=False) must reproduce the original fixed-resize behavior.
        nac, ac = self._vol((20, 16, 16), 1), self._vol((24, 16, 16), 2)
        rng = np.random.RandomState(3)
        nb, ab = data.sample_pair_volumes_full(nac, ac, 3, self.SIZE, rng, augment=False)
        self.assertEqual(tuple(nb.shape), (3, 1, self.SIZE, self.SIZE, self.SIZE))
        self.assertEqual(tuple(ab.shape), (3, 1, self.SIZE, self.SIZE, self.SIZE))
        # Every entry equals the deterministic whole-volume resize (no diversity).
        rn, ra = data._resize_volume(nac, self.SIZE), data._resize_volume(ac, self.SIZE)
        for i in range(3):
            self.assertTrue(torch.allclose(nb[i], rn, atol=1e-6))
            self.assertTrue(torch.allclose(ab[i], ra, atol=1e-6))

    def test_pair_augment_applies_same_transform_to_both_halves(self):
        # If NAC and AC are identical, the augmented pair must stay identical
        # (same crop/flip/rotation applied to both) -- preserves correspondence.
        vol = self._vol((22, 16, 16), 7)
        rng = np.random.RandomState(5)
        nb, ab = data.sample_pair_volumes_full(vol.clone(), vol.clone(), 6, self.SIZE, rng, augment=True)
        self.assertTrue(torch.allclose(nb, ab, atol=1e-6))

    def test_pair_augment_produces_diversity(self):
        # Augmented samples from the SAME patient must differ across the batch,
        # unlike the fixed-resize path. This is the core of the fix.
        nac, ac = self._vol((20, 16, 16), 1), self._vol((20, 16, 16), 1)
        rng = np.random.RandomState(0)
        nb, _ = data.sample_pair_volumes_full(nac, ac, 8, self.SIZE, rng, augment=True)
        # Count distinct cubes; expect clearly more than one.
        flat = nb.reshape(8, -1)
        distinct = {tuple(torch.round(flat[i] * 1e4).to(torch.int64).tolist()) for i in range(8)}
        self.assertGreater(len(distinct), 1)

    def test_pair_augment_deterministic_with_seed(self):
        nac, ac = self._vol((18, 16, 16), 1), self._vol((21, 16, 16), 2)
        a = data.sample_pair_volumes_full(nac, ac, 4, self.SIZE, np.random.RandomState(42), augment=True)
        b = data.sample_pair_volumes_full(nac, ac, 4, self.SIZE, np.random.RandomState(42), augment=True)
        self.assertTrue(torch.allclose(a[0], b[0], atol=1e-6))
        self.assertTrue(torch.allclose(a[1], b[1], atol=1e-6))

    # --- sample_volumes_full (single channel, ae3d) ---
    def test_volumes_full_augment_shape_and_diversity(self):
        vol = self._vol((20, 16, 16), 9)
        rng = np.random.RandomState(2)
        out = data.sample_volumes_full([vol], 8, self.SIZE, rng, augment=True)
        self.assertEqual(tuple(out.shape), (8, 1, self.SIZE, self.SIZE, self.SIZE))
        flat = out.reshape(8, -1)
        distinct = {tuple(torch.round(flat[i] * 1e4).to(torch.int64).tolist()) for i in range(8)}
        self.assertGreater(len(distinct), 1)

    def test_volumes_full_no_augment_is_fixed(self):
        vol = self._vol((20, 16, 16), 9)
        out = data.sample_volumes_full([vol], 5, self.SIZE, np.random.RandomState(0), augment=False)
        ref = data._resize_volume(vol, self.SIZE)
        for i in range(5):
            self.assertTrue(torch.allclose(out[i], ref, atol=1e-6))


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestMultiPlaneSlicing(unittest.TestCase):
    SIZE = 8

    def test_slice_at_each_axis_shape(self):
        vol = torch.arange(10 * 12 * 9, dtype=torch.float32).reshape(10, 12, 9)
        for axis in (0, 1, 2):
            s = data._slice_at(vol, 0.5, self.SIZE, axis=axis)
            self.assertEqual(tuple(s.shape), (1, self.SIZE, self.SIZE))

    def test_sample_slices_uses_all_three_planes(self):
        vol = torch.rand(10, 12, 9)
        pool = np.linspace(0.0, 1.0, 128, endpoint=False)
        rng = np.random.RandomState(0)
        seen = set()
        orig = data._slice_at

        def spy(v, t, size, axis=0):
            seen.add(axis)
            return orig(v, t, size, axis)

        data._slice_at = spy
        try:
            for _ in range(60):
                data.sample_slices([vol], pool, 4, self.SIZE, rng)
        finally:
            data._slice_at = orig
        self.assertEqual(seen, {0, 1, 2})

    def test_sample_pairs_same_plane_for_both(self):
        # Identical NAC/AC -> paired slices identical (same plane + position).
        vol = torch.rand(14, 12, 9)
        pool = np.linspace(0.0, 1.0, 128, endpoint=False)
        nb, ab = data.sample_pairs(vol.clone(), vol.clone(), pool, 8, self.SIZE, np.random.RandomState(1))
        self.assertEqual(tuple(nb.shape), (8, 1, self.SIZE, self.SIZE))
        self.assertTrue(torch.allclose(nb, ab, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
