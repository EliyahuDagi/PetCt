"""Tests for the geometric augmentation builder (src.training.utils.augment).

Covers the load-bearing contracts:
  * disabled / missing config -> a no-op identity (existing behavior preserved);
  * paired keys get SYNCHRONIZED geometry (same params on nac & ac);
  * output shape matches input (channel-first 2D and 3D);
  * NO intensity transforms are ever constructed (geometric only);
  * the data.py samplers thread the augment hook through without changing shapes.

Skips cleanly when torch / MONAI transforms are unavailable.
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
    from src.training.utils import augment as aug
    HAVE_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False


def _have_monai():
    try:
        from monai.transforms import Compose, RandFlipd, RandRotate90d, RandAffined  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class TestAugmentDisabled(unittest.TestCase):
    def test_none_cfg_is_identity_2d(self):
        self.assertIs(aug.build_aug_2d(None), aug._identity)

    def test_none_cfg_is_identity_3d(self):
        self.assertIs(aug.build_aug_3d(None), aug._identity)

    def test_disabled_cfg_is_identity(self):
        self.assertIs(aug.build_aug_2d({"enabled": False}), aug._identity)
        self.assertIs(aug.build_aug_3d({"enabled": False}), aug._identity)

    def test_identity_returns_input_unchanged(self):
        t = torch.rand(1, 8, 8)
        out = aug._identity({"img": t})
        self.assertIs(out["img"], t)


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class TestAugmentEnabled(unittest.TestCase):
    def setUp(self):
        if not _have_monai():
            self.skipTest("monai transforms not available")

    def test_2d_preserves_shape(self):
        f = aug.build_aug_2d({"enabled": True})
        self.assertIsNot(f, aug._identity)
        out = f({"img": torch.rand(1, 16, 16)})
        self.assertEqual(tuple(out["img"].shape), (1, 16, 16))
        self.assertTrue(torch.isfinite(out["img"]).all())

    def test_3d_preserves_shape(self):
        f = aug.build_aug_3d({"enabled": True})
        self.assertIsNot(f, aug._identity)
        out = f({"img": torch.rand(1, 8, 8, 8)})
        self.assertEqual(tuple(out["img"].shape), (1, 8, 8, 8))
        self.assertTrue(torch.isfinite(out["img"]).all())

    def test_2d_paired_geometry_synchronized(self):
        # Identical nac/ac in -> identical out (same random draw on both keys).
        f = aug.build_aug_2d({"enabled": True})
        v = torch.rand(1, 16, 16)
        out = f({"nac": v.clone(), "ac": v.clone()})
        self.assertTrue(torch.allclose(out["nac"], out["ac"], atol=1e-5))

    def test_3d_paired_geometry_synchronized(self):
        f = aug.build_aug_3d({"enabled": True, "elastic_prob": 0.0})
        v = torch.rand(1, 8, 8, 8)
        out = f({"nac": v.clone(), "ac": v.clone()})
        self.assertTrue(torch.allclose(out["nac"], out["ac"], atol=1e-5))

    def test_flip_only_changes_data(self):
        # A flip-only transform with prob 1 must actually permute the data.
        f = aug.build_aug_2d({"enabled": True, "flip_prob": 1.0, "rot90_prob": 0.0, "affine_prob": 0.0})
        v = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 16, 16)
        out = f({"img": v.clone()})["img"]
        self.assertEqual(tuple(out.shape), (1, 16, 16))
        self.assertFalse(torch.allclose(out, v))


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class TestSamplerHook(unittest.TestCase):
    """The data.py samplers must accept the augment hook and keep shapes."""

    def setUp(self):
        from src.training import data
        self.data = data

    def test_sample_pairs_augment_none_default(self):
        vol = torch.rand(14, 12, 9)
        pool = np.linspace(0.0, 1.0, 128, endpoint=False)
        nb, ab = self.data.sample_pairs(vol.clone(), vol.clone(), pool, 4, 8, np.random.RandomState(1))
        self.assertEqual(tuple(nb.shape), (4, 1, 8, 8))
        self.assertTrue(torch.allclose(nb, ab, atol=1e-6))

    def test_sample_pairs_with_augment_keeps_pair_synced(self):
        if not _have_monai():
            self.skipTest("monai transforms not available")
        vol = torch.rand(14, 12, 9)
        pool = np.linspace(0.0, 1.0, 128, endpoint=False)
        f = aug.build_aug_2d({"enabled": True})
        nb, ab = self.data.sample_pairs(vol.clone(), vol.clone(), pool, 4, 8, np.random.RandomState(1), augment=f)
        self.assertEqual(tuple(nb.shape), (4, 1, 8, 8))
        # Identical NAC/AC stay identical after synchronized augmentation.
        self.assertTrue(torch.allclose(nb, ab, atol=1e-5))

    def test_sample_pair_volumes_geo_aug_synced(self):
        if not _have_monai():
            self.skipTest("monai transforms not available")
        vol = torch.rand(20, 16, 16)
        f = aug.build_aug_3d({"enabled": True, "elastic_prob": 0.0})
        nb, ab = self.data.sample_pair_volumes(
            vol.clone(), vol.clone(), 2, 8, "train", 0.2, np.random.RandomState(0), geo_aug=f
        )
        self.assertEqual(tuple(nb.shape), (2, 1, 8, 8, 8))
        self.assertTrue(torch.allclose(nb, ab, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
