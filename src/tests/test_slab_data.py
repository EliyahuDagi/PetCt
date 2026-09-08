"""CPU unit tests for the slab-mode samplers in ``src/training/data.py``.

NAC = non-attenuation-corrected PET, AC = attenuation-corrected PET.

Slab mode resizes a volume in-plane only, keeps every native slice, and trains on
random contiguous depth slabs. These tests pin the shapes, the determinism under a
seeded ``numpy.random.RandomState``, and the fact that an un-augmented slab is a
plain depth crop of the resized volume. Both families are covered: the paired
samplers the flow model uses, and the single-volume ones the autoencoder uses (it
pools whatever PET volumes a patient has, paired or not).

``reuse_split_json`` is tested here too: it is the guard that stops one run from
training on another run's held-out patients.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from src.training.data import (
    _resize_volume,
    resize_pair_native_depth,
    resize_volume_native_depth,
    reuse_split_json,
    sample_pair_slabs,
    sample_pair_slabs_band,
    sample_pair_volume_native,
    sample_pair_volume_native_band,
    sample_volume_native,
    sample_volume_slabs,
)


def _vol(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g)


class TestResizeVolume(unittest.TestCase):
    def test_int_gives_cube(self):
        out = _resize_volume(_vol((10, 20, 24), 0), 8)
        self.assertEqual(tuple(out.shape), (1, 8, 8, 8))

    def test_tuple_gives_requested_depth_rows_cols(self):
        out = _resize_volume(_vol((10, 20, 24), 0), (10, 16, 16))
        self.assertEqual(tuple(out.shape), (1, 10, 16, 16))

    def test_int_and_matching_tuple_agree(self):
        v = _vol((10, 20, 24), 1)
        self.assertTrue(torch.equal(_resize_volume(v, 8), _resize_volume(v, (8, 8, 8))))

    def test_bad_tuple_rejected(self):
        with self.assertRaises(ValueError):
            _resize_volume(_vol((4, 4, 4), 0), (4, 4))


class TestResizePairNativeDepth(unittest.TestCase):
    def test_depth_follows_ac_and_inplane_follows_size(self):
        ac = _vol((12, 30, 30), 0)
        nac = _vol((13, 28, 26), 1)  # one extra slice and a different in-plane grid
        nac_r, ac_r = resize_pair_native_depth(nac, ac, 16)
        self.assertEqual(tuple(ac_r.shape), (1, 12, 16, 16))
        self.assertEqual(tuple(nac_r.shape), (1, 12, 16, 16))
        self.assertEqual(ac_r.dtype, torch.float32)

    def test_missing_volume_rejected(self):
        with self.assertRaises(ValueError):
            resize_pair_native_depth(None, _vol((4, 8, 8), 0), 8)


class TestSamplePairSlabs(unittest.TestCase):
    def test_shapes_and_dtype(self):
        nac, ac = _vol((12, 30, 30), 0), _vol((12, 30, 30), 1)
        n, a = sample_pair_slabs(nac, ac, 3, 16, 4, np.random.RandomState(0), augment=False)
        self.assertEqual(tuple(n.shape), (3, 1, 4, 16, 16))
        self.assertEqual(tuple(a.shape), (3, 1, 4, 16, 16))
        self.assertEqual(n.dtype, torch.float32)
        self.assertEqual(a.dtype, torch.float32)

    def test_deterministic_for_same_seed(self):
        nac, ac = _vol((12, 30, 30), 0), _vol((12, 30, 30), 1)
        n1, a1 = sample_pair_slabs(nac, ac, 2, 16, 4, np.random.RandomState(0), augment=True)
        n2, a2 = sample_pair_slabs(nac, ac, 2, 16, 4, np.random.RandomState(0), augment=True)
        self.assertTrue(torch.equal(n1, n2))
        self.assertTrue(torch.equal(a1, a2))

    def test_unaugmented_slab_is_a_depth_crop_of_the_resized_pair(self):
        nac, ac = _vol((12, 30, 30), 0), _vol((12, 30, 30), 1)
        nac_r, ac_r = resize_pair_native_depth(nac, ac, 16)
        n, a = sample_pair_slabs(nac, ac, 4, 16, 4, np.random.RandomState(3), augment=False)
        for b in range(a.shape[0]):
            offsets = [z0 for z0 in range(ac_r.shape[1] - 4 + 1)
                       if torch.allclose(a[b], ac_r[:, z0:z0 + 4])]
            self.assertTrue(offsets, "AC slab %d is not a contiguous depth crop" % b)
            # The NAC slab must come from the SAME depth offset (paired crop).
            self.assertTrue(any(torch.allclose(n[b], nac_r[:, z0:z0 + 4]) for z0 in offsets),
                            "NAC slab %d does not share the AC slab's depth offset" % b)

    def test_short_volume_is_stretched_to_slab_depth(self):
        nac, ac = _vol((3, 20, 20), 0), _vol((3, 20, 20), 1)
        n, a = sample_pair_slabs(nac, ac, 2, 16, 4, np.random.RandomState(0), augment=True)
        self.assertEqual(tuple(n.shape), (2, 1, 4, 16, 16))
        self.assertEqual(tuple(a.shape), (2, 1, 4, 16, 16))


class TestSamplePairVolumeNative(unittest.TestCase):
    def test_shape_is_batch_of_one_native_depth(self):
        nac, ac = _vol((13, 30, 30), 0), _vol((12, 30, 30), 1)
        n, a = sample_pair_volume_native(nac, ac, 16)
        self.assertEqual(tuple(n.shape), (1, 1, 12, 16, 16))
        self.assertEqual(tuple(a.shape), (1, 1, 12, 16, 16))

    def test_equals_the_resized_pair(self):
        nac, ac = _vol((12, 30, 30), 0), _vol((12, 30, 30), 1)
        nac_r, ac_r = resize_pair_native_depth(nac, ac, 16)
        n, a = sample_pair_volume_native(nac, ac, 16)
        self.assertTrue(torch.equal(n[0], nac_r))
        self.assertTrue(torch.equal(a[0], ac_r))


class TestResizeVolumeNativeDepth(unittest.TestCase):
    """Single-volume resize: rows and columns follow ``size``, depth is left alone."""

    def test_inplane_follows_size_and_depth_is_native(self):
        out = resize_volume_native_depth(_vol((13, 30, 28), 0), 16)
        self.assertEqual(tuple(out.shape), (1, 13, 16, 16))
        self.assertEqual(out.dtype, torch.float32)

    def test_missing_volume_rejected(self):
        with self.assertRaises(ValueError):
            resize_volume_native_depth(None, 8)


class TestSampleVolumeSlabs(unittest.TestCase):
    """The autoencoder's training sampler: one volume per sample from the pool."""

    def test_shapes_and_dtype(self):
        pool = [_vol((12, 30, 30), 0), _vol((12, 28, 26), 1)]
        out = sample_volume_slabs(pool, 3, 16, 4, np.random.RandomState(0), augment=False)
        self.assertEqual(tuple(out.shape), (3, 1, 4, 16, 16))
        self.assertEqual(out.dtype, torch.float32)

    def test_unaugmented_slab_is_a_depth_crop_of_a_pooled_volume(self):
        pool = [_vol((12, 30, 30), 0), _vol((12, 28, 26), 1)]
        resized = [resize_volume_native_depth(v, 16) for v in pool]
        out = sample_volume_slabs(pool, 4, 16, 4, np.random.RandomState(3), augment=False)
        for b in range(out.shape[0]):
            found = any(torch.allclose(out[b], r[:, z0:z0 + 4])
                        for r in resized for z0 in range(r.shape[1] - 4 + 1))
            self.assertTrue(found, "slab %d is not a contiguous depth crop of a pooled volume" % b)

    def test_deterministic_for_same_seed(self):
        pool = [_vol((12, 30, 30), 0), _vol((12, 28, 26), 1)]
        a = sample_volume_slabs(pool, 2, 16, 4, np.random.RandomState(0), augment=True)
        b = sample_volume_slabs(pool, 2, 16, 4, np.random.RandomState(0), augment=True)
        self.assertTrue(torch.equal(a, b))

    def test_short_volume_is_stretched_to_slab_depth(self):
        out = sample_volume_slabs([_vol((3, 20, 20), 0)], 2, 16, 4,
                                  np.random.RandomState(0), augment=True)
        self.assertEqual(tuple(out.shape), (2, 1, 4, 16, 16))

    def test_augmentation_keeps_the_shape(self):
        pool = [_vol((12, 30, 30), 0)]
        plain = sample_volume_slabs(pool, 3, 16, 4, np.random.RandomState(1), augment=False)
        flipped = sample_volume_slabs(pool, 3, 16, 4, np.random.RandomState(1), augment=True)
        self.assertEqual(tuple(flipped.shape), tuple(plain.shape))

    def test_empty_pool_rejected(self):
        with self.assertRaises(ValueError):
            sample_volume_slabs([None], 1, 16, 4, np.random.RandomState(0))


class TestSampleVolumeNative(unittest.TestCase):
    """The autoencoder's validation sampler: the whole volume, every native slice."""

    def test_every_native_slice_is_returned(self):
        out = sample_volume_native(_vol((13, 30, 30), 0), 16)
        self.assertEqual(tuple(out.shape), (1, 1, 13, 16, 16))

    def test_equals_the_resized_volume(self):
        v = _vol((12, 30, 30), 0)
        self.assertTrue(torch.equal(sample_volume_native(v, 16)[0],
                                    resize_volume_native_depth(v, 16)))


class TestReuseSplitJson(unittest.TestCase):
    """Reusing another run's partition, so this run cannot train on its test patients."""

    PATIENTS = ["/data/a", "/data/b", "/data/c", "/data/d", "/data/e"]

    def _split_file(self, tmp, payload):
        path = os.path.join(tmp, "split.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    def test_extras_go_to_training_when_asked(self):
        # The file lists 3 of the 5 patients; the autoencoder pools unpaired volumes, so
        # the other 2 were never in that run's enumeration and are safe to train on.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._split_file(tmp, {"train": ["/data/a"], "val": ["/data/b"],
                                          "test": ["/data/c"]})
            train, val, test = reuse_split_json(path, self.PATIENTS, None, extra_to_train=True)
            self.assertEqual(sorted(train), [0, 3, 4])
            self.assertEqual(val, [1])
            self.assertEqual(test, [2])

    def test_extras_are_dropped_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._split_file(tmp, {"train": ["/data/a"], "val": ["/data/b"],
                                          "test": ["/data/c"]})
            train, val, test = reuse_split_json(path, self.PATIENTS, None)
            self.assertEqual(train, [0])
            self.assertEqual(val, [1])
            self.assertEqual(test, [2])

    def test_missing_test_patient_raises(self):
        # A test patient that cannot be found now makes the reused split unreproducible,
        # so the run must stop rather than quietly hold out a different set.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._split_file(tmp, {"train": ["/data/a"], "val": ["/data/b"],
                                          "test": ["/data/gone"]})
            with self.assertRaises(ValueError):
                reuse_split_json(path, self.PATIENTS, None, extra_to_train=True)

    def test_not_a_split_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._split_file(tmp, {"train": ["/data/a"]})
            with self.assertRaises(ValueError):
                reuse_split_json(path, self.PATIENTS, None)


class TestBandVariants(unittest.TestCase):
    """Single-patient fallback: train slabs from the first part of the depth axis,
    validation on the last ``val_fraction`` of the slices."""

    def test_band_slabs_and_band_volume(self):
        nac, ac = _vol((20, 30, 30), 0), _vol((20, 30, 30), 1)
        n, a = sample_pair_slabs_band(nac, ac, 2, 16, 4, "train", 0.2, np.random.RandomState(0), augment=False)
        self.assertEqual(tuple(n.shape), (2, 1, 4, 16, 16))
        self.assertEqual(tuple(a.shape), (2, 1, 4, 16, 16))
        nv, av = sample_pair_volume_native_band(nac, ac, 16, "val", 0.2)
        # The val band is the last 20% of 20 slices = slices 16..19.
        self.assertEqual(tuple(nv.shape), (1, 1, 4, 16, 16))
        self.assertEqual(tuple(av.shape), (1, 1, 4, 16, 16))
        _, ac_r = resize_pair_native_depth(nac, ac, 16)
        self.assertTrue(torch.equal(av[0], ac_r[:, 16:20]))


if __name__ == "__main__":
    unittest.main()
