"""CPU tests for the SSD pre-cache (src/training/precache.py) + transparent loader.

No real DICOM and no GPU: ``load_patient_by_path`` is monkeypatched with a fake
that returns deterministic synthetic volumes, so we can assert the cache round-trips
arrays within dtype tolerance, the loader reads the cache without touching DICOM on
a hit, falls back / write-throughs on a miss, honors skip-vs-force, writes
atomically, and answers pairing from sidecars.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import numpy as np
    import torch
    from src.training import data, precache
    HAVE_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_DEPS = False


def _fake_vols(patient_path, has_ac=True, has_nac=True, z=6, y=8, x=8):
    """Deterministic synthetic patient dict keyed by path (no DICOM)."""
    seed = abs(hash(patient_path)) % (2 ** 31)

    def _v(tag):
        gg = torch.Generator().manual_seed((seed + hash(tag)) % (2 ** 31))
        return torch.rand(z, y, x, generator=gg, dtype=torch.float32)

    return {
        "ct": None,
        "pet_ac": _v("ac") if has_ac else None,
        "pet_nac": _v("nac") if has_nac else None,
        "spacing": (3.0, 2.0, 2.0),
        "origin": (1.0, 2.0, 3.0),
        "patient_path": patient_path,
    }


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestPrecacheRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmp, "cache")
        self.patient = os.path.join(self.tmp, "patient_0")
        os.makedirs(self.patient)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_float16_roundtrip_within_tolerance(self):
        vols = _fake_vols(self.patient)
        precache.write_entry(self.cache_dir, vols, dtype="float16")
        out = precache.load_entry(self.cache_dir, self.patient, to_tensor=True)
        # Shape + keys + metadata preserved.
        self.assertEqual(set(out), {"ct", "pet_ac", "pet_nac", "spacing", "origin", "patient_path"})
        self.assertIsNone(out["ct"])
        self.assertEqual(out["spacing"], (3.0, 2.0, 2.0))
        self.assertEqual(out["origin"], (1.0, 2.0, 3.0))
        self.assertEqual(out["patient_path"], self.patient)
        self.assertEqual(tuple(out["pet_ac"].shape), tuple(vols["pet_ac"].shape))
        # float16 storage: equal within float16 precision (cast back to float32).
        self.assertTrue(torch.allclose(out["pet_ac"], vols["pet_ac"], atol=1e-3))
        self.assertTrue(torch.allclose(out["pet_nac"], vols["pet_nac"], atol=1e-3))
        self.assertEqual(out["pet_ac"].dtype, torch.float32)

    def test_float32_roundtrip_exact(self):
        vols = _fake_vols(self.patient)
        precache.write_entry(self.cache_dir, vols, dtype="float32")
        out = precache.load_entry(self.cache_dir, self.patient, to_tensor=True)
        self.assertTrue(torch.equal(out["pet_ac"], vols["pet_ac"]))
        self.assertTrue(torch.equal(out["pet_nac"], vols["pet_nac"]))

    def test_missing_modality_flags(self):
        vols = _fake_vols(self.patient, has_nac=False)
        precache.write_entry(self.cache_dir, vols, dtype="float16")
        side = precache.read_sidecar(self.cache_dir, self.patient)
        self.assertTrue(side["has_ac"])
        self.assertFalse(side["has_nac"])
        out = precache.load_entry(self.cache_dir, self.patient, to_tensor=True)
        self.assertIsNone(out["pet_nac"])
        self.assertIsNotNone(out["pet_ac"])


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestTransparentLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmp, "cache")
        self.patient = os.path.join(self.tmp, "patient_0")
        os.makedirs(self.patient)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        # Ensure DicomModel patch (if any) is removed.
        sys.modules.pop("_precache_test_dicom", None)

    def _patch_dicom_raise(self):
        """Make the DICOM path explode so a cache hit that touches it is detectable."""
        import src.DataViewer.model as viewer_model

        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError("DICOM path must not be hit on a cache hit")

        self._orig_dm = viewer_model.DicomModel
        viewer_model.DicomModel = _Boom

    def _restore_dicom(self):
        import src.DataViewer.model as viewer_model
        viewer_model.DicomModel = self._orig_dm

    def test_hit_does_not_touch_dicom(self):
        precache.write_entry(self.cache_dir, _fake_vols(self.patient), dtype="float16")
        self._patch_dicom_raise()
        try:
            out = data.load_patient_by_path(
                self.patient, load_ct=False, run_segmentation=False, cache_dir=self.cache_dir
            )
        finally:
            self._restore_dicom()
        self.assertIsNotNone(out["pet_ac"])
        self.assertEqual(out["patient_path"], self.patient)

    def test_miss_falls_back_to_dicom(self):
        # No cache entry; the loader must fall through to load_patient_data. Patch
        # DicomModel with a fake that returns synthetic volumes (no real DICOM).
        import src.DataViewer.model as viewer_model

        ref = _fake_vols(self.patient)

        class _FakeDM:
            def __init__(self):
                self.patient_list = []
                self.current_patient_index = 0
                self.ct_volume = None
                self.pet_volume = ref["pet_ac"].numpy()
                self.pet_nac_volume = ref["pet_nac"].numpy()

            def load_patient_data(self, path, load_ct=True, run_segmentation=True):
                self.patient_list = [path]

            def get_voxel_spacing(self):
                return (3.0, 2.0, 2.0)

            def get_origin(self):
                return (1.0, 2.0, 3.0)

        orig = viewer_model.DicomModel
        viewer_model.DicomModel = _FakeDM
        try:
            out = data.load_patient_by_path(
                self.patient, load_ct=False, run_segmentation=False, cache_dir=self.cache_dir
            )
        finally:
            viewer_model.DicomModel = orig
        self.assertIsNotNone(out["pet_ac"])
        # Nothing was written (write_through defaulted False).
        self.assertFalse(precache.is_fresh(self.cache_dir, self.patient))

    def test_write_through_populates_on_miss(self):
        import src.DataViewer.model as viewer_model
        ref = _fake_vols(self.patient)

        class _FakeDM:
            def __init__(self):
                self.patient_list = []
                self.ct_volume = None
                self.pet_volume = ref["pet_ac"].numpy()
                self.pet_nac_volume = ref["pet_nac"].numpy()

            def load_patient_data(self, path, load_ct=True, run_segmentation=True):
                self.patient_list = [path]

            def get_voxel_spacing(self):
                return (3.0, 2.0, 2.0)

            def get_origin(self):
                return (1.0, 2.0, 3.0)

        orig = viewer_model.DicomModel
        viewer_model.DicomModel = _FakeDM
        try:
            data.load_patient_by_path(
                self.patient, load_ct=False, run_segmentation=False,
                cache_dir=self.cache_dir, write_through=True,
            )
        finally:
            viewer_model.DicomModel = orig
        self.assertTrue(precache.is_fresh(self.cache_dir, self.patient))

    def test_disabled_cache_is_pure_dicom(self):
        # cache_dir=None must never consult the cache even if one exists on disk.
        precache.write_entry(self.cache_dir, _fake_vols(self.patient), dtype="float16")
        self.assertFalse(precache.is_fresh(self.cache_dir + "_other", self.patient))
        # resolve_cache_dir(None) with no env var -> None (disabled).
        os.environ.pop(precache.CACHE_ENV_VAR, None)
        self.assertIsNone(precache.resolve_cache_dir(None))


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestPrecacheCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmp, "cache")
        self.root = os.path.join(self.tmp, "root")
        self.patients = []
        for i in range(3):
            p = os.path.join(self.root, f"patient_{i}")
            os.makedirs(p)
            self.patients.append(p)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _loader(self, calls):
        def _l(patient_path, device=None, load_ct=True, run_segmentation=True):
            calls.append(patient_path)
            return _fake_vols(patient_path)
        return _l

    def test_skip_existing_vs_force(self):
        calls = []
        loader = self._loader(calls)
        s1 = precache.precache([self.root], cache_dir=self.cache_dir, dtype="float16",
                               workers=1, loader=loader, log=lambda *a: None)
        self.assertEqual(s1["cached"], 3)
        self.assertEqual(len(calls), 3)
        # Second run: all fresh -> skipped, no loader calls.
        calls.clear()
        s2 = precache.precache([self.root], cache_dir=self.cache_dir, dtype="float16",
                               workers=1, loader=loader, log=lambda *a: None)
        self.assertEqual(s2["skipped"], 3)
        self.assertEqual(len(calls), 0)
        # Force rebuild: loader called again.
        calls.clear()
        s3 = precache.precache([self.root], cache_dir=self.cache_dir, dtype="float16",
                               workers=1, force=True, loader=loader, log=lambda *a: None)
        self.assertEqual(s3["cached"], 3)
        self.assertEqual(len(calls), 3)

    def test_empty_patient_not_cached(self):
        def _l(patient_path, device=None, load_ct=True, run_segmentation=True):
            return _fake_vols(patient_path, has_ac=False, has_nac=False)
        s = precache.precache([self.root], cache_dir=self.cache_dir, workers=1,
                              loader=_l, log=lambda *a: None)
        self.assertEqual(s["empty"], 3)
        self.assertEqual(s["cached"], 0)
        for p in self.patients:
            self.assertFalse(precache.is_fresh(self.cache_dir, p))

    def test_failed_patient_counted(self):
        def _l(patient_path, device=None, load_ct=True, run_segmentation=True):
            raise RuntimeError("boom")
        s = precache.precache([self.root], cache_dir=self.cache_dir, workers=1,
                              loader=_l, log=lambda *a: None)
        self.assertEqual(s["failed"], 3)
        self.assertEqual(s["cached"], 0)

    def test_atomic_write_no_partial_on_failure(self):
        # Simulate a crash mid-sidecar-write: the npz tmp/replace already ran, but
        # if json.dump fails the sidecar must not exist, so is_fresh() stays False.
        vols = _fake_vols(self.patients[0])
        orig_dump = precache.json.dump

        def _boom_dump(*a, **k):
            raise OSError("disk full")

        precache.json.dump = _boom_dump
        try:
            with self.assertRaises(OSError):
                precache.write_entry(self.cache_dir, vols, dtype="float16")
        finally:
            precache.json.dump = orig_dump
        # No final sidecar -> not a valid/fresh entry; no leftover .tmp sidecar.
        self.assertFalse(precache.is_fresh(self.cache_dir, self.patients[0]))
        _, side = precache.entry_paths(self.cache_dir, self.patients[0])
        self.assertFalse(os.path.exists(side + ".tmp"))


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestPairingFromSidecar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmp, "cache")
        self.root = os.path.join(self.tmp, "root")
        self.paths = []
        for i in range(4):
            p = os.path.join(self.root, f"patient_{i}")
            os.makedirs(p)
            self.paths.append(p)
        # patient_0, patient_2 paired; patient_1 ac-only; patient_3 nac-only.
        precache.write_entry(self.cache_dir, _fake_vols(self.paths[0]))
        precache.write_entry(self.cache_dir, _fake_vols(self.paths[1], has_nac=False))
        precache.write_entry(self.cache_dir, _fake_vols(self.paths[2]))
        precache.write_entry(self.cache_dir, _fake_vols(self.paths[3], has_ac=False))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_filter_paired_from_sidecar_no_dicom(self):
        import src.DataViewer.model as viewer_model

        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError("pairing must read sidecars, not DICOM")

        orig = viewer_model.DicomModel
        viewer_model.DicomModel = _Boom
        try:
            paired = data.filter_paired_patients(self.paths, cache_dir=self.cache_dir)
        finally:
            viewer_model.DicomModel = orig
        self.assertEqual(paired, [self.paths[0], self.paths[2]])


if __name__ == "__main__":
    unittest.main()
