"""CPU tests for the async patient-prefetch cache + paired-patient manifest cache.

No real DICOM and no GPU: a fake loader returns deterministic synthetic volumes
keyed by patient path so we can assert the prefetching cache returns the SAME
volumes as the synchronous path, covers all indices each epoch, actually loads
ahead, shuts down cleanly (including early stop), and propagates worker errors.
"""

import os
import sys
import threading
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import numpy as np
    import torch
    from src.training import data
    from src.training.data import (
        PrefetchingPatientCache,
        filter_paired_patients_cached,
        load_paired_cache,
        write_paired_cache,
    )
    HAVE_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_DEPS = False


def _fake_loader_factory(delay=0.0, calls=None, lock=None):
    """Return a load_patient_by_path-compatible fake.

    Volumes are deterministic from the path string so the same path always yields
    identical tensors (lets us compare sync vs prefetch). ``calls`` (a list) records
    load order; ``delay`` makes each load slow to exercise overlap/look-ahead.
    """
    def _vol(path, tag):
        seed = (abs(hash((path, tag))) % (2 ** 31))
        g = torch.Generator().manual_seed(seed)
        return torch.rand(4, 5, 5, generator=g, dtype=torch.float32)

    def _loader(patient_path, device=None, load_ct=True, run_segmentation=True):
        if delay:
            time.sleep(delay)
        if calls is not None:
            with (lock or threading.Lock()):
                calls.append(patient_path)
        out = {
            "ct": None,
            "pet_ac": _vol(patient_path, "ac"),
            "pet_nac": _vol(patient_path, "nac"),
            "spacing": (3.0, 2.0, 2.0),
            "origin": (0.0, 0.0, 0.0),
            "patient_path": patient_path,
        }
        if device is not None:
            for k in ("ct", "pet_ac", "pet_nac"):
                if out[k] is not None:
                    out[k] = out[k].to(device)
        return out

    return _loader


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestPrefetchCache(unittest.TestCase):
    def setUp(self):
        self.paths = [f"patient_{i}" for i in range(8)]

    def test_same_volumes_as_sync(self):
        loader = _fake_loader_factory()
        sync = PrefetchingPatientCache(self.paths, device=None, prefetch=0, loader=loader)
        for i in range(len(self.paths)):
            v = sync.get(i)
            ref = loader(self.paths[i])
            self.assertTrue(torch.equal(v["pet_ac"], ref["pet_ac"]))
            self.assertTrue(torch.equal(v["pet_nac"], ref["pet_nac"]))

    def test_prefetch_returns_correct_volumes(self):
        loader = _fake_loader_factory()
        rng = np.random.RandomState(0)
        cache = PrefetchingPatientCache(
            self.paths, device=None, prefetch=3,
            train_indices=list(range(len(self.paths))), rng=rng, loader=loader,
        ).start()
        try:
            for _ in range(len(self.paths)):
                idx, vols = cache.next_train()
                ref = loader(self.paths[idx])
                self.assertTrue(torch.equal(vols["pet_ac"], ref["pet_ac"]))
                self.assertEqual(vols["patient_path"], self.paths[idx])
        finally:
            cache.close()

    def test_shuffled_epoch_covers_all_indices(self):
        loader = _fake_loader_factory()
        rng = np.random.RandomState(1)
        cache = PrefetchingPatientCache(
            self.paths, device=None, prefetch=2,
            train_indices=list(range(len(self.paths))), rng=rng, loader=loader,
        ).start()
        try:
            seen = set()
            for _ in range(len(self.paths)):
                idx, _ = cache.next_train()
                seen.add(idx)
            self.assertEqual(seen, set(range(len(self.paths))))
        finally:
            cache.close()

    def test_actually_loads_ahead(self):
        # Slow loader: by the time the consumer pops the first item, the worker
        # should already have loaded more than one patient (look-ahead).
        calls = []
        lock = threading.Lock()
        loader = _fake_loader_factory(delay=0.05, calls=calls, lock=lock)
        rng = np.random.RandomState(2)
        cache = PrefetchingPatientCache(
            self.paths, device=None, prefetch=3,
            train_indices=list(range(len(self.paths))), rng=rng, loader=loader,
        ).start()
        try:
            cache.next_train()  # pop one
            time.sleep(0.3)     # let the worker fill the buffer
            with lock:
                loaded = len(calls)
            # With prefetch depth 3 the worker loads ahead of the single consumed item.
            self.assertGreater(loaded, 1)
        finally:
            cache.close()

    def test_clean_shutdown_early_stop(self):
        loader = _fake_loader_factory(delay=0.02)
        rng = np.random.RandomState(3)
        cache = PrefetchingPatientCache(
            self.paths, device=None, prefetch=4,
            train_indices=list(range(len(self.paths))), rng=rng, loader=loader,
        ).start()
        cache.next_train()  # consume one, then stop early
        cache.close()
        self.assertFalse(cache._worker is not None and cache._worker.is_alive())
        cache.close()  # idempotent

    def test_worker_exception_propagates(self):
        def _boom(patient_path, device=None, load_ct=True, run_segmentation=True):
            raise RuntimeError("loader boom")

        rng = np.random.RandomState(4)
        cache = PrefetchingPatientCache(
            self.paths, device=None, prefetch=2,
            train_indices=list(range(len(self.paths))), rng=rng, loader=_boom,
        ).start()
        try:
            with self.assertRaises(RuntimeError):
                # Drain until the error sentinel surfaces (should not hang).
                for _ in range(len(self.paths) + 2):
                    cache.next_train()
        finally:
            cache.close()

    def test_disabled_prefetch_no_thread(self):
        loader = _fake_loader_factory()
        cache = PrefetchingPatientCache(self.paths, device=None, prefetch=0, loader=loader).start()
        self.assertIsNone(cache._worker)
        # next_train still works synchronously over the shuffled epoch.
        idx, vols = cache.next_train()
        self.assertIn(idx, range(len(self.paths)))
        cache.close()


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestPairedManifestCache(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        # Create real folders so mtime-keying works.
        self.paths = []
        for i in range(5):
            p = os.path.join(self.tmp, f"patient_{i}")
            os.makedirs(p)
            self.paths.append(p)
        self.cache_path = os.path.join(self.tmp, "paired.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch_filter(self, paired_subset, counter):
        def _fake(patient_paths, device=None, log=None):
            counter["n"] += 1
            return list(paired_subset)
        self._orig = data.filter_paired_patients
        data.filter_paired_patients = _fake

    def _restore_filter(self):
        data.filter_paired_patients = self._orig

    def test_missing_manifest_triggers_scan_and_write(self):
        counter = {"n": 0}
        paired = self.paths[:3]
        self._patch_filter(paired, counter)
        try:
            out = filter_paired_patients_cached(self.paths, cache_path=self.cache_path)
            self.assertEqual(out, paired)
            self.assertEqual(counter["n"], 1)            # scanned once
            self.assertTrue(os.path.exists(self.cache_path))
        finally:
            self._restore_filter()

    def test_cache_hit_skips_scan(self):
        counter = {"n": 0}
        paired = self.paths[:3]
        self._patch_filter(paired, counter)
        try:
            filter_paired_patients_cached(self.paths, cache_path=self.cache_path)
            self.assertEqual(counter["n"], 1)
            # Second call with same roots/mtimes reuses the manifest (no rescan).
            out = filter_paired_patients_cached(self.paths, cache_path=self.cache_path)
            self.assertEqual(out, paired)
            self.assertEqual(counter["n"], 1)
        finally:
            self._restore_filter()

    def test_rescan_forces_rebuild(self):
        counter = {"n": 0}
        self._patch_filter(self.paths[:3], counter)
        try:
            filter_paired_patients_cached(self.paths, cache_path=self.cache_path)
            filter_paired_patients_cached(self.paths, cache_path=self.cache_path, rescan=True)
            self.assertEqual(counter["n"], 2)  # rescan ignored the manifest
        finally:
            self._restore_filter()

    def test_staleness_forces_rebuild(self):
        counter = {"n": 0}
        self._patch_filter(self.paths[:3], counter)
        try:
            filter_paired_patients_cached(self.paths, cache_path=self.cache_path)
            # Touch a patient folder to change its mtime -> manifest key no longer matches.
            time.sleep(0.01)
            os.utime(self.paths[0], None)
            new_mtime = time.time() + 5
            os.utime(self.paths[0], (new_mtime, new_mtime))
            out = filter_paired_patients_cached(self.paths, cache_path=self.cache_path)
            self.assertEqual(counter["n"], 2)  # rescanned because stale
            self.assertEqual(out, self.paths[:3])
        finally:
            self._restore_filter()

    def test_write_then_read_roundtrip(self):
        paired = self.paths[1:4]
        write_paired_cache(self.cache_path, self.paths, paired)
        out = load_paired_cache(self.cache_path, self.paths)
        self.assertEqual(out, paired)


if __name__ == "__main__":
    unittest.main()
