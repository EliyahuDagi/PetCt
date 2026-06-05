"""Tests for make_patient_split (by-patient holdout) and ae_pool_volumes.

These exercise src.training.data, which imports torch; the suite skips cleanly
when torch is unavailable.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch  # noqa: F401
    from src.training.data import ae_pool_volumes, make_patient_split
    HAVE_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestMakePatientSplit(unittest.TestCase):
    def test_single_patient_signals_within_patient_fallback(self):
        self.assertEqual(make_patient_split(1, 0.2, seed=0), ([0], [0]))

    def test_deterministic(self):
        a = make_patient_split(10, 0.3, seed=7)
        b = make_patient_split(10, 0.3, seed=7)
        self.assertEqual(a, b)

    def test_disjoint_and_cover_all(self):
        train, val = make_patient_split(10, 0.3, seed=1)
        self.assertEqual(set(train) & set(val), set())
        self.assertEqual(sorted(train + val), list(range(10)))

    def test_guarantees_at_least_one_each_when_multi(self):
        for n in (2, 3, 5, 20):
            for vf in (0.0, 0.01, 0.5, 0.99, 1.0):
                train, val = make_patient_split(n, vf, seed=3)
                self.assertGreaterEqual(len(train), 1, f"n={n} vf={vf}")
                self.assertGreaterEqual(len(val), 1, f"n={n} vf={vf}")

    def test_different_seeds_differ(self):
        a = make_patient_split(20, 0.25, seed=1)
        b = make_patient_split(20, 0.25, seed=2)
        self.assertNotEqual(a, b)

    def test_invalid_num_patients(self):
        with self.assertRaises(ValueError):
            make_patient_split(0, 0.2, seed=0)


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestAePoolVolumes(unittest.TestCase):
    def _vol(self):
        return torch.zeros(2, 3, 3)

    def test_pet_prefers_pet_pair(self):
        vols = {"pet_ac": self._vol(), "pet_nac": self._vol(), "ct": self._vol()}
        pool = ae_pool_volumes(vols, "pet")
        self.assertEqual(len(pool), 2)

    def test_pet_falls_back_to_ct(self):
        vols = {"pet_ac": None, "pet_nac": None, "ct": self._vol()}
        pool = ae_pool_volumes(vols, "pet")
        self.assertEqual(len(pool), 1)

    def test_pet_unpaired_keeps_single(self):
        vols = {"pet_ac": self._vol(), "pet_nac": None, "ct": self._vol()}
        pool = ae_pool_volumes(vols, "pet")
        self.assertEqual(len(pool), 1)

    def test_ct_modality(self):
        vols = {"pet_ac": self._vol(), "pet_nac": self._vol(), "ct": self._vol()}
        self.assertEqual(len(ae_pool_volumes(vols, "ct")), 1)

    def test_nothing_available(self):
        vols = {"pet_ac": None, "pet_nac": None, "ct": None}
        self.assertEqual(ae_pool_volumes(vols, "pet"), [])


if __name__ == "__main__":
    unittest.main()
