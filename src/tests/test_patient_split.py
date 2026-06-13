"""Tests for make_patient_split (by-patient holdout) and ae_pool_volumes.

These exercise src.training.data, which imports torch; the suite skips cleanly
when torch is unavailable.
"""

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch  # noqa: F401
    from src.training.data import (
        ae_pool_volumes,
        make_patient_split,
        make_patient_split3,
        write_split_json,
    )
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
class TestMakePatientSplit3(unittest.TestCase):
    def test_single_patient_signals_within_patient_fallback(self):
        # Mirrors make_patient_split: ([0],[0]) + empty test set.
        self.assertEqual(make_patient_split3(1, 0.2, 0.1, seed=0), ([0], [0], []))

    def test_two_patients_no_test(self):
        # Not enough patients for a disjoint test holdout: 1 train / 1 val / 0 test.
        train, val, test = make_patient_split3(2, 0.2, 0.1, seed=5)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(test, [])
        self.assertEqual(sorted(train + val), [0, 1])

    def test_deterministic(self):
        a = make_patient_split3(20, 0.2, 0.1, seed=7)
        b = make_patient_split3(20, 0.2, 0.1, seed=7)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a = make_patient_split3(20, 0.2, 0.1, seed=1)
        b = make_patient_split3(20, 0.2, 0.1, seed=2)
        self.assertNotEqual(a, b)

    def test_pairwise_disjoint_and_cover_all(self):
        for n in (3, 5, 10, 17, 50):
            train, val, test = make_patient_split3(n, 0.2, 0.1, seed=3)
            self.assertEqual(set(train) & set(val), set(), f"train/val overlap n={n}")
            self.assertEqual(set(train) & set(test), set(), f"train/test overlap n={n}")
            self.assertEqual(set(val) & set(test), set(), f"val/test overlap n={n}")
            self.assertEqual(sorted(train + val + test), list(range(n)), f"cover n={n}")

    def test_at_least_one_each_when_three_or_more(self):
        for n in (3, 4, 10, 30):
            for vf in (0.0, 0.01, 0.5, 0.99, 1.0):
                for tf in (0.0, 0.01, 0.5, 0.99, 1.0):
                    train, val, test = make_patient_split3(n, vf, tf, seed=11)
                    self.assertGreaterEqual(len(train), 1, f"n={n} vf={vf} tf={tf}")
                    self.assertGreaterEqual(len(val), 1, f"n={n} vf={vf} tf={tf}")
                    self.assertGreaterEqual(len(test), 1, f"n={n} vf={vf} tf={tf}")

    def test_test_patients_never_in_train_or_val(self):
        # The core safety property: no test patient leaks into training/validation.
        train, val, test = make_patient_split3(40, 0.15, 0.15, seed=9)
        for t in test:
            self.assertNotIn(t, train)
            self.assertNotIn(t, val)

    def test_nested_consistent_ordering_with_2way(self):
        # Same (n, seed) -> same shuffled order, so the 2-way val set is exactly the
        # 3-way test+val (the head of the shuffled order) for matching fractions.
        n, seed = 30, 4
        _, val2 = make_patient_split(n, 0.3, seed)
        train3, val3, test3 = make_patient_split3(n, 0.2, 0.1, seed)
        self.assertEqual(set(val2), set(val3) | set(test3))

    def test_invalid_num_patients(self):
        with self.assertRaises(ValueError):
            make_patient_split3(0, 0.2, 0.1, seed=0)


@unittest.skipUnless(HAVE_DEPS, "torch not available")
class TestWriteSplitJson(unittest.TestCase):
    def test_roundtrip_paths_and_schema(self):
        patients = [f"/data/patient_{i}" for i in range(10)]
        train, val, test = make_patient_split3(10, 0.2, 0.1, seed=2)
        with tempfile.TemporaryDirectory() as d:
            path = write_split_json(d, "diff2d", patients, train, val, test,
                                    0.2, 0.1, seed=2)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data["task"], "diff2d")
        self.assertEqual(data["val_fraction"], 0.2)
        self.assertEqual(data["test_fraction"], 0.1)
        self.assertEqual(data["seed"], 2)
        self.assertEqual(data["num_patients"], 10)
        self.assertEqual(data["train"], [patients[i] for i in train])
        self.assertEqual(data["val"], [patients[i] for i in val])
        self.assertEqual(data["test"], [patients[i] for i in test])
        # The persisted path sets stay pairwise disjoint.
        self.assertEqual(set(data["train"]) & set(data["test"]), set())
        self.assertEqual(set(data["val"]) & set(data["test"]), set())


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
