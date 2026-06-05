"""Tests for the torch-free patient enumeration in src.training.dataset_index.

Covers the single-patient-folder heuristic, multi-subdir roots, multi-root
concatenation + de-duplication, ordering, and the missing_ok behavior.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.dataset_index import PATIENT_MARKERS, enumerate_patients


def _mk(*parts):
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


class TestEnumeratePatients(unittest.TestCase):
    def test_single_patient_folder_heuristic(self):
        # A root that directly contains a marker subdir (e.g. DICOM/) is itself one patient.
        with tempfile.TemporaryDirectory() as tmp:
            _mk(tmp, "DICOM")
            _mk(tmp, "Segmentation")
            patients = enumerate_patients(tmp)
            self.assertEqual(patients, [tmp])

    def test_marker_match_is_case_insensitive(self):
        for marker in ("DICOM", "ct", "Pet", "SECTRA"):
            with tempfile.TemporaryDirectory() as tmp:
                _mk(tmp, marker)
                self.assertEqual(enumerate_patients(tmp), [tmp])
        # Sanity: the contract markers are all present.
        self.assertIn("dicom", PATIENT_MARKERS)
        self.assertIn("sectra", PATIENT_MARKERS)

    def test_multi_subdir_root(self):
        # No marker subdir -> each immediate subdir is a patient, sorted.
        with tempfile.TemporaryDirectory() as tmp:
            p_b = _mk(tmp, "patientB")
            p_a = _mk(tmp, "patientA")
            p_c = _mk(tmp, "patientC")
            self.assertEqual(enumerate_patients(tmp), [p_a, p_b, p_c])

    def test_multi_root_concatenation_preserves_root_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root1 = _mk(tmp, "root1")
            root2 = _mk(tmp, "root2")
            r1b = _mk(root1, "b")
            r1a = _mk(root1, "a")
            r2z = _mk(root2, "z")
            r2y = _mk(root2, "y")
            patients = enumerate_patients([root1, root2])
            # root1's patients (sorted) come before root2's (sorted).
            self.assertEqual(patients, [r1a, r1b, r2y, r2z])

    def test_dedup_across_roots(self):
        # The same patient path reachable via two roots appears once.
        with tempfile.TemporaryDirectory() as tmp:
            root = _mk(tmp, "root")
            pa = _mk(root, "a")
            pb = _mk(root, "b")
            patients = enumerate_patients([root, root])
            self.assertEqual(patients, [pa, pb])

    def test_single_str_arg(self):
        with tempfile.TemporaryDirectory() as tmp:
            pa = _mk(tmp, "a")
            self.assertEqual(enumerate_patients(tmp), [pa])

    def test_missing_root_raises_by_default(self):
        with self.assertRaises(ValueError):
            enumerate_patients(os.path.join(os.sep, "no", "such", "root_xyz"))

    def test_missing_root_skipped_when_missing_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = _mk(tmp, "root")
            pa = _mk(real, "a")
            missing = os.path.join(tmp, "does_not_exist")
            patients = enumerate_patients([missing, real], missing_ok=True)
            self.assertEqual(patients, [pa])

    def test_all_missing_missing_ok_returns_empty(self):
        patients = enumerate_patients(["/nope/a", "/nope/b"], missing_ok=True)
        self.assertEqual(patients, [])


if __name__ == "__main__":
    unittest.main()
