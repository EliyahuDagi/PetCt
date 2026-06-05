"""CPU smoke-of-the-smoke: run src.training.smoke end-to-end with synthetic data.

Monkeypatches the DICOM loader in each stage so the smoke runner exercises the
full ae2d->ae3d->diff2d->ft3d chain on CPU at tiny sizes, then checks the JSON
report has a per-stage entry with a timing for every stage.
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
    import torch
    from src.training import infer, smoke
    from src.training.train import train_ae2d, train_ae3d, train_diff2d, train_ft3d
    HAVE_DEPS = True
except Exception as _e:  # pragma: no cover - environment dependent
    HAVE_DEPS = False


def _synthetic_loader(z=24, y=40, x=40):
    def _vol():
        return torch.rand(z, y, x, dtype=torch.float32)

    def _loader(data_dir, patient_index, device=None):
        return {"ct": _vol(), "pet_ac": _vol(), "pet_nac": _vol(),
                "spacing": (3.0, 2.0, 2.0), "origin": (0.0, 0.0, 0.0), "patient_path": "synthetic"}

    return _loader


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestSmokeCpu(unittest.TestCase):
    def setUp(self):
        loader = _synthetic_loader()
        self._origs = {}
        # Patch the loader everywhere the smoke run touches it: the four train
        # stages and the inference CLI (smoke runs inference after training).
        for mod in (train_ae2d, train_ae3d, train_diff2d, train_ft3d, infer):
            self._origs[mod] = mod.load_patient_volumes
            mod.load_patient_volumes = loader
        self._argv = sys.argv

    def tearDown(self):
        for mod, fn in self._origs.items():
            mod.load_patient_volumes = fn
        sys.argv = self._argv
        import logging
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)

    def test_smoke_report(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            report_path = os.path.join(tmp, "smoke_report.json")
            work_dir = os.path.join(tmp, "work")
            sys.argv = ["smoke", "--data_dir", "synthetic", "--device", "cpu",
                        "--steps", "4", "--size", "32", "--crop3d", "16",
                        "--batch2d", "2", "--batch3d", "1",
                        "--plan_epochs", "1", "--plan_steps_per_epoch", "10",
                        "--work_dir", work_dir, "--report", report_path]
            rc = smoke.main()

            self.assertTrue(os.path.exists(report_path), "report not written")
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            stages = {s["stage"]: s for s in report["stages"]}
            self.assertEqual(set(stages), {"ae2d", "ae3d", "diff2d", "ft3d"})
            for name, s in stages.items():
                self.assertTrue(s["ok"], f"{name} failed: {s.get('error')}")
                self.assertIsNotNone(s["ms_per_step"], f"{name} has no ms/step")
                self.assertIsNotNone(s.get("eta_seconds"), f"{name} has no ETA")
            # Inference ran for every stage and succeeded.
            infer_stages = {r["stage"]: r for r in report["inference"]}
            self.assertEqual(set(infer_stages), {"ae2d", "ae3d", "diff2d", "ft3d"})
            for name, r in infer_stages.items():
                self.assertTrue(r["ok"], f"infer {name} failed: {r.get('error')}")
            self.assertTrue(report["all_ok"])
            self.assertIsNotNone(report["eta_total_seconds"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
