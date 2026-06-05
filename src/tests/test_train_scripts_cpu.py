"""End-to-end CPU smoke test of the training scripts + inference CLI.

Monkeypatches the DICOM loader (load_patient_volumes) with synthetic volumes so
the full pipeline runs without real data: train ae2d -> diff2d -> ft3d (with
inflation), then run inference for each task. Verifies metrics.jsonl, best.pt and
the predicted/GT .npy outputs are produced with the right shapes.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    from src.training.train import train_ae2d, train_ae3d, train_diff2d, train_ft3d
    from src.training import infer
    from src.training.utils.metrics import read_metrics
    HAVE_DEPS = True
except Exception as _e:  # pragma: no cover - environment dependent
    HAVE_DEPS = False
    _IMPORT_ERROR = _e


def _synthetic_vols(z=24, y=40, x=40):
    def _vol():
        return torch.rand(z, y, x, dtype=torch.float32)

    def _loader(data_dir, patient_index, device=None):
        return {
            "ct": _vol(),
            "pet_ac": _vol(),
            "pet_nac": _vol(),
            "spacing": (3.0, 2.0, 2.0),
            "origin": (0.0, 0.0, 0.0),
            "patient_path": "synthetic",
        }

    return _loader


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestTrainScriptsCpu(unittest.TestCase):
    def setUp(self):
        self.loader = _synthetic_vols()
        self._origs = {}
        for mod in (train_ae2d, train_ae3d, train_diff2d, train_ft3d, infer):
            self._origs[mod] = mod.load_patient_volumes
            mod.load_patient_volumes = self.loader
        self._argv = sys.argv

    def tearDown(self):
        for mod, fn in self._origs.items():
            mod.load_patient_volumes = fn
        sys.argv = self._argv
        # Release the root-logger file handle left open by the last training run
        # so the temp directory can be removed on Windows.
        import logging
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)

    def _run(self, main_fn, argv):
        sys.argv = ["prog"] + argv
        main_fn()

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae_dir = os.path.join(tmp, "ae2d")
            ae3_dir = os.path.join(tmp, "ae3d")
            d2_dir = os.path.join(tmp, "diff2d")
            d3_dir = os.path.join(tmp, "ft3d")
            ae_ckpt = os.path.join(ae_dir, "best.pt")
            ae3_ckpt = os.path.join(ae3_dir, "best.pt")
            d2_ckpt = os.path.join(d2_dir, "best.pt")
            d3_ckpt = os.path.join(d3_dir, "best.pt")
            common = ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
                      "--steps_per_epoch", "2", "--val_every", "1", "--val_batches", "1",
                      "--batch_size", "2"]

            # --- ae2d ---
            self._run(train_ae2d.main, common + ["--slice_size", "32", "--save_dir", ae_dir])
            self.assertTrue(os.path.exists(ae_ckpt), "ae2d best.pt missing")
            rows = read_metrics(os.path.join(ae_dir, "metrics.jsonl"))
            self.assertTrue(any(r["phase"] == "train" for r in rows))
            self.assertTrue(any(r["phase"] == "val" for r in rows))

            # --- ae3d (inflate 2D AE -> 3D AE, fine-tune on volumes) ---
            self._run(train_ae3d.main, common + ["--crop_size", "16", "--ae2d_ckpt", ae_ckpt, "--save_dir", ae3_dir])
            self.assertTrue(os.path.exists(ae3_ckpt), "ae3d best.pt missing")
            rows = read_metrics(os.path.join(ae3_dir, "metrics.jsonl"))
            self.assertTrue(any(r["phase"] == "train" for r in rows))

            # --- diff2d (uses frozen 2D ae) ---
            self._run(train_diff2d.main, common + ["--latent_size", "32", "--ae_ckpt", ae_ckpt, "--save_dir", d2_dir])
            self.assertTrue(os.path.exists(d2_ckpt), "diff2d best.pt missing")
            rows = read_metrics(os.path.join(d2_dir, "metrics.jsonl"))
            self.assertTrue(any(r["phase"] == "val" and "l1" in r for r in rows))

            # --- ft3d (frozen 3D ae + inflate diffusion UNet from diff2d) ---
            self._run(train_ft3d.main, common + ["--latent_size", "16", "--ae_ckpt", ae3_ckpt,
                                                  "--inflate_from", d2_ckpt, "--save_dir", d3_dir,
                                                  "--steps_per_epoch", "1", "--val_every", "1"])
            self.assertTrue(os.path.exists(d3_ckpt), "ft3d best.pt missing")

            # --- inference for each task ---
            infer_root = os.path.join(tmp, "infer")
            self._run(infer.main, ["--task", "ae2d", "--data_dir", "synthetic", "--device", "cpu",
                                   "--slice", "5", "--ae_ckpt", ae_ckpt, "--size", "32",
                                   "--out", os.path.join(infer_root, "ae2d")])
            self._run(infer.main, ["--task", "ae3d", "--data_dir", "synthetic", "--device", "cpu",
                                   "--ae_ckpt", ae3_ckpt, "--size", "16",
                                   "--out", os.path.join(infer_root, "ae3d")])
            self._run(infer.main, ["--task", "diff2d", "--data_dir", "synthetic", "--device", "cpu",
                                   "--slice", "5", "--ae_ckpt", ae_ckpt, "--diff_ckpt", d2_ckpt,
                                   "--size", "32", "--ddim_steps", "2", "--out", os.path.join(infer_root, "diff2d")])
            self._run(infer.main, ["--task", "ft3d", "--data_dir", "synthetic", "--device", "cpu",
                                   "--ae_ckpt", ae3_ckpt, "--diff_ckpt", d3_ckpt, "--size", "16",
                                   "--ddim_steps", "2", "--out", os.path.join(infer_root, "ft3d")])

            import numpy as np
            for task, ndim in (("ae2d", 2), ("ae3d", 3), ("diff2d", 2), ("ft3d", 3)):
                base = os.path.join(infer_root, task)
                pred = np.load(os.path.join(base, "pred.npy"))
                gt = np.load(os.path.join(base, "gt.npy"))
                self.assertEqual(pred.ndim, ndim, f"{task} pred ndim")
                self.assertEqual(pred.shape, gt.shape, f"{task} pred/gt shape mismatch")
                self.assertTrue(os.path.exists(os.path.join(base, "meta.json")))


if __name__ == "__main__":
    unittest.main()
