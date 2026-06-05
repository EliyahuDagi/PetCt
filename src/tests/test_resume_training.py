"""Verify --resume restores model/optimizer/RNG and reproduces the exact run.

Trains ae2d uninterrupted for N steps, then trains a fresh run for N/2 steps,
resumes it for the remaining steps, and asserts the final model weights and the
recorded train losses of the resumed tail match the uninterrupted reference
bit-for-bit. This confirms resume continues from the exact stopping point.
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
    from src.training.train import train_ae2d
    from src.training.utils.checkpointing import load_checkpoint
    from src.training.utils.metrics import read_metrics
    HAVE_DEPS = True
except Exception as _e:  # pragma: no cover - environment dependent
    HAVE_DEPS = False


def _loader(data_dir, patient_index, device=None):
    g = torch.Generator().manual_seed(1234)

    def _vol():
        return torch.rand(24, 40, 40, generator=g, dtype=torch.float32)

    return {
        "ct": _vol(), "pet_ac": _vol(), "pet_nac": _vol(),
        "spacing": (3.0, 2.0, 2.0), "origin": (0.0, 0.0, 0.0),
        "patient_path": "synthetic",
    }


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestResumeTraining(unittest.TestCase):
    def setUp(self):
        self._orig = train_ae2d.load_patient_volumes
        train_ae2d.load_patient_volumes = _loader
        self._argv = sys.argv

    def tearDown(self):
        train_ae2d.load_patient_volumes = self._orig
        sys.argv = self._argv
        import logging
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)

    def _run(self, argv):
        sys.argv = ["prog"] + argv
        train_ae2d.main()

    def test_resume_matches_uninterrupted(self):
        # 2 epochs x 4 steps = 8 steps; a checkpoint is written every 2 steps.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ref_dir = os.path.join(tmp, "ref")
            res_dir = os.path.join(tmp, "res")
            base = ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "2",
                    "--steps_per_epoch", "4", "--val_every", "2", "--val_batches", "1",
                    "--batch_size", "2", "--slice_size", "32"]

            # Reference: one uninterrupted run.
            self._run(base + ["--save_dir", ref_dir])
            ref = load_checkpoint(os.path.join(ref_dir, "last.pt"))

            # Interrupted run: crash on entering step 5, after step 4's periodic
            # checkpoint was written (last.pt -> 5 completed steps). This mirrors a
            # real Ctrl-C between checkpoints.
            orig_step = train_ae2d.train_step
            calls = {"n": 0}

            def crashing(model, batch, optimizer):
                if calls["n"] == 5:
                    raise KeyboardInterrupt
                calls["n"] += 1
                return orig_step(model, batch, optimizer)

            train_ae2d.train_step = crashing
            try:
                with self.assertRaises(KeyboardInterrupt):
                    self._run(base + ["--save_dir", res_dir])
            finally:
                train_ae2d.train_step = orig_step

            interrupted = load_checkpoint(os.path.join(res_dir, "last.pt"))
            self.assertEqual(interrupted["step"], 5, "expected a periodic checkpoint at 5 steps")

            # Resume the interrupted run to completion.
            self._run(base + ["--save_dir", res_dir, "--resume"])
            res = load_checkpoint(os.path.join(res_dir, "last.pt"))

            self.assertEqual(ref["step"], res["step"])
            for k in ref["model"]:
                self.assertTrue(
                    torch.equal(ref["model"][k], res["model"][k]),
                    f"weight {k} diverged after resume",
                )

            # Train losses for the resumed tail (steps >= 5) must match the reference
            # bit-for-bit -- proves model+optimizer+RNG were restored exactly.
            def tail(d):
                return {r["step"]: r["loss"] for r in read_metrics(os.path.join(d, "metrics.jsonl"))
                        if r["phase"] == "train" and r["step"] >= 5}
            self.assertEqual(tail(ref_dir), tail(res_dir))


if __name__ == "__main__":
    unittest.main()
