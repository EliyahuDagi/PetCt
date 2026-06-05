import argparse
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training import smoke


def _args():
    return argparse.Namespace(
        data_dir="/data", patient_index=0, device="cpu", steps=3,
        size=128, crop3d=64, batch2d=4, batch3d=1,
    )


class TestStageArgv(unittest.TestCase):
    def test_ae3d_inflates_from_ae2d_with_crop_size(self):
        argv = smoke._stage_argv("ae3d", _args(), "/work")
        self.assertIn("--crop_size", argv)
        self.assertIn("--ae2d_ckpt", argv)
        self.assertEqual(argv[argv.index("--ae2d_ckpt") + 1], os.path.join("/work", "ae2d", "best.pt"))
        self.assertNotIn("--slice_size", argv)

    def test_ft3d_uses_3d_ae_and_inflates_unet(self):
        argv = smoke._stage_argv("ft3d", _args(), "/work")
        self.assertEqual(argv[argv.index("--ae_ckpt") + 1], os.path.join("/work", "ae3d", "best.pt"))
        self.assertEqual(argv[argv.index("--inflate_from") + 1], os.path.join("/work", "diff2d", "best.pt"))

    def test_diff2d_uses_2d_ae(self):
        argv = smoke._stage_argv("diff2d", _args(), "/work")
        self.assertEqual(argv[argv.index("--ae_ckpt") + 1], os.path.join("/work", "ae2d", "best.pt"))


class TestInferArgv(unittest.TestCase):
    def test_3d_tasks_use_3d_ae_and_crop_size(self):
        for task in ("ae3d", "ft3d"):
            argv = smoke._infer_argv(task, _args(), "/work")
            self.assertEqual(argv[argv.index("--ae_ckpt") + 1], os.path.join("/work", "ae3d", "best.pt"))
            self.assertEqual(argv[argv.index("--size") + 1], "64")

    def test_2d_tasks_use_2d_ae(self):
        for task in ("ae2d", "diff2d"):
            argv = smoke._infer_argv(task, _args(), "/work")
            self.assertEqual(argv[argv.index("--ae_ckpt") + 1], os.path.join("/work", "ae2d", "best.pt"))
            self.assertEqual(argv[argv.index("--size") + 1], "128")

    def test_diffusion_tasks_pass_diff_ckpt(self):
        for task in ("diff2d", "ft3d"):
            argv = smoke._infer_argv(task, _args(), "/work")
            self.assertIn("--diff_ckpt", argv)
            self.assertIn("--ddim_steps", argv)

    def test_ae_tasks_omit_diff_ckpt(self):
        for task in ("ae2d", "ae3d"):
            argv = smoke._infer_argv(task, _args(), "/work")
            self.assertNotIn("--diff_ckpt", argv)

    def test_prereqs_cover_all_stages(self):
        self.assertEqual(set(smoke.INFER_PREREQS), set(smoke.STAGES))


if __name__ == "__main__":
    unittest.main()
