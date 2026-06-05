import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The TrainViewer model must be importable WITHOUT tkinter or torch.
from src.TrainViewer import model as tvm


class TestPathTranslation(unittest.TestCase):
    def test_drive_letter(self):
        self.assertEqual(tvm.win_to_wsl_path(r"C:\Users\algo\x"), "/mnt/c/Users/algo/x")
        self.assertEqual(tvm.win_to_wsl_path("D:/data/set"), "/mnt/d/data/set")

    def test_passthrough(self):
        self.assertEqual(tvm.win_to_wsl_path("/mnt/c/already"), "/mnt/c/already")
        self.assertEqual(tvm.win_to_wsl_path("~/petct/.venv"), "~/petct/.venv")

    def test_torch_free_import(self):
        # Run in a clean subprocess: importing TrainViewer.model must not pull in
        # torch (the shared pytest process imports torch from other tests).
        import subprocess
        code = (
            "import sys; import src.TrainViewer.model; "
            "sys.exit(1 if 'torch' in sys.modules else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
        self.assertEqual(result.returncode, 0, "importing TrainViewer.model pulled in torch")


class TestMetricsReader(unittest.TestCase):
    def test_incremental_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.jsonl")
            reader = tvm.MetricsReader(path)
            self.assertEqual(reader.read_new(), [])  # absent file

            with open(path, "w", encoding="utf-8") as f:
                f.write('{"step": 0, "loss": 1.0}\n')
            rows = reader.read_new()
            self.assertEqual(len(rows), 1)
            self.assertEqual(reader.read_new(), [])  # nothing new

            with open(path, "a", encoding="utf-8") as f:
                f.write('{"step": 1, "loss": 0.5}\n')
            rows = reader.read_new()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 1)

    def test_partial_line_held_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.jsonl")
            reader = tvm.MetricsReader(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"step": 0}\n{"step": 1')  # second line partial
            rows = reader.read_new()
            self.assertEqual(len(rows), 1)
            # Complete the partial line; it should now be returned.
            with open(path, "a", encoding="utf-8") as f:
                f.write(', "loss": 0.2}\n')
            rows = reader.read_new()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 1)

    def test_reset_on_truncate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.jsonl")
            reader = tvm.MetricsReader(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"step": 5}\n{"step": 6}\n')
            self.assertEqual(len(reader.read_new()), 2)
            # New run truncates the file.
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"step": 0}\n')
            rows = reader.read_new()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 0)


class TestCommandBuilding(unittest.TestCase):
    def setUp(self):
        self.cfg = tvm.TrainViewerConfig()
        self.cfg.venv_path = "~/petct/.venv"
        self.cfg.project_path = "/mnt/c/proj"
        self.cfg.distro = "Ubuntu"
        self.cfg.size = 128
        self.cfg.data_dirs = [r"C:\data"]

    def test_ae2d_uses_slice_size(self):
        r = tvm.WslRunner()
        inner = r.build_inner("ae2d", self.cfg, "/tmp/outputs")
        self.assertIn("launcher ae2d", inner)
        self.assertIn("--slice_size 128", inner)
        self.assertIn("--data_dir /mnt/c/data", inner)
        self.assertNotIn("--latent_size", inner)

    def test_diff2d_uses_latent_size(self):
        r = tvm.WslRunner()
        inner = r.build_inner("diff2d", self.cfg, "/tmp/outputs")
        self.assertIn("--latent_size 128", inner)

    def test_ae3d_uses_crop_size(self):
        r = tvm.WslRunner()
        inner = r.build_inner("ae3d", self.cfg, "/tmp/outputs")
        self.assertIn("launcher ae3d", inner)
        self.assertIn("--crop_size 128", inner)
        self.assertNotIn("--slice_size", inner)
        self.assertNotIn("--latent_size", inner)

    def test_ft3d_inflate_only_when_present(self):
        r = tvm.WslRunner()
        with tempfile.TemporaryDirectory() as out:
            inner = r.build_inner("ft3d", self.cfg, out)
            self.assertNotIn("--inflate_from", inner)
            d2 = os.path.join(out, "diff2d")
            os.makedirs(d2)
            open(os.path.join(d2, "best.pt"), "wb").close()
            inner2 = r.build_inner("ft3d", self.cfg, out)
            self.assertIn("--inflate_from outputs/diff2d/best.pt", inner2)

    def test_multi_root_data_dir_emission(self):
        # Every root appears in WSL form, space-joined, after one --data_dir flag.
        self.cfg.data_dirs = [r"C:\data\a", r"D:\sets\b"]
        r = tvm.WslRunner()
        inner = r.build_inner("ae2d", self.cfg, "/tmp/outputs")
        self.assertIn("--data_dir /mnt/c/data/a /mnt/d/sets/b", inner)
        # Exactly one --data_dir flag (multi-value, not repeated).
        self.assertEqual(inner.count("--data_dir"), 1)

    def test_infer_cmd_omits_diff_ckpt_for_ae(self):
        r = tvm.InferenceRunner()
        inner = r.build_inner("ae2d", self.cfg, [r"C:\data"], 0, 3, "/tmp/outputs")
        self.assertIn("--task ae2d", inner)
        self.assertNotIn("--diff_ckpt", inner)
        inner_d = r.build_inner("diff2d", self.cfg, [r"C:\data"], 0, 3, "/tmp/outputs")
        self.assertIn("--diff_ckpt outputs/diff2d/best.pt", inner_d)

    def test_infer_multi_root_and_fallback(self):
        r = tvm.InferenceRunner()
        # Explicit infer roots are emitted.
        inner = r.build_inner("ae2d", self.cfg, [r"C:\d1", r"E:\d2"], 0, 0, "/tmp/outputs")
        self.assertIn("--data_dir /mnt/c/d1 /mnt/e/d2", inner)
        # Empty infer roots fall back to the train cfg.data_dirs.
        self.cfg.data_dirs = [r"C:\trainroot"]
        inner_fb = r.build_inner("ae2d", self.cfg, [], 0, 0, "/tmp/outputs")
        self.assertIn("--data_dir /mnt/c/trainroot", inner_fb)


class TestSmokeCommand(unittest.TestCase):
    def setUp(self):
        self.cfg = tvm.TrainViewerConfig()
        self.cfg.venv_path = "~/petct/.venv"
        self.cfg.project_path = "/mnt/c/proj"
        self.cfg.distro = "Ubuntu-24.04"
        self.cfg.size = 128
        self.cfg.batch_size = 4
        self.cfg.epochs = 10
        self.cfg.data_dirs = [r"C:\data"]

    def test_smoke_inner(self):
        r = tvm.SmokeRunner()
        inner = r.build_inner(self.cfg)
        self.assertIn("python -m src.training.smoke", inner)
        self.assertIn("--data_dir /mnt/c/data", inner)
        self.assertIn("--size 128", inner)
        self.assertIn("--batch2d 4", inner)
        self.assertIn("--plan_epochs 10", inner)
        self.assertIn("--report outputs/smoke_report.json", inner)

    def test_smoke_multi_root(self):
        self.cfg.data_dirs = [r"C:\a", r"D:\b"]
        r = tvm.SmokeRunner()
        inner = r.build_inner(self.cfg)
        self.assertIn("--data_dir /mnt/c/a /mnt/d/b", inner)
        self.assertEqual(inner.count("--data_dir"), 1)

    def test_smoke_load_report(self):
        import json as _json
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "smoke_report.json"), "w", encoding="utf-8") as f:
                _json.dump({"all_ok": True, "stages": []}, f)
            r = tvm.SmokeRunner()
            report, err = r.load_report(out)
            self.assertIsNone(err)
            self.assertTrue(report["all_ok"])


class TestConfigDataDirs(unittest.TestCase):
    def test_seed_from_single_data_dir(self):
        cfg = tvm.TrainViewerConfig(data_dir=r"C:\one", data_dirs=[])
        self.assertEqual(cfg.data_dirs, [r"C:\one"])

    def test_explicit_data_dirs_not_overwritten(self):
        cfg = tvm.TrainViewerConfig(data_dir=r"C:\one", data_dirs=[r"C:\a", r"C:\b"])
        self.assertEqual(cfg.data_dirs, [r"C:\a", r"C:\b"])


class TestSettingsRoundTrip(unittest.TestCase):
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as out:
            m = tvm.TrainModel(outputs_root=out)
            m.config.epochs = 7
            m.config.venv_path = "/opt/venv"
            m.config.data_dirs = [r"C:\a", r"C:\b"]
            m.save_settings()
            m2 = tvm.TrainModel(outputs_root=out)
            self.assertEqual(m2.config.epochs, 7)
            self.assertEqual(m2.config.venv_path, "/opt/venv")
            self.assertEqual(m2.config.data_dirs, [r"C:\a", r"C:\b"])

    def test_legacy_settings_migration(self):
        # An old settings file with only data_dir (no data_dirs) seeds data_dirs.
        import json as _json
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, tvm.TrainModel.SETTINGS_NAME), "w", encoding="utf-8") as f:
                _json.dump({"data_dir": r"C:\legacy", "epochs": 3}, f)
            m = tvm.TrainModel(outputs_root=out)
            self.assertEqual(m.config.data_dir, r"C:\legacy")
            self.assertEqual(m.config.data_dirs, [r"C:\legacy"])
            self.assertEqual(m.config.epochs, 3)


if __name__ == "__main__":
    unittest.main()
