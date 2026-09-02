"""Tests for the 3D flow port: sampling-helper routing, gradient accumulation, and
the ft3d flow wiring.

Fast routing tests (no torch training) exercise the shared
``src.training.utils.translate.sample_nac_to_ac`` helper with a channel-recording
stub model. The heavier CPU end-to-end tests (grad accumulation, flow wiring) mirror
``test_train_scripts_cpu.py`` and are guarded by ``skipUnless(HAVE_DEPS)``.
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
    from src.training.train import train_ae2d, train_ae3d, train_ft3d
    from src.training.utils.metrics import read_metrics
    HAVE_DEPS = True
except Exception as _e:  # pragma: no cover - environment dependent
    HAVE_DEPS = False
    _IMPORT_ERROR = _e

# The routing test only needs torch + the schedule/helper (no monai-generative
# models), so import them under their own guard.
try:
    import torch  # noqa: F811
    from src.training.utils.sampling import DiffusionSchedule
    from src.training.utils.translate import sample_nac_to_ac
    HAVE_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False


if HAVE_TORCH:
    class _ChannelRecorder(torch.nn.Module):
        """Stub UNet recording the channel count of the LAST input it saw.

        Returns ``out_channels`` channels so its output matches the AC-latent shape
        the samplers expect (flow velocity / epsilon both have C channels).
        """

        def __init__(self, out_channels):
            super().__init__()
            self.out_channels = int(out_channels)
            self.last_in_channels = None
            self.n_calls = 0

        def forward(self, x, t):
            self.last_in_channels = int(x.shape[1])
            self.n_calls += 1
            return torch.zeros((x.shape[0], self.out_channels, *x.shape[2:]),
                               dtype=x.dtype, device=x.device)


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class TestSampleNacToAcRouting(unittest.TestCase):
    """sample_nac_to_ac routing: flow feeds C channels (no concat) for 2D AND 3D;
    epsilon feeds 2C (concat). Flow ignores a non-trivial guidance_scale."""

    def _run_flow(self, shape):
        sched = DiffusionSchedule(schedule="cosine")
        c = shape[1]
        rec = _ChannelRecorder(out_channels=c)
        cond = torch.randn(shape)
        out = sample_nac_to_ac(
            rec, sched, cond, {"prediction_type": "flow"},
            num_steps=4, spacing="linear", guidance_scale=1.0, clip_x0=None, tag="ft3d",
        )
        self.assertEqual(out.shape, shape)
        # Flow feeds ONLY x_t (C channels), never the 2C concat.
        self.assertEqual(rec.last_in_channels, c)
        return rec

    def test_flow_feeds_C_2d(self):
        self._run_flow((2, 4, 8, 8))

    def test_flow_feeds_C_3d(self):
        self._run_flow((2, 3, 4, 8, 8))

    def test_epsilon_feeds_2C_2d(self):
        sched = DiffusionSchedule(schedule="cosine")
        c = 4
        rec = _ChannelRecorder(out_channels=c)
        cond = torch.randn(2, c, 8, 8)
        sample_nac_to_ac(
            rec, sched, cond, {"prediction_type": "epsilon"},
            num_steps=4, spacing="linear", guidance_scale=1.0, clip_x0=4.0, tag="diff2d",
        )
        self.assertEqual(rec.last_in_channels, 2 * c)  # concat [x_t | NAC]

    def test_epsilon_feeds_2C_3d(self):
        sched = DiffusionSchedule(schedule="cosine")
        c = 3
        rec = _ChannelRecorder(out_channels=c)
        cond = torch.randn(2, c, 4, 8, 8)
        sample_nac_to_ac(
            rec, sched, cond, {"prediction_type": "epsilon"},
            num_steps=4, spacing="linear", guidance_scale=1.0, clip_x0=4.0, tag="ft3d",
        )
        self.assertEqual(rec.last_in_channels, 2 * c)

    def test_flow_ignores_guidance_scale(self):
        # guidance_scale != 1.0 in flow mode must not crash and must NOT double the
        # input channels (no CFG concat doubling); it stays at C.
        sched = DiffusionSchedule(schedule="cosine")
        c = 4
        rec = _ChannelRecorder(out_channels=c)
        cond = torch.randn(2, c, 8, 8)
        sample_nac_to_ac(
            rec, sched, cond, {"prediction_type": "flow"},
            num_steps=4, spacing="linear", guidance_scale=3.0, clip_x0=None, tag="ft3d",
        )
        self.assertEqual(rec.last_in_channels, c)

    def test_missing_prediction_type_defaults_to_epsilon(self):
        sched = DiffusionSchedule(schedule="cosine")
        c = 4
        rec = _ChannelRecorder(out_channels=c)
        cond = torch.randn(2, c, 8, 8)
        sample_nac_to_ac(
            rec, sched, cond, {},
            num_steps=4, spacing="linear", guidance_scale=1.0, clip_x0=4.0, tag="diff2d",
        )
        self.assertEqual(rec.last_in_channels, 2 * c)


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
class TestFt3dFlowAndGradAccum(unittest.TestCase):
    """CPU end-to-end: bootstrap a tiny ae3d, then run ft3d to check grad accumulation
    (one optimizer step per global_step) and the flow wiring (in_channels = C)."""

    def setUp(self):
        self.loader = _synthetic_vols()
        self._origs = {}
        for mod in (train_ae2d, train_ae3d, train_ft3d):
            self._origs[mod] = mod.load_patient_volumes
            mod.load_patient_volumes = self.loader
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

    def _run(self, main_fn, argv):
        sys.argv = ["prog"] + argv
        main_fn()

    def _bootstrap_ae3d(self, tmp):
        """Train tiny ae2d then inflate/fine-tune ae3d; return the ae3d checkpoint."""
        ae_dir = os.path.join(tmp, "ae2d")
        ae3_dir = os.path.join(tmp, "ae3d")
        ae_ckpt = os.path.join(ae_dir, "best.pt")
        ae3_ckpt = os.path.join(ae3_dir, "best.pt")
        common = ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
                  "--steps_per_epoch", "2", "--val_every", "1", "--val_batches", "1",
                  "--batch_size", "2", "--perceptual_weight", "0"]
        self._run(train_ae2d.main, common + ["--slice_size", "32", "--save_dir", ae_dir])
        self._run(train_ae3d.main, common + ["--crop_size", "16", "--ae2d_ckpt", ae_ckpt,
                                             "--save_dir", ae3_dir])
        self.assertTrue(os.path.exists(ae3_ckpt), "ae3d bootstrap failed")
        return ae3_ckpt

    def _write_config(self, tmp, **kv):
        """Write a tiny ft3d yaml with the given overrides; return its path."""
        import yaml
        cfg = {
            "seed": 42,
            "batch_size": 1,
            "learning_rate": 1.0e-4,
            "ema_decay": 0.0,
            "perceptual_weight": 0.0,
            "model": {"num_channels": [8, 16], "attention_levels": [False, False],
                      "num_res_blocks": 1},
        }
        cfg.update(kv)
        path = os.path.join(tmp, "ft3d_test.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return path

    def test_grad_accum_one_optimizer_step_per_global_step(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae3_ckpt = self._bootstrap_ae3d(tmp)
            d3_dir = os.path.join(tmp, "ft3d")
            cfg = self._write_config(tmp, grad_accum_steps=3)

            # Count optimizer steps by wrapping Adam.step (proves accumulation: ONE
            # optimizer step per global_step regardless of grad_accum micro-batches).
            calls = {"n": 0}
            orig_step = torch.optim.Adam.step

            def counting_step(self, *a, **k):
                calls["n"] += 1
                return orig_step(self, *a, **k)

            torch.optim.Adam.step = counting_step
            try:
                self._run(train_ft3d.main,
                          ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
                           "--steps_per_epoch", "2", "--val_every", "1", "--val_batches", "1",
                           "--latent_size", "16", "--ae_ckpt", ae3_ckpt, "--config", cfg,
                           "--save_dir", d3_dir])
            finally:
                torch.optim.Adam.step = orig_step

            # 2 global_steps => exactly 2 optimizer steps despite grad_accum_steps=3.
            self.assertEqual(calls["n"], 2,
                             f"expected 1 optimizer step/global_step (2), got {calls['n']}")
            self.assertTrue(os.path.exists(os.path.join(d3_dir, "best.pt")))

    def test_ft3d_flow_wiring(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae3_ckpt = self._bootstrap_ae3d(tmp)
            d3_dir = os.path.join(tmp, "ft3d_flow")
            cfg = self._write_config(tmp, prediction_type="flow", latent_scale=1.0)
            self._run(train_ft3d.main,
                      ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
                       "--steps_per_epoch", "2", "--val_every", "1", "--val_batches", "1",
                       "--latent_size", "16", "--ae_ckpt", ae3_ckpt, "--config", cfg,
                       "--save_dir", d3_dir])
            ckpt_path = os.path.join(d3_dir, "best.pt")
            self.assertTrue(os.path.exists(ckpt_path))

            from src.training.utils.checkpointing import load_checkpoint
            state = load_checkpoint(ckpt_path)
            saved_cfg = state.get("config", {})
            self.assertEqual(str(saved_cfg.get("prediction_type")).lower(), "flow")
            latent_channels = int(saved_cfg.get("latent_channels"))
            # Flow feeds ONLY x_t => in_channels == C, not 2C.
            self.assertEqual(int(saved_cfg["model"]["in_channels"]), latent_channels)

            rows = read_metrics(os.path.join(d3_dir, "metrics.jsonl"))
            self.assertTrue(any(r["phase"] == "val" and "l1" in r for r in rows),
                            "no val row with 'l1' written")


    def test_latent_intensity_weight_reaches_flow_loss(self):
        """`latent_weight: intensity` must hand flow_loss a real per-position weight map --
        and the DEFAULT config must still hand it None (the plain-mean path)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae3_ckpt = self._bootstrap_ae3d(tmp)
            seen = []
            orig = train_ft3d.flow_loss

            def spy(*a, **kw):
                seen.append(kw.get("weight_map"))
                return orig(*a, **kw)

            def run(save_dir, **cfg_kv):
                cfg = self._write_config(tmp, prediction_type="flow", latent_scale=1.0,
                                         **cfg_kv)
                train_ft3d.flow_loss = spy
                try:
                    self._run(train_ft3d.main,
                              ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
                               "--steps_per_epoch", "1", "--val_every", "1",
                               "--val_batches", "1", "--latent_size", "16",
                               "--ae_ckpt", ae3_ckpt, "--config", cfg,
                               "--save_dir", os.path.join(tmp, save_dir)])
                finally:
                    train_ft3d.flow_loss = orig

            run("ft3d_iwlat", latent_weight="intensity", bg_weight=0.1,
                iw_gamma=1.0, iw_percentile=99.0, iw_source="ac")
            self.assertTrue(seen, "flow_loss was never called")
            w = seen[0]
            self.assertIsNotNone(w, "latent_weight: intensity passed no weight map")
            self.assertEqual(int(w.shape[1]), 1, "weight must broadcast over latent channels")
            self.assertGreaterEqual(float(w.min()), 0.1 - 1e-6)
            self.assertLessEqual(float(w.max()), 1.0 + 1e-6)
            self.assertGreater(float(w.max()), float(w.min()), "weight map is constant")

            # Regression guard: without the knob the original unweighted path must run.
            seen.clear()
            run("ft3d_plain")
            self.assertTrue(seen)
            self.assertIsNone(seen[0], "default config must not weight the velocity MSE")


class TestFt3dFlowYamlSanity(unittest.TestCase):
    """ft3d_flow.yaml config sanity (no torch needed)."""

    def test_flow_yaml_fields(self):
        import yaml
        path = os.path.join(ROOT, "src", "training", "configs", "ft3d_flow.yaml")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(str(cfg.get("prediction_type")).lower(), "flow")
        self.assertEqual(float(cfg.get("latent_scale")), 1.0)
        self.assertEqual(float(cfg.get("perceptual_weight")), 0.0)
        self.assertEqual(int(cfg["model"]["spatial_dims"]), 3)

    def test_iwlat_arm_yaml_fields(self):
        """Both PET-value-weighting arms: the knob is set, and nothing else is on.

        A typo here costs a multi-hour run that silently trains the control, which is
        exactly how an ablation arm becomes worthless.
        """
        import yaml
        for name, channels in (("ft3d_ft_iwlat", 8), ("ft3d_2x_ft_iwlat", 16)):
            with self.subTest(config=name):
                path = os.path.join(ROOT, "src", "training", "configs", name + ".yaml")
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                self.assertEqual(str(cfg.get("prediction_type")).lower(), "flow")
                self.assertEqual(str(cfg.get("latent_weight")), "intensity")
                self.assertLess(float(cfg.get("bg_weight")), 1.0)
                self.assertIn(str(cfg.get("iw_source")), ("ac", "nac", "union"))
                self.assertIn(str(cfg.get("iw_pool")), ("avg", "max"))
                self.assertGreater(float(cfg.get("iw_gamma")), 0.0)
                self.assertGreater(float(cfg.get("iw_clip")), 0.0)
                self.assertGreater(float(cfg.get("iw_percentile")), 0.0)
                self.assertLessEqual(float(cfg.get("iw_percentile")), 100.0)
                # Single-variable: no other loss term rides along.
                self.assertEqual(float(cfg.get("perceptual_weight")), 0.0)
                self.assertEqual(float(cfg.get("quant_weight")), 0.0)
                self.assertEqual(str(cfg.get("quant_loss")), "none")
                self.assertEqual(str(cfg.get("flow_loss_weighting")), "none")
                self.assertEqual(int(cfg["model"]["in_channels"]), channels)
                self.assertEqual(int(cfg["model"]["out_channels"]), channels)


if __name__ == "__main__":
    unittest.main()
