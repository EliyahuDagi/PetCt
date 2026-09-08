"""CPU end-to-end smoke tests for slab-mode 3D autoencoder training.

NAC = non-attenuation-corrected PET, AC = attenuation-corrected PET; the autoencoder
pools both and does not need them paired.

Slab mode: volumes are resized in-plane only, every native slice is kept, and the 3D
autoencoder downsamples in-plane only, so it solves exactly the problem the 2D
autoencoder solves. Centre-inflated from the trained 2D autoencoder, it therefore starts
as an exact copy of it. These tests train a tiny 2D autoencoder, then a tiny slab-mode 3D
autoencoder warm-started from it, and check two things:

1. the checkpoint carries ``model.anisotropic`` and the slab keys, so inference and
   evaluation rebuild the same network from the embedded config;
2. with zero optimizer steps the 3D autoencoder reconstructs a volume identically to the
   2D one applied slice by slice -- the property that makes the whole exercise honest.

Mirrors ``test_ft3d_slab_smoke.py`` (synthetic loader monkeypatched into every entry
point) and is guarded by ``skipUnless(HAVE_DEPS)``.
"""

import math
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    from src.training.models.anisotropic import is_inplane_only
    from src.training.models.autoencoder2d import SliceWiseAutoencoder, build_autoencoder_2d
    from src.training.models.autoencoder3d import build_autoencoder_3d
    from src.training.train import train_ae2d, train_ae3d
    from src.training.utils.checkpointing import load_checkpoint
    from src.training.utils.metrics import read_metrics
    HAVE_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_DEPS = False

# Synthetic volume: 12 native slices (kept as is), 40x40 in-plane (resized to 32x32).
Z, Y, X = 12, 40, 40
INPLANE = 32
SLAB_DEPTH = 4
SLAB_WINDOW = 6   # fixed centre window used for validation

# Tiny 2D autoencoder, so the inflated 3D one is small enough to run on the CPU in
# seconds. One downsample level: 32x32 in-plane becomes a 16x16 latent.
TINY_AE = {"in_channels": 1, "out_channels": 1, "block_out_channels": [8, 16],
           "num_res_blocks": 1, "norm_num_groups": 4}
LATENT_CHANNELS = 4


def _synthetic_vols(z=Z, y=Y, x=X):
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


def _expected_lr_factor(step, warmup, total, min_ratio=0.0):
    """The multiplier ``build_lr_scheduler`` applies to the base rate at optimizer step ``step``.

    Same arithmetic as ``src/training/utils/schedule.py``: a linear ramp ``(step+1)/warmup``
    while ``step < warmup``, then a cosine from 1 down to ``min_ratio`` over the remaining
    ``total - warmup`` steps. Written out here so the test states the curve it expects
    instead of trusting the module it is checking.
    """
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestAe3dSlabSmoke(unittest.TestCase):
    """Train ae2d -> slab ae3d (inflated, in-plane only) on CPU."""

    COMMON = ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
              "--val_every", "1", "--val_batches", "1", "--batch_size", "2",
              "--perceptual_weight", "0"]

    def setUp(self):
        self.loader = _synthetic_vols()
        self._origs = {}
        for mod in (train_ae2d, train_ae3d):
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

    def _write_yaml(self, path, cfg):
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return path

    def _train_ae2d(self, tmp, steps="2"):
        """Tiny 2D autoencoder: the warm-start source. Returns its checkpoint path."""
        ae_dir = os.path.join(tmp, "ae2d")
        ae_ckpt = os.path.join(ae_dir, "best.pt")
        cfg = self._write_yaml(os.path.join(tmp, "ae2d_test.yaml"), {
            "seed": 42, "batch_size": 2, "learning_rate": 1.0e-4,
            "perceptual_weight": 0.0, "latent_channels": LATENT_CHANNELS,
            "model": dict(TINY_AE),
        })
        self._run(train_ae2d.main, self.COMMON + ["--steps_per_epoch", steps,
                                                 "--slice_size", str(INPLANE),
                                                 "--config", cfg, "--save_dir", ae_dir])
        self.assertTrue(os.path.exists(ae_ckpt), "ae2d bootstrap failed")
        return ae_ckpt

    def _slab_argv(self, ae_ckpt, save_dir, steps):
        return ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
                "--steps_per_epoch", steps, "--val_every", "1", "--val_batches", "1",
                "--batch_size", "2", "--perceptual_weight", "0",
                "--ae2d_ckpt", ae_ckpt, "--slab_depth", str(SLAB_DEPTH),
                "--slab_inplane_size", str(INPLANE), "--slab_window", str(SLAB_WINDOW),
                "--save_dir", save_dir]

    def test_slab_train_writes_a_rebuildable_checkpoint(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae_ckpt = self._train_ae2d(tmp)

            ae3_dir = os.path.join(tmp, "ae3d_slab")
            ae3_ckpt = os.path.join(ae3_dir, "best.pt")
            self._run(train_ae3d.main, self._slab_argv(ae_ckpt, ae3_dir, "2"))
            self.assertTrue(os.path.exists(ae3_ckpt), "slab ae3d wrote no best.pt")

            state = load_checkpoint(ae3_ckpt)
            saved = state.get("config", {})
            self.assertEqual(int(saved.get("slab_depth")), SLAB_DEPTH)
            self.assertEqual(int(saved.get("slab_inplane_size")), INPLANE)
            self.assertEqual(int(saved.get("slab_window")), SLAB_WINDOW)
            self.assertTrue(bool(saved["model"].get("anisotropic")),
                            "model.anisotropic not recorded: inference would rebuild a "
                            "depth-compressing autoencoder instead")
            # The architecture is inherited from the 2D autoencoder, not invented here.
            self.assertEqual(list(saved["model"]["block_out_channels"]),
                             list(TINY_AE["block_out_channels"]))
            self.assertEqual(int(saved.get("latent_channels")), LATENT_CHANNELS)

            # What inference and evaluation do: rebuild from the embedded config alone.
            rebuilt = build_autoencoder_3d(saved)
            rebuilt.load_state_dict(state["model"])
            self.assertTrue(is_inplane_only(rebuilt))

            rows = read_metrics(os.path.join(ae3_dir, "metrics.jsonl"))
            self.assertTrue(any(r["phase"] == "val" and "recon_l1" in r for r in rows),
                            "no val row with 'recon_l1' written")

    def test_slab_autoencoder_starts_as_the_2d_one_slice_by_slice(self):
        """Zero optimizer steps: the trainer's own build must equal the 2D autoencoder.

        Everything here goes through ``train_ae3d.main`` -- the same config assembly,
        the same warm start (which prefers the weight-averaged shadow when there is one)
        and the same centre inflation a real run uses -- and the model is then rebuilt
        from the written checkpoint exactly as inference rebuilds it. If any of that
        drifts, this test fails even though the model builder alone is still correct.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae_ckpt = self._train_ae2d(tmp)

            # steps_per_epoch 0: nothing is trained, so last.pt holds the inflated
            # weights untouched -- the state the run starts from.
            ae3_dir = os.path.join(tmp, "ae3d_slab_step0")
            self._run(train_ae3d.main, self._slab_argv(ae_ckpt, ae3_dir, "0"))
            state = load_checkpoint(os.path.join(ae3_dir, "last.pt"))
            self.assertEqual(int(state.get("step", -1)), 0, "expected an untrained checkpoint")

            ae3d = build_autoencoder_3d(state["config"])
            ae3d.load_state_dict(state["model"])
            ae3d.eval()

            # The reference is built from the SAME weights the trainer inflated:
            # load_ae2d_config prefers the weight-averaged shadow when the checkpoint has
            # one, so reading the checkpoint by hand here could compare against a
            # different set of weights and hide a real mismatch.
            ae2d_config, ae2d_weights, _ = train_ae3d.load_ae2d_config(ae_ckpt)
            ae2d = build_autoencoder_2d(ae2d_config)
            ae2d.load_state_dict(ae2d_weights)
            reference = SliceWiseAutoencoder(ae2d).eval()

            x = torch.rand(1, 1, 5, INPLANE, INPLANE)
            with torch.no_grad():
                mu3, sigma3 = ae3d.encode(x)
                mu2, sigma2 = reference.encode(x)
                recon3 = ae3d.decode(mu3)
                recon2 = reference.decode(mu2)
            # Guard against a vacuous pass (two identical all-zero outputs).
            self.assertGreater(float(recon2.abs().max()), 0.0)
            self.assertEqual(tuple(mu3.shape), tuple(mu2.shape))
            self.assertEqual(tuple(recon3.shape), tuple(x.shape))
            for name, a, b in (("latent mean", mu3, mu2),
                               ("latent spread", sigma3, sigma2),
                               ("reconstruction", recon3, recon2)):
                self.assertTrue(
                    torch.allclose(a, b, atol=1e-5),
                    "%s differs from the 2D autoencoder applied slice by slice: "
                    "largest difference %.3e" % (name, float((a - b).abs().max())))

    def test_learning_rate_warms_up_then_follows_the_schedule(self):
        """The first update is taken at a small fraction of the configured rate.

        Why: the inflated autoencoder starts as an exact copy of the 2D one, and a
        full-size first Adam step (every weight moved by the whole learning rate, the
        zero depth taps included) was measured to destroy that start (validation loss
        0.0030 -> 0.0151 after one step, 2026-09-06). The trainer now uses the same
        ``build_lr_scheduler`` schedule as the flow trainer: a linear ramp over
        ``lr_warmup_steps`` optimizer steps, then a cosine decay from the configured rate
        down to ``lr_min_ratio`` (default 0) of it at the last step.

        The schedule DECAYS, so the last step is not at the configured rate. Read back
        from the ``lr`` key of the train rows in ``metrics.jsonl``, the checks are:

        * step 0 is well below the configured rate (< 0.5x) and above zero;
        * the ramp never goes down during the warmup;
        * the last warmup step (index ``warmup - 1``) equals the configured rate to 1e-9;
        * the last step equals the configured rate times the cosine factor to 1e-9
          (with warmup 4 of 6 steps that factor is exactly 0.5).

        Run in slab mode and in cube mode: nothing about the warmup is slab-specific.
        """
        lr, warmup, steps = 1.0e-4, 4, 6
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae_ckpt = self._train_ae2d(tmp)
            cfg = self._write_yaml(os.path.join(tmp, "ae3d_lr_test.yaml"), {
                "seed": 42, "learning_rate": lr, "lr_warmup_steps": warmup,
                "perceptual_weight": 0.0,
            })
            slab_dir = os.path.join(tmp, "ae3d_slab_lr")
            cube_dir = os.path.join(tmp, "ae3d_cube_lr")
            runs = {
                "slab": (slab_dir, self._slab_argv(ae_ckpt, slab_dir, str(steps)) + ["--config", cfg]),
                # Cube mode: the same invocation test_train_scripts_cpu uses (the depth
                # band is RESIZED to a 16^3 cube, so 12 native slices are enough).
                "cube": (cube_dir, self.COMMON + ["--steps_per_epoch", str(steps), "--crop_size", "16",
                                                  "--ae2d_ckpt", ae_ckpt, "--save_dir", cube_dir,
                                                  "--config", cfg]),
            }
            for mode, (save_dir, argv) in runs.items():
                with self.subTest(mode=mode):
                    self._run(train_ae3d.main, argv)
                    rows = read_metrics(os.path.join(save_dir, "metrics.jsonl"))
                    train_rows = sorted((r for r in rows if r["phase"] == "train"),
                                        key=lambda r: r["step"])
                    self.assertEqual([r["step"] for r in train_rows], list(range(steps)))
                    self.assertTrue(all("lr" in r for r in train_rows),
                                    "train rows do not carry the 'lr' key")
                    lrs = [float(r["lr"]) for r in train_rows]

                    self.assertGreater(lrs[0], 0.0)
                    self.assertLess(lrs[0], 0.5 * lr,
                                    "step 0 ran at %.3e, not a warmed-up fraction of %.3e" % (lrs[0], lr))
                    for a, b in zip(lrs[:warmup - 1], lrs[1:warmup]):
                        self.assertLessEqual(a, b, "learning rate went down during the warmup")
                    self.assertAlmostEqual(lrs[warmup - 1], lr, delta=1e-9)

                    expected = [lr * _expected_lr_factor(s, warmup, steps) for s in range(steps)]
                    for s, (got, want) in enumerate(zip(lrs, expected)):
                        self.assertAlmostEqual(
                            got, want, delta=1e-9,
                            msg="step %d: lr %.6e, schedule prescribes %.6e" % (s, got, want))
                    # The schedule decays after the warmup: the last step sits below the
                    # configured rate (here exactly half of it).
                    self.assertLess(lrs[-1], lr)
                    self.assertAlmostEqual(lrs[-1], 0.5 * lr, delta=1e-9)


class TestAe3dSlabYamlSanity(unittest.TestCase):
    """ae3d_slab.yaml sanity (no torch needed): a typo here costs a multi-hour run."""

    def test_slab_yaml_fields(self):
        import yaml
        path = os.path.join(ROOT, "src", "training", "configs", "ae3d_slab.yaml")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertGreater(int(cfg.get("slab_depth")), 0)
        self.assertEqual(int(cfg.get("slab_inplane_size")), 128)
        self.assertGreaterEqual(int(cfg.get("slab_window")), int(cfg.get("slab_depth")))
        self.assertTrue(bool(cfg["model"].get("anisotropic")))
        self.assertEqual(int(cfg["model"]["spatial_dims"]), 3)
        self.assertEqual(int(cfg.get("latent_channels")), 8)
        self.assertEqual(list(cfg["model"]["block_out_channels"]), [64, 128, 256])
        self.assertEqual(int(cfg["model"]["num_res_blocks"]), 2)
        self.assertEqual(float(cfg.get("learning_rate")), 5.0e-5)
        # Warmup is what protects the exact 2D start from the first Adam update.
        self.assertEqual(int(cfg.get("lr_warmup_steps")), 500)
        self.assertEqual(str(cfg.get("ae2d_ckpt")), "outputs/ae2d_p/best.pt")
        self.assertEqual(str(cfg.get("output_dir")), "outputs/ae3d_slab")
        # Depth-resampling augmentations must be off: slab mode keeps native slices intact.
        self.assertEqual(float(cfg["augment"]["affine_prob"]), 0.0)
        self.assertEqual(float(cfg["augment"]["elastic_prob"]), 0.0)


if __name__ == "__main__":
    unittest.main()
