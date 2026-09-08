"""CPU end-to-end smoke test for slab-mode ft3d training and inference.

NAC = non-attenuation-corrected PET, AC = attenuation-corrected PET.

Slab mode: volumes are resized in-plane only, every native slice is kept, the frozen
autoencoder makes one latent slice per image slice, and the 3D flow UNet (downsampling
in-plane only) is trained on short depth slabs and run on the whole volume through
overlapping depth windows. This test trains a tiny 2D autoencoder, a tiny 2D flow model
to warm start from, then a tiny slab-mode ft3d, and finally runs inference on the whole
synthetic volume. It mirrors ``test_ft3d_flow.py`` (synthetic loader monkeypatched
into every entry point) and is guarded by ``skipUnless(HAVE_DEPS)``.

The last class covers which autoencoders slab mode accepts: the 2D one applied slice by
slice, and a 3D one built in-plane only (depth kept). A depth-compressing 3D
autoencoder is still refused, and the refusal names both acceptable options.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import numpy as np
    import torch
    from src.training import infer
    from src.training.models.anisotropic import is_inplane_only
    from src.training.models.autoencoder2d import SliceWiseAutoencoder, build_autoencoder_2d
    from src.training.models.autoencoder3d import build_autoencoder_3d
    from src.training.train import train_ae2d, train_diff2d, train_ft3d
    from src.training.utils.checkpointing import load_checkpoint
    from src.training.utils.metrics import read_metrics
    HAVE_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_DEPS = False

# Synthetic volume: 12 native slices (kept as is), 40x40 in-plane (resized to 32x32).
Z, Y, X = 12, 40, 40
INPLANE = 32
SLAB_DEPTH = 4
SLAB_WINDOW = 6   # whole-volume windows of 6 slices ...
SLAB_STRIDE = 3   # ... advancing 3 slices, so 12 slices need several overlapping windows

# Same tiny UNet for the 2D warm-start source and the 3D slab model, so every
# convolution inflates (in/out channels and in-plane kernel sizes match).
TINY_UNET = {"num_channels": [8, 16], "attention_levels": [False, True],
             "num_res_blocks": 1, "norm_num_groups": 4}

# Tiny 3D autoencoder used by the acceptance tests below. With anisotropic true it
# downsamples in-plane only and keeps every slice; with it false it is the cube
# autoencoder, which compresses depth as well.
AE3D_LATENT_CHANNELS = 4
AE3D_MODEL = {"in_channels": 1, "out_channels": 1, "block_out_channels": [8, 16],
              "num_res_blocks": 1, "norm_num_groups": 4, "attention_levels": [False, False]}
TINY_AE2D = {"latent_channels": 2,
             "model": {"in_channels": 1, "out_channels": 1,
                       "block_out_channels": [4, 8], "num_res_blocks": 1}}


def _ae3d_config(anisotropic):
    cfg = {"latent_channels": AE3D_LATENT_CHANNELS, "model": dict(AE3D_MODEL)}
    cfg["model"]["anisotropic"] = bool(anisotropic)
    return cfg


def _write_ae3d_ckpt(path, anisotropic):
    """Save a tiny 3D autoencoder checkpoint and return its path.

    Written directly instead of running train_ae3d: what slab mode inspects is the model
    that ``build_autoencoder_3d`` makes from this config, and a real autoencoder run
    would add minutes to the test without changing anything it checks.
    """
    cfg = _ae3d_config(anisotropic)
    model = build_autoencoder_3d(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "task": "ae3d"}, path)
    return path


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


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestFt3dSlabSmoke(unittest.TestCase):
    """Train ae2d -> diff2d (flow) -> slab ft3d (inflated, anisotropic) -> infer, all on CPU."""

    COMMON = ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
              "--steps_per_epoch", "2", "--val_every", "1", "--val_batches", "1",
              "--batch_size", "2", "--perceptual_weight", "0"]

    def setUp(self):
        self.loader = _synthetic_vols()
        self._origs = {}
        for mod in (train_ae2d, train_diff2d, train_ft3d, infer):
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

    def test_slab_train_inflate_and_infer(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            # 1) Tiny 2D autoencoder: the frozen slice-wise bottleneck of slab mode.
            ae_dir = os.path.join(tmp, "ae2d")
            ae_ckpt = os.path.join(ae_dir, "best.pt")
            self._run(train_ae2d.main, self.COMMON + ["--slice_size", str(INPLANE), "--save_dir", ae_dir])
            self.assertTrue(os.path.exists(ae_ckpt), "ae2d bootstrap failed")

            # 2) Tiny 2D flow model: the warm-start source for the 3D slab model.
            d2_dir = os.path.join(tmp, "diff2d_flow")
            d2_ckpt = os.path.join(d2_dir, "best.pt")
            d2_cfg = self._write_yaml(os.path.join(tmp, "diff2d_test.yaml"), {
                "seed": 42, "batch_size": 2, "learning_rate": 1.0e-4, "ema_decay": 0.0,
                "perceptual_weight": 0.0, "prediction_type": "flow", "latent_scale": 1.0,
                "model": dict(TINY_UNET),
            })
            self._run(train_diff2d.main, self.COMMON + ["--latent_size", str(INPLANE), "--ae_ckpt", ae_ckpt,
                                                       "--config", d2_cfg, "--save_dir", d2_dir])
            self.assertTrue(os.path.exists(d2_ckpt), "diff2d flow bootstrap failed")

            # 3) Slab-mode ft3d: 2D AE slice by slice, slabs of 4 native slices, in-plane
            #    32, anisotropic UNet inflated from the 2D flow checkpoint.
            d3_dir = os.path.join(tmp, "ft3d_slab")
            d3_ckpt = os.path.join(d3_dir, "best.pt")
            d3_cfg = self._write_yaml(os.path.join(tmp, "ft3d_slab_test.yaml"), {
                "seed": 42, "batch_size": 2, "learning_rate": 1.0e-4, "ema_decay": 0.0,
                "perceptual_weight": 0.0, "prediction_type": "flow", "latent_scale": 1.0,
                "slab_window": SLAB_WINDOW, "slab_stride": SLAB_STRIDE,
                "model": dict(TINY_UNET),
            })
            self._run(train_ft3d.main,
                      ["--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
                       "--steps_per_epoch", "2", "--val_every", "1", "--val_batches", "1",
                       "--ae2d_ckpt", ae_ckpt, "--slab_depth", str(SLAB_DEPTH), "--anisotropic",
                       "--latent_size", str(INPLANE), "--inflate_from", d2_ckpt,
                       "--config", d3_cfg, "--save_dir", d3_dir])
            self.assertTrue(os.path.exists(d3_ckpt), "slab ft3d wrote no best.pt")

            saved = load_checkpoint(d3_ckpt).get("config", {})
            self.assertEqual(str(saved.get("ae_mode")), "2d_slicewise")
            self.assertEqual(int(saved.get("slab_depth")), SLAB_DEPTH)
            self.assertEqual(int(saved.get("slab_window")), SLAB_WINDOW)
            self.assertEqual(int(saved.get("slab_stride")), SLAB_STRIDE)
            self.assertEqual(int(saved.get("slab_inplane_size")), INPLANE)
            self.assertEqual(str(saved.get("prediction_type")).lower(), "flow")
            self.assertTrue(bool(saved["model"].get("anisotropic")), "model.anisotropic not recorded")
            latent_channels = int(saved.get("latent_channels"))
            # Flow feeds ONLY the interpolant => in_channels == C (the 2D AE's latent channels).
            self.assertEqual(int(saved["model"]["in_channels"]), latent_channels)
            self.assertEqual(int(saved["model"]["out_channels"]), latent_channels)

            rows = read_metrics(os.path.join(d3_dir, "metrics.jsonl"))
            self.assertTrue(any(r["phase"] == "val" and "l1" in r for r in rows),
                            "no val row with 'l1' written")

            # 4) Inference on the whole synthetic volume: the depth comes out native (12),
            #    in-plane at the recorded slab_inplane_size (no --size given), and the
            #    NAC input is saved next to the prediction.
            out_dir = os.path.join(tmp, "infer")
            self._run(infer.main,
                      ["--task", "ft3d", "--ae_ckpt", ae_ckpt, "--diff_ckpt", d3_ckpt,
                       "--data_dir", "synthetic", "--device", "cpu", "--ddim_steps", "2",
                       "--out", out_dir])
            pred = np.load(os.path.join(out_dir, "pred.npy"))
            self.assertEqual(tuple(pred.shape), (Z, INPLANE, INPLANE))
            gt = np.load(os.path.join(out_dir, "gt.npy"))
            self.assertEqual(tuple(gt.shape), (Z, INPLANE, INPLANE))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "nac.npy")))
            self.assertTrue(np.isfinite(pred).all(), "prediction has non-finite values")


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestFt3dSlabFixedValSet(unittest.TestCase):
    """Slab-mode validation scores a FIXED set of patients, in a fixed order, every pass.

    The multi-patient branch that builds this set is not reachable with the synthetic
    single-root loader above (``enumerate_patients`` finds no patients), so the two
    helpers behind it are checked directly.
    """

    VAL_IDX = [3, 8, 11, 14, 20, 27, 31, 35, 40, 42]

    def test_fixed_list_depends_only_on_seed_and_is_distinct(self):
        a = train_ft3d._fixed_val_indices(self.VAL_IDX, 4, seed=42)
        b = train_ft3d._fixed_val_indices(self.VAL_IDX, 4, seed=42)   # what a --resume run derives
        self.assertEqual(a, b, "the fixed val set must depend only on (seed, val_idx)")
        self.assertEqual(len(a), 4)
        self.assertEqual(len(set(a)), 4, "indices must be distinct")
        self.assertTrue(set(a) <= set(self.VAL_IDX))
        # Fewer val patients than --val_batches: every val patient, each exactly once.
        self.assertEqual(sorted(train_ft3d._fixed_val_indices(self.VAL_IDX, 50, seed=42)),
                         sorted(self.VAL_IDX))

    def test_two_validation_passes_draw_the_same_sequence(self):
        cursor = train_ft3d._FixedValCursor(train_ft3d._fixed_val_indices(self.VAL_IDX, 4, seed=7))
        first = [cursor.next_index() for _ in range(4)]
        cursor.reset()   # what _validate does at the start of every pass
        second = [cursor.next_index() for _ in range(4)]
        self.assertEqual(first, second)
        self.assertEqual(first, cursor.indices)


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestSlabAutoencoderAcceptance(unittest.TestCase):
    """Slab mode takes any autoencoder that keeps every slice, and only those.

    Two kinds keep every slice: the 2D autoencoder applied slice by slice, and a 3D
    autoencoder built in-plane only (it downsamples in-plane and never along depth, so
    the latent has one slice per image slice, with depth context the 2D one cannot see).
    The cube 3D autoencoder compresses depth and must still be refused, with a message
    that names both acceptable options.
    """

    def setUp(self):
        self.loader = _synthetic_vols()
        self._orig = train_ft3d.load_patient_volumes
        train_ft3d.load_patient_volumes = self.loader
        self._argv = sys.argv

    def tearDown(self):
        train_ft3d.load_patient_volumes = self._orig
        sys.argv = self._argv
        import logging
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)

    def _write_cfg(self, path):
        import yaml
        cfg = {"seed": 42, "batch_size": 2, "learning_rate": 1.0e-4, "ema_decay": 0.0,
               "perceptual_weight": 0.0, "prediction_type": "flow", "latent_scale": 1.0,
               "slab_window": SLAB_WINDOW, "slab_stride": SLAB_STRIDE,
               "model": dict(TINY_UNET)}
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return path

    def _run_slab(self, ae_ckpt, cfg_path, save_dir):
        # No --ae2d_ckpt: ae_mode stays "3d", so the autoencoder is judged by what it is.
        sys.argv = ["prog", "--data_dir", "synthetic", "--device", "cpu", "--epochs", "1",
                    "--steps_per_epoch", "1", "--val_every", "1", "--val_batches", "1",
                    "--batch_size", "2", "--ae_ckpt", ae_ckpt, "--slab_depth", str(SLAB_DEPTH),
                    "--anisotropic", "--latent_size", str(INPLANE), "--config", cfg_path,
                    "--save_dir", save_dir]
        train_ft3d.main()

    def test_depth_is_read_from_the_built_model(self):
        inplane = build_autoencoder_3d(_ae3d_config(anisotropic=True))
        cube = build_autoencoder_3d(_ae3d_config(anisotropic=False))
        self.assertTrue(is_inplane_only(inplane))
        self.assertFalse(is_inplane_only(cube))
        self.assertTrue(train_ft3d.ae_preserves_depth(inplane, "3d"))
        self.assertFalse(train_ft3d.ae_preserves_depth(cube, "3d"))
        # The claim is real, not just a marker: five slices in, five latent slices out
        # for the in-plane one, fewer for the cube one.
        x = torch.randn(1, 1, 5, INPLANE, INPLANE)
        with torch.no_grad():
            self.assertEqual(int(inplane.encode(x)[0].shape[2]), 5)
            self.assertLess(int(cube.encode(x)[0].shape[2]), 5)
        # The 2D autoencoder run slice by slice keeps depth by construction.
        wrapper = SliceWiseAutoencoder(build_autoencoder_2d(TINY_AE2D))
        self.assertTrue(train_ft3d.ae_preserves_depth(wrapper, "2d_slicewise"))
        self.assertIn("slice by slice", train_ft3d.ae_description(wrapper, "2d_slicewise"))
        self.assertIn("in-plane-only", train_ft3d.ae_description(inplane, "3d"))
        self.assertIn("depth-compressing", train_ft3d.ae_description(cube, "3d"))

    def test_slab_mode_accepts_an_inplane_only_3d_autoencoder(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae_ckpt = _write_ae3d_ckpt(os.path.join(tmp, "ae3d_inplane", "best.pt"),
                                       anisotropic=True)
            save_dir = os.path.join(tmp, "ft3d_slab_ae3d")
            self._run_slab(ae_ckpt, self._write_cfg(os.path.join(tmp, "ft3d.yaml")), save_dir)
            ckpt = os.path.join(save_dir, "best.pt")
            self.assertTrue(os.path.exists(ckpt), "slab ft3d on the in-plane 3D AE wrote no best.pt")
            saved = load_checkpoint(ckpt).get("config", {})
            self.assertEqual(str(saved.get("ae_mode")), "3d")
            self.assertEqual(int(saved.get("slab_depth")), SLAB_DEPTH)
            self.assertEqual(int(saved.get("slab_inplane_size")), INPLANE)
            # Latent channels come from the autoencoder, not from the diffusion config.
            self.assertEqual(int(saved.get("latent_channels")), AE3D_LATENT_CHANNELS)
            rows = read_metrics(os.path.join(save_dir, "metrics.jsonl"))
            self.assertTrue(any(r["phase"] == "val" and "l1" in r for r in rows),
                            "no val row with 'l1' written")

    def test_slab_mode_refuses_a_depth_compressing_3d_autoencoder(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ae_ckpt = _write_ae3d_ckpt(os.path.join(tmp, "ae3d_cube", "best.pt"),
                                       anisotropic=False)
            save_dir = os.path.join(tmp, "ft3d_slab_cube")
            with self.assertRaises(ValueError) as caught:
                self._run_slab(ae_ckpt, self._write_cfg(os.path.join(tmp, "ft3d.yaml")), save_dir)
            message = str(caught.exception)
            # Both ways out must be named, or the message is not actionable.
            self.assertIn("2d_slicewise", message)
            self.assertIn("anisotropic", message)
            self.assertIn("depth-compressing 3D autoencoder", message)
            self.assertIn(str(SLAB_DEPTH), message)


class TestFt3dSlabYamlSanity(unittest.TestCase):
    """ft3d_slab_flow.yaml sanity (no torch needed): a typo here costs a multi-hour run."""

    def test_slab_yaml_fields(self):
        import yaml
        path = os.path.join(ROOT, "src", "training", "configs", "ft3d_slab_flow.yaml")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(str(cfg.get("prediction_type")).lower(), "flow")
        self.assertEqual(str(cfg.get("ae_mode")), "2d_slicewise")
        self.assertGreater(int(cfg.get("slab_depth")), 0)
        self.assertGreaterEqual(int(cfg.get("slab_window")), int(cfg.get("slab_depth")))
        self.assertLessEqual(int(cfg.get("slab_stride")), int(cfg.get("slab_window")))
        self.assertEqual(int(cfg.get("slab_inplane_size")), 128)
        self.assertEqual(float(cfg.get("perceptual_weight")), 0.0)
        self.assertTrue(bool(cfg["model"].get("anisotropic")))
        self.assertEqual(int(cfg["model"]["spatial_dims"]), 3)
        self.assertEqual(int(cfg["model"]["in_channels"]), int(cfg.get("latent_channels")))
        self.assertEqual(int(cfg["model"]["out_channels"]), int(cfg.get("latent_channels")))
        # Depth-resampling augmentations must be off: slab mode keeps native slices intact.
        self.assertEqual(float(cfg["augment"]["affine_prob"]), 0.0)
        self.assertEqual(float(cfg["augment"]["elastic_prob"]), 0.0)
        self.assertEqual(str(cfg.get("output_dir")), "outputs/ft3d_slab_flow")


if __name__ == "__main__":
    unittest.main()
