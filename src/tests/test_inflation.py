import os
import sys
import tempfile
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.training.models.inflation import inflate_conv2d_to_3d, map_state_dict_2d_to_3d

# The trainer-level warm-start test needs the tiny MONAI UNets and the ft3d trainer
# module. Guarded so the two pure-tensor tests above still run without them.
try:
    from src.training.models.diffusion2d import build_diffusion_2d
    from src.training.models.diffusion3d import build_diffusion_3d
    from src.training.train.train_ft3d import _inflation_source, inflate_and_load
    from src.training.utils.checkpointing import load_checkpoint
    HAVE_TRAINER_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_TRAINER_DEPS = False

# Tiny matching 2D/3D UNets (same shape as test_inflation_freeze.py): small channels,
# one attention-free downsample, so every 2D conv inflates into the 3D net on CPU.
_UNET_MODEL = {
    "in_channels": 4,
    "out_channels": 4,
    "num_channels": [8, 16],
    "attention_levels": [False, False],
    "num_res_blocks": 1,
}


def _unet_cfg():
    return {"model": dict(_UNET_MODEL)}


class TestInflation(unittest.TestCase):
    def test_inflate_conv2d_to_3d_center(self):
        weight_2d = torch.randn(2, 3, 3, 3)
        weight_3d = inflate_conv2d_to_3d(weight_2d, kernel_depth=3)
        self.assertEqual(weight_3d.shape, (2, 3, 3, 3, 3))
        self.assertTrue(torch.allclose(weight_3d[:, :, 1, :, :], weight_2d))
        self.assertTrue(torch.allclose(weight_3d[:, :, 0, :, :], torch.zeros_like(weight_2d)))
        self.assertTrue(torch.allclose(weight_3d[:, :, 2, :, :], torch.zeros_like(weight_2d)))

    def test_map_state_dict_2d_to_3d(self):
        state_2d = {
            "conv.weight": torch.randn(2, 3, 3, 3),
            "bias": torch.randn(2),
        }
        state_3d = {
            "conv.weight": torch.zeros(2, 3, 3, 3, 3),
            "bias": torch.zeros(2),
        }
        mapped, missing = map_state_dict_2d_to_3d(state_2d, state_3d)
        self.assertIn("conv.weight", mapped)
        self.assertIn("bias", mapped)
        self.assertEqual(len(missing), 0)

    @unittest.skipUnless(HAVE_TRAINER_DEPS, "torch / monai-generative / ft3d trainer not importable")
    def test_inflation_source_prefers_ema_shadow(self):
        """The ft3d --inflate_from warm start must inflate the EMA shadow, not the raw weights.

        Evaluation/inference load the EMA shadow, so inflating the raw weights gave a 3D
        start that differed from the evaluated 2D model. Fake checkpoint: EMA = raw + 0.5
        on every float tensor, so the two sources are trivially distinguishable.
        """
        torch.manual_seed(0)
        try:
            m2d = build_diffusion_2d(_unet_cfg())
            m3d = build_diffusion_3d(_unet_cfg())
        except ImportError as exc:  # monai-generative missing at build time
            self.skipTest(f"monai-generative not available: {exc}")

        raw_sd = {k: v.detach().clone() for k, v in m2d.state_dict().items()}
        ema_sd = {k: (v + 0.5) if v.is_floating_point() else v.clone() for k, v in raw_sd.items()}
        float_keys = [k for k, v in raw_sd.items() if v.is_floating_point()]
        self.assertTrue(float_keys)

        ckpt = {
            "model": raw_sd,
            "ema": {"decay": 0.999, "shadow": ema_sd},
            "config": {"prediction_type": "flow", "model": dict(_UNET_MODEL)},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.pt")
            torch.save(ckpt, path)
            state = load_checkpoint(path)

        # (a)+(b): with an EMA shadow present, it is the selected source.
        src_sd, src_name = _inflation_source(state)
        self.assertEqual(src_name, "EMA shadow")
        self.assertEqual(set(src_sd.keys()), set(ema_sd.keys()))
        for k, v in ema_sd.items():
            self.assertTrue(torch.equal(src_sd[k], v), f"{k}: selected source != EMA shadow")
        for k in float_keys:
            self.assertFalse(torch.equal(src_sd[k], raw_sd[k]), f"{k}: selected source == raw model")

        # (c): without the "ema" key, fall back to the raw weights.
        state_no_ema = {k: v for k, v in state.items() if k != "ema"}
        fallback_sd, fallback_name = _inflation_source(state_no_ema)
        self.assertEqual(fallback_name, "raw model")
        self.assertEqual(set(fallback_sd.keys()), set(raw_sd.keys()))
        for k, v in raw_sd.items():
            self.assertTrue(torch.equal(fallback_sd[k], v), f"{k}: fallback != raw model")

        # (d): after inflating the selected source, the centre depth slice of an inflated
        # conv holds the EMA tensor (same centre convention as inflate_conv2d_to_3d).
        inflate_and_load(m3d, src_sd)
        sd3d = m3d.state_dict()
        conv_keys = [k for k, v in ema_sd.items()
                     if v.ndim == 4 and k in sd3d and sd3d[k].ndim == 5]
        self.assertTrue(conv_keys, "no conv was inflated into the 3D UNet")
        key = conv_keys[0]
        w3d = sd3d[key]
        center = w3d.shape[2] // 2
        self.assertTrue(torch.equal(w3d[:, :, center], ema_sd[key]),
                        f"{key}: centre slice != EMA tensor")
        self.assertFalse(torch.equal(w3d[:, :, center], raw_sd[key]),
                         f"{key}: centre slice == raw tensor (EMA not used)")


if __name__ == "__main__":
    unittest.main()
