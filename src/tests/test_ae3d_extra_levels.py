"""CPU tests for the opt-in ``--extra_levels`` deepening of the 3D AE (B2 config).

Verifies the config helper pads every per-level list and that the resulting 3D AE
(a) loads via the existing strict=False inflation path with the fresh level left
at init, and (b) compresses a 128^3 input to a 16^3 latent (factor 8) -- the B2
target, with ft3d's latent grid unchanged. Channels are kept tiny for CPU.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    from src.training.train.train_ae3d import append_extra_levels
    from src.training.models.autoencoder2d import build_autoencoder_2d
    from src.training.models.autoencoder3d import build_autoencoder_3d, ae3d_encode
    from src.training.models.inflation import map_state_dict_2d_to_3d
    HAVE_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestExtraLevels(unittest.TestCase):
    def test_helper_pads_all_lists(self):
        cfg = {
            "block_out_channels": [8, 16, 32],
            "attention_levels": [False, False, False],
            "num_res_blocks": [1, 1, 1],
        }
        append_extra_levels(cfg, 1)
        self.assertEqual(len(cfg["block_out_channels"]), 4)
        self.assertEqual(cfg["block_out_channels"][-1], 64)  # last width doubled
        self.assertEqual(len(cfg["attention_levels"]), 4)
        self.assertEqual(len(cfg["num_res_blocks"]), 4)
        self.assertFalse(cfg["attention_levels"][-1])

    def test_helper_noop_for_zero(self):
        cfg = {"block_out_channels": [8, 16, 32]}
        before = list(cfg["block_out_channels"])
        append_extra_levels(cfg, 0)
        self.assertEqual(cfg["block_out_channels"], before)

    def test_extra_level_is_one_longer_and_inflates(self):
        # 2D AE (3 levels) -> 3D AE deepened to 4 levels. The 2D->3D inflation must
        # still succeed; the fresh 4th level has no 2D counterpart (left at init).
        base = {
            "latent_channels": 2,
            "model": {
                "in_channels": 1,
                "out_channels": 1,
                "block_out_channels": [4, 8, 16],
                "num_res_blocks": 1,
            },
        }
        ae2d = build_autoencoder_2d(base)

        cfg3d = {"latent_channels": 2, "model": dict(base["model"])}
        cfg3d["model"]["block_out_channels"] = list(base["model"]["block_out_channels"])
        append_extra_levels(cfg3d["model"], 1)
        self.assertEqual(
            len(cfg3d["model"]["block_out_channels"]),
            len(base["model"]["block_out_channels"]) + 1,
        )

        ae3d = build_autoencoder_3d(cfg3d)
        mapped, missing = map_state_dict_2d_to_3d(ae2d.state_dict(), ae3d.state_dict())
        ae3d.load_state_dict(mapped, strict=False)  # must not raise
        # Some 2D params have no shape-compatible 3D target (the extra level adds
        # parameters with no 2D source) -> mapping is partial, which is expected.
        self.assertLess(len(mapped), len(ae3d.state_dict()))

    def test_latent_grid_is_16_for_128_input(self):
        # 4 levels -> 3 downsamples -> /8. A 128^3 crop must yield a 16^3 latent.
        cfg3d = {
            "latent_channels": 2,
            "model": {
                "in_channels": 1,
                "out_channels": 1,
                "block_out_channels": [4, 8, 16],
                "num_res_blocks": 1,
            },
        }
        append_extra_levels(cfg3d["model"], 1)
        ae3d = build_autoencoder_3d(cfg3d)
        ae3d.eval()
        x = torch.zeros(1, 1, 128, 128, 128)
        z = ae3d_encode(ae3d, x)
        self.assertEqual(tuple(z.shape[2:]), (16, 16, 16))

    def test_latent_grid_factor4_without_extra_level(self):
        # Sanity: the unmodified 3-level AE compresses /4 (64^3 -> 16^3), so the
        # extra level is what keeps the B2 latent at 16^3 from a 128^3 input.
        cfg3d = {
            "latent_channels": 2,
            "model": {
                "in_channels": 1,
                "out_channels": 1,
                "block_out_channels": [4, 8, 16],
                "num_res_blocks": 1,
            },
        }
        ae3d = build_autoencoder_3d(cfg3d)
        ae3d.eval()
        z = ae3d_encode(ae3d, torch.zeros(1, 1, 64, 64, 64))
        self.assertEqual(tuple(z.shape[2:]), (16, 16, 16))


if __name__ == "__main__":
    unittest.main()
