import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.training.models.autoencoder2d import build_autoencoder_2d
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode, build_autoencoder_3d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.models.inflation import map_state_dict_2d_to_3d

AE_CONFIG = {
    "latent_channels": 4,
    "model": {
        "in_channels": 1,
        "out_channels": 1,
        "block_out_channels": [16, 32, 64],
        "num_res_blocks": 1,
    },
}


class TestTrainingModels(unittest.TestCase):
    def test_autoencoder_2d_shapes(self):
        config = {
            "latent_channels": 4,
            "model": {
                "in_channels": 1,
                "out_channels": 1,
                "block_out_channels": [16, 32, 64],
                "num_res_blocks": 1,
            },
        }
        model = build_autoencoder_2d(config)
        x = torch.randn(1, 1, 32, 32)
        recon, z_mu, z_sigma = model(x)
        self.assertEqual(tuple(recon.shape), tuple(x.shape))
        self.assertEqual(z_mu.shape[0], x.shape[0])
        self.assertEqual(z_sigma.shape[0], x.shape[0])

    def test_diffusion_2d_shapes(self):
        config = {
            "model": {
                "in_channels": 4,
                "out_channels": 4,
                "num_channels": [16, 32, 64],
                "attention_levels": [False, True, True],
                "num_res_blocks": 1,
            }
        }
        model = build_diffusion_2d(config)
        x = torch.randn(1, 4, 32, 32)
        t = torch.randint(0, 1000, (1,))
        y = model(x, t)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_diffusion_3d_shapes(self):
        config = {
            "model": {
                "in_channels": 4,
                "out_channels": 4,
                "num_channels": [16, 32, 64],
                "attention_levels": [False, True, True],
                "num_res_blocks": 1,
            }
        }
        model = build_diffusion_3d(config)
        x = torch.randn(1, 4, 16, 16, 16)
        t = torch.randint(0, 1000, (1,))
        y = model(x, t)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_autoencoder_3d_compresses_depth(self):
        ae = build_autoencoder_3d(AE_CONFIG)
        vol = torch.randn(1, 1, 16, 16, 16)
        z = ae3d_encode(ae, vol)
        recon = ae3d_decode(ae, z)
        # 3-level AE -> 4x downsample in ALL dims, including depth (16 -> 4).
        self.assertEqual(z.shape[1], 4)  # latent_channels
        self.assertEqual(tuple(z.shape[2:]), (4, 4, 4))
        self.assertEqual(tuple(recon.shape), tuple(vol.shape))

    def test_ae_2d_inflates_fully_into_3d(self):
        ae2d = build_autoencoder_2d(AE_CONFIG)
        ae3d = build_autoencoder_3d(AE_CONFIG)
        mapped, missing = map_state_dict_2d_to_3d(ae2d.state_dict(), ae3d.state_dict())
        # Same architecture differing only in spatial_dims -> every param maps.
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(mapped), len(ae3d.state_dict()))
        res = ae3d.load_state_dict(mapped, strict=False)
        self.assertEqual(len(res.missing_keys), 0)
        self.assertEqual(len(res.unexpected_keys), 0)


if __name__ == "__main__":
    unittest.main()
