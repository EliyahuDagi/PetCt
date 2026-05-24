import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.training.models.autoencoder2d import build_autoencoder_2d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d


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


if __name__ == "__main__":
    unittest.main()
