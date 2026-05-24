import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.training.train import train_ae2d, train_diff2d, train_ft3d
from src.training.models.autoencoder2d import build_autoencoder_2d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d


class TestTrainingSteps(unittest.TestCase):
    def test_ae2d_train_eval(self):
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
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        batch = torch.randn(2, 1, 32, 32)
        train_metrics = train_ae2d.train_step(model, batch, optimizer)
        eval_metrics = train_ae2d.eval_step(model, batch)
        self.assertIn("loss", train_metrics)
        self.assertIn("loss", eval_metrics)

    def test_diff2d_train_eval(self):
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
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        latents = torch.randn(2, 4, 32, 32)
        train_metrics = train_diff2d.train_step(model, latents, optimizer)
        eval_metrics = train_diff2d.eval_step(model, latents)
        self.assertIn("loss", train_metrics)
        self.assertIn("loss", eval_metrics)

    def test_ft3d_train_eval(self):
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
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        latents = torch.randn(1, 4, 16, 16, 16)
        train_metrics = train_ft3d.train_step(model, latents, optimizer)
        eval_metrics = train_ft3d.eval_step(model, latents)
        self.assertIn("loss", train_metrics)
        self.assertIn("loss", eval_metrics)


if __name__ == "__main__":
    unittest.main()
