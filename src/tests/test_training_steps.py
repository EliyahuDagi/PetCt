import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np

from src.training.train import train_ae2d, train_ae3d, train_diff2d, train_ft3d
from src.training.data import sample_volumes
from src.training.models.autoencoder2d import build_autoencoder_2d
from src.training.models.autoencoder3d import build_autoencoder_3d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.utils.sampling import DiffusionSchedule


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

    def test_diff2d_conditioned_loss(self):
        # NAC->AC conditioning: model input is concat(noisy_AC, NAC) -> 2*latent_channels.
        config = {
            "latent_channels": 4,
            "model": {
                "in_channels": 8,
                "out_channels": 4,
                "num_channels": [16, 32, 64],
                "attention_levels": [False, True, True],
                "num_res_blocks": 1,
            },
        }
        model = build_diffusion_2d(config)
        schedule = DiffusionSchedule()
        ac_lat = torch.randn(2, 4, 32, 32)
        nac_lat = torch.randn(2, 4, 32, 32)
        loss, x_t, t, pred, noise = train_diff2d.diffusion_loss(model, schedule, ac_lat, nac_lat)
        self.assertEqual(pred.shape, ac_lat.shape)
        self.assertTrue(torch.isfinite(loss))

    def test_ae3d_train_eval(self):
        config = {
            "latent_channels": 4,
            "model": {
                "in_channels": 1,
                "out_channels": 1,
                "block_out_channels": [16, 32, 64],
                "num_res_blocks": 1,
            },
        }
        model = build_autoencoder_3d(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        batch = torch.randn(1, 1, 16, 16, 16)
        train_metrics = train_ae3d.train_step(model, batch, optimizer)
        eval_metrics = train_ae3d.eval_step(model, batch)
        self.assertIn("recon_l1", train_metrics)
        self.assertIn("loss", eval_metrics)

    def test_sample_volumes_train_val_disjoint_shape(self):
        rng = np.random.RandomState(0)
        vols = [torch.rand(20, 24, 24), torch.rand(20, 24, 24)]  # pooled NAC + AC
        train_batch = sample_volumes(vols, batch_size=2, size=16, phase="train", val_fraction=0.25, rng=rng)
        val_batch = sample_volumes(vols, batch_size=2, size=16, phase="val", val_fraction=0.25, rng=rng)
        self.assertEqual(tuple(train_batch.shape), (2, 1, 16, 16, 16))
        self.assertEqual(tuple(val_batch.shape), (2, 1, 16, 16, 16))

    def test_ft3d_conditioned_loss(self):
        config = {
            "latent_channels": 4,
            "model": {
                "in_channels": 8,
                "out_channels": 4,
                "num_channels": [16, 32, 48],
                "attention_levels": [False, False, True],
                "num_res_blocks": 1,
            },
        }
        model = build_diffusion_3d(config)
        schedule = DiffusionSchedule()
        ac_lat = torch.randn(1, 4, 16, 16, 16)
        nac_lat = torch.randn(1, 4, 16, 16, 16)
        loss, x_t, t, pred, noise = train_ft3d.diffusion_loss(model, schedule, ac_lat, nac_lat)
        self.assertEqual(pred.shape, ac_lat.shape)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
