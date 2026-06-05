import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.training.utils.sampling import DiffusionSchedule
from src.training.utils.schedule import EMA, build_lr_scheduler


class TestNoiseSchedule(unittest.TestCase):
    def test_cosine_alphas_cumprod_monotonic(self):
        sched = DiffusionSchedule(schedule="cosine")
        acp = sched.alphas_cumprod
        self.assertEqual(acp.shape[0], 1000)
        # alphas_cumprod must be strictly decreasing and within (0, 1].
        self.assertTrue(torch.all(acp[1:] <= acp[:-1] + 1e-6))
        self.assertGreater(float(acp[0]), float(acp[-1]))
        self.assertGreater(float(acp[-1]), 0.0)  # not zero unless ZTSNR is requested
        self.assertLessEqual(float(acp[0]), 1.0)

    def test_linear_still_supported(self):
        sched = DiffusionSchedule(schedule="linear")
        self.assertEqual(sched.alphas_cumprod.shape[0], 1000)

    def test_zero_terminal_snr_drives_terminal_acp_to_zero(self):
        sched = DiffusionSchedule(schedule="cosine", rescale_zero_terminal_snr=True)
        self.assertAlmostEqual(float(sched.alphas_cumprod[-1]), 0.0, places=5)

    def test_q_sample_shape_and_finite(self):
        sched = DiffusionSchedule(schedule="cosine")
        x0 = torch.randn(3, 4, 8, 8)
        t = torch.tensor([0, 500, 999])
        noise = torch.randn_like(x0)
        x_t = sched.q_sample(x0, t, noise)
        self.assertEqual(x_t.shape, x0.shape)
        self.assertTrue(torch.isfinite(x_t).all())

    def test_min_snr_weights(self):
        sched = DiffusionSchedule(schedule="cosine")
        t = torch.tensor([0, 999])
        w = sched.min_snr_weights(t, gamma=5.0)
        # Low-noise step (t=0, high SNR) is down-weighted below 1; weights positive.
        self.assertTrue(torch.all(w > 0))
        self.assertLess(float(w[0]), 1.0 + 1e-4)
        self.assertEqual(w.shape, t.shape)


class TestKarrasSampling(unittest.TestCase):
    def test_karras_spacing_runs_and_differs_from_linear(self):
        sched = DiffusionSchedule(schedule="cosine")
        lin = sched._timesteps_for_spacing(10, "linear")
        kar = sched._timesteps_for_spacing(10, "karras")
        self.assertEqual(len(lin), 10)
        self.assertEqual(len(kar), 10)
        self.assertNotEqual(lin, kar)  # Karras concentrates steps at low noise

    def test_ddim_sample_karras(self):
        sched = DiffusionSchedule(schedule="cosine")

        def model_fn(x, t):
            return torch.zeros_like(x)

        out = sched.ddim_sample(model_fn, (2, 4, 8, 8), device="cpu", num_steps=8, spacing="karras")
        self.assertEqual(out.shape, (2, 4, 8, 8))
        self.assertTrue(torch.isfinite(out).all())


class TestEMA(unittest.TestCase):
    def test_ema_tracks_and_swaps(self):
        model = torch.nn.Linear(4, 4)
        ema = EMA(model, decay=0.5)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)  # move live weights away from the EMA snapshot
        ema.update(model)
        # Inside the context the model holds EMA weights; outside it is restored.
        live = model.weight.detach().clone()
        with ema.average_parameters(model):
            swapped = model.weight.detach().clone()
        restored = model.weight.detach().clone()
        self.assertFalse(torch.allclose(swapped, live))
        self.assertTrue(torch.allclose(restored, live))

    def test_ema_state_roundtrip(self):
        model = torch.nn.Linear(4, 4)
        ema = EMA(model, decay=0.9)
        state = ema.state_dict()
        ema2 = EMA(model, decay=0.1)
        ema2.load_state_dict(state)
        self.assertEqual(ema2.decay, 0.9)


class TestLRScheduler(unittest.TestCase):
    def test_none_when_unconfigured(self):
        opt = torch.optim.Adam(torch.nn.Linear(2, 2).parameters(), lr=1e-3)
        self.assertIsNone(build_lr_scheduler(opt, {}))

    def test_warmup_then_decay(self):
        opt = torch.optim.Adam(torch.nn.Linear(2, 2).parameters(), lr=1e-3)
        sched = build_lr_scheduler(opt, {"lr_total_steps": 100, "lr_warmup_steps": 10, "lr_min_ratio": 0.0})
        self.assertIsNotNone(sched)
        lr_start = opt.param_groups[0]["lr"]
        for _ in range(5):
            opt.step()
            sched.step()
        lr_mid_warmup = opt.param_groups[0]["lr"]
        self.assertGreater(lr_mid_warmup, lr_start)  # still ramping up
        for _ in range(95):
            opt.step()
            sched.step()
        lr_end = opt.param_groups[0]["lr"]
        self.assertLess(lr_end, lr_mid_warmup)  # decayed by the end


if __name__ == "__main__":
    unittest.main()
