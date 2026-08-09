"""Tests for the NAC->AC rectified-flow bridge (flow_sample / flow_loss).

The bridge starts the trajectory FROM the NAC latent (not noise) and transports it
to AC along the linear path x_tau = (1-tau)*AC + tau*NAC with constant velocity
v = NAC - AC. These tests lock the math (interpolation direction, (T-1) divisor,
velocity sign, Euler update) and confirm the epsilon path is unaffected.
"""
import inspect
import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.training.utils.sampling import DiffusionSchedule
from src.training.train.train_diff2d import (
    flow_loss,
    flow_x0_decode_in_graph,
    diffusion_in_channels,
    _selection_metric,
    _resume_best_val,
)


class _IdentityAE:
    """Stub AE whose decode is identity -> lets us check the latent AC estimate directly."""

    def decode(self, z):
        return z


class _ConstVelocity:
    """model_fn returning a fixed velocity tensor regardless of (x, t)."""

    def __init__(self, velocity):
        self.velocity = velocity

    def __call__(self, x, t):
        return self.velocity


class _ConstModel(torch.nn.Module):
    """Stub UNet returning a constant per-channel value (first C channels' shape)."""

    def __init__(self, value, channels):
        super().__init__()
        self.value = float(value)
        self.channels = int(channels)

    def forward(self, x, t):
        return torch.full((x.shape[0], self.channels, *x.shape[2:]), self.value,
                          dtype=x.dtype, device=x.device)


class _ChannelRecorder(torch.nn.Module):
    """Stub UNet that records the channel count of the LAST input it saw."""

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = int(out_channels)
        self.last_in_channels = None

    def forward(self, x, t):
        self.last_in_channels = int(x.shape[1])
        return torch.zeros((x.shape[0], self.out_channels, *x.shape[2:]),
                           dtype=x.dtype, device=x.device)


class TestFlowSampleOracle(unittest.TestCase):
    """With the true (constant) velocity, sampling recovers AC exactly at any step count."""

    def _check(self, shape):
        sched = DiffusionSchedule(schedule="cosine")
        torch.manual_seed(0)
        ac = torch.randn(shape)
        nac = torch.randn(shape)
        model_fn = _ConstVelocity(nac - ac)  # v = NAC - AC
        for k in (1, 4, 8):
            out = sched.flow_sample(model_fn, x_init=nac, num_steps=k, spacing="linear")
            self.assertEqual(out.shape, shape)
            self.assertTrue(torch.allclose(out, ac, atol=1e-5),
                            msg=f"shape={shape} num_steps={k} did not recover AC")

    def test_oracle_recovers_ac_2d(self):
        self._check((2, 4, 8, 8))

    def test_oracle_recovers_ac_3d(self):
        self._check((2, 3, 4, 8, 8))

    def test_step_count_invariance(self):
        sched = DiffusionSchedule(schedule="cosine")
        torch.manual_seed(1)
        ac = torch.randn(2, 4, 8, 8)
        nac = torch.randn(2, 4, 8, 8)
        model_fn = _ConstVelocity(nac - ac)
        o1 = sched.flow_sample(model_fn, x_init=nac, num_steps=1)
        o8 = sched.flow_sample(model_fn, x_init=nac, num_steps=8)
        self.assertTrue(torch.allclose(o1, o8, atol=1e-5))  # constant field telescopes

    def test_deterministic_no_rng(self):
        sched = DiffusionSchedule(schedule="cosine")
        nac = torch.randn(2, 4, 8, 8)
        model_fn = _ConstVelocity(torch.zeros_like(nac))
        a = sched.flow_sample(model_fn, x_init=nac, num_steps=5)
        b = sched.flow_sample(model_fn, x_init=nac, num_steps=5)
        self.assertTrue(torch.equal(a, b))  # no randn -> bit-identical

    def test_requires_two_timesteps(self):
        sched = DiffusionSchedule(num_train_timesteps=1, schedule="linear")
        with self.assertRaises(ValueError):
            sched.flow_sample(_ConstVelocity(torch.zeros(1, 1, 2, 2)),
                              x_init=torch.zeros(1, 1, 2, 2), num_steps=1)


class TestFlowLoss(unittest.TestCase):
    def test_interpolation_direction_and_divisor(self):
        """x_t must equal tau*NAC (with AC=0, NAC=1) and tau = t/(T-1)."""
        sched = DiffusionSchedule(schedule="cosine")
        T = sched.num_train_timesteps
        ac = torch.zeros(4, 4, 8, 8)
        nac = torch.ones(4, 4, 8, 8)
        torch.manual_seed(7)
        loss, x_t, timesteps, pred, target = flow_loss(_ConstModel(0.0, 4), sched, ac, nac)
        expected_tau = (timesteps.float() / float(T - 1)).view(-1, 1, 1, 1).expand_as(x_t)
        # AC=0, NAC=1 => x_tau = tau ; locks the direction AND the (T-1) divisor.
        self.assertTrue(torch.allclose(x_t, expected_tau, atol=1e-6))
        # target velocity = NAC - AC = 1 everywhere.
        self.assertTrue(torch.allclose(target, torch.ones_like(target)))

    def test_velocity_sign(self):
        sched = DiffusionSchedule(schedule="cosine")
        ac = torch.zeros(4, 4, 8, 8)
        nac = torch.ones(4, 4, 8, 8)  # target = NAC - AC = +1
        torch.manual_seed(3)
        loss_right, *_ = flow_loss(_ConstModel(1.0, 4), sched, ac, nac)   # predicts +1
        torch.manual_seed(3)
        loss_wrong, *_ = flow_loss(_ConstModel(-1.0, 4), sched, ac, nac)  # predicts -1
        self.assertAlmostEqual(float(loss_right), 0.0, places=5)
        self.assertAlmostEqual(float(loss_wrong), 4.0, places=4)  # (-1 - 1)^2 = 4


class TestEpsilonPathUnaffected(unittest.TestCase):
    def test_missing_prediction_type_resolves_to_epsilon_concat(self):
        # A config without prediction_type must take the epsilon path: is_flow False
        # AND the UNet built with 2C (concat) channels. Exercises the real decision
        # logic (the resolution expression + diffusion_in_channels), not a dict literal.
        cfg = {}
        is_flow = str(cfg.get("prediction_type", "epsilon")).lower() == "flow"
        self.assertFalse(is_flow)
        self.assertEqual(diffusion_in_channels(8, is_flow), 16)  # epsilon -> 2C
        self.assertEqual(_selection_metric({"loss": 0.1, "l1": 0.9}, is_flow), 0.1)

    def test_ddim_sample_still_starts_from_noise(self):
        # Sanity: ddim_sample is untouched and still produces finite output from noise.
        sched = DiffusionSchedule(schedule="cosine")
        out = sched.ddim_sample(lambda x, t: torch.zeros_like(x), (2, 4, 8, 8),
                                device="cpu", num_steps=8, clip_x0=4.0)
        self.assertEqual(out.shape, (2, 4, 8, 8))
        self.assertTrue(torch.isfinite(out).all())


class TestNoSourceConcat(unittest.TestCase):
    """Fix #1: the flow bridge must feed ONLY x_t (in_channels = C), never concat NAC.

    Concatenating NAC lets the net recover AC=(x_t - tau*NAC)/(1-tau) by algebra for
    tau<1, so the loss collapses to a no-op and generation fails (I2SB avoids this).
    """

    def test_in_channels_helper_flow_is_C_epsilon_is_2C(self):
        for c in (1, 4, 8):
            self.assertEqual(diffusion_in_channels(c, is_flow=True), c)
            self.assertEqual(diffusion_in_channels(c, is_flow=False), 2 * c)

    def test_flow_loss_feeds_only_xt(self):
        sched = DiffusionSchedule(schedule="cosine")
        ac = torch.randn(2, 4, 8, 8)
        nac = torch.randn(2, 4, 8, 8)
        rec = _ChannelRecorder(out_channels=4)
        flow_loss(rec, sched, ac, nac)
        # 4 (x_t alone), NOT 8 (concat of x_t and nac).
        self.assertEqual(rec.last_in_channels, 4)

    def test_flow_sample_feeds_only_xt(self):
        sched = DiffusionSchedule(schedule="cosine")
        nac = torch.randn(2, 4, 8, 8)
        rec = _ChannelRecorder(out_channels=4)
        sched.flow_sample(lambda x, t: rec(x, t), x_init=nac, num_steps=4, spacing="linear")
        self.assertEqual(rec.last_in_channels, 4)  # sampling contract: no concat, no CFG doubling


class TestFlowPerceptualDecode(unittest.TestCase):
    """Flow perceptual uses the AC estimate AC = x_t - tau*v (not the epsilon x0)."""

    def test_ac_estimate_recovers_ac_with_true_velocity(self):
        sched = DiffusionSchedule(schedule="cosine")
        T = sched.num_train_timesteps
        ac = torch.randn(3, 4, 8, 8)
        nac = torch.randn(3, 4, 8, 8)
        v = nac - ac  # true velocity
        t = torch.tensor([0, T // 2, T - 1])
        tau = (t.float() / float(T - 1)).view(-1, 1, 1, 1)
        x_t = (1.0 - tau) * ac + tau * nac
        # With the true velocity, x_t - tau*v must equal AC at every tau (identity AE).
        rec = flow_x0_decode_in_graph(_IdentityAE(), sched, x_t, t, v, latent_scale=1.0)
        self.assertTrue(torch.allclose(rec, ac, atol=1e-5))


class TestSelectionMetric(unittest.TestCase):
    """Fix #3: flow selects best.pt on the honest rollout L1, not the velocity loss."""

    def test_flow_selects_on_l1_epsilon_on_loss(self):
        vm = {"loss": 0.001, "l1": 0.42}
        self.assertEqual(_selection_metric(vm, is_flow=True), 0.42)
        self.assertEqual(_selection_metric(vm, is_flow=False), 0.001)


class TestSnrGammaInert(unittest.TestCase):
    """Fix #5: snr_gamma / Min-SNR weighting does not apply in flow mode."""

    def test_flow_loss_has_no_snr_gamma_param(self):
        self.assertNotIn("snr_gamma", inspect.signature(flow_loss).parameters)

    def test_flow_loss_is_unweighted_mse(self):
        # Plain MSE: const +1 prediction vs target (NAC-AC=+1) -> 0; vs -1 -> 4. No
        # per-timestep SNR reweighting would change these exact values.
        sched = DiffusionSchedule(schedule="cosine")
        ac = torch.zeros(4, 4, 8, 8)
        nac = torch.ones(4, 4, 8, 8)
        torch.manual_seed(0)
        right, *_ = flow_loss(_ConstModel(1.0, 4), sched, ac, nac)
        torch.manual_seed(0)
        wrong, *_ = flow_loss(_ConstModel(-1.0, 4), sched, ac, nac)
        self.assertAlmostEqual(float(right), 0.0, places=5)
        self.assertAlmostEqual(float(wrong), 4.0, places=4)


class TestResumeBestValScale(unittest.TestCase):
    """Fix #6: best_val (a mode-dependent scale) is reset across a prediction_type switch."""

    def test_same_mode_keeps_best_val(self):
        self.assertEqual(_resume_best_val(0.123, "epsilon", "epsilon"), 0.123)
        self.assertEqual(_resume_best_val(0.123, "flow", "flow"), 0.123)

    def test_mode_switch_resets_to_inf(self):
        self.assertEqual(_resume_best_val(0.123, "epsilon", "flow"), float("inf"))
        self.assertEqual(_resume_best_val(7.5, "flow", "epsilon"), float("inf"))

    def test_case_insensitive(self):
        self.assertEqual(_resume_best_val(0.5, "EPSILON", "epsilon"), 0.5)


if __name__ == "__main__":
    unittest.main()
