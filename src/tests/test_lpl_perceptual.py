"""Tests for the LPL (latent decoder-feature) perceptual backend and the flow
loss-shaping knobs (U-shaped tau sampling, RFPP (1-tau) weighting).

No MONAI / no downloads: a minimal fake AE mimics the two attributes LPL relies on
(``post_quant_conv`` and ``decoder.blocks``), so these run on CPU everywhere.
"""

import unittest

import torch
import torch.nn as nn

from src.training.train.train_diff2d import (
    flow_ac_estimate_latent,
    flow_loss,
    perceptual_term,
    sample_flow_timesteps,
    unscale_latent,
)
from src.training.utils.perceptual import (
    LatentDecoderPerceptualLoss,
    build_perceptual_loss,
)
from src.training.utils.sampling import DiffusionSchedule


class _FakeUpsample(nn.Module):
    """Named so LPL's auto-tap logic finds it (matches on the class name)."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class _FakeDecoder(nn.Module):
    """blocks: Conv | Res | Res | Upsample | Res | Res | GroupNorm | Conv (8 blocks).

    Mirrors the MONAI AutoencoderKL decoder shape: one flat ModuleList, each called
    as ``blk(h)``, with the resolution-doubling Upsample near the end.
    """

    def __init__(self, ch=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv3d(ch, ch, 3, padding=1),   # 0
            nn.Conv3d(ch, ch, 3, padding=1),   # 1
            nn.Conv3d(ch, ch, 3, padding=1),   # 2
            _FakeUpsample(ch),                 # 3  <- last (only) Upsample
            nn.Conv3d(ch, ch, 3, padding=1),   # 4
            nn.Conv3d(ch, ch, 3, padding=1),   # 5
            nn.GroupNorm(1, ch),               # 6
            nn.Conv3d(ch, 1, 3, padding=1),    # 7
        ])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class _FakeAE(nn.Module):
    def __init__(self, latent_ch=4, ch=4):
        super().__init__()
        self.post_quant_conv = nn.Conv3d(latent_ch, ch, 1)
        self.decoder = _FakeDecoder(ch)

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))


def _frozen_ae():
    ae = _FakeAE()
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


class TestLplBackend(unittest.TestCase):
    def setUp(self):
        self.ae = _frozen_ae()
        self.lpl = LatentDecoderPerceptualLoss(self.ae)
        self.z = torch.randn(2, 4, 4, 4, 4)

    def test_auto_taps_stop_before_last_upsample(self):
        # The whole point of LPL here is skipping the full-resolution tail: no tap may
        # be at or beyond the last Upsample (index 3 in the fake decoder).
        self.assertTrue(self.lpl.taps, "expected at least one tap")
        self.assertLess(max(self.lpl.taps), 3)
        self.assertNotIn(0, self.lpl.taps, "block 0 is a near-linear map; skip it")

    def test_registers_no_parameters(self):
        # The borrowed AE must NOT join this module's parameter tree, or the optimizer
        # / EMA / checkpoint would pick up the frozen AE weights.
        self.assertEqual(len(list(self.lpl.parameters())), 0)
        self.assertEqual(len(list(self.lpl.buffers())), 0)

    def test_consumes_latents_flag(self):
        self.assertTrue(LatentDecoderPerceptualLoss.consumes_latents)

    def test_identical_latents_zero(self):
        self.assertLess(float(self.lpl(self.z, self.z)), 1e-6)

    def test_different_latents_positive(self):
        self.assertGreater(float(self.lpl(self.z + 0.5, self.z)), 0.0)

    def test_scalar_output_5d(self):
        out = self.lpl(self.z + 0.1, self.z)
        self.assertEqual(out.dim(), 0)

    def test_grad_reaches_prediction_not_ae(self):
        v = torch.randn_like(self.z).requires_grad_(True)
        loss = self.lpl(self.z + 0.3 * v, self.z)
        loss.backward()
        self.assertIsNotNone(v.grad)
        self.assertGreater(float(v.grad.abs().sum()), 0.0)
        for p in self.ae.parameters():
            self.assertIsNone(p.grad, "frozen AE must not accumulate grads")

    def test_target_branch_is_detached(self):
        # A grad-carrying TARGET must not propagate: only the prediction is optimized.
        # With a non-grad prediction the whole loss must therefore have NO graph at all.
        t = torch.randn_like(self.z).requires_grad_(True)
        loss = self.lpl(self.z, t)
        self.assertFalse(loss.requires_grad, "target branch leaked into the graph")
        # Sanity: a grad-carrying PREDICTION does build a graph.
        p = torch.randn_like(self.z).requires_grad_(True)
        self.assertTrue(self.lpl(p, self.z).requires_grad)

    def test_early_stop_skips_tail_blocks(self):
        # Blocks after the deepest tap must never be executed.
        seen = []
        for i, blk in enumerate(self.ae.decoder.blocks):
            blk.register_forward_hook(lambda m, inp, out, i=i: seen.append(i))
        self.lpl(self.z + 0.1, self.z)
        self.assertEqual(max(seen), self.lpl.last_tap)
        for i in range(self.lpl.last_tap + 1, len(self.ae.decoder.blocks)):
            self.assertNotIn(i, seen)

    def test_explicit_taps_and_weights(self):
        lpl = LatentDecoderPerceptualLoss(self.ae, taps=[1, 2], tap_weights=[1.0, 0.5])
        self.assertEqual(lpl.taps, [1, 2])
        self.assertGreater(float(lpl(self.z + 0.2, self.z)), 0.0)

    def test_bad_taps_raise(self):
        with self.assertRaises(ValueError):
            LatentDecoderPerceptualLoss(self.ae, taps=[99])
        with self.assertRaises(ValueError):
            LatentDecoderPerceptualLoss(self.ae, taps=[1, 2], tap_weights=[1.0])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.lpl(self.z, torch.randn(2, 4, 2, 2, 2))

    def test_factory_raises_instead_of_silent_none(self):
        # A silent None would masquerade as the perceptual arm of an A/B while actually
        # training the plain-MSE control -- LPL needs no weights, so it must fail loudly.
        with self.assertRaises(Exception):
            build_perceptual_loss("lpl", ae=object())

    def test_factory_ignores_inapplicable_kwargs(self):
        # The train scripts pass weights_path/slices_per_plane for the VGG path.
        loss = build_perceptual_loss("lpl", ae=self.ae, weights_path=None, taps=None)
        self.assertIsInstance(loss, LatentDecoderPerceptualLoss)


class TestFlowPerceptualRouting(unittest.TestCase):
    """perceptual_term must hand latents to LPL and stay UNGATED in flow mode."""

    def setUp(self):
        self.ae = _frozen_ae()
        self.lpl = LatentDecoderPerceptualLoss(self.ae)
        self.sched = DiffusionSchedule(num_train_timesteps=100)
        self.ac_lat = torch.randn(2, 4, 4, 4, 4)
        self.nac_lat = torch.randn(2, 4, 4, 4, 4)

    def _xt_pred(self, t_val):
        t = torch.full((2,), t_val, dtype=torch.long)
        tau = self.sched._broadcast(t.float() / 99.0, self.ac_lat)
        x_t = (1 - tau) * self.ac_lat + tau * self.nac_lat
        pred = (self.nac_lat - self.ac_lat).clone().requires_grad_(True)
        return x_t, t, pred

    def test_flow_is_ungated_at_max_timestep(self):
        # tau=1 (pure NAC) is exactly where the flow term matters most. The epsilon
        # path returns None there (FIX B gate); flow must NOT.
        x_t, t, pred = self._xt_pred(99)
        term, val = perceptual_term(
            self.lpl, self.ae, self.sched, x_t, t, pred, None, 0.1,
            is_flow=True, ac_lat=self.ac_lat,
        )
        self.assertIsNotNone(term, "flow perceptual must be active at tau=1")

    def test_epsilon_still_gated_at_max_timestep(self):
        x_t, t, pred = self._xt_pred(99)
        term, val = perceptual_term(
            self.lpl, self.ae, self.sched, x_t, t, pred, None, 0.1,
            is_flow=False, perceptual_active_frac=0.7, ac_lat=self.ac_lat,
        )
        self.assertIsNone(term, "epsilon path must gate out high-noise timesteps")

    def test_oracle_velocity_gives_zero_perceptual(self):
        # With the TRUE velocity the AC estimate is exact at every tau, so the term
        # must vanish -- at tau=1 just as much as at tau=0.5.
        for t_val in (0, 50, 99):
            x_t, t, pred = self._xt_pred(t_val)
            term, val = perceptual_term(
                self.lpl, self.ae, self.sched, x_t, t, pred, None, 0.1,
                is_flow=True, ac_lat=self.ac_lat,
            )
            self.assertLess(val, 1e-6, f"expected ~0 at t={t_val}, got {val}")

    def test_latent_backend_requires_ac_lat(self):
        x_t, t, pred = self._xt_pred(50)
        with self.assertRaises(ValueError):
            perceptual_term(self.lpl, self.ae, self.sched, x_t, t, pred, None, 0.1,
                            is_flow=True, ac_lat=None)

    def test_disabled_weight_is_noop(self):
        x_t, t, pred = self._xt_pred(50)
        self.assertEqual(
            perceptual_term(self.lpl, self.ae, self.sched, x_t, t, pred, None, 0.0,
                            is_flow=True, ac_lat=self.ac_lat),
            (None, 0.0),
        )

    def test_gradient_scales_with_tau(self):
        """d(x_tau - tau*v)/dv = -tau, so the term's gradient must grow with tau.

        This is the quantitative reason the flow path is ungated: a low-t gate would
        keep exactly the samples whose gradient is weakest.
        """
        mags = []
        for t_val in (10, 50, 90):
            x_t, t, pred = self._xt_pred(t_val)
            # Perturb away from the oracle so there is a non-zero gradient.
            pred = (pred + 0.5).detach().requires_grad_(True)
            term, _ = perceptual_term(
                self.lpl, self.ae, self.sched, x_t, t, pred, None, 1.0,
                is_flow=True, ac_lat=self.ac_lat,
            )
            term.backward()
            mags.append(float(pred.grad.abs().mean()))
        self.assertLess(mags[0], mags[1])
        self.assertLess(mags[1], mags[2])


class TestFlowTimestepSampling(unittest.TestCase):
    def setUp(self):
        self.sched = DiffusionSchedule(num_train_timesteps=1000)

    def test_uniform_is_the_default_and_in_range(self):
        t = sample_flow_timesteps(self.sched, 4096, torch.device("cpu"))
        self.assertEqual(t.dtype, torch.long)
        self.assertGreaterEqual(int(t.min()), 0)
        self.assertLessEqual(int(t.max()), 999)

    def test_ushaped_in_range_and_integer(self):
        t = sample_flow_timesteps(self.sched, 4096, torch.device("cpu"), dist="ushaped", a=4.0)
        self.assertEqual(t.dtype, torch.long)
        self.assertGreaterEqual(int(t.min()), 0)
        self.assertLessEqual(int(t.max()), 999)

    def test_ushaped_is_denser_at_both_ends(self):
        n = 40000
        t = sample_flow_timesteps(self.sched, n, torch.device("cpu"), dist="ushaped", a=4.0).float()
        # Compare the two outer fifths against the middle fifth.
        low = float((t < 200).float().mean())
        mid = float(((t >= 400) & (t < 600)).float().mean())
        high = float((t >= 800).float().mean())
        self.assertGreater(low, mid, "U-shape must over-sample the low end")
        self.assertGreater(high, mid, "U-shape must over-sample the high end")
        # Uniform would give 0.2 in each fifth.
        self.assertLess(mid, 0.2)

    def test_zero_sharpness_degenerates_to_uniform(self):
        t = sample_flow_timesteps(self.sched, 256, torch.device("cpu"), dist="ushaped", a=0.0)
        self.assertGreaterEqual(int(t.min()), 0)

    def test_unknown_dist_raises(self):
        with self.assertRaises(ValueError):
            sample_flow_timesteps(self.sched, 8, torch.device("cpu"), dist="bogus")


class _ConstVelocityModel(torch.nn.Module):
    """Returns the oracle velocity (NAC-AC) plus a fixed offset."""

    def __init__(self, vel, offset=0.0):
        super().__init__()
        self.vel = vel
        self.offset = offset

    def forward(self, x, t):
        return self.vel + self.offset


class TestFlowLossWeighting(unittest.TestCase):
    def setUp(self):
        self.sched = DiffusionSchedule(num_train_timesteps=100)
        self.ac = torch.zeros(64, 2, 4, 4)
        self.nac = torch.ones(64, 2, 4, 4)
        self.model = _ConstVelocityModel(self.nac - self.ac, offset=1.0)  # error = 1 everywhere

    def test_default_is_unweighted_mse(self):
        torch.manual_seed(0)
        loss, *_ = flow_loss(self.model, self.sched, self.ac, self.nac)
        # Constant per-element error of 1.0 -> plain MSE is exactly 1.0.
        self.assertAlmostEqual(float(loss), 1.0, places=5)

    def test_rfpp_weighting_downweights_high_tau(self):
        torch.manual_seed(0)
        loss, *_ = flow_loss(self.model, self.sched, self.ac, self.nac, loss_weighting="rfpp")
        # Weight is (1-tau) with tau ~ U[0,1] -> expectation ~0.5 of the unweighted 1.0.
        self.assertLess(float(loss), 1.0)
        self.assertGreater(float(loss), 0.2)

    def test_unknown_weighting_raises(self):
        with self.assertRaises(ValueError):
            flow_loss(self.model, self.sched, self.ac, self.nac, loss_weighting="bogus")

    def test_ushaped_tau_does_not_break_the_bridge(self):
        # With the ORACLE velocity the loss must stay ~0 regardless of tau sampling.
        oracle = _ConstVelocityModel(self.nac - self.ac, offset=0.0)
        loss, *_ = flow_loss(oracle, self.sched, self.ac, self.nac, tau_dist="ushaped")
        self.assertLess(float(loss), 1e-10)


class TestFlowAcEstimateLatent(unittest.TestCase):
    def test_oracle_recovers_ac_at_every_tau(self):
        sched = DiffusionSchedule(num_train_timesteps=100)
        ac = torch.randn(3, 2, 4, 4)
        nac = torch.randn(3, 2, 4, 4)
        v = nac - ac
        for t_val in (0, 1, 50, 99):
            t = torch.full((3,), t_val, dtype=torch.long)
            tau = sched._broadcast(t.float() / 99.0, ac)
            x_t = (1 - tau) * ac + tau * nac
            est = flow_ac_estimate_latent(sched, x_t, t, v)
            self.assertTrue(torch.allclose(est, ac, atol=1e-5), f"failed at t={t_val}")

    def test_unscale_latent_is_noop_at_one(self):
        z = torch.randn(2, 3)
        self.assertTrue(torch.equal(unscale_latent(z, 1.0), z))
        self.assertTrue(torch.equal(unscale_latent(z, 0.0), z))
        self.assertTrue(torch.allclose(unscale_latent(z, 2.0), z / 2.0))


if __name__ == "__main__":
    unittest.main()
