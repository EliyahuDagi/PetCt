"""CPU tests for the pluggable perceptual loss (src.training.utils.perceptual).

These never download VGG weights: ``VGGPerceptualLoss`` is monkeypatched with a
tiny fake feature extractor so the loss math, 2D/3D dispatch, gradient flow and
graceful-disable paths are all exercised offline.
"""

import os
import sys
import unittest
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from src.training.utils import perceptual as P


def _fake_features():
    """A tiny stand-in for vgg16().features: a flat conv/relu Sequential whose
    indices line up with the block-end indices the loss slices on (0..15)."""
    layers = []
    in_ch = 3
    for i in range(16):
        if i in (0, 5, 10):  # conv at the start of each "block"
            layers.append(nn.Conv2d(in_ch, 4, 3, padding=1))
            in_ch = 4
        else:
            layers.append(nn.ReLU(inplace=False))
    return nn.Sequential(*layers)


def _build_fake(slices_per_plane=2):
    # Patch the loader so VGGPerceptualLoss uses the fake extractor (no download).
    orig = P._load_vgg16_features
    P._load_vgg16_features = lambda weights_path=None: _fake_features()
    try:
        return P.VGGPerceptualLoss(slices_per_plane=slices_per_plane)
    finally:
        P._load_vgg16_features = orig


class TestPerceptual(unittest.TestCase):
    def test_scalar_nonnegative_2d(self):
        loss = _build_fake()
        a = torch.rand(2, 1, 24, 24)
        b = torch.rand(2, 1, 24, 24)
        out = loss(a, b)
        self.assertEqual(out.dim(), 0)
        self.assertGreaterEqual(float(out), 0.0)

    def test_identical_inputs_zero(self):
        loss = _build_fake()
        a = torch.rand(2, 1, 24, 24)
        out = loss(a, a.clone())
        self.assertLess(float(out), 1e-5)

    def test_different_inputs_positive(self):
        loss = _build_fake()
        a = torch.rand(2, 1, 24, 24)
        b = torch.rand(2, 1, 24, 24) + 0.5
        self.assertGreater(float(loss(a, b)), 0.0)

    def test_3d_dispatch(self):
        loss = _build_fake()
        a = torch.rand(1, 1, 8, 24, 24)
        b = torch.rand(1, 1, 8, 24, 24)
        out = loss(a, b)
        self.assertEqual(out.dim(), 0)
        self.assertGreaterEqual(float(out), 0.0)
        # Identical 3D inputs -> ~0.
        self.assertLess(float(loss(a, a.clone())), 1e-5)

    def test_gradients_flow_to_input_2d(self):
        loss = _build_fake()
        a = torch.rand(2, 1, 24, 24, requires_grad=True)
        b = torch.rand(2, 1, 24, 24)
        loss(a, b).backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0.0)

    def test_gradients_flow_to_input_3d(self):
        loss = _build_fake()
        a = torch.rand(1, 1, 8, 24, 24, requires_grad=True)
        b = torch.rand(1, 1, 8, 24, 24)
        loss(a, b).backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0.0)

    def test_backbone_params_frozen(self):
        loss = _build_fake()
        self.assertFalse(any(p.requires_grad for p in loss.parameters()))
        # train() must not flip it out of eval.
        loss.train()
        self.assertFalse(loss.training)

    def test_factory_returns_none_when_unavailable(self):
        orig = P._load_vgg16_features
        P._load_vgg16_features = lambda weights_path=None: None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertIsNone(P.build_perceptual_loss("vgg"))
                self.assertIsNone(P.build_perceptual_loss("medical_sam"))
                self.assertIsNone(P.build_perceptual_loss("nope"))
        finally:
            P._load_vgg16_features = orig

    def test_fix_a_intensity_sensitivity(self):
        # Shared (target-derived) normalization makes an intensity-scaled/shifted pred
        # yield a clearly nonzero loss (the old per-sample independent normalization
        # was intensity-invariant and gave ~0 here).
        loss = _build_fake()
        target = torch.rand(2, 1, 24, 24) + 0.5
        self.assertGreater(float(loss(target * 0.5, target)), 1e-3)
        self.assertGreater(float(loss(target + 0.3, target)), 1e-3)
        # Identical inputs still ~0.
        self.assertLess(float(loss(target, target.clone())), 1e-5)

    def test_fix_c_channel_norm_constant_input_no_nan(self):
        # LPIPS-style per-block channel normalization must run for 2D and 3D on a
        # near-constant input (all-0.5) without NaN -- the 1e-10 eps guards the norm.
        loss = _build_fake()
        a2d = torch.full((1, 1, 24, 24), 0.5)
        out2d = loss(a2d, a2d.clone())
        self.assertFalse(torch.isnan(out2d).any())
        self.assertLess(float(out2d), 1e-4)
        a3d = torch.full((1, 1, 8, 24, 24), 0.5)
        out3d = loss(a3d, a3d.clone())
        self.assertFalse(torch.isnan(out3d).any())
        self.assertLess(float(out3d), 1e-4)

    def test_factory_builds_with_fake(self):
        orig = P._load_vgg16_features
        P._load_vgg16_features = lambda weights_path=None: _fake_features()
        try:
            loss = P.build_perceptual_loss("vgg", slices_per_plane=2)
            self.assertIsNotNone(loss)
            self.assertGreaterEqual(float(loss(torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16))), 0.0)
        finally:
            P._load_vgg16_features = orig


class TestPerceptualWiredIntoTrainStep(unittest.TestCase):
    """The in-graph x0 decode + perceptual term must push gradients to the UNet
    through the FROZEN AE decoder (the whole point of decoding in-graph)."""

    def _deps(self):
        try:
            from src.training.models.autoencoder2d import build_autoencoder_2d
            from src.training.models.diffusion2d import build_diffusion_2d
            from src.training.utils.sampling import DiffusionSchedule
            from src.training.train import train_diff2d
            return build_autoencoder_2d, build_diffusion_2d, DiffusionSchedule, train_diff2d
        except Exception:
            return None

    def test_grad_reaches_unet_through_frozen_ae(self):
        deps = self._deps()
        if deps is None:
            self.skipTest("monai-generative not available")
        build_ae, build_unet, Schedule, train_diff2d = deps

        ae = build_ae({"latent_channels": 2, "model": {
            "in_channels": 1, "out_channels": 1, "block_out_channels": [4, 8], "num_res_blocks": 1}})
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)

        unet = build_unet({"latent_channels": 2, "model": {
            "in_channels": 4, "out_channels": 2, "num_channels": [4, 8],
            "attention_levels": [False, False], "num_res_blocks": 1}})
        schedule = Schedule()

        # Probe the real latent spatial size from the AE so the UNet input matches.
        with torch.no_grad():
            z = ae.encode(torch.zeros(1, 1, 32, 32))
            z = z[0] if isinstance(z, (tuple, list)) else z
        h = w = z.shape[-1]
        ac_lat = torch.randn(1, 2, h, w)
        nac_lat = torch.randn(1, 2, h, w)
        ac_img = torch.rand(1, 1, 32, 32)

        perceptual = _build_fake(slices_per_plane=2)

        _, x_t, timesteps, pred, _ = train_diff2d.diffusion_loss(unet, schedule, ac_lat, nac_lat)
        # frac=1.0 -> gating never excludes (keep all t < T) so this gradient-flow
        # check is deterministic regardless of the randomly sampled timestep.
        pterm, pval = train_diff2d.perceptual_term(
            perceptual, ae, schedule, x_t, timesteps, pred, ac_img, weight=0.1,
            perceptual_active_frac=1.0)
        self.assertIsNotNone(pterm)
        self.assertGreaterEqual(pval, 0.0)
        pterm.backward()
        grads = [p.grad for p in unet.parameters() if p.grad is not None]
        self.assertTrue(grads, "perceptual term produced no UNet gradients")
        # AE stayed frozen: no AE param accumulated a grad.
        self.assertFalse(any(p.grad is not None for p in ae.parameters()))

    def test_fix_b_gates_high_noise_timesteps(self):
        # FIX B: high-noise (low-SNR) timesteps -> the x0 estimate is garbage, so the
        # perceptual term is gated OFF (returns (None, 0.0)); low-noise t -> active.
        deps = self._deps()
        if deps is None:
            self.skipTest("monai-generative not available")
        build_ae, build_unet, Schedule, train_diff2d = deps

        ae = build_ae({"latent_channels": 2, "model": {
            "in_channels": 1, "out_channels": 1, "block_out_channels": [4, 8], "num_res_blocks": 1}})
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        unet = build_unet({"latent_channels": 2, "model": {
            "in_channels": 4, "out_channels": 2, "num_channels": [4, 8],
            "attention_levels": [False, False], "num_res_blocks": 1}})
        schedule = Schedule()

        with torch.no_grad():
            z = ae.encode(torch.zeros(1, 1, 32, 32))
            z = z[0] if isinstance(z, (tuple, list)) else z
        h = w = z.shape[-1]
        ac_lat = torch.randn(2, 2, h, w)
        nac_lat = torch.randn(2, 2, h, w)
        ac_img = torch.rand(2, 1, 32, 32)
        perceptual = _build_fake(slices_per_plane=2)

        _, x_t, timesteps, pred, _ = train_diff2d.diffusion_loss(unet, schedule, ac_lat, nac_lat)

        T = schedule.num_train_timesteps
        # All-high timesteps (full noise) -> gated off.
        high = torch.full_like(timesteps, T - 1)
        pterm, pval = train_diff2d.perceptual_term(
            perceptual, ae, schedule, x_t, high, pred, ac_img, weight=0.1,
            perceptual_active_frac=0.7)
        self.assertIsNone(pterm)
        self.assertEqual(pval, 0.0)
        # All-low timesteps (high SNR) -> active, positive.
        low = torch.zeros_like(timesteps)
        pterm, pval = train_diff2d.perceptual_term(
            perceptual, ae, schedule, x_t, low, pred, ac_img, weight=0.1,
            perceptual_active_frac=0.7)
        self.assertIsNotNone(pterm)
        self.assertGreater(pval, 0.0)

    def test_disabled_weight_is_noop(self):
        deps = self._deps()
        if deps is None:
            self.skipTest("monai-generative not available")
        _, _, _, train_diff2d = deps
        # weight 0 -> (None, 0.0) regardless of perceptual object.
        pterm, pval = train_diff2d.perceptual_term(object(), None, None, None, None, None, None, weight=0.0)
        self.assertIsNone(pterm)
        self.assertEqual(pval, 0.0)
        # None perceptual -> also a no-op.
        pterm, pval = train_diff2d.perceptual_term(None, None, None, None, None, None, None, weight=0.1)
        self.assertIsNone(pterm)


if __name__ == "__main__":
    unittest.main()
