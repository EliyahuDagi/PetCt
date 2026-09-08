"""CPU tests for the in-plane-only ("anisotropic") 3D mode.

The claim under test: a 3D network built with ``model.anisotropic: true`` and
warm-started by centre inflation of its 2D twin equals the 2D twin applied to
each slice, to float precision. Also covered: depth is kept through every
level, the slice-wise wrapper around the 2D autoencoder matches the 2D model
and lets gradients through its decoder, the sliding depth window stitches a
linear model exactly, and the surgery is idempotent and refuses 2D models.
Channels are tiny so everything runs on the CPU in seconds.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    from src.training.models.anisotropic import (
        SliceWiseGroupNorm,
        is_inplane_only,
        make_inplane_only,
    )
    from src.training.models.autoencoder2d import SliceWiseAutoencoder, build_autoencoder_2d
    from src.training.models.autoencoder3d import build_autoencoder_3d
    from src.training.models.diffusion2d import build_diffusion_2d
    from src.training.models.diffusion3d import build_diffusion_3d
    from src.training.models.inflation import map_state_dict_2d_to_3d
    from src.training.utils.sliding import blend_weights, depth_windowed_model_fn, depth_windows
    HAVE_DEPS = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_DEPS = False


UNET_MODEL = {
    "in_channels": 4,
    "out_channels": 4,
    "num_channels": [8, 16],
    "attention_levels": [False, True],
    "num_res_blocks": 1,
    "norm_num_groups": 4,
}

AE_CONFIG = {
    "latent_channels": 2,
    "model": {
        "in_channels": 1,
        "out_channels": 1,
        "block_out_channels": [4, 8],
        "num_res_blocks": 1,
    },
}


def _with_anisotropic(config):
    """Copy of ``config`` with ``model.anisotropic: true``."""
    out = dict(config)
    out["model"] = dict(config["model"])
    out["model"]["anisotropic"] = True
    return out


def _fill_zero_params(model, seed=0):
    """Give small random values to every all-zero parameter.

    MONAI zero-initialises the UNet output convolution, every residual block's
    second convolution and the transformer output projection. Left as they are,
    both the 2D and the 3D UNet would output exactly zero and the equality test
    would pass for the wrong reason.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if not bool((p != 0).any()):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1)


def _inflate(model2d, model3d):
    """Load the 2D weights into the 3D model by centre inflation; strict loading must work."""
    mapped, missing = map_state_dict_2d_to_3d(model2d.state_dict(), model3d.state_dict())
    model3d.load_state_dict(mapped, strict=True)
    return missing


def _max_abs_diff(a, b):
    return float((a - b).abs().max().item())


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestAnisotropicUNet(unittest.TestCase):
    def _pair(self):
        torch.manual_seed(0)
        unet2d = build_diffusion_2d({"model": dict(UNET_MODEL)})
        _fill_zero_params(unet2d)
        unet3d = build_diffusion_3d(_with_anisotropic({"model": dict(UNET_MODEL)}))
        missing = _inflate(unet2d, unet3d)
        unet2d.eval()
        unet3d.eval()
        return unet2d, unet3d, missing

    def test_unet_inplane_exact_after_inflation(self):
        unet2d, unet3d, missing = self._pair()
        self.assertEqual(missing, [])
        self.assertTrue(is_inplane_only(unet3d))

        x = torch.randn(2, 4, 5, 16, 16)
        t = torch.randint(0, 1000, (2,))
        with torch.no_grad():
            y3d = unet3d(x, t)
            self.assertEqual(tuple(y3d.shape), tuple(x.shape))
            # Guard against a vacuous pass: the output must not be all zero.
            self.assertGreater(float(y3d.abs().max().item()), 0.0)
            for k in range(x.shape[2]):
                y2d_k = unet2d(x[:, :, k], t)
                self.assertTrue(
                    torch.allclose(y3d[:, :, k], y2d_k, atol=1e-5),
                    "slice %d differs from the 2D UNet: max abs diff %.3e"
                    % (k, _max_abs_diff(y3d[:, :, k], y2d_k)),
                )

    def test_unet_depth_preserved_through_levels(self):
        _, unet3d, _ = self._pair()
        x = torch.randn(1, 4, 7, 16, 16)  # odd depth: no level may round it
        t = torch.randint(0, 1000, (1,))
        with torch.no_grad():
            y = unet3d(x, t)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_make_inplane_only_idempotent_and_rejects_2d(self):
        unet2d = build_diffusion_2d({"model": dict(UNET_MODEL)})
        with self.assertRaises(ValueError):
            make_inplane_only(unet2d)
        self.assertFalse(is_inplane_only(unet2d))

        _, unet3d, _ = self._pair()
        x = torch.randn(1, 4, 3, 16, 16)
        t = torch.randint(0, 1000, (1,))
        with torch.no_grad():
            y_once = unet3d(x, t)
        keys_once = list(unet3d.state_dict().keys())
        returned = make_inplane_only(unet3d)  # second call: must change nothing
        self.assertIs(returned, unet3d)
        self.assertTrue(is_inplane_only(unet3d))
        self.assertEqual(list(unet3d.state_dict().keys()), keys_once)
        with torch.no_grad():
            y_twice = unet3d(x, t)
        self.assertTrue(torch.equal(y_once, y_twice))
        # Every GroupNorm was swapped, none is left as the plain class.
        norms = [m for m in unet3d.modules() if isinstance(m, torch.nn.GroupNorm)]
        self.assertGreater(len(norms), 0)
        self.assertTrue(all(isinstance(m, SliceWiseGroupNorm) for m in norms))
        # State-dict keys are identical to a plain (non-anisotropic) 3D build.
        plain3d = build_diffusion_3d({"model": dict(UNET_MODEL)})
        self.assertEqual(list(plain3d.state_dict().keys()), keys_once)


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestAnisotropicAutoencoder(unittest.TestCase):
    def test_ae_inplane_exact_after_inflation(self):
        torch.manual_seed(0)
        ae2d = build_autoencoder_2d(AE_CONFIG)
        ae3d = build_autoencoder_3d(_with_anisotropic(AE_CONFIG))
        missing = _inflate(ae2d, ae3d)
        self.assertEqual(missing, [])
        self.assertTrue(is_inplane_only(ae3d))
        ae2d.eval()
        ae3d.eval()

        x = torch.randn(1, 1, 3, 16, 16)
        with torch.no_grad():
            z_mu, z_sigma = ae3d.encode(x)
            self.assertEqual(tuple(z_mu.shape), (1, 2, 3, 8, 8))
            self.assertEqual(tuple(z_sigma.shape), (1, 2, 3, 8, 8))
            for k in range(x.shape[2]):
                mu2d, sigma2d = ae2d.encode(x[:, :, k])
                self.assertTrue(
                    torch.allclose(z_mu[:, :, k], mu2d, atol=1e-5),
                    "encode slice %d: max abs diff %.3e" % (k, _max_abs_diff(z_mu[:, :, k], mu2d)),
                )
                self.assertTrue(
                    torch.allclose(z_sigma[:, :, k], sigma2d, atol=1e-5),
                    "encode sigma slice %d: max abs diff %.3e"
                    % (k, _max_abs_diff(z_sigma[:, :, k], sigma2d)),
                )
            recon3d = ae3d.decode(z_mu)
            self.assertEqual(tuple(recon3d.shape), tuple(x.shape))
            for k in range(x.shape[2]):
                recon2d = ae2d.decode(z_mu[:, :, k])
                self.assertTrue(
                    torch.allclose(recon3d[:, :, k], recon2d, atol=1e-5),
                    "decode slice %d: max abs diff %.3e" % (k, _max_abs_diff(recon3d[:, :, k], recon2d)),
                )

    def test_slicewise_autoencoder_matches_2d(self):
        torch.manual_seed(0)
        ae2d = build_autoencoder_2d(AE_CONFIG).eval()
        wrapper = SliceWiseAutoencoder(ae2d, chunk_slices=2).eval()
        self.assertEqual(wrapper.latent_channels, 2)

        x = torch.randn(2, 1, 5, 16, 16)
        with torch.no_grad():
            z_mu, z_sigma = wrapper.encode(x)
            self.assertEqual(tuple(z_mu.shape), (2, 2, 5, 8, 8))
            self.assertEqual(tuple(z_sigma.shape), (2, 2, 5, 8, 8))
            for b in range(x.shape[0]):
                for k in range(x.shape[2]):
                    mu2d, sigma2d = ae2d.encode(x[b:b + 1, :, k])
                    self.assertTrue(torch.allclose(z_mu[b:b + 1, :, k], mu2d, atol=1e-5))
                    self.assertTrue(torch.allclose(z_sigma[b:b + 1, :, k], sigma2d, atol=1e-5))
            recon = wrapper.decode(z_mu)
            self.assertEqual(tuple(recon.shape), tuple(x.shape))
            for b in range(x.shape[0]):
                for k in range(x.shape[2]):
                    recon2d = ae2d.decode(z_mu[b:b + 1, :, k])
                    self.assertTrue(torch.allclose(recon[b:b + 1, :, k], recon2d, atol=1e-5))
            # forward == decode(encode(x)[0])
            self.assertTrue(torch.allclose(wrapper(x), recon, atol=1e-6))

        # Gradients must flow back through the (frozen) decoder into the latent.
        z = z_mu.detach().clone().requires_grad_(True)
        wrapper.decode(z).sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestDepthWindows(unittest.TestCase):
    def test_depth_windows_cover_and_end_at_depth(self):
        windows = depth_windows(11, 4, 2)
        covered = set()
        for start, end in windows:
            self.assertEqual(end - start, 4)
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(11)))
        self.assertEqual(windows[-1][1], 11)
        self.assertEqual(windows[0], (0, 4))
        # Depth not deeper than the window: one window over everything.
        self.assertEqual(depth_windows(4, 4, 2), [(0, 4)])
        self.assertEqual(depth_windows(3, 4, 2), [(0, 3)])
        # Stride exactly tiling the depth: no extra window is appended.
        self.assertEqual(depth_windows(8, 4, 2), [(0, 4), (2, 6), (4, 8)])

    def test_blend_weights(self):
        w = blend_weights(5)
        self.assertEqual(tuple(w.shape), (5,))
        self.assertTrue(bool((w > 0).all()))
        self.assertEqual(int(w.argmax().item()), 2)
        self.assertAlmostEqual(float(w[2].item()), 1.0, places=6)
        self.assertAlmostEqual(float(w[0].item()), float(w[4].item()), places=6)
        self.assertTrue(torch.equal(blend_weights(3, kind="flat"), torch.ones(3)))
        self.assertTrue(torch.equal(blend_weights(1), torch.ones(1)))
        with self.assertRaises(ValueError):
            blend_weights(3, kind="gaussian")

    def test_depth_windowed_model_fn(self):
        calls = {"n": 0}

        def model_fn(x, t):
            calls["n"] += 1
            return x * 2

        f = depth_windowed_model_fn(model_fn, window=4, stride=2)
        x = torch.randn(2, 3, 11, 5, 5)
        t = torch.zeros(2, dtype=torch.long)
        y = f(x, t)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(
            torch.allclose(y, 2 * x, atol=1e-6),
            "stitched output differs from 2x: max abs diff %.3e" % _max_abs_diff(y, 2 * x),
        )
        self.assertEqual(calls["n"], len(depth_windows(11, 4, 2)))

        # Depth not deeper than the window: the model is called once on the whole tensor.
        calls["n"] = 0
        x_small = torch.randn(1, 3, 4, 5, 5)
        y_small = f(x_small, t[:1])
        self.assertEqual(calls["n"], 1)
        self.assertTrue(torch.equal(y_small, 2 * x_small))

        # Flat blending stitches the linear model exactly too.
        f_flat = depth_windowed_model_fn(model_fn, window=4, stride=3, blend="flat")
        self.assertTrue(torch.allclose(f_flat(x, t), 2 * x, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
