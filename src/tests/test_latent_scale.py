"""Tests for the Stable-Diffusion-style latent-scale normalization.

The diffusion stages multiply AE latents by ``scale`` on encode and divide on
decode so the sampler (which starts from N(0,1)) operates on ~unit-std latents.
These tests pin the three invariants that make that safe:

  1. ``scale=1.0`` (and any non-positive scale) is an exact no-op -> AE training
     and reconstruction stages are byte-identical to before.
  2. encode(scale)/decode(scale) round-trips: decode undoes encode's scaling, so
     the reconstructed image is independent of ``scale``.
  3. multiplying a latent by ``1/std`` yields ~unit-std latents (the whole point).

Runs on CPU with a tiny AE so it is cheap.
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.models.autoencoder2d import ae_decode, ae_encode, build_autoencoder_2d


def _tiny_ae():
    config = {
        "latent_channels": 4,
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "block_out_channels": [8, 8],
            "num_res_blocks": 1,
            "attention_levels": [False, False],
            "norm_num_groups": 8,
        },
    }
    return build_autoencoder_2d(config).eval()


def _img(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 1, 16, 16, generator=g)


def test_default_scale_is_noop():
    ae = _tiny_ae()
    x = _img()
    z1 = ae_encode(ae, x)            # default scale=1.0
    z_explicit = ae_encode(ae, x, scale=1.0)
    z_nonpos = ae_encode(ae, x, scale=0.0)   # non-positive -> treated as no-op
    assert torch.equal(z1, z_explicit)
    assert torch.equal(z1, z_nonpos)


def test_encode_multiplies_decode_divides_roundtrip():
    ae = _tiny_ae()
    x = _img(seed=1)
    scale = 0.37
    z_unscaled = ae_encode(ae, x)
    z_scaled = ae_encode(ae, x, scale=scale)
    # encode multiplies by scale
    assert torch.allclose(z_scaled, z_unscaled * scale, atol=1e-5)
    # decode(scale) divides it back, so the decoded image matches the no-scale path
    recon_scaled = ae_decode(ae, z_scaled, scale=scale)
    recon_plain = ae_decode(ae, z_unscaled)
    assert torch.allclose(recon_scaled, recon_plain, atol=1e-5)


def test_scaling_normalizes_latent_std():
    ae = _tiny_ae()
    x = _img(seed=2)
    z = ae_encode(ae, x)
    std = float(z.std())
    assert std > 0
    scale = 1.0 / (std + 1e-8)
    z_norm = ae_encode(ae, x, scale=scale)
    assert abs(float(z_norm.std()) - 1.0) < 1e-3


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
