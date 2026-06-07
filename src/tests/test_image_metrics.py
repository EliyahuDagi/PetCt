import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.utils.image_metrics import (
    image_quality_metrics,
    mae,
    mean_relative_bias,
    nrmse,
    psnr,
    ssim,
)


def _rand2d(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 1, 16, 16, generator=g)


def _rand3d(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 1, 6, 12, 12, generator=g)


def _finite(x):
    return isinstance(x, float) and math.isfinite(x)


def test_identical_2d():
    gt = _rand2d()
    pred = gt.clone()
    assert abs(ssim(pred, gt) - 1.0) < 1e-4
    assert nrmse(pred, gt) < 1e-6
    assert mae(pred, gt) < 1e-6
    assert abs(mean_relative_bias(pred, gt)) < 1e-6
    assert psnr(pred, gt) == float("inf")


def test_identical_3d():
    gt = _rand3d()
    pred = gt.clone()
    assert abs(ssim(pred, gt) - 1.0) < 1e-4
    assert nrmse(pred, gt) < 1e-6
    assert mae(pred, gt) < 1e-6
    assert abs(mean_relative_bias(pred, gt)) < 1e-6


def test_offset_bias_sign_and_ssim_drop_2d():
    gt = _rand2d(seed=1)
    offset = 0.2
    pred = gt + offset
    rb = mean_relative_bias(pred, gt)
    # Positive offset -> positive relative bias; magnitude ~ offset / fg_mean.
    fg = gt[gt > 0.05 * float(gt.max())]
    expected = offset / float(fg.mean())
    assert rb > 0
    assert abs(rb - expected) < 1e-3
    assert ssim(pred, gt) < 1.0
    # A negative offset flips the sign.
    assert mean_relative_bias(gt - offset, gt) < 0


def test_offset_bias_3d():
    gt = _rand3d(seed=2)
    pred = gt + 0.1
    assert mean_relative_bias(pred, gt) > 0
    assert ssim(pred, gt) < 1.0


def test_shapes_return_finite_floats():
    for builder in (_rand2d, _rand3d):
        gt = builder(seed=3)
        pred = builder(seed=4)
        m = image_quality_metrics(pred, gt)
        assert set(m) == {"psnr", "ssim", "nrmse", "mae", "rel_bias"}
        for k, v in m.items():
            assert _finite(v), (k, v)


def test_half_precision_inputs_upcast():
    gt = _rand2d(seed=5).half()
    pred = _rand2d(seed=6).half()
    m = image_quality_metrics(pred, gt)
    for v in m.values():
        assert isinstance(v, float)
    assert _finite(m["ssim"])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
