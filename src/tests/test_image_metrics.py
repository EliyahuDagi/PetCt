import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.utils.image_metrics import (
    clamp_unit,
    hot_band_relative_error,
    image_quality_metrics,
    mae,
    max_relative_error,
    mean_relative_bias,
    nrmse,
    percentile_relative_error,
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
        assert set(m) == {
            "psnr", "ssim", "nrmse", "mae", "rel_bias",
            "max_rel_error", "p95_rel_error", "hot_band_rel_error",
            "voxel_r2", "reg_slope", "reg_intercept",
            "ba_mean_bias", "ba_loa_lower", "ba_loa_upper",
        }
        for k, v in m.items():
            assert _finite(v), (k, v)


def test_half_precision_inputs_upcast():
    gt = _rand2d(seed=5).half()
    pred = _rand2d(seed=6).half()
    m = image_quality_metrics(pred, gt)
    for v in m.values():
        assert isinstance(v, float)
    assert _finite(m["ssim"])


# --- clamp_unit + the robust peak metrics -----------------------------------------
# Context: normalize_volume (src/training/data.py) percentile-normalizes AND CLIPS every
# volume to [0,1], so a GT foreground max is always exactly 1.0 while a decoded
# prediction is unbounded. That makes max_rel_error a pure "how far above 1.0 did the
# decoder ring" statistic rather than a lesion-SUVmax measure.


def _clipped_gt_with_lesion(seed=7, size=24):
    """A GT volume shaped like a real normalize_volume output: clipped, with a plateau."""
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(1, 1, size, size, size, generator=g) * 0.6
    gt[0, 0, 8:14, 8:14, 8:14] = 1.5          # a hot lesion, above the clip point
    return gt.clamp(0.0, 1.0)                  # <- what normalize_volume does


def test_clamp_unit_is_optional_and_exact():
    x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.8])
    assert torch.equal(clamp_unit(x, enabled=False), x)
    assert float(clamp_unit(x).min()) == 0.0
    assert float(clamp_unit(x).max()) == 1.0
    # device/dtype preserved (it runs inside the model path, not the CPU metric path)
    h = x.to(torch.float16)
    assert clamp_unit(h).dtype == torch.float16


def test_single_spike_breaks_max_but_not_robust_metrics():
    gt = _clipped_gt_with_lesion()
    pred = gt.clone()
    pred[0, 0, 2, 2, 2] = 1.8                  # ONE spurious overshoot voxel
    # GT max is exactly 1.0, so max_rel_error degenerates to (pred_max - 1).
    assert abs(max_relative_error(pred, gt) - 0.8) < 1e-4
    # The robust metrics ignore a single voxel entirely.
    assert abs(percentile_relative_error(pred, gt)) < 1e-3
    assert abs(hot_band_relative_error(pred, gt)) < 1e-3
    # And clamping removes the overshoot outright, with no retraining involved.
    assert abs(max_relative_error(clamp_unit(pred), gt)) < 1e-6


def test_high_percentiles_are_blind_to_clamped_overshoot():
    """Why the default q is 95 and not 99.9.

    normalize_volume clips at p99, so a few percent of every GT volume sits at exactly
    1.0 and GT p98/p99/p99.9 are all 1.0. A prediction that OVER-shoots and is then
    clamped to the valid range also has p99.9 == 1.0, so the high-percentile metric
    reads exactly 0 and is blind to the error -- and over-shoot is precisely this
    pipeline's failure mode (the AE decoder's linear output conv rings above 1.0).
    q=95 sits below the plateau and still sees it.
    """
    gt = _clipped_gt_with_lesion()
    fg = gt[gt > 0.05 * float(gt.max())]
    assert float(torch.quantile(fg, 0.999)) == 1.0, "top percentile should be saturated"
    q95 = float(torch.quantile(fg, 0.95))
    assert q95 < 1.0, "q=95 must stay below the saturation plateau to be informative"

    # Over-predict the high-uptake region, then clamp to the valid range as the eval
    # path now does.
    pred = clamp_unit(gt * 1.4)
    assert abs(percentile_relative_error(pred, gt, q=99.9)) < 1e-6, \
        "q=99.9 is blind here: both GT and clamped pred saturate at 1.0"
    assert percentile_relative_error(pred, gt, q=95.0) > 0.01
    assert hot_band_relative_error(pred, gt) > 0.01


def test_hot_band_detects_under_and_over_estimation():
    gt = _clipped_gt_with_lesion()
    assert abs(hot_band_relative_error(gt, gt)) < 1e-6
    assert hot_band_relative_error(gt * 0.8, gt) < 0      # under-estimate -> negative
    assert hot_band_relative_error(gt * 1.2, gt) > 0      # over-estimate  -> positive


def test_robust_metrics_work_in_2d():
    g = torch.Generator().manual_seed(11)
    gt = (torch.rand(1, 1, 32, 32, generator=g) * 0.6)
    gt[0, 0, 10:16, 10:16] = 1.5
    gt = gt.clamp(0.0, 1.0)
    assert _finite(percentile_relative_error(gt, gt))
    assert _finite(hot_band_relative_error(gt, gt))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
