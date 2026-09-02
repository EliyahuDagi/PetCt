"""Tests for the quantitative-fidelity losses and the fine-tune plumbing they need.

These target the MEASURED failure of ft3d_flow_p: ``pred ~= 0.84*gt + 0.057`` (dynamic range
compressed ~16%, hot tissue under-estimated ~10%). Each loss must be ~0 for a perfect
prediction, clearly positive for that shrunk one, and prefer the correct range.
"""

import json
import logging
import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.train.train_ft3d import _reuse_split
from src.training.utils.quant_losses import (
    build_quant_loss,
    expectile_loss,
    intensity_weighted_l1,
    slope_penalty,
)


def _gt_with_lesion(seed=0, size=12):
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(1, 1, size, size, size, generator=g) * 0.6
    gt[0, 0, 4:8, 4:8, 4:8] = 1.5
    return gt.clamp(0.0, 1.0)


class TestSlopePenalty(unittest.TestCase):
    def setUp(self):
        self.gt = _gt_with_lesion()
        self.shrunk = (0.84 * self.gt + 0.057).clamp(0, 1)

    def test_perfect_is_zero(self):
        self.assertLess(float(slope_penalty(self.gt, self.gt)), 1e-8)

    def test_shrunk_is_penalized(self):
        self.assertGreater(float(slope_penalty(self.shrunk, self.gt)), 1e-3)

    def test_prefers_correct_range_over_shrunk(self):
        self.assertLess(float(slope_penalty(self.gt, self.gt)),
                        float(slope_penalty(self.shrunk, self.gt)))

    def test_penalty_grows_as_slope_departs_from_one(self):
        vals = [float(slope_penalty((a * self.gt).clamp(0, 1), self.gt)) for a in (0.95, 0.85, 0.70)]
        self.assertLess(vals[0], vals[1])
        self.assertLess(vals[1], vals[2])

    def test_invariant_to_a_pure_offset(self):
        # A constant shift changes the INTERCEPT, not the slope, so this term must ignore
        # it -- that is precisely why it is paired with the recon loss rather than replacing it.
        a = float(slope_penalty(self.gt, self.gt))
        b = float(slope_penalty(self.gt + 0.05, self.gt))
        self.assertAlmostEqual(a, b, places=6)

    def test_variance_term_punishes_oversmoothing(self):
        # Same slope, less variance: blur toward the mean but rescale to keep covariance.
        blurred = self.gt * 0.5 + self.gt.mean() * 0.5
        plain = float(slope_penalty(blurred, self.gt, variance_weight=0.0))
        withvar = float(slope_penalty(blurred, self.gt, variance_weight=1.0))
        self.assertGreater(withvar, plain)

    def test_gradient_reaches_pred_only(self):
        p = torch.nn.Parameter(self.shrunk.clone())
        g = self.gt.clone().requires_grad_(True)
        slope_penalty(p, g).backward()
        self.assertIsNotNone(p.grad)
        self.assertGreater(float(p.grad.abs().sum()), 0.0)
        self.assertIsNone(g.grad, "gt must be detached")

    def test_degenerate_flat_gt_is_safe(self):
        flat = torch.full((1, 1, 4, 4, 4), 0.5)
        self.assertTrue(torch.isfinite(slope_penalty(flat, flat)))


class TestIntensityWeighted(unittest.TestCase):
    def setUp(self):
        self.gt = _gt_with_lesion(seed=1)

    def test_perfect_is_zero(self):
        self.assertLess(float(intensity_weighted_l1(self.gt, self.gt)), 1e-8)

    def test_hot_errors_cost_more_than_equal_cold_errors(self):
        """The defining property. Controlled for voxel COUNT and error MAGNITUDE."""
        flat = self.gt.flatten()
        fg_idx = torch.nonzero(flat > 0.05 * flat.max(), as_tuple=True)[0]
        order = fg_idx[torch.argsort(flat[fg_idx])]
        k = 40
        # Exact INDEX selection, not a threshold: the lesion is a plateau of identical
        # values, so `gt >= thr` would grab the whole plateau and the two perturbations
        # would touch different numbers of voxels -- measuring count, not weighting.
        cold_idx, hot_idx = order[:k], order[-k:]
        eh, ec = self.gt.flatten().clone(), self.gt.flatten().clone()
        eh[hot_idx] -= 0.1
        ec[cold_idx] -= 0.1
        eh, ec = eh.view_as(self.gt), ec.view_as(self.gt)
        # Plain L1 sees these as ~equal; the weighted loss must not.
        self.assertAlmostEqual(float((eh - self.gt).abs().mean()),
                               float((ec - self.gt).abs().mean()), places=3)
        self.assertGreater(float(intensity_weighted_l1(eh, self.gt)),
                           float(intensity_weighted_l1(ec, self.gt)))

    def test_lam_zero_reduces_to_plain_masked_l1(self):
        pred = self.gt * 0.8
        mask = self.gt > 0.05 * float(self.gt.max())
        plain = float((pred[mask] - self.gt[mask]).abs().mean())
        self.assertAlmostEqual(float(intensity_weighted_l1(pred, self.gt, lam=0.0)), plain, places=6)

    def test_gradient_reaches_pred(self):
        p = torch.nn.Parameter((self.gt * 0.8).clone())
        intensity_weighted_l1(p, self.gt).backward()
        self.assertGreater(float(p.grad.abs().sum()), 0.0)


class TestExpectile(unittest.TestCase):
    def setUp(self):
        self.gt = _gt_with_lesion(seed=2)

    def test_perfect_is_zero(self):
        self.assertLess(float(expectile_loss(self.gt, self.gt)), 1e-8)

    def test_q_half_is_symmetric(self):
        d = 0.1
        under = float(expectile_loss(self.gt - d, self.gt, q=0.5))
        over = float(expectile_loss(self.gt + d, self.gt, q=0.5))
        self.assertAlmostEqual(under, over, places=8)

    def test_high_q_punishes_underprediction_more(self):
        d = 0.1
        under = float(expectile_loss(self.gt - d, self.gt, q=0.7))
        over = float(expectile_loss(self.gt + d, self.gt, q=0.7))
        self.assertGreater(under, over)
        # The ratio is exactly q/(1-q) for a uniform-magnitude error.
        self.assertAlmostEqual(under / over, 0.7 / 0.3, places=4)

    def test_invalid_q_raises(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                expectile_loss(self.gt, self.gt, q=bad)


class TestFactory(unittest.TestCase):
    def test_none_disables(self):
        for name in (None, "", "none"):
            self.assertIsNone(build_quant_loss(name))

    def test_unknown_raises_rather_than_silently_disabling(self):
        # A silent no-op would train the control while the log claims otherwise, which
        # quietly invalidates an ablation arm.
        with self.assertRaises(ValueError):
            build_quant_loss("bogus")

    def test_builds_and_forwards_kwargs(self):
        gt = _gt_with_lesion(seed=3)
        term = build_quant_loss("expectile", q=0.7)
        d = 0.1
        self.assertGreater(float(term(gt - d, gt)), float(term(gt + d, gt)))

    def test_none_valued_kwargs_are_dropped(self):
        # The train script passes every knob, most of them None for a given loss.
        term = build_quant_loss("slope", variance_weight=None, lam=None, gamma=None, q=None)
        gt = _gt_with_lesion(seed=4)
        self.assertLess(float(term(gt, gt)), 1e-8)


class TestReuseSplit(unittest.TestCase):
    """The split-reuse plumbing that makes the ablation leak-free."""

    def setUp(self):
        self.logger = logging.getLogger("test_reuse_split")
        self.logger.addHandler(logging.NullHandler())

    def _write(self, obj):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        self.addCleanup(os.unlink, path)
        return path

    def test_maps_paths_to_indices(self):
        patients = ["/p/a", "/p/b", "/p/c", "/p/d"]
        path = self._write({"train": ["/p/c"], "val": ["/p/a"], "test": ["/p/d"]})
        tr, va, te = _reuse_split(path, patients, self.logger)
        self.assertEqual((tr, va, te), ([2], [0], [3]))

    def test_missing_test_patient_raises(self):
        # Silently dropping a test patient would make the run incomparable to the split
        # it claims to reuse -- the entire reason for the flag.
        path = self._write({"train": ["/p/a"], "val": [], "test": ["/p/zzz"]})
        with self.assertRaises(ValueError):
            _reuse_split(path, ["/p/a", "/p/b"], self.logger)

    def test_missing_train_patient_is_tolerated(self):
        path = self._write({"train": ["/p/a", "/p/gone"], "val": [], "test": ["/p/b"]})
        tr, va, te = _reuse_split(path, ["/p/a", "/p/b"], self.logger)
        self.assertEqual((tr, va, te), ([0], [], [1]))

    def test_malformed_file_raises(self):
        path = self._write({"nope": []})
        with self.assertRaises(ValueError):
            _reuse_split(path, ["/p/a"], self.logger)


if __name__ == "__main__":
    unittest.main()


class TestLatentOccupancyWeight(unittest.TestCase):
    """Background down-weighting: rebalance WHERE the loss is measured, not by brightness."""

    def _vol(self, size=8, hot=1.0):
        v = torch.zeros(2, 1, size, size, size)
        v[:, :, 2:6, 2:6, 2:6] = hot        # a centred "body"; the rest is air
        return v

    def test_disabled_returns_none(self):
        from src.training.utils.quant_losses import latent_occupancy_weight
        v = self._vol()
        for bg in (None, 1.0, 1.5):
            self.assertIsNone(latent_occupancy_weight([v], (4, 4, 4), bg_weight=bg))

    def test_shape_and_range(self):
        from src.training.utils.quant_losses import latent_occupancy_weight
        w = latent_occupancy_weight([self._vol()], (4, 4, 4), bg_weight=0.1)
        self.assertEqual(tuple(w.shape), (2, 1, 4, 4, 4))
        self.assertGreaterEqual(float(w.min()), 0.1 - 1e-6)
        self.assertLessEqual(float(w.max()), 1.0 + 1e-6)

    def test_air_gets_bg_weight_and_body_more(self):
        from src.training.utils.quant_losses import latent_occupancy_weight
        w = latent_occupancy_weight([self._vol()], (4, 4, 4), bg_weight=0.1)[0, 0]
        self.assertAlmostEqual(float(w[0, 0, 0]), 0.1, places=5)   # corner = pure air
        self.assertGreater(float(w[1:3, 1:3, 1:3].mean()), 0.5)    # centre = body

    def test_weight_is_INDEPENDENT_of_brightness(self):
        """The defining property: scaling the volume's intensity must not change the weight.

        This is what separates it from intensity weighting and is why it cannot reward
        saturation -- it encodes WHERE anatomy is, never how bright.
        """
        from src.training.utils.quant_losses import latent_occupancy_weight
        a = latent_occupancy_weight([self._vol(hot=1.0)], (4, 4, 4), bg_weight=0.2)
        b = latent_occupancy_weight([self._vol(hot=0.3)], (4, 4, 4), bg_weight=0.2)
        self.assertTrue(torch.allclose(a, b), "weight must not depend on intensity")

    def test_masks_are_unioned_across_volumes(self):
        from src.training.utils.quant_losses import latent_occupancy_weight
        left = torch.zeros(1, 1, 8, 8, 8); left[:, :, :, :, :4] = 1.0
        right = torch.zeros(1, 1, 8, 8, 8); right[:, :, :, :, 4:] = 1.0
        both = latent_occupancy_weight([left, right], (4, 4, 4), bg_weight=0.1)
        only = latent_occupancy_weight([left], (4, 4, 4), bg_weight=0.1)
        self.assertGreater(float(both.mean()), float(only.mean()))
        self.assertAlmostEqual(float(both.min()), 1.0, places=5)  # union covers everything

    def test_none_volumes_are_ignored(self):
        from src.training.utils.quant_losses import latent_occupancy_weight
        w = latent_occupancy_weight([self._vol(), None], (4, 4, 4), bg_weight=0.1)
        self.assertIsNotNone(w)
        self.assertIsNone(latent_occupancy_weight([None], (4, 4, 4), bg_weight=0.1))

    def test_negative_bg_weight_raises(self):
        from src.training.utils.quant_losses import latent_occupancy_weight
        with self.assertRaises(ValueError):
            latent_occupancy_weight([self._vol()], (4, 4, 4), bg_weight=-0.1)


class TestFlowLossWeightMapIsNoOp(unittest.TestCase):
    """weight_map=None must reproduce the ORIGINAL loss bit-exactly.

    Three ablation arms were still training when this parameter was added, and each arm
    launches a fresh python process -- so the default path must be provably unchanged or
    those arms would silently run different code than the ones already finished.
    """

    def setUp(self):
        from src.training.utils.sampling import DiffusionSchedule
        self.sched = DiffusionSchedule(num_train_timesteps=50)
        self.ac = torch.zeros(4, 2, 4, 4)
        self.nac = torch.ones(4, 2, 4, 4)

    class _Const(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, x, t):
            return torch.full_like(x, self.v)

    def test_default_matches_explicit_unit_weight(self):
        from src.training.train.train_diff2d import flow_loss
        m = self._Const(0.5)
        torch.manual_seed(7)
        a, *_ = flow_loss(m, self.sched, self.ac, self.nac)
        torch.manual_seed(7)
        ones = torch.ones(4, 1, 4, 4)
        b, *_ = flow_loss(m, self.sched, self.ac, self.nac, weight_map=ones)
        self.assertAlmostEqual(float(a), float(b), places=10)

    def test_uniform_weight_of_any_scale_is_neutral(self):
        # Normalizing by the mean weight is what keeps the effective LR unchanged.
        from src.training.train.train_diff2d import flow_loss
        m = self._Const(0.5)
        torch.manual_seed(9)
        a, *_ = flow_loss(m, self.sched, self.ac, self.nac)
        torch.manual_seed(9)
        b, *_ = flow_loss(m, self.sched, self.ac, self.nac,
                          weight_map=torch.full((4, 1, 4, 4), 0.37))
        self.assertAlmostEqual(float(a), float(b), places=10)

    def test_nonuniform_weight_changes_the_loss(self):
        """Down-weighting a HIGH-error region must lower the loss.

        The error field has to vary spatially for this to be detectable: a normalized
        weighted mean of a UNIFORM field equals its unweighted mean for any weights, so a
        constant model against a constant target cannot show the effect at all.
        """
        from src.training.train.train_diff2d import flow_loss
        m = self._Const(0.0)
        # target = nac - ac varies along H, so the squared error does too.
        nac = torch.zeros(4, 2, 4, 4)
        nac[:, :, :2] = 4.0          # big error in the top half
        nac[:, :, 2:] = 0.5          # small error in the bottom half
        w = torch.ones(4, 1, 4, 4)
        w[:, :, :2] = 0.1            # ...and down-weight exactly that top half
        torch.manual_seed(11)
        a, *_ = flow_loss(m, self.sched, self.ac, nac)
        torch.manual_seed(11)
        b, *_ = flow_loss(m, self.sched, self.ac, nac, weight_map=w)
        self.assertLess(float(b), float(a),
                        "down-weighting the high-error region must reduce the loss")


class TestLatentIntensityWeight(unittest.TestCase):
    """PET-value weighting of the LATENT velocity MSE (the cheap sibling of the iw arm)."""

    def _vol(self, size=8, body=0.3, hot=1.0):
        """A dim 'body' block with a small hot lesion inside it; the rest is air."""
        v = torch.zeros(2, 1, size, size, size)
        v[:, :, 2:6, 2:6, 2:6] = body
        v[:, :, 4:5, 4:5, 4:5] = hot
        return v

    def test_disabled_returns_none(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        v = self._vol()
        for floor in (None, 1.0, 1.5):
            self.assertIsNone(latent_intensity_weight([v], (4, 4, 4), floor=floor))

    def test_shape_and_range(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        w = latent_intensity_weight([self._vol()], (4, 4, 4), floor=0.1)
        self.assertEqual(tuple(w.shape), (2, 1, 4, 4, 4))
        self.assertGreaterEqual(float(w.min()), 0.1 - 1e-6)
        self.assertLessEqual(float(w.max()), 1.0 + 1e-6)

    def test_air_gets_the_floor_and_hot_tissue_more(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        w = latent_intensity_weight([self._vol()], (4, 4, 4), floor=0.1)[0, 0]
        self.assertAlmostEqual(float(w[0, 0, 0]), 0.1, places=5)     # corner = pure air
        self.assertGreater(float(w[2, 2, 2]), float(w[1, 1, 1]))     # lesion cell > body cell

    def test_weight_RISES_with_brightness(self):
        """The defining difference from the occupancy weight, which ignores intensity."""
        from src.training.utils.quant_losses import (
            latent_intensity_weight,
            latent_occupancy_weight,
        )
        dim = torch.zeros(1, 1, 8, 8, 8)
        dim[:, :, 2:6, 2:6, 2:6] = 0.2
        dim[:, :, 2:3, 2:3, 2:3] = 1.0          # one hot corner fixes the normalizer
        bright = dim.clone()
        bright[:, :, 4:6, 4:6, 4:6] = 1.0       # brighten part of the SAME anatomy

        cell = (slice(None), slice(None), slice(2, 3), slice(2, 3), slice(2, 3))
        a = latent_intensity_weight([dim], (4, 4, 4), floor=0.1)[cell]
        b = latent_intensity_weight([bright], (4, 4, 4), floor=0.1)[cell]
        self.assertGreater(float(b), float(a))
        # ...whereas the occupancy weight is unchanged: same anatomy, same mask.
        oa = latent_occupancy_weight([dim], (4, 4, 4), bg_weight=0.1)[cell]
        ob = latent_occupancy_weight([bright], (4, 4, 4), bg_weight=0.1)[cell]
        self.assertAlmostEqual(float(oa), float(ob), places=6)

    def test_a_single_outlier_does_not_collapse_the_rest(self):
        """The point of the percentile normalizer: one runaway voxel must not zero the map.

        With a max-normalizer a lone voxel 50x the tissue level would divide every other
        weight down to the floor. The p99-of-foreground scale is unmoved by it.
        """
        from src.training.utils.quant_losses import latent_intensity_weight
        clean = torch.zeros(1, 1, 16, 16, 16)
        clean[:, :, 4:12, 4:12, 4:12] = 0.4
        spiked = clean.clone()
        spiked[:, :, 0, 0, 0] = 50.0            # an injection-site / decoder-spike outlier

        kw = dict(floor=0.1, percentile=99.0)
        a = latent_intensity_weight([clean], (4, 4, 4), **kw)
        b = latent_intensity_weight([spiked], (4, 4, 4), **kw)
        body = (slice(None), slice(None), slice(1, 3), slice(1, 3), slice(1, 3))
        self.assertGreater(float(b[body].mean()), 0.9 * float(a[body].mean()))

        # A pure max-normalizer (percentile=100) is exactly the failure mode being avoided.
        c = latent_intensity_weight([spiked], (4, 4, 4), floor=0.1, percentile=100.0)
        self.assertLess(float(c[body].mean()), 0.2 * float(a[body].mean()))

    def test_clip_bounds_the_ramp(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        v = torch.zeros(1, 1, 16, 16, 16)
        v[:, :, 4:12, 4:12, 4:12] = 0.3         # 512 foreground voxels set the p99...
        v[:, :, 8, 8, 8] = 3.0                  # ...so this one is 10x above it
        w = latent_intensity_weight([v], (16, 16, 16), floor=0.0, clip=1.0)
        self.assertLessEqual(float(w.max()), 1.0 + 1e-6)
        w2 = latent_intensity_weight([v], (16, 16, 16), floor=0.0, clip=4.0)
        self.assertAlmostEqual(float(w2.max()), 4.0, places=4)

    def test_gamma_concentrates_on_the_hottest(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        v = self._vol()
        lo = latent_intensity_weight([v], (4, 4, 4), floor=0.0, gamma=1.0)
        hi = latent_intensity_weight([v], (4, 4, 4), floor=0.0, gamma=3.0)
        # gamma>1 pushes mid-uptake weights down relative to the peak.
        self.assertLess(float(hi.mean()) / float(hi.max()), float(lo.mean()) / float(lo.max()))

    def test_max_pool_preserves_a_small_lesion(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        v = torch.zeros(1, 1, 8, 8, 8)
        v[:, :, 2:6, 2:6, 2:6] = 0.1
        v[:, :, 4, 4, 4] = 1.0                  # one hot voxel inside a 2^3 latent block
        avg = latent_intensity_weight([v], (4, 4, 4), floor=0.0, pool="avg")
        mx = latent_intensity_weight([v], (4, 4, 4), floor=0.0, pool="max")
        self.assertAlmostEqual(float(mx[0, 0, 2, 2, 2]), 1.0, places=5)
        self.assertLess(float(avg[0, 0, 2, 2, 2]), 0.5)

    def test_works_in_2d(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        v = torch.zeros(3, 1, 16, 16)
        v[:, :, 4:12, 4:12] = 0.5
        w = latent_intensity_weight([v], (4, 4), floor=0.1)
        self.assertEqual(tuple(w.shape), (3, 1, 4, 4))

    def test_per_sample_normalization(self):
        """A dim patient and a bright one must get the SAME map -- the scale is per-sample."""
        from src.training.utils.quant_losses import latent_intensity_weight
        v = torch.zeros(2, 1, 8, 8, 8)
        v[0, :, 2:6, 2:6, 2:6] = 0.2
        v[1, :, 2:6, 2:6, 2:6] = 0.9            # same anatomy, 4.5x brighter
        w = latent_intensity_weight([v], (4, 4, 4), floor=0.1)
        self.assertTrue(torch.allclose(w[0], w[1], atol=1e-5))

    def test_all_zero_volume_is_safe(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        w = latent_intensity_weight([torch.zeros(1, 1, 8, 8, 8)], (4, 4, 4), floor=0.1)
        self.assertTrue(torch.isfinite(w).all())
        self.assertAlmostEqual(float(w.max()), 0.1, places=5)

    def test_no_gradient_reaches_the_weight(self):
        """The map is built from GT only; it must never be a path the model can optimize."""
        from src.training.utils.quant_losses import latent_intensity_weight
        v = self._vol().requires_grad_(True)
        w = latent_intensity_weight([v], (4, 4, 4), floor=0.1)
        self.assertFalse(w.requires_grad)

    def test_bad_args_raise(self):
        from src.training.utils.quant_losses import latent_intensity_weight
        v = self._vol()
        with self.assertRaises(ValueError):
            latent_intensity_weight([v], (4, 4, 4), floor=-0.1)
        with self.assertRaises(ValueError):
            latent_intensity_weight([v], (4, 4, 4), clip=0.0)
        with self.assertRaises(ValueError):
            latent_intensity_weight([v], (4, 4, 4), gamma=0.0)
        with self.assertRaises(ValueError):
            latent_intensity_weight([v], (4, 4, 4), pool="median")


class TestBuildLatentWeight(unittest.TestCase):
    """The config-driven factory the train scripts call."""

    def setUp(self):
        self.ac = torch.zeros(1, 1, 8, 8, 8)
        self.ac[:, :, 2:6, 2:6, 2:6] = 1.0          # AC anatomy on the left-ish block
        self.nac = torch.zeros(1, 1, 8, 8, 8)
        self.nac[:, :, 2:6, 2:6, 6:8] = 1.0         # NAC anatomy somewhere else

    def test_none_disables(self):
        from src.training.utils.quant_losses import build_latent_weight
        self.assertIsNone(build_latent_weight("none"))
        self.assertIsNone(build_latent_weight(None))

    def test_unknown_raises(self):
        from src.training.utils.quant_losses import build_latent_weight
        with self.assertRaises(ValueError):
            build_latent_weight("brightness")

    def test_occupancy_default_is_the_previous_behaviour(self):
        from src.training.utils.quant_losses import (
            build_latent_weight,
            latent_occupancy_weight,
        )
        fn = build_latent_weight("occupancy", bg_weight=0.1)
        got = fn(self.ac, self.nac, (4, 4, 4))
        want = latent_occupancy_weight([self.ac, self.nac], (4, 4, 4), bg_weight=0.1)
        self.assertTrue(torch.allclose(got, want))

    def test_occupancy_off_at_unit_weight(self):
        from src.training.utils.quant_losses import build_latent_weight
        self.assertIsNone(build_latent_weight("occupancy", bg_weight=1.0)(self.ac, self.nac, (4, 4, 4)))

    def test_source_selects_the_volume(self):
        from src.training.utils.quant_losses import build_latent_weight
        kw = dict(bg_weight=0.1)
        w_ac = build_latent_weight("intensity", source="ac", **kw)(self.ac, self.nac, (4, 4, 4))
        w_nac = build_latent_weight("intensity", source="nac", **kw)(self.ac, self.nac, (4, 4, 4))
        w_u = build_latent_weight("intensity", source="union", **kw)(self.ac, self.nac, (4, 4, 4))
        self.assertFalse(torch.allclose(w_ac, w_nac))
        self.assertGreater(float(w_u.mean()), float(w_ac.mean()))
        self.assertGreater(float(w_u.mean()), float(w_nac.mean()))

    def test_unknown_source_raises(self):
        from src.training.utils.quant_losses import build_latent_weight
        with self.assertRaises(ValueError):
            build_latent_weight("intensity", source="both")

    def test_intensity_map_is_usable_by_flow_loss(self):
        """End-to-end shape contract: the map must broadcast over the latent channel dim."""
        from src.training.train.train_diff2d import flow_loss
        from src.training.utils.quant_losses import build_latent_weight
        from src.training.utils.sampling import DiffusionSchedule

        class _Const(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        sched = DiffusionSchedule(num_train_timesteps=50)
        ac_lat = torch.zeros(1, 3, 4, 4, 4)
        nac_lat = torch.ones(1, 3, 4, 4, 4)
        w = build_latent_weight("intensity", bg_weight=0.1)(self.ac, self.nac, ac_lat.shape[2:])
        self.assertEqual(tuple(w.shape), (1, 1, 4, 4, 4))
        loss, *_ = flow_loss(_Const(), sched, ac_lat, nac_lat, weight_map=w)
        self.assertTrue(torch.isfinite(loss))
