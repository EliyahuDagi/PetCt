"""Tests for the gradient-masked center-freeze warm-start (Make-A-Video / Video-LDM).

Builds tiny 2D and 3D diffusion UNets with MATCHING channels, inflates the 2D prior
into the 3D net, then exercises ``plan_inflation`` + ``build_center_freeze_plan`` +
``CenterFreeze`` on CPU:

  - inflation is function-preserving for convs (center depth slice == 2D weight,
    off-center == 0);
  - the freeze zeroes the center-slice gradient of masked convs, freezes the other
    2D-derived params, and leaves fresh params trainable;
  - one optimizer step pins the center slice + frozen params but moves off-center taps;
  - unfreeze restores full trainability;
  - the plan survives a JSON round-trip and reproduces the freeze on a fresh model
    (i.e. it is sufficient for --resume without the 2D checkpoint).

Guarded by ``skipUnless(HAVE_DEPS)`` (needs torch + monai-generative), mirroring
``test_ft3d_flow.py`` / ``test_training_models.py``.
"""

import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    from src.training.models.diffusion2d import build_diffusion_2d
    from src.training.models.diffusion3d import build_diffusion_3d
    from src.training.models.inflation import (
        CenterFreeze,
        build_center_freeze_plan,
        plan_inflation,
    )
    HAVE_DEPS = True
except Exception as _e:  # pragma: no cover - environment dependent
    HAVE_DEPS = False
    _IMPORT_ERROR = _e


# Tiny matching 2D/3D UNets: small channels, one attention-free downsample, so a
# forward on a small 3D volume runs fast on CPU and every 2D param maps into 3D.
_LATENT_C = 4
_MODEL = {
    "in_channels": _LATENT_C,
    "out_channels": _LATENT_C,
    "num_channels": [8, 16],
    "attention_levels": [False, False],
    "num_res_blocks": 1,
}


def _cfg():
    return {"model": dict(_MODEL)}


def _forward_backward(model):
    """Random 3D forward + scalar loss + backward. Returns the loss tensor."""
    x = torch.randn(1, _LATENT_C, 8, 16, 16)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t)
    loss = (out ** 2).mean()
    loss.backward()
    return loss


def _named(model):
    return dict(model.named_parameters())


def _reachable_param_names():
    """Names of params that actually receive a gradient in a plain (unfrozen) backward.

    MONAI's ``DiffusionModelUNet`` carries a couple of params that are NOT wired into
    the forward pass (e.g. ``middle_block.attention.proj_attn.{weight,bias}``); those
    get ``None`` grad regardless of freezing. Assertions about "unfreeze restores
    grads" must be scoped to reachable params only, else they falsely flag this quirk.
    """
    torch.manual_seed(0)
    m = build_diffusion_3d(_cfg())
    with torch.no_grad():
        dict(m.named_parameters())["out.2.conv.weight"].normal_(0.0, 0.1)
    _forward_backward(m)
    return {n for n, p in m.named_parameters() if p.grad is not None}


@unittest.skipUnless(HAVE_DEPS, "torch / monai-generative not available")
class TestInflationFreeze(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m2d = build_diffusion_2d(_cfg())
        self.m3d = build_diffusion_3d(_cfg())
        # MONAI zero-initializes the final output conv (out.2.conv.weight), so a
        # freshly-built UNet outputs identically zero -> zero loss -> zero gradients
        # everywhere, which would hide the freeze behavior. Perturb it to mimic a
        # TRAINED diff2d checkpoint (the real inflation source) so the network is a
        # non-degenerate function and gradients actually flow through the depth taps.
        with torch.no_grad():
            dict(self.m2d.named_parameters())["out.2.conv.weight"].normal_(0.0, 0.1)
        self.state_2d = self.m2d.state_dict()
        # Params genuinely used in the forward (excludes MONAI's unused proj_attn).
        self.reachable = _reachable_param_names()
        self.plan_inf = plan_inflation(self.state_2d, self.m3d.state_dict())
        # Load the inflated 2D prior into the 3D net (mirrors inflate_and_load).
        self.m3d.load_state_dict(self.plan_inf["mapped"], strict=False)
        # Matching architectures => convs inflate (center-slice warm-start).
        self.assertTrue(self.plan_inf["inflated_conv_keys"], "no convs were inflated")
        self.freeze_plan = build_center_freeze_plan(self.m3d, self.plan_inf["missing"])
        # A meaningful test needs at least one masked conv (kd>1) and one frozen param.
        self.assertTrue(self.freeze_plan["masked_conv_keys"], "no masked convs")
        self.assertTrue(self.freeze_plan["frozen_keys"], "no frozen params")

    # --- 1. inflation is function-preserving for convs ---------------------------
    def test_inflation_center_slice_and_zero_off_center(self):
        params = _named(self.m3d)
        for name in self.plan_inf["inflated_conv_keys"]:
            src = self.state_2d[name]
            w3d = params[name]
            self.assertEqual(w3d.ndim, 5)
            center = w3d.shape[2] // 2
            self.assertTrue(torch.allclose(w3d[:, :, center], src),
                            f"{name}: center slice != 2D weight")
            for d in range(w3d.shape[2]):
                if d == center:
                    continue
                self.assertEqual(float(w3d[:, :, d].abs().sum()), 0.0,
                                 f"{name}: off-center slice d={d} not zero")

    # --- 2. gradient behavior after apply + backward -----------------------------
    def test_grad_masking_freeze_and_fresh(self):
        freeze = CenterFreeze()
        summary = freeze.apply(self.m3d, self.freeze_plan)
        self.assertEqual(summary["n_masked"], len(self.freeze_plan["masked_conv_keys"]))
        self.assertEqual(summary["n_frozen"], len(self.freeze_plan["frozen_keys"]))

        _forward_backward(self.m3d)
        params = _named(self.m3d)

        # (a) masked convs: center-slice grad exactly zero; off-center grad nonzero
        #     somewhere across the masked set.
        off_center_total = 0.0
        for name in self.freeze_plan["masked_conv_keys"]:
            g = params[name].grad
            self.assertIsNotNone(g, f"{name}: masked conv has no grad")
            center = g.shape[2] // 2
            self.assertEqual(float(g[:, :, center].abs().sum()), 0.0,
                             f"{name}: center-slice grad not zeroed")
            for d in range(g.shape[2]):
                if d != center:
                    off_center_total += float(g[:, :, d].abs().sum())
        self.assertGreater(off_center_total, 0.0, "no off-center grad anywhere")

        # (b) frozen params: requires_grad False, grad None.
        for name in self.freeze_plan["frozen_keys"]:
            p = params[name]
            self.assertFalse(p.requires_grad, f"{name}: frozen param still requires grad")
            self.assertIsNone(p.grad, f"{name}: frozen param has a grad")

        # (c) fresh params (if any) receive grads.
        for name in self.freeze_plan["fresh_keys"]:
            self.assertIsNotNone(params[name].grad, f"{name}: fresh param has no grad")

    # --- 2b. optimizer.step pins center + frozen, moves off-center ---------------
    def test_optimizer_step_pins_center_and_frozen(self):
        freeze = CenterFreeze()
        freeze.apply(self.m3d, self.freeze_plan)
        optimizer = torch.optim.Adam(self.m3d.parameters(), lr=1e-2)
        params = _named(self.m3d)

        masked = self.freeze_plan["masked_conv_keys"]
        frozen = self.freeze_plan["frozen_keys"]
        centers = {n: params[n].shape[2] // 2 for n in masked}
        center_before = {n: params[n][:, :, centers[n]].detach().clone() for n in masked}
        offcenter_before = {n: params[n].detach().clone() for n in masked}
        frozen_before = {n: params[n].detach().clone() for n in frozen}

        _forward_backward(self.m3d)
        optimizer.step()

        params = _named(self.m3d)
        # Center slice unchanged (grad was masked to 0 -> Adam update 0).
        for n in masked:
            self.assertTrue(torch.equal(params[n][:, :, centers[n]], center_before[n]),
                            f"{n}: center slice moved despite the mask")
        # Frozen params unchanged.
        for n in frozen:
            self.assertTrue(torch.equal(params[n], frozen_before[n]),
                            f"{n}: frozen param moved")
        # At least one off-center slice changed.
        moved = any(not torch.equal(params[n], offcenter_before[n]) for n in masked)
        self.assertTrue(moved, "no off-center taps moved after optimizer.step")

    # --- 3. unfreeze restores full trainability ----------------------------------
    def test_unfreeze_releases_backbone(self):
        freeze = CenterFreeze()
        freeze.apply(self.m3d, self.freeze_plan)
        # Prime one masked step so the freeze is demonstrably active first.
        _forward_backward(self.m3d)
        self.m3d.zero_grad(set_to_none=True)

        freeze.unfreeze()
        for name in self.freeze_plan["frozen_keys"]:
            self.assertTrue(_named(self.m3d)[name].requires_grad,
                            f"{name}: still frozen after unfreeze")

        _forward_backward(self.m3d)
        params = _named(self.m3d)
        # Previously-frozen params that are actually used in the forward now receive
        # grads (skip MONAI's structurally-unused proj_attn, which is never reachable).
        checked = 0
        for name in self.freeze_plan["frozen_keys"]:
            if name not in self.reachable:
                continue
            self.assertIsNotNone(params[name].grad, f"{name}: no grad after unfreeze")
            checked += 1
        self.assertGreater(checked, 0, "no reachable frozen params to verify unfreeze")
        # Center-slice grad is no longer forced to zero (hook removed).
        center_total = 0.0
        for name in self.freeze_plan["masked_conv_keys"]:
            g = params[name].grad
            center = g.shape[2] // 2
            center_total += float(g[:, :, center].abs().sum())
        self.assertGreater(center_total, 0.0,
                           "center-slice grad still zero after unfreeze")

    # --- 4. resume-safety: plan survives JSON + reproduces freeze on fresh model --
    def test_plan_json_roundtrip_reproduces_freeze(self):
        # The plan is pure strings -> JSON-serializable and embeddable in a checkpoint.
        blob = json.dumps(self.freeze_plan)
        restored = json.loads(blob)
        self.assertEqual(set(restored.keys()),
                         {"masked_conv_keys", "frozen_keys", "fresh_keys"})

        # Rebuild a FRESH 3D model (NO 2D checkpoint / no inflation) and apply the
        # round-tripped plan: the freeze must behave identically from names alone.
        torch.manual_seed(1)
        fresh = build_diffusion_3d(_cfg())
        freeze = CenterFreeze()
        freeze.apply(fresh, restored)

        _forward_backward(fresh)
        params = _named(fresh)
        for name in restored["masked_conv_keys"]:
            g = params[name].grad
            self.assertIsNotNone(g, f"{name}: masked conv has no grad on fresh model")
            center = g.shape[2] // 2
            self.assertEqual(float(g[:, :, center].abs().sum()), 0.0,
                             f"{name}: center-slice grad not zeroed on fresh model")
        for name in restored["frozen_keys"]:
            p = params[name]
            self.assertFalse(p.requires_grad, f"{name}: not frozen on fresh model")
            self.assertIsNone(p.grad, f"{name}: frozen param has grad on fresh model")


if __name__ == "__main__":
    unittest.main()
