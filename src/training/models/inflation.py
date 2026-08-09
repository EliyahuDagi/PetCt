import torch


def inflate_conv2d_to_3d(weight_2d, kernel_depth: int) -> torch.Tensor:
    if weight_2d.ndim != 4:
        raise ValueError("Expected 4D Conv2d weight.")
    out_ch, in_ch, k_h, k_w = weight_2d.shape
    weight_3d = torch.zeros((out_ch, in_ch, kernel_depth, k_h, k_w), dtype=weight_2d.dtype)
    center = kernel_depth // 2
    weight_3d[:, :, center, :, :] = weight_2d
    return weight_3d


def plan_inflation(state_2d, state_3d):
    """Plan a 2D->3D state-dict inflation without mutating any model.

    Returns a dict::

        {
            "mapped":             name -> tensor to load into the 3D model (either a
                                  shape-equal 1:1 copy, or a 4D->5D center-inflated
                                  Conv weight),
            "missing":            list of state_2d keys that could NOT be mapped
                                  (absent in the 3D model, or a structural mismatch),
            "inflated_conv_keys": subset of ``mapped`` produced by the 4D->5D
                                  center-inflation path (the ones whose CENTER depth
                                  slice is the real 2D prior and off-center is zero),
            "matched_keys":       subset of ``mapped`` that were shape-equal 1:1
                                  (norm affines, biases, attention/time-embed params).
        }

    The center-inflation only triggers when out/in channels and the (H, W) kernel
    dims match; a layer that differs structurally (e.g. a different num_channels per
    level) is left to its fresh init and reported in ``missing``.
    """
    mapped = {}
    missing = []
    inflated_conv_keys = []
    matched_keys = []
    for name, tensor in state_2d.items():
        if name not in state_3d:
            missing.append(name)
            continue
        target = state_3d[name]
        if tensor.shape == target.shape:
            mapped[name] = tensor
            matched_keys.append(name)
            continue
        # Inflate Conv2d -> Conv3d only when out/in channels and the H,W kernel
        # dims match; otherwise the layer differs structurally (e.g. different
        # num_channels per level) and must be left to its fresh init.
        if (
            tensor.ndim == 4
            and target.ndim == 5
            and tensor.shape[0] == target.shape[0]
            and tensor.shape[1] == target.shape[1]
            and tensor.shape[2] == target.shape[3]
            and tensor.shape[3] == target.shape[4]
        ):
            mapped[name] = inflate_conv2d_to_3d(tensor, target.shape[2])
            inflated_conv_keys.append(name)
            continue
        missing.append(name)

    return {
        "mapped": mapped,
        "missing": missing,
        "inflated_conv_keys": inflated_conv_keys,
        "matched_keys": matched_keys,
    }


def map_state_dict_2d_to_3d(state_2d, state_3d):
    # Thin back-compat wrapper: existing callers (train_ft3d/train_ae3d inflate_and_load,
    # tests) rely on the (mapped, missing) tuple. Behavior is byte-identical to the
    # pre-refactor implementation.
    plan = plan_inflation(state_2d, state_3d)
    return plan["mapped"], plan["missing"]


def build_center_freeze_plan(model, missing_keys):
    """Derive a JSON-serializable freeze plan from the MODEL alone + the fresh keys.

    Implements the structural half of the "gradient-masked center-freeze" warm-start
    (Make-A-Video / Video-LDM): after inflating a 2D diffusion prior into a 3D UNet,
    the 2D spatial prior (center depth slice) and the new depth-axis capacity
    (off-center slices) live in the SAME monolithic 3D conv weight, so they cannot be
    separated with ``requires_grad_``. This function classifies every model parameter
    into one of three groups:

    - ``masked_conv_keys``: conv weights with ``ndim == 5 and shape[2] > 1`` that were
      NOT freshly initialized (not in ``missing``). Their CENTER depth slice holds the
      pinned 2D prior; :class:`CenterFreeze` masks the center-slice gradient so only
      the zero-initialized off-center depth taps learn.
    - ``frozen_keys``: every remaining non-fresh param (norm affines, biases,
      time-embed MLPs, attention projections, and 1x1x1 convs which carry no depth
      axis). These are pinned with ``requires_grad_(False)``.
    - ``fresh_keys``: the ``missing`` params that actually exist in the model. These
      never received a 2D weight, so they stay FULLY trainable.

    Deriving the plan from the model + ``missing`` (strings only) is what lets
    ``--resume`` re-establish the exact freeze from the checkpoint-embedded plan,
    without needing the original 2D checkpoint.
    """
    missing = set(missing_keys or [])
    masked_conv_keys = []
    frozen_keys = []
    fresh_keys = []
    for name, param in model.named_parameters():
        if name in missing:
            # Freshly-initialized (no 2D weight mapped in): stay fully trainable.
            fresh_keys.append(name)
            continue
        if param.ndim == 5 and param.shape[2] > 1:
            masked_conv_keys.append(name)
        else:
            frozen_keys.append(name)
    return {
        "masked_conv_keys": masked_conv_keys,
        "frozen_keys": frozen_keys,
        "fresh_keys": fresh_keys,
    }


class CenterFreeze:
    """Apply / release a gradient-masked center-freeze from a plan.

    ``apply(model, plan)``:
      - for each ``masked_conv_keys`` param, registers a backward hook that zeroes the
        gradient on the center depth slice ``[:, :, kd // 2]`` (pinning the 2D prior;
        Adam then builds no momentum there) while letting the off-center taps learn;
      - for each ``frozen_keys`` param, sets ``requires_grad_(False)``.

    ``unfreeze()`` removes every hook and restores ``requires_grad_(True)`` on the
    frozen params, so subsequent optimizer steps fine-tune the whole net.

    Operate on the RAW (uncompiled) module's ``named_parameters`` -- the hooks/flags
    live on the real leaf tensors that a ``torch.compile`` wrapper shares.
    """

    def __init__(self):
        self._handles = []
        self._frozen_params = []
        self.n_masked = 0
        self.n_frozen = 0
        self.n_fresh = 0

    def apply(self, model, plan):
        params = dict(model.named_parameters())
        self._handles = []
        self._frozen_params = []
        self.n_masked = 0
        self.n_frozen = 0
        self.n_fresh = 0

        for name in plan.get("masked_conv_keys", []):
            param = params.get(name)
            if param is None:
                continue
            center = param.shape[2] // 2
            # ones_like matches shape/device/dtype/memory-format of the param, so the
            # elementwise (g * mask) in the hook stays layout-consistent.
            mask = torch.ones_like(param)
            mask[:, :, center] = 0
            handle = param.register_hook(lambda grad, m=mask: grad * m)
            self._handles.append(handle)
            self.n_masked += 1

        for name in plan.get("frozen_keys", []):
            param = params.get(name)
            if param is None:
                continue
            param.requires_grad_(False)
            self._frozen_params.append(param)
            self.n_frozen += 1

        for name in plan.get("fresh_keys", []):
            if name in params:
                self.n_fresh += 1

        return self.summary()

    def unfreeze(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        for param in self._frozen_params:
            param.requires_grad_(True)
        self._frozen_params = []

    @property
    def n_trainable(self):
        # Tensors that still receive gradients while the freeze is active: the
        # center-masked convs (off-center still trains) plus the fresh params.
        return self.n_masked + self.n_fresh

    def summary(self):
        return {
            "n_masked": self.n_masked,
            "n_frozen": self.n_frozen,
            "n_fresh": self.n_fresh,
            "n_trainable": self.n_trainable,
        }
