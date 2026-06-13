"""Config-driven GEOMETRIC data augmentation for the NAC->AC training pipeline.

Builds MONAI dictionary transforms (``Compose`` of ``RandFlipd`` / ``RandRotate90d``
/ ``RandAffined`` / ``Rand3DElasticd``) and wraps them in a tiny callable that maps a
dict of single-sample tensors -> the same dict augmented. Because the transforms are
*dictionary* transforms applied to all keys at once, a single random draw is shared
across every key in the dict, so a paired ``{"nac": ..., "ac": ...}`` sample gets the
SAME geometry on both halves (correspondence preserved). Single-volume AE stages pass
``{"img": ...}``.

GEOMETRIC ONLY: no intensity-domain transforms are ever built here (no scale/shift
intensity, no Gaussian noise, etc.) -- by design and per the pipeline decision.

Everything defaults OFF: :func:`build_aug_2d` / :func:`build_aug_3d` return a no-op
identity callable when the config block is missing or ``enabled`` is false, so the
existing samplers / CPU end-to-end tests behave exactly as before unless a config
explicitly opts in.

Tensor shape contract
---------------------
The samplers operate per-sample on channel-first tensors:
  * 2D: ``(C, H, W)``  (the samplers' ``(1, size, size)`` slices)
  * 3D: ``(C, D, H, W)`` (the samplers' ``(1, S, S, S)`` cubes)
The returned callable accepts/returns tensors of exactly that shape; it adds no batch
dimension. MONAI ``Rand*d`` transforms are channel-first, matching this directly.
"""

import math


# Default augmentation knobs. A config ``augment:`` block may override any subset;
# anything omitted falls back to these. ``enabled: false`` short-circuits to a no-op.
_DEFAULTS_2D = {
    "enabled": False,
    "flip_prob": 0.5,       # per spatial axis (H, then W)
    "rot90_prob": 0.5,      # 90-degree rotations in the H-W plane
    "affine_prob": 0.3,     # mild rotate/scale/translate
    "affine_rotate_deg": 5.0,
    "affine_scale": 0.05,
    "affine_translate": 0.02,  # fraction of the axis length
}

_DEFAULTS_3D = {
    "enabled": False,
    "flip_prob": 0.5,       # per spatial axis (D, H, W)
    "rot90_prob": 0.5,      # 90-degree rotations in the axial (H-W) plane
    "affine_prob": 0.3,     # mild rotate/scale/translate
    "affine_rotate_deg": 7.0,
    "affine_scale": 0.1,
    "affine_translate": 0.02,
    "elastic_prob": 0.0,    # very-mild elastic; off by default
    "elastic_sigma_range": [1.0, 2.0],
    "elastic_magnitude_range": [1.0, 2.0],
}


def _merged(cfg, defaults):
    """Merge a user ``augment`` block over the per-dim defaults (shallow)."""
    out = dict(defaults)
    if isinstance(cfg, dict):
        out.update({k: v for k, v in cfg.items() if v is not None})
    return out


def _identity(sample):
    """No-op transform: return the dict of tensors unchanged."""
    return sample


def _import_dict_transforms():
    """Import the MONAI dictionary transforms, returning a name->class dict.

    Imported lazily so this module stays importable on a torch-free host and so a
    disabled config never forces the MONAI dependency.
    """
    from monai.transforms import (  # type: ignore
        Compose,
        RandAffined,
        RandFlipd,
        RandRotate90d,
    )

    out = {
        "Compose": Compose,
        "RandAffined": RandAffined,
        "RandFlipd": RandFlipd,
        "RandRotate90d": RandRotate90d,
    }
    try:  # Rand3DElasticd is only needed when elastic is enabled.
        from monai.transforms import Rand3DElasticd  # type: ignore

        out["Rand3DElasticd"] = Rand3DElasticd
    except Exception:  # pragma: no cover - optional
        out["Rand3DElasticd"] = None
    return out


class _MonaiDictAug:
    """Callable wrapping a MONAI ``Compose`` over a fixed set of dict keys.

    A single ``__call__`` runs the same Compose on all provided keys, so all keys
    share one random draw (synchronized geometry across paired NAC/AC). The keys are
    inferred from the input dict at call time, so the same builder works for both
    ``{"img": ...}`` and ``{"nac": ..., "ac": ...}``.
    """

    def __init__(self, transforms):
        from monai.transforms import Compose  # type: ignore

        self._compose = Compose(transforms)

    def __call__(self, sample):
        # MONAI dict transforms operate on the keys named when constructed; we
        # construct them with the union of possible keys, so any subset present is
        # transformed and absent keys are simply ignored (allow_missing_keys=True).
        return self._compose(dict(sample))


# Keys the transforms are wired for; any subset may appear in a given call.
_AUG_KEYS = ("img", "nac", "ac")


def build_aug_2d(cfg):
    """Build a 2D geometric augmentation callable from a config ``augment`` block.

    ``cfg`` is the parsed ``augment:`` dict (or None). Returns a callable mapping a
    dict of ``(C, H, W)`` tensors to the augmented dict, or a no-op identity when
    augmentation is disabled / cfg is missing. Synchronizes geometry across all keys
    (so paired ``nac``/``ac`` get identical transforms).
    """
    c = _merged(cfg, _DEFAULTS_2D)
    if not c.get("enabled", False):
        return _identity

    t = _import_dict_transforms()
    keys = list(_AUG_KEYS)
    transforms = []

    flip_prob = float(c.get("flip_prob", 0.0) or 0.0)
    if flip_prob > 0:
        # Spatial axes for a (C,H,W) tensor are 0 (H) and 1 (W).
        transforms.append(t["RandFlipd"](keys=keys, prob=flip_prob, spatial_axis=0, allow_missing_keys=True))
        transforms.append(t["RandFlipd"](keys=keys, prob=flip_prob, spatial_axis=1, allow_missing_keys=True))

    rot90_prob = float(c.get("rot90_prob", 0.0) or 0.0)
    if rot90_prob > 0:
        transforms.append(
            t["RandRotate90d"](keys=keys, prob=rot90_prob, max_k=3, spatial_axes=(0, 1), allow_missing_keys=True)
        )

    affine_prob = float(c.get("affine_prob", 0.0) or 0.0)
    if affine_prob > 0:
        rot = math.radians(float(c.get("affine_rotate_deg", 0.0) or 0.0))
        scale = float(c.get("affine_scale", 0.0) or 0.0)
        trans = float(c.get("affine_translate", 0.0) or 0.0)
        transforms.append(
            t["RandAffined"](
                keys=keys,
                prob=affine_prob,
                rotate_range=(rot,),           # single in-plane rotation for 2D
                scale_range=(scale, scale),
                translate_range=(trans, trans),
                mode="bilinear",
                padding_mode="zeros",
                allow_missing_keys=True,
            )
        )

    if not transforms:
        return _identity
    return _MonaiDictAug(transforms)


def build_aug_3d(cfg):
    """Build a 3D geometric augmentation callable from a config ``augment`` block.

    ``cfg`` is the parsed ``augment:`` dict (or None). Returns a callable mapping a
    dict of ``(C, D, H, W)`` tensors to the augmented dict, or a no-op identity when
    disabled / missing. Layers MONAI ``RandFlipd`` (each axis), ``RandRotate90d``
    (axial H-W plane), a mild ``RandAffined`` and an optional very-mild
    ``Rand3DElasticd`` on top of the existing crop/resize done by the samplers.
    Geometry is synchronized across all keys (paired NAC/AC identical).
    """
    c = _merged(cfg, _DEFAULTS_3D)
    if not c.get("enabled", False):
        return _identity

    t = _import_dict_transforms()
    keys = list(_AUG_KEYS)
    transforms = []

    flip_prob = float(c.get("flip_prob", 0.0) or 0.0)
    if flip_prob > 0:
        # Spatial axes for a (C,D,H,W) tensor are 0 (D), 1 (H), 2 (W).
        for ax in (0, 1, 2):
            transforms.append(t["RandFlipd"](keys=keys, prob=flip_prob, spatial_axis=ax, allow_missing_keys=True))

    rot90_prob = float(c.get("rot90_prob", 0.0) or 0.0)
    if rot90_prob > 0:
        # Rotate within the axial plane (H, W) = spatial axes (1, 2).
        transforms.append(
            t["RandRotate90d"](keys=keys, prob=rot90_prob, max_k=3, spatial_axes=(1, 2), allow_missing_keys=True)
        )

    affine_prob = float(c.get("affine_prob", 0.0) or 0.0)
    if affine_prob > 0:
        rot = math.radians(float(c.get("affine_rotate_deg", 0.0) or 0.0))
        scale = float(c.get("affine_scale", 0.0) or 0.0)
        trans = float(c.get("affine_translate", 0.0) or 0.0)
        transforms.append(
            t["RandAffined"](
                keys=keys,
                prob=affine_prob,
                rotate_range=(rot, rot, rot),     # per-axis rotation for 3D
                scale_range=(scale, scale, scale),
                translate_range=(trans, trans, trans),
                # MONAI RandAffine resampling uses torch grid_sample, whose 3D
                # interpolation is selected with "bilinear" (trilinear is not a
                # valid grid_sample mode); this gives trilinear behavior in 3D.
                mode="bilinear",
                padding_mode="zeros",
                allow_missing_keys=True,
            )
        )

    elastic_prob = float(c.get("elastic_prob", 0.0) or 0.0)
    if elastic_prob > 0 and t.get("Rand3DElasticd") is not None:
        sigma = tuple(c.get("elastic_sigma_range", [1.0, 2.0]))
        mag = tuple(c.get("elastic_magnitude_range", [1.0, 2.0]))
        transforms.append(
            t["Rand3DElasticd"](
                keys=keys,
                prob=elastic_prob,
                sigma_range=sigma,
                magnitude_range=mag,
                mode="bilinear",
                padding_mode="zeros",
                allow_missing_keys=True,
            )
        )

    if not transforms:
        return _identity
    return _MonaiDictAug(transforms)
