import torch


def inflate_conv2d_to_3d(weight_2d, kernel_depth: int) -> torch.Tensor:
    if weight_2d.ndim != 4:
        raise ValueError("Expected 4D Conv2d weight.")
    out_ch, in_ch, k_h, k_w = weight_2d.shape
    weight_3d = torch.zeros((out_ch, in_ch, kernel_depth, k_h, k_w), dtype=weight_2d.dtype)
    center = kernel_depth // 2
    weight_3d[:, :, center, :, :] = weight_2d
    return weight_3d


def map_state_dict_2d_to_3d(state_2d, state_3d):
    mapped = {}
    missing = []
    for name, tensor in state_2d.items():
        if name not in state_3d:
            missing.append(name)
            continue
        target = state_3d[name]
        if tensor.shape == target.shape:
            mapped[name] = tensor
            continue
        if tensor.ndim == 4 and target.ndim == 5:
            mapped[name] = inflate_conv2d_to_3d(tensor, target.shape[2])
            continue
        missing.append(name)

    return mapped, missing
