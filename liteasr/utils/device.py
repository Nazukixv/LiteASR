"""Device utilities."""

import torch


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    return obj
