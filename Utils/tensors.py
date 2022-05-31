import numpy as np
import torch as th


def to_numpy(tensor: th.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()


def clamp(x, lb, ub) -> th.Tensor:
    if not isinstance(x, th.Tensor):
        x = th.tensor(x, dtype=th.float64)
    if not isinstance(ub, th.Tensor):
        ub = th.tensor(ub, dtype=th.float64)
    if not isinstance(lb, th.Tensor):
        lb = th.tensor(lb, dtype=th.float64)

    lb = lb.to(x)
    ub = ub.to(x)

    x = th.min(th.max(x, lb), ub)
    x = th.where(lb < ub, x, (lb + ub) / 2)
    return x