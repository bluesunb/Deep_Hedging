import numpy as np
import torch as th
import torch.nn as nn
import random

from typing import Type, List, Optional, Dict, Any

def set_seed(seed=42):
    th.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def to_numpy(tensor: th.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()

def to_tensor(numpy: np.ndarray) -> th.Tensor:
    return th.from_numpy(numpy).to(th.float32)


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


def create_module(input_dim: int,
                  output_dim: int,
                  net_arch: List[int],
                  activation_fn: Optional[Type[nn.Module]] = nn.ReLU,
                  squash_output: bool = False,
                  net_kwargs: Optional[Dict[str, Any]] = None, ) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
        - element in net_arch
            - int : Linear layer output dim
            - tuple(Type, str) : not-linear layer & name; used for search network kwargs in net_kwargs
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param net_kwargs: network kwargs of non-linear layer in net_arch
    :return:
    """
    modules = []
    last_layer_dim = input_dim
    for i, net in enumerate(net_arch):
        if isinstance(net, int):
            modules.append(nn.Linear(last_layer_dim, net))
            if i < len(net_arch) - 1 or output_dim > 0:
                if activation_fn is not None:
                    modules.append(activation_fn())
            last_layer_dim = net
        elif isinstance(net, tuple):
            net_class, name = net
            class_kwargs = net_kwargs.get(name + "_kwargs", {})
            modules.append(net_class(**class_kwargs))

    if output_dim > 0:
        modules.append(nn.Linear(last_layer_dim, output_dim))

    if squash_output:
        modules.append(nn.Tanh())

    return modules
