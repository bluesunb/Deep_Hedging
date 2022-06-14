import torch as th
import torch.nn as nn
import torch.nn.functional as F

from Utils.prices_torch import european_call_delta
from Utils.tensors import create_module, clamp, to_numpy

from typing import Optional, Dict, List, Union, Tuple, Any


class NoTransactionBand(nn.Module):
    def __init__(self,
                 features_in: int,
                 net_arch: Optional[List[Union[Tuple[nn.Module, str], int]]],
                 activation_fn: Optional[nn.Module]=nn.ReLU,
                 net_kwargs: Optional[Dict[str, Any]]=None,
                 squash: bool=False):
        super(NoTransactionBand, self).__init__()
        mlp = create_module(features_in, 2, net_arch,
                            activation_fn=activation_fn,
                            net_kwargs=net_kwargs,
                            squash_output=squash)

        self.squash = squash
        self.mlp = nn.Sequential(*mlp)

    def forward(self, obs, prev_hedge=None):
        if prev_hedge is None:
            prev_hedge = obs[..., 3]

        moneyness, expiry, volatility = [obs[..., i] for i in range(3)]

        delta = european_call_delta(moneyness, expiry, volatility)   # [0, 1]
        # delta = th.tensor(delta).to(actions)

        actions = self.mlp(obs)     # [-1, 1]
        lb = delta - F.leaky_relu(actions[..., 0])      # [-1, 1.a]
        ub = delta + F.leaky_relu(actions[..., 1])      # [-1, 1.a]

        if self.squash:
            prev_hedge_scaled = 2.0 * prev_hedge - 1.0
            hedge = 0.5 * (clamp(prev_hedge_scaled, lb, ub)) + 0.5
        else:
            hedge = clamp(prev_hedge, lb, ub)

        return hedge

    def predict(self, obs):
        return self.forward(obs), None
