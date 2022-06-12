import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import random

from Utils.tensors import create_module, clamp, to_numpy, set_seed
from Utils.prices_torch import european_call_delta

class NoTransactionBandNet(nn.Module):
    def __init__(self, in_features=4, squash=False):
        super(NoTransactionBandNet, self).__init__()

        mlp = create_module(in_features, 2, [32, 32, 32, 32], activation_fn=nn.ReLU, squash_output=squash)
        self.mlp = nn.Sequential(*mlp)
        self.squash = squash

    def forward(self, obs, prev_hedge=None):
        """
        :param obs: (n_paths, 3)
        :param prev_hedge: (n_paths, 1)
        """
        log_moneyness, time_expiry, volatility = obs[:, 0], obs[:, 1], obs[:, 2]
        prev_hedge = obs[:, 3]      # [0, 1]
        no_cost_delta = european_call_delta(log_moneyness, time_expiry, volatility)   # [0, 1]

        band_width = self.mlp(obs)      # [-1, 1]
        lb = no_cost_delta - F.relu(band_width[:, 0])     # [-1, 1.a]
        ub = no_cost_delta + F.relu(band_width[:, 1])     # [-1, 1.a]

        if self.squash:
            prev_hedge_scaled = 2.0 * prev_hedge - 1.0
            hedge = 0.5*(clamp(prev_hedge_scaled, lb, ub)) + 0.5
        else:
            hedge = clamp(prev_hedge, lb, ub)

        return hedge


from Env.env_torch import BSMarket
def compute_pnl2(hedging_model: nn.Module,
                env: BSMarket,
                recurrent: bool=False) -> th.Tensor:

    total_pnl = 0.0
    obs = env.reset()
    done, info = False, {}
    while not done:
        action = hedging_model(obs)
        obs, reward, done, info = env.step(action)
        total_pnl += reward

    return total_pnl

def get_reward(pnl, coeff=0.2):
    # print(f'mean: {pnl.mean():.4f}\t\tstd:{pnl.std():.4f}')
    return pnl.mean() - coeff * pnl.std()

from tqdm import tqdm

def fit2(model: nn.Module,
         env: BSMarket,
         steps: int) ->list:

    optimizer = th.optim.Adam(model.parameters())

    loss_history = []
    progress = tqdm(range(steps))
    for epoch in progress:
        optimizer.zero_grad()
        pnl = compute_pnl2(model, env)
        loss = -th.mean(pnl) + 0.2 * th.std(pnl)
        # loss = pnl_entropic_loss(pnl)

        loss.backward()
        optimizer.step()

        progress.desc = f'Loss={loss:.5f}'
        loss_history.append(loss.item())

    return loss_history


set_seed()
env = BSMarket(50000, cost=1e-3, payoff_coeff=0.0)

set_seed()
model = NoTransactionBandNet(in_features=4, squash=True)

losses = fit2(model, env, steps=100)