import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sb

from Env.env_torch import BSMarketTorch

env_config = \
{'cost': 1e-2,
 'dividend': 0.0,
 'drift': 0.0,
 'freq': 1,
 'gen_name': 'gbm',
 'init_price': 1.0,
 'maturity': 30,
 'n_assets': 1000,
 'payoff': 'european',
 'payoff_coeff': 1.0,
 'period_unit': 365,
 'random_drift': False,
 'random_vol': False,
 'reward_fn': 'mean var',
 'reward_fn_kwargs': {},
 'reward_mode': 'pnl',
 'risk_free_interest': 0.0,
 'strike': 1.0,
 'volatility': 0.2}

env = BSMarketTorch(**env_config)

def step_reward(env, model):
    obs = env.get_obs()
    action = model(obs)
    obs, reward, done, info = env.step(action)
    return obs, env.raw_reward, done, info

def episode_reward(env, model):
    obs = env.reset()
    done, info = False, {}
    while not done:
        action = model(obs)
        obs, reward, done, info = env.step(action)

    return obs, env.raw_reward, done, info


from tqdm import tqdm
from Env.rewards_torch import mean_variance_reward, cvar_reward, pnl_entropic_reward

def train(env: BSMarketTorch, model: nn.Module, epochs: int = 100, reward_name='episode', loss_name='mean var', es=None,
          **kwargs):
    env.reset()
    losses = []
    progress = tqdm(range(epochs))
    optimizer = th.optim.Adam(model.parameters())
    done = False
    for epoch in progress:
        optimizer.zero_grad()
        if reward_name == 'step':
            obs, raw_reward, done, info = step_reward(env, model)
        elif reward_name == 'episode':
            obs, raw_reward, done, info = episode_reward(env, model)
        else:
            raise ValueError

        print(f'EPOCH: {epoch}')
        print(f"obs: {obs['obs'].requires_grad}")
        print(f"prev: {obs['prev_hedge'].requires_grad}")
        print(f"raw: {raw_reward.requires_grad}")

        if loss_name == 'mean var':
            loss = -mean_variance_reward(raw_reward, **kwargs.get('loss_kwargs', {}))
        elif loss_name == 'cvar':
            loss = -cvar_reward(raw_reward, **kwargs.get('loss_kwargs', {}))
        elif loss_name == 'entropy':
            loss = -pnl_entropic_reward(raw_reward, **kwargs.get('loss_kwargs', {}))

        print(f"loss: {loss.requires_grad}")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if done:
            obs = env.reset()

        if es:
            es(loss, model)

            if es.early_stop:
                print("Early stopping")
                break

    return losses

from Utils.tensors import create_module, clamp, set_seed
from Utils.prices_torch import european_call_delta

class FFN(nn.Module):
    def __init__(self, in_features, net_arch, activation_fn=nn.ReLU, net_kwargs=None):
        super(FFN, self).__init__()
        mlp = create_module(in_features, 1, net_arch, activation_fn, squash_output=True, net_kwargs=net_kwargs)
        self.mlp = nn.Sequential(*mlp)
        self.flatten = nn.Flatten(0)

    def forward(self, obs):
        action = self.mlp(obs['obs'])
        action = self.flatten(action)
        return (action + 1.0) * 0.5


class NTB(nn.Module):
    def __init__(self, in_features, net_arch, activation_fn=nn.ReLU, net_kwargs=None):
        super(NTB, self).__init__()
        mlp = create_module(in_features, 2, net_arch, activation_fn, squash_output=False, net_kwargs=net_kwargs)
        self.mlp = nn.Sequential(*mlp)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten(0)

    def forward(self, obs):
        prev_hedge = obs['prev_hedge']
        action = self.mlp(obs['obs'])

        moneyness, expiry, volatility, drift = [obs['obs'][..., i] for i in range(4)]
        delta = european_call_delta(moneyness, expiry, volatility, drift).to(action)

        scaler = 2.0 - 1e-5
        delta_unscaled = (delta * scaler - scaler / 2).atanh()

        if th.isinf(delta_unscaled).any():
            raise ValueError('inf value passed!')

        lb = self.tanh(delta_unscaled - F.leaky_relu(action[..., 0]))   # [-inf, inf] - [0, inf] = [ -inf, inf]
        ub = self.tanh(delta_unscaled + F.leaky_relu(action[..., 1]))   # [-inf, inf] + [0, inf] = [-inf, inf]

        prev_hedge_unscaled = 2.0 * prev_hedge - 1.0
        action = clamp(prev_hedge_unscaled, lb, ub)

        action = self.flatten(action)

        return (action + 1.0) * 0.5

from pprint import pprint
set_seed()

env = BSMarketTorch(**env_config)
pprint(env_config)

bn = (nn.BatchNorm1d, 'bn')

model_config = {'in_features': 4,
                'net_arch': [32, bn, 32, bn, 32, bn, 32],
                'activation_fn': nn.ReLU,
                'net_kwargs': {'bn_kwargs':{'num_features': 32}}}

set_seed()
ffn = FFN(**model_config)
set_seed()
ntb = NTB(**model_config)

print(ffn)
print(ntb)

train_config = {'epochs': 5,
                # 'reward_name': 'episode ',
                'reward_name': 'step',
                'loss_name': 'entropy'
                }

ffn_loss = train(env, ffn, **train_config)
ntb_loss = train(env, ntb, **train_config)