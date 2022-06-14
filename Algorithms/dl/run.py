import numpy as np
import torch as th


import torch as th
from Utils.prices_torch import geometric_brownian_motion, european_call_price, european_call_delta
from Utils.tensors import set_seed

gbm_kwargs = dict(n_paths=1000,
                  n_periods=31,
                  drift=0.0,
                  volatility=0.2,
                  init_price=1.0,
                  dt=1/365)

set_seed(65)
prices = geometric_brownian_motion(**gbm_kwargs)

expiry = (gbm_kwargs['n_periods'] -1 - th.arange(len(prices))[:, None])*gbm_kwargs['dt']
expiry = th.where(expiry==0, th.tensor([1e-6]), expiry)
volatility = th.full_like(expiry, gbm_kwargs['volatility'])
options, delta = european_call_price(prices, expiry, volatility, delta_return=True)


import torch.nn.functional as F

def get_reward(prices, options, actions, cost=0.2):
    reward = 0
    i = 0
    prev_action = 0
    info = {'option_gain': 0,
            'asset_gain': 0}
    while i < len(prices) - 1:
        option_gain = options[i]-options[i+1]
        asset_gain = - cost*prices[i]*(actions[i] - prev_action) + actions[i]*(prices[i+1]-prices[i])
        # reward += (options[i+1]-options[i]) - cost*prices[i]*(actions[i] - prev_action) + actions[i]*(prices[i+1]-prices[i])
        reward += option_gain + asset_gain

        info['option_gain'] += option_gain.mean()
        info['asset_gain'] += asset_gain.mean()
        prev_action = actions[i]
        i += 1
    option_gain = options[i-1] -F.relu(options[-1] - 1.0)
    asset_gain = (-cost)*prev_action*prices[i]
    # reward += -F.relu(options[-1] - options[0]) + (1-cost)*prev_action*prices[i]
    reward += option_gain + asset_gain

    info['option_gain'] += option_gain.mean()
    info['asset_gain'] += asset_gain.mean()
    info['raw_reward'] = reward
    return reward.mean(), info

print(get_reward(prices, options, delta, cost=0.2)[0])
print(get_reward(prices, options, delta[1:], cost=0.2)[0])


from Env.env_torch import BSMarketTorch
set_seed(65)
env = BSMarketTorch(n_assets=1000, cost=1e-3, reward_fn='raw', reward_fn_kwargs={})