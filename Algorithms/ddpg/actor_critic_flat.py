from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.td3.policies import BasePolicy

from Utils.prices_torch import european_call_delta
from Utils.tensors import clamp, create_module

class CustomFlatActor(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[Union[Tuple, int]],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            net_kwargs: Optional[Dict[str, Any]] = None,
            ntb_mode: bool = False,
    ):
        super(CustomFlatActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.ntb_mode = ntb_mode

        action_dim = 2 if ntb_mode else 1
        actor_net = create_module(features_dim, action_dim,
                                  net_arch, activation_fn, squash_output=False, net_kwargs=net_kwargs)
        self.mu = nn.Sequential(*actor_net)
        self.tanh = nn.Tanh()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super(CustomFlatActor, self)._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                ntb_mode=self.ntb_mode,
            )
        )
        return data

    def ntb_forward(self, obs: th.Tensor, action: th.Tensor, prev_hedge: th.Tensor):
        moneyness, expiry, volatility, drift = [obs[..., i] for i in range(4)]
        delta = european_call_delta(moneyness, expiry, volatility, drift).to(action)    # [0, 1]

        scaler = 2.0-1e-5
        delta_unscaled = (delta*scaler - scaler/2).atanh()

        if th.isinf(delta_unscaled).any():
            raise ValueError('inf value passed!')

        lb = self.tanh(delta - F.leaky_relu(action[..., 0]))   # [0, 1] - [-0.a, 1] = [-1, 1.a]
        ub = self.tanh(delta + F.leaky_relu(action[..., 1]))   # [0, 1] + [-0.a, 1] = [-0.a, 1.a]

        prev_hedge_unscaled = 2.0 * prev_hedge - 1.0
        action = clamp(prev_hedge_unscaled, lb, ub)

        return action

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        action = self.mu(features)

        if self.ntb_mode:
            action = self.ntb_forward(obs['obs'], action, obs['prev_hedge'])
        else:
            action = self.tanh(action)

        return action

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation)

class CustomFlatCritic(BaseModel):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            net_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []

        action_dim = get_action_dim(action_space)   # 1
        for idx in range(n_critics):
            q_net = create_module(features_dim + action_dim, 1,
                                  net_arch, activation_fn, net_kwargs=net_kwargs)
            q_net = nn.Sequential(*q_net)

            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=-1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs)
        q_val = self.q_networks[0](th.cat([features, actions], dim=-1))
        return q_val