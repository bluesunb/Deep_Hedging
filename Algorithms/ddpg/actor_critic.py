from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.td3.policies import BasePolicy

from Utils.prices_torch import european_call_delta
from Utils.tensors import clamp, to_numpy, create_module


class CustomActor(BasePolicy):
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
        super(CustomActor, self).__init__(
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
                                  net_arch, activation_fn, squash_output=True, net_kwargs=net_kwargs)
        self.mu = nn.Sequential(*actor_net)
        self.flatten = nn.Flatten(-2)  # due to action_dim = 1 so last dim of mu(action) will 1

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super(CustomActor, self)._get_constructor_parameters()

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
        # prev_hedge = obs[..., 3]

        moneyness, expiry, volatility, drift = [obs[..., i] for i in range(4)]
        delta = european_call_delta(moneyness, expiry, volatility, drift).to(action)
        # assert th.all(delta - european_call_delta(moneyness, expiry, volatility) < 1e-6)
        # delta = th.tensor(delta).to(action)     # [0, 1]

        lb = delta - F.leaky_relu(action[..., 0])       # [-1, 1]
        ub = delta + F.leaky_relu(action[..., 1])

        prev_hedge_scaled = 2.0 * prev_hedge - 1.0      # [-1, 1]
        action = clamp(prev_hedge_scaled, lb, ub)       # [-1, 1]

        return th.clip(action, -1., 1.)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        action = self.mu(features)

        if self.ntb_mode:
            action = self.ntb_forward(obs['obs'], action, obs['prev_hedge'])
        else:
            action = self.flatten(action)

        return action

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation)


class FlattenActor(BasePolicy):
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
        super(FlattenActor, self).__init__(
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
                                  net_arch, activation_fn, squash_output=True, net_kwargs=net_kwargs)
        self.mu = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super(FlattenActor, self)._get_constructor_parameters()

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

    def ntb_forward(self, obs: th.Tensor, action: th.Tensor):
        moneyness, expiry, volatility, prev_hedge = [obs[..., [i]] for i in range(4)]
        delta = european_call_delta(moneyness, expiry, volatility).to(action)
        # assert th.all(delta - european_call_delta(moneyness, expiry, volatility) < 1e-6)
        # delta = th.tensor(delta).to(action)     # [0, 1]

        lb = delta - F.leaky_relu(action[..., [0]])       # [-1, 1]
        ub = delta + F.leaky_relu(action[..., [1]])

        prev_hedge_scaled = 2.0 * prev_hedge - 1.0      # [-1, 1]
        hedge = clamp(prev_hedge_scaled, lb, ub)       # [-1, 1]

        return th.clip(hedge, -1., 1.)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        action = self.mu(features)

        if self.ntb_mode:
            action = self.ntb_forward(obs, action)

        return action

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation)


class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.
    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

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

        flatten_action_dim = get_action_dim(action_space)
        for idx in range(n_critics):
            q_net = create_module(features_dim + 1, 1,
                                  net_arch, activation_fn, net_kwargs=net_kwargs)
            # q_net.append(nn.Flatten(-2))
            # q_net.append(nn.Linear(flatten_action_dim, 1))
            q_net = nn.Sequential(*q_net)

            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, th.unsqueeze(actions, -1)], dim=-1)
        # return tuple(q_net(qvalue_input) for q_net in self.q_networks)
        return tuple(th.mean(q_net(qvalue_input), dim=-2) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs)
        q_val = self.q_networks[0](th.cat([features, th.unsqueeze(actions, -1)], dim=-1))
        # return q_val
        return th.mean(q_val, dim=-2)


class FlattenCritic(BaseModel):
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

        for idx in range(n_critics):
            q_net = create_module(features_dim + 1, 1,
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
        return self.q_networks[0](th.cat([features, actions], dim=-1))
