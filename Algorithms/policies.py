from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy, BasePolicy
from stable_baselines3.common.policies import BaseModel

from Utils.tensors import clamp, to_numpy
from Utils.prices import european_call_delta


class CustomActor(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
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

        # action_dim = get_action_dim(self.action_space)
        action_dim = 1
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)
        self.flatten = nn.Flatten(-2, -1)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        action = self.mu(features)
        return self.flatten(action)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation)


class NoTransactionBandActor(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super(NoTransactionBandActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        # action_dim = get_action_dim(self.action_space)
        action_dim = 2
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)

        # Deterministic action
        self.mu = nn.Sequential(*actor_net)
        self.flatten = nn.Flatten(-2, -1)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        moneyness, expiry, volatility = obs[..., 0], obs[..., 1], obs[..., 2]
        prev_hedge = obs[..., 3]
        no_cost_delta = european_call_delta(to_numpy(moneyness),
                                            to_numpy(expiry),
                                            to_numpy(volatility)) * 2 - 1

        features = self.extract_features(obs)
        action = self.mu(features)
        no_cost_delta = th.tensor(no_cost_delta, dtype=th.float64).to(action)

        # TD Policy는 squash_output이 True이므로, action이 -1~1로 나와야 한다.
        # 즉,-1<action<1 이므로 -1<no_cost_delta<1 로 정해서 0.5배 해야함.
        lb = no_cost_delta - action[..., 0]
        ub = no_cost_delta + action[..., 1]
        hedge = clamp(prev_hedge, lb/2, ub/2)

        return hedge

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
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # action_dim = get_action_dim(self.action_space)
        action_dim = 1

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, th.unsqueeze(actions, -1)], dim=-1)
        return tuple(th.sum(q_net(qvalue_input), dim=-2) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        q_val = self.q_networks[0](th.cat([features, th.unsqueeze(actions, -1)], dim=-1))
        return th.sum(q_val, dim=-2)


class DoubleTD3Policy(TD3Policy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            actor: str = "mlp",
    ):
        self.actor_name = actor
        super(DoubleTD3Policy, self).__init__(observation_space,
                                              action_space,
                                              lr_schedule,
                                              net_arch=net_arch,
                                              activation_fn=activation_fn,
                                              features_extractor_class=features_extractor_class,
                                              features_extractor_kwargs=features_extractor_kwargs,
                                              normalize_images=normalize_images,
                                              optimizer_class=optimizer_class,
                                              optimizer_kwargs=optimizer_kwargs,
                                              n_critics=n_critics,
                                              share_features_extractor=share_features_extractor)

        self.critic2, self.critic2_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        if self.share_features_extractor:
            self.critic2 = self.make_critic(features_extractor=self.actor.features_extractor)
            self.critic2_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            self.critic2 = self.make_critic(features_extractor=None)
            self.critic2_target = self.make_critic(features_extractor=None)

        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2.optimizer = self.optimizer_class(self.critic2.parameters(), lr=lr_schedule(1),
                                                      **self.optimizer_kwargs)

        self.critic2_target.set_training_mode(False)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        self.critic2.set_training_mode(mode)

    def _get_actor(self):
        if self.actor_name == 'mlp':
            return CustomActor
        elif self.actor_name == 'no-transaction band':
            return NoTransactionBandActor
        else:
            raise ValueError(f'actor name : {self.actor_name} not found')

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return self._get_actor()(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)
