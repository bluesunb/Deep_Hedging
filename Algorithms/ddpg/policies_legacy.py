from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    get_actor_critic_arch
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy, BasePolicy
from stable_baselines3.common.policies import BaseModel

from Utils.prices import european_call_delta
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
                                  net_arch, activation_fn, squash_output=(not ntb_mode), net_kwargs=net_kwargs)
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
                # ntb_mode=self.ntb_mode,
            )
        )
        return data

    def ntb_forward(self, obs: th.Tensor, action: th.Tensor):
        # prev_hedge = obs[..., 3]

        moneyness, expiry, volatility, prev_hedge = [obs[..., i] for i in range(4)]
        delta = european_call_delta(to_numpy(moneyness),
                                    to_numpy(expiry),
                                    to_numpy(volatility))
        delta = th.tensor(delta).to(action)

        scale = lambda x, low, high: 2.0 * ((x - low) / (high - low)) - 1.0

        lb = th.clamp(delta - th.tanh(F.leaky_relu(action[..., 0])), -1., 1.,)
        ub = th.clamp(delta + th.tanh(F.leaky_relu(action[..., 1])), -1., 1.,)
        prev_hedge_scaled = scale(prev_hedge, 0, 1)
        action = clamp(prev_hedge_scaled, lb, ub)
        return action

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        action = self.mu(features)

        if self.ntb_mode:
            action = self.ntb_forward(obs, action)
        else:
            action = self.flatten(action)

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


class DoubleDDPGPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            actor_net_kwargs: Optional[Dict[str, Any]] = None,
            critic_net_kwargs: Optional[Dict[str, Any]] = None,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            ntb_mode: bool = False,
    ):
        super(DoubleDDPGPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            # net_arch = {'pi': [(nn.BatchNorm1d, 'bn'), 16],
            #             'qf': [(nn.BatchNorm1d, 'bn'), 16]}
            net_arch = [(nn.BatchNorm1d, 'bn'), 16]
            actor_net_kwargs = {'bn_kwargs': {'num_features': get_action_dim(action_space)}}
            critic_net_kwargs = actor_net_kwargs.copy()

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.actor_net_kwargs = actor_net_kwargs
        self.critic_net_kwargs = critic_net_kwargs
        self.ntb_mode = ntb_mode

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "net_kwargs": actor_net_kwargs,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "net_kwargs": critic_net_kwargs,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.critic2, self.critic2_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            self.critic2 = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extactor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
            self.critic2_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic2 = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)
            self.critic2_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1),
                                                     **self.optimizer_kwargs)
        self.critic2.optimizer = self.optimizer_class(self.critic2.parameters(), lr=lr_schedule(1),
                                                      **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)
        self.critic2_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                acator_net_kwargs = self.actor_net_kwargs,
                critic_net_kwargs = self.critic_net_kwargs,
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
                ntb_mode=self.ntb_mode
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs, ntb_mode=self.ntb_mode).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.critic2.set_training_mode(mode)
        self.training = mode
