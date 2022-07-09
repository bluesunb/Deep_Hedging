from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import BasePolicy

from Algorithms.ddpg.actor_critic_flat import (
    CustomFlatActor, CustomFlatCritic
)


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], ...]:

    if isinstance(net_arch, list):
        actor_arch, critic_arch, critic2_arch = net_arch, net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        assert "qf2" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
        critic2_arch = net_arch['qf2']
    return actor_arch, critic_arch, critic2_arch


class QDDPGPolicy(BasePolicy):
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
        super(QDDPGPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            features_out = features_extractor_kwargs['features_out']
            net_arch = [(nn.BatchNorm1d, 'bn'), 16]
            actor_net_kwargs = {'bn_kwargs': {'num_features': features_out}}
            critic_net_kwargs = actor_net_kwargs.copy()

        actor_arch, critic_arch, critic2_arch = get_actor_critic_arch(net_arch)

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
                "share_features_extractor": share_features_extractor
            }
        )
        self.critic2_kwargs = self.critic_kwargs.copy()
        self.critic2_kwargs.update({
            "net_arch": critic2_arch
        })

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.critic2, self.critic2_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            self.critic2 = self.make_critic2(features_extractor=self.actor.features_extractor)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
            self.critic2_target = self.make_critic2(features_extractor=self.actor_target.features_extractor)

        else:
            self.critic = self.make_critic(features_extractor=None)
            self.critic2 = self.make_critic2(features_extractor=None)
            self.critic_target = self.make_critic2(features_extractor=None)
            self.critic2_target = self.make_critic2(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1),
                                                     **self.optimizer_kwargs)
        self.critic2.optimizer = self.optimizer_class(self.critic2.parameters(), lr=lr_schedule(1),
                                                      **self.optimizer_kwargs)

        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)
        self.critic2_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                acator_net_kwargs=self.actor_net_kwargs,
                critic_net_kwargs=self.critic_net_kwargs,
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
                ntb_mode=self.ntb_mode,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomFlatActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomFlatActor(**actor_kwargs, ntb_mode=self.ntb_mode).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomFlatCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomFlatCritic(**critic_kwargs).to(self.device)

    def make_critic2(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomFlatCritic:
        critic2_kwargs = self._update_features_extractor(self.critic2_kwargs, features_extractor)
        return CustomFlatCritic(**critic2_kwargs).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.critic2.set_training_mode(mode)
        self.training = mode

