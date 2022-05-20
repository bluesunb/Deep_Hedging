from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy

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
    ):
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
        self.critic2.optimizer = self.optimizer_class(self.critic2.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        self.critic2_target.set_training_mode(False)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        self.critic2.set_training_mode(mode)

