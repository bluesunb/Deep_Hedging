import gym
import torch as th
from torch import nn

from typing import List, Dict, Type
from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp

"""
feature_extractor는 Actor, Critic Network에 들어가는 obs를 미리 preprocessing 한다.
"""


class MarketObsExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.Space,
                 features_in: int,
                 features_out: int,
                 net_arch: Optional[List[int]],
                 activation_fn: nn.Module):

        super(MarketObsExtractor, self).__init__(observation_space,
                                                 features_dim=features_out)
        # batchNorm1d 는 분포의 단위를 2번째(idx:1) 차원으로 본다.
        # nn.BatchNorm1d(4)([[1,2,3,4],[5,6,7,8]]) = [[a...],[b...]]
        # nn.BatchNorm1d(3)([[[1,2,3,4],[5,6,7,8]], [[1,3,5,7],[2,4,6,8]]]) = [[[a,..],[a..]], [[b,..],[b,..]]]

        modules = []
        n_assets = observation_space.shape[0]

        if len(net_arch) > 0:
            modules.append(nn.BatchNorm1d(n_assets))
            modules.append(nn.Linear(features_in, net_arch[0]))
            modules.append(activation_fn())

        for idx in range(len(net_arch) - 1):
            # modules.append(nn.BatchNorm1d(num_features=net_arch[idx]))
            modules.append(nn.BatchNorm1d(n_assets))
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activation_fn())

        if features_out > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_in
            if modules:
                modules.append(nn.BatchNorm1d(n_assets))
            modules.append(nn.Linear(last_layer_dim, features_out))

        self.layers = nn.Sequential(*modules)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # observation : Dict[np.ndarray(bs ,n_assets, n_features)]
        # obs = th.unsqueeze(observations, dim=0)     # (1, bs, n_assets, n_features)
        # obs = self.bn(observations)
        # obs = self.layers(obs)
        return self.layers(observations)


class BatchNormExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.Space,
                 features_in: int,
                 features_out: int,
                 net_arch: Optional[List[int]],
                 activation_fn: Type[nn.Module]):
        super(BatchNormExtractor, self).__init__(observation_space=observation_space,
                                                 features_dim=observation_space.shape[0] * features_out)

        self.bn = nn.BatchNorm1d(observation_space.shape[0])
        self.mlp = nn.Sequential(*create_mlp(features_in, features_out, net_arch, activation_fn))
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.count = 0

    def forward(self, observations: th.Tensor) -> th.Tensor:
        out = self.bn(observations)
        out = self.mlp(out)
        self.count += 1
        return self.flatten(out)


