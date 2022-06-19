from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, create_sde_features_extractor
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim

from Utils.prices_torch import european_call_delta
from Utils.tensors import create_module, clamp, to_numpy

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class CustomActor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
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

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        self.ntb_mode = ntb_mode

        action_dim = 1
        mu_action_dim = 2 if ntb_mode else 1
        flatten_action_dim = get_action_dim(self.action_space)
        # latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        latent_pi_net = create_module(features_dim, -1, net_arch, activation_fn, net_kwargs=net_kwargs)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = [dim for dim in net_arch if isinstance(dim, int)][-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            latent_sde_dim = last_layer_dim
            # Separate features extractor for gSDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                    features_dim, sde_net_arch, activation_fn
                )

            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=latent_sde_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(flatten_action_dim)
            self.mu = nn.Sequential(nn.Linear(last_layer_dim, mu_action_dim))
            # self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Sequential(nn.Linear(last_layer_dim, 1))
            # self.log_std = nn.Linear(last_layer_dim, action_dim)

        self.flatten = nn.Flatten(-2)
        # self.tanh = nn.Tanh()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                sde_net_arch=self.sde_net_arch,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.ntb_mode:

            moneyness, expiry, volatility, prev_hedge = [obs[..., i] for i in range(self.observation_space.shape[-1])]
            delta = european_call_delta(moneyness, expiry, volatility).to(mean_actions)
            delta = ((2-2e-6) * delta - (1-1e-6)).atanh()       # [-1, 1].atanh

            lb = delta - F.leaky_relu(mean_actions[..., 0])  # [-1, 1].atanh - [0, inf]
            ub = delta + F.leaky_relu(mean_actions[..., 1])  # [-1, 1].atanh + [0, inf]

            prev_hedge_scaled = (2-2e-6) * prev_hedge - (1-1e-6)  # [-1, 1]
            mean_actions = clamp(prev_hedge_scaled.atanh(), lb, ub)  # [-1, 1].atanh => action_dist.log_prob 에서 tanh 적용
        else:
            mean_actions = self.flatten(mean_actions)

        if self.use_sde:
            latent_sde = latent_pi
            if self.sde_features_extractor is not None:
                latent_sde = self.sde_features_extractor(features)
            return mean_actions, self.log_std, dict(latent_sde=latent_sde)
        # Unstructured exploration (Original implementation)
        log_std = self.flatten(self.log_std(latent_pi))
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)


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

        # action_dim = get_action_dim(self.action_space)
        action_dim = 1

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []

        for idx in range(n_critics):
            q_net = create_module(features_dim + action_dim, 1,
                                  net_arch, activation_fn, net_kwargs=net_kwargs)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, th.unsqueeze(actions, -1)], dim=-1)
        return tuple(th.mean(q_net(qvalue_input), dim=-2) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        q_val = self.q_networks[0](th.cat([features, th.unsqueeze(actions, -1)], dim=-1))
        return th.mean(q_val, dim=-2)