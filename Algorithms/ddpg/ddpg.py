from typing import Any, Dict, Optional, Tuple, Type, Union, List

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.td3 import TD3

from Env.buffers import CustomReplayBuffer


class DoubleDDPG(TD3):
    """
    Deep Deterministic Policy Gradient (DDPG).

    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Note: we treat DDPG as a special case of its successor TD3.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Union[ReplayBuffer, CustomReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        mean_coeff: float = 1.0,
        std_coeff: float = 0.02,
    ):

        super(DoubleDDPG, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        self.mean_coeff = mean_coeff
        self.std_coeff = std_coeff

        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.critic2 = self.policy.critic2
        self.critic2_target = self.policy.critic2_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Optimizers
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic2.optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        actor_losses, critic_losses, critic2_losses = [], [], []
        mean_cost_losses, std_cost_losses = [], []
        actor_actions_mean = []
        actor_actions_std = []

        for _ in range(gradient_steps):
            # Increase num of updates
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)   # (n_batches, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                next_q2_values = th.cat(self.critic2_target(replay_data.next_observations, next_actions), dim=1)
                next_q2_values, _ = th.min(next_q2_values, dim=1, keepdim=True)
                target_q2_values = replay_data.rewards2 + (1 - replay_data.dones) * self.gamma * next_q2_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            current_q2_values = self.critic2(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            critic2_loss = sum([F.mse_loss(current_q2, target_q2_values) for current_q2 in current_q2_values])
            critic2_losses.append(critic2_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            self.critic2.optimizer.zero_grad()
            critic_loss.backward()
            critic2_loss.backward()
            self.critic.optimizer.step()
            self.critic2.optimizer.step()

            # Delayed Policy update
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                mean_cost_loss = -self.critic.q1_forward(replay_data.observations,
                                                         self.actor(replay_data.observations)).mean()

                # std_cost_loss = th.abs(
                #     self.critic2.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean() -
                #     mean_cost_loss ** 2
                # )
                # std_cost_loss = std_cost_loss.sqrt() * self.std_coeff
                # std_cost_loss = th.abs(0.002 -
                #     self.critic2.q1_forward(replay_data.observations, self.actor(replay_data.observations))
                # ).mean()

                std_cost_loss = F.mse_loss(
                    th.full_like(replay_data.rewards2, 0.002),
                    self.critic2.q1_forward(replay_data.observations, self.actor(replay_data.observations))
                )

                actor_loss = self.mean_coeff * mean_cost_loss + self.std_coeff * std_cost_loss
                actor_losses.append(actor_loss.item())
                mean_cost_losses.append(-mean_cost_loss.item())
                std_cost_losses.append(std_cost_loss.item())

                with th.no_grad():
                    actor_actions = self.actor(replay_data.observations)    # [bs, 1000]
                    actor_actions_mean.append(th.mean(actor_actions.mean(dim=-1)).item())
                    # actor_actions_std.append(th.mean(actor_actions.std(dim=-1)).item())
                    actor_actions_std.append(th.mean(actor_actions.max(dim=-1).values).item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.critic2.parameters(), self.critic2_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/mean_cost_loss", np.mean(mean_cost_losses))
            self.logger.record("train/std_cost_loss", np.mean(std_cost_losses))
            self.logger.record("train/action_mean", np.mean(actor_actions_mean))
            self.logger.record("train/action_std", np.mean(actor_actions_std))

        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/critic2_loss", np.mean(critic2_losses))

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "DDPG",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(DoubleDDPG, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        save_params = super(DoubleDDPG, self)._excluded_save_params()
        return save_params + ["critic2", "critic2_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, _ = super()._get_torch_save_params()
        state_dicts += ['critic2.optimizer']
        return state_dicts, _

