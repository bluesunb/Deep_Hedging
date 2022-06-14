import numpy as np
import torch as th
import gym
import gym.spaces as spaces

from pprint import pprint

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from typing import Dict, Any, List, Callable, Optional

from Env import rewards_torch
from Utils.prices_torch import european_option_payoff, lookback_option_payoff
from Utils.prices_torch import geometric_brownian_motion, european_call_price, european_call_delta
from Utils.tensors import to_tensor, to_numpy

# from stable_baselines3.ddpg import DDPG
# from stable_baselines3.ddpg.policies import MlpPolicy


class BSMarketTorch(gym.Env):
    def __init__(self,
                 n_assets: int,
                 cost: float,
                 maturity: int = 30,
                 freq: float = 1,
                 period_unit: int = 365,
                 drift: float = 0.0,
                 volatility: float = 0.2,
                 init_price: float = 1.0,
                 risk_free_interest: float = 0.0,
                 strike: float = 1.0,
                 dividend: float = 0.0,
                 reward_fn: str = "mean var",
                 reward_fn_kwargs: Optional[Dict[str, Any]] = None,
                 payoff: str = "european",
                 gen_name: str = "gbm",
                 reward_mode: str = "pnl",
                 payoff_coeff: float = 1.0):

        super(BSMarketTorch, self).__init__()
        self.n_assets = n_assets
        self.transaction_cost = cost
        self.cost = self.transaction_cost

        self.period_unit = period_unit
        self.dt = freq / period_unit        # 0.5 / 365
        self.maturity = maturity / period_unit   # 30 / 365
        self.n_periods = int(maturity / freq) + 1     # 30 / 0.5 = 60 = self.maturity / self.dt + 1

        self.drift = drift
        self.volatility = volatility
        self.init_price = init_price

        self.risk_free_interest = risk_free_interest
        self.strike = strike
        self.dividend = dividend

        self.payoff = self.get_payoff_fn(payoff)

        self.reward_fn = self.get_reward_fn(reward_fn)
        self.reward_fn_kwargs = {} if reward_fn_kwargs is None else reward_fn_kwargs

        self.price_generator = self.get_price_generator(gen_name)
        self.reward_mode = reward_mode      # one of pnl, cashflow
        self.payoff_coeff = payoff_coeff

        self.now = 0

        self.underlying_prices = None
        self.option_prices = None

        self.hold = None
        # self.position = None
        self.delta = None

        self.reset()

        # moneyness, expiry, volatility, prev_hedge
        self.observation_space = spaces.Box(0, np.inf, shape=(n_assets, 4))
        self.action_space = spaces.Box(0, 1, shape=(n_assets, ))

        print("env 'BSMarket was created!")

    def seed(self, seed=None):
        np.random.seed(seed)
        th.manual_seed(seed)

    def reset(self, initialize="zero") -> th.Tensor:
        if initialize == 'zero':
            self.hold = th.zeros(self.n_assets)
        elif initialize == 'std':
            self.hold = th.randn(self.n_assets)

        self.now = 0
        # (n_periods, n_assets)
        self.underlying_prices = self.price_generator(self.n_assets,
                                                      self.n_periods,
                                                      self.drift,
                                                      self.volatility,
                                                      self.init_price,
                                                      self.dt)

        moneyness = self.underlying_prices/self.strike
        expiry = th.linspace(self.maturity, 0, self.n_periods)[:, None]
        expiry = th.where(expiry==0, th.tensor([1e-6]), expiry)

        self.option_prices, self.delta = european_call_price(
                                                 moneyness,
                                                 expiry,
                                                 self.volatility,
                                                 risk_free_interest=self.risk_free_interest,
                                                 strike=self.strike,
                                                 dividend=self.dividend,
                                                 delta_return=True)

        return self.get_obs()

    def get_obs(self) -> th.Tensor:
        moneyness = self.underlying_prices[self.now][:, None] / self.strike      # (n_periods+1, n_assets)
        expiry = th.full_like(moneyness, self.maturity - self.now*self.dt)
        expiry = th.where(expiry==0, th.tensor([1e-6]), expiry)
        volatility = th.full_like(moneyness, self.volatility)
        prev_hedge = self.hold[:, None]

        obs = th.cat([moneyness, expiry, volatility, prev_hedge], dim=-1)

        return obs

    def render(self, mode='human'):
        print(f'now: {self.now}')
        print(self.get_obs())

    def step(self, action: th.Tensor, render=False) -> GymStepReturn:
        """
        step - reward는 scalar로 전달되어야 하므로 n_assets의 reward에 대해 mean-variance measure를 취함
        """
        assert th.all(action >= 0) and th.all(action <= 1), f'min:{th.min(action)}, max:{th.max(action)}'
        assert action.shape == (self.n_assets, )

        step_return = None
        if self.reward_mode == 'pnl':
            step_return = self.pnl_step(action)

        if render:
            self.render()

        return step_return

    def pnl_step(self, action: th.Tensor) -> GymStepReturn:
        """
        action for holdings
        :param action: holdings : (0, 1)
        """
        done, info = False, {}

        # action = action.flatten()
        now_underlying, underlying = self.underlying_prices[self.now:self.now+2]
        now_option, option = self.option_prices[self.now:self.now+2]

        # transaction cost
        cost = self.transaction_cost * th.abs(action - self.hold) * now_underlying
        # gain from price movement
        price_gain = action * (underlying - now_underlying)
        payoff = self.payoff_coeff * (now_option - option)

        self.now += 1
        self.hold = action.detach()

        if self.now == self.n_periods - 1:  # 만약 다음 step이 maturity라면, 다음 step에는 처분밖에 못하므로
            # call option payoff를 빼주는 이유는 seller 입장에서 option 행사는 손해이기 때문
            payoff += now_option - self.payoff(self.option_prices, self.strike) - \
                      self.transaction_cost * underlying * action
            info['msg'] = 'MATURITY'
            done = True
        # else:
        #     # payoff = self.payoff_coeff*(option - now_option)
        #     payoff = now_option - option
        #     print(payoff.mean())

        raw_reward = payoff + price_gain - cost
        # reward = np.mean(raw_reward) - self.transaction_cost*np.std(raw_reward)
        reward = self.reward_fn(raw_reward, **self.reward_fn_kwargs)
        info['raw_reward'] = to_numpy(raw_reward)
        info['mean_square_reward'] = np.mean(info['raw_reward'] ** 2)

        return self.get_obs(), reward, done, info

    def pnl_step2(self, action: th.Tensor) -> GymStepReturn:
        """
        action for holdings
        :param action: holdings : (0, 1)
        """
        done, info = False, {}
        now_underlying = self.underlying_prices[self.now]
        now_option = self.option_prices[self.now]

        self.now += 1

        underlying = self.underlying_prices[self.now]
        option = self.option_prices[self.now]

        price_gain = action * (underlying - now_underlying)
        cost = self.transaction_cost * now_underlying * th.abs(action - self.hold)
        if self.now == self.n_periods - 1:
            payoff = now_option - self.payoff(self.underlying_prices, self.strike) - \
                     self.transaction_cost * action * underlying
            done = True
        else:
            payoff = now_option - option

        reward = payoff + price_gain - cost
        reward = self.reward_fn(reward, **self.reward_fn_kwargs)

        return self.get_obs(), reward, done, info

    @staticmethod
    def get_payoff_fn(payoff_name):
        if payoff_name == "european":
            return european_option_payoff

        elif payoff_name == "lookback":
            return lookback_option_payoff

        else:
            raise ValueError(f"payoff name not found: {payoff_name}")

    @staticmethod
    def get_price_generator(gen_name):
        if gen_name == "gbm":
            return geometric_brownian_motion
        else:
            raise ValueError(f"price generator name not found: {gen_name}")

    @staticmethod
    def get_reward_fn(reward_fn):
        if reward_fn == 'mean var':
            return rewards_torch.mean_variance_reward
        elif reward_fn == 'entropy':
            return rewards_torch.pnl_entropic_reward
        elif reward_fn == 'raw':
            return rewards_torch.raw_reward
        else:
            raise ValueError(f"reward function not found: {reward_fn}")


class BSMarketEvalTorch(BSMarketTorch):
    def __init__(self, **env_kwargs):
        super(BSMarketEvalTorch, self).__init__(**env_kwargs)

    def step(self, action: th.Tensor, render=False) -> GymStepReturn:
        new_obs, reward, done, info = super(BSMarketEvalTorch, self).step(action, render)
        reward -= self.transaction_cost * th.sqrt(info['mean_square_reward'] - reward ** 2)
        return new_obs, reward, done, info

    def pnl_eval(self, model=None):
        obs = self.reset()
        reward, done, info = 0, False, {}
        total_raw_reward = 0
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=False)
            else:
                action = self.action_space.sample()
            obs, reward, done, info = self.step(action)
            total_raw_reward += info['raw_reward']

        return total_raw_reward

    def delta_eval(self):
        obs = self.reset()
        reward, done, info = 0, False, {}
        total_raw_reward = 0
        i = 0
        while not done:
            # action = self.delta[i].copy()
            moneyness, expiry, volatility = [obs[..., j] for j in range(3)]
            action = european_call_delta(moneyness, expiry, volatility)
            assert np.all(abs(action - self.delta[i]) < 1e-5)
            obs, reward, done, info = self.step(action)
            total_raw_reward += info['raw_reward']
            i += 1

        return total_raw_reward

    def delta_eval2(self):
        obs = self.reset()
        reward, done, info = 0, False, {}
        total_raw_reward = 0
        i = 1
        while not done:
            action = self.delta[i].copy()
            # moneyness, expiry, volatility = [obs[..., j] for j in range(3)]
            # action = european_call_delta(moneyness, expiry, volatility)
            # assert np.all(abs(action - self.delta[i]) < 1e-5)
            obs, reward, done, info = self.step(action)
            total_raw_reward += info['raw_reward']
            i += 1

        return total_raw_reward