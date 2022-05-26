import numpy as np
import torch as th
import gym
import gym.spaces as spaces

from pprint import pprint

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from typing import Dict, Any, List

from Utils.prices import european_option_payoff, lookback_option_payoff
from Utils.prices import geometric_brownian_motion, european_call_price

from stable_baselines3.ddpg import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy


class BSMarket(gym.Env):
    def __init__(self,
                 n_assets: int,
                 cost: float,
                 n_periods: int = 30,
                 freq: float = 1,
                 period_unit: int = 365,
                 drift: float = 0.0,
                 volatility: float = 0.2,
                 init_price: float = 1.0,
                 risk_free_interest: float = 0.0,
                 strike: float = 1.0,
                 dividend: float = 0.0,
                 payoff: str = "european",
                 gen_name: str = "gbm",
                 reward_mode: str = "pnl"):

        super(BSMarket, self).__init__()
        self.n_assets = n_assets
        self.transaction_cost = cost
        self.cost = self.transaction_cost

        self.period_unit = period_unit
        self.n_periods = int(n_periods / freq)     # 30 / 0.5 = 60
        self.dt = freq / period_unit        # 0.5 / 365
        self.maturity = n_periods / period_unit  # 30 / 365

        self.drift = drift
        self.volatility = volatility
        self.init_price = init_price

        self.risk_free_interest = risk_free_interest
        self.strike = strike
        self.dividend = dividend

        self.payoff = self.get_payoff_fn(payoff)
        self.price_generator = self.get_price_generator(gen_name)
        self.reward_mode = reward_mode      # one of pnl, cashflow

        self.now = 0

        self.underlying_prices = None
        self.option_prices = None
        self.hedge = None
        self.delta = None

        self.reset()

        # moneyness, expiry, volatility, prev_hedge
        self.observation_space = spaces.Box(0, np.inf, shape=(n_assets, 4))
        self.action_space = spaces.Box(0, 1, shape=(n_assets, ))

        print("env 'BSMarket was created!")

    def seed(self, seed=None):
        np.random.seed(seed)
        th.manual_seed(seed)

    def reset(self, initialize="zero") -> GymObs:
        if initialize == 'zero':
            self.hedge = np.zeros(self.n_assets)
        elif initialize == 'std':
            self.hedge = np.random.randn(self.n_assets)

        self.now = 0
        # (n_periods, n_assets)
        self.underlying_prices = self.price_generator(self.n_assets,
                                                      self.n_periods+1,
                                                      self.drift,
                                                      self.volatility,
                                                      self.init_price,
                                                      self.dt)

        moneyness = self.underlying_prices/self.strike
        expiry = self.maturity - np.arange(len(self.underlying_prices))[:, None] * self.dt

        self.option_prices, self.delta = european_call_price(
                                                 moneyness,
                                                 expiry,
                                                 self.volatility,
                                                 risk_free_interest=self.risk_free_interest,
                                                 strike=self.strike,
                                                 dividend=self.dividend,
                                                 delta_return=True)
        return self.get_obs()

    def get_obs(self) -> GymObs:
        moneyness = self.underlying_prices[self.now][:, None] / self.strike      # (n_periods+1, n_assets)
        expiry = np.full_like(moneyness, (self.n_periods - self.now) * self.dt)
        volatility = np.full_like(moneyness, self.volatility)
        prev_hedge = self.hedge.copy()[:, None]

        obs = np.c_[moneyness, expiry, volatility, prev_hedge]

        return obs

    def render(self, mode='human'):
        print(f'now: {self.now}')
        print(self.get_obs())

    def step(self, action: np.ndarray, render=False) -> GymStepReturn:
        """
        step - reward는 scalar로 전달되어야 하므로 n_assets의 reward에 대해 mean-variance measure를 취함
        """
        assert np.all(action >= 0), f"action: {action} out of range"
        # assert action.shape == (self.n_assets, 1)

        step_return = None
        if self.reward_mode == 'pnl':
            step_return = self.pnl_step(action)

        if render:
            self.render()

        return step_return

    def pnl_step(self, action: np.ndarray) -> GymStepReturn:
        done, info = False, {}

        action = action.flatten()
        now_underlying, underlying = self.underlying_prices[self.now:self.now+2]
        now_option, option = self.option_prices[self.now:self.now+2]

        # transaction cost
        cost = self.transaction_cost * np.abs(action - self.hedge) * now_underlying
        # gain from price movement
        price_gain = action * (underlying - now_underlying)

        if self.now == self.n_periods - 1:  # 만약 다음 step이 maturity라면, 다음 step에는 처분밖에 못하므로
            # call option payoff를 빼주는 이유는 seller 입장에서 option 행사는 손해이기 때문
            payoff = - self.payoff(option, self.strike) - self.transaction_cost * underlying * action
            info['msg'] = 'MATURITY'
            done = True
        else:
            payoff = option - now_option

        self.now += 1
        self.hedge = action

        raw_reward = payoff + price_gain - cost
        reward = np.mean(raw_reward) - 2.0*np.std(raw_reward)
        info['raw_reward'] = raw_reward

        return self.get_obs(), reward, done, info

    def pnl_eval(self, model):
        obs = self.reset()
        reward, done, info = 0, False, {}
        total_raw_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            action = model.policy.unscale_action(action)
            obs, reward, done, info = self.step(action)
            total_raw_reward += info['raw_reward']

        return total_raw_reward

    def delta_eval(self):
        obs = self.reset()
        reward, done, info = 0, False, {}
        total_raw_reward = 0
        i = 1
        while not done:
            action = self.delta[i].copy()
            obs, reward, done, info = self.step(action)
            total_raw_reward += info['raw_reward']
            i += 1

        return total_raw_reward

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