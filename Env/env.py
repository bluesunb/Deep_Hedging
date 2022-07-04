import numpy as np
import torch as th
import gym
import gym.spaces as spaces
from gym.utils import seeding

from pprint import pprint

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from typing import Dict, Any, List, Callable, Optional

from Env import rewards
from Utils.prices import european_option_payoff, lookback_option_payoff
from Utils.prices import geometric_brownian_motion, european_call_price, european_call_delta

# from stable_baselines3.ddpg import DDPG
# from stable_baselines3.ddpg.policies import MlpPolicy


class BSMarket(gym.Env):
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
                 payoff_coeff: float = 1.0,
                 random_drift: bool = False,
                 random_vol: bool = False,
                 ntb_mode: bool = False):

        self.random_drift = random_drift
        self.random_vol = random_vol

        super(BSMarket, self).__init__()
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
        self.reset_count = 0

        self.underlying_prices = None
        self.option_prices = None

        self.hold = None
        # self.position = None
        self.delta = None
        self.raw_reward = None

        self.ntb_mode = ntb_mode
        self.reset()

        # moneyness, expiry, volatility, prev_hedge
        # self.observation_space = spaces.Box(0, np.inf, shape=(n_assets, 4) if n_assets > 1 else (4, ))
        if self.ntb_mode:
            self.observation_space = \
                spaces.Dict({'obs': spaces.Box(0, np.inf, shape=(n_assets, 4) if n_assets > 1 else (4, )),
                             'prev_hedge': spaces.Box(0, 1, shape=(n_assets, ))})
        else:
            self.observation_space = \
                spaces.Dict({'obs': spaces.Box(0, np.inf, shape=(n_assets, 5) if n_assets > 1 else (5,))})

        self.action_space = spaces.Box(-1., 1., shape=(n_assets, ))

        print("env 'BSMarket was created!")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        th.manual_seed(seed)
        return [seed]

    def reset(self, initialize="zero") -> GymObs:
        if initialize == 'zero':
            self.hold = np.zeros(self.n_assets)
        elif initialize == 'std':
            self.hold = np.random.randn(self.n_assets)

        self.raw_reward = np.zeros(self.n_assets)

        if self.random_drift:
            # self.drift = np.random.choice(np.arange(6)*2/10)      # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            self.drift = self.reset_count // 5 % 5 * 0.2
        if self.random_vol:
            # self.volatility = np.random.choice(np.arange(1,6)*2/10)     # [0.2, 0.4, 0.6, 0.8, 1.0]
            self.volatility = 2*self.reset_count % 10 * 0.1 + 0.2

        self.now = 0
        self.reset_count += 1
        # (n_periods, n_assets)
        self.underlying_prices = self.price_generator(self.n_assets,
                                                      self.n_periods,
                                                      self.drift,
                                                      self.volatility,
                                                      self.init_price,
                                                      self.dt)

        moneyness = self.underlying_prices/self.strike
        expiry = np.linspace(self.maturity, 0, self.n_periods)[:, None]
        expiry[np.where(expiry == 0)] = 1e-6

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
        if self.n_assets > 1:
            moneyness = self.underlying_prices[self.now][:, None] / self.strike      # (n_periods+1, n_assets)
        else:
            moneyness = self.underlying_prices[self.now] / self.strike

        prev_hedge = self.hold.copy()
        expiry = np.full_like(moneyness, self.maturity - self.now*self.dt)
        expiry[np.where(expiry == 0)] = 1e-6

        if not isinstance(self.volatility, np.ndarray):
            volatility = np.full_like(moneyness, self.volatility)
        else:
            volatility = self.volatility.reshape(-1, 1)

        if not isinstance(self.drift, np.ndarray):
            drift = np.full_like(moneyness, self.drift)
        else:
            drift = self.drift.reshape(-1, 1)

        if self.ntb_mode:
            obs = {'obs': np.hstack([moneyness, expiry, volatility, drift]),
                   'prev_hedge': prev_hedge}
        else:
            obs = {'obs': np.hstack([moneyness, expiry, volatility, drift, prev_hedge[:, None]])}

        return obs

    def render(self, mode='human'):
        print(f'now: {self.now}')
        print(self.get_obs())

    def step(self, action: np.ndarray, render=False) -> GymStepReturn:
        """
        Note! reward must be returned in scalar
        """
        # assert np.all(action >= 0) and np.all(action <= 1), f'min:{np.min(action)}, max:{np.max(action)}'
        if len(action.shape) > 1:
            action = action.flatten()

        step_return = None
        if self.reward_mode == 'pnl':
            step_return = self.pnl_step(action)

        elif self.reward_mode == 'cash':
            step_return = self.cashflow_pnl(action)

        else:
            raise ValueError(f'Env step reward mode error: {self.reward_mode}')

        if render:
            self.render()

        return step_return

    def pnl_step(self, action: np.ndarray) -> GymStepReturn:
        """
        action for holdings
        :param action: holdings : (0, 1)
        """
        done, info = False, {}
        now_underlying, underlying = self.underlying_prices[self.now:self.now + 2]
        now_option, option = self.option_prices[self.now:self.now + 2]

        price_gain = action * (underlying - now_underlying)
        cost = self.transaction_cost * now_underlying * np.abs(action - self.hold)

        self.now += 1
        self.hold = action

        if self.now == self.n_periods - 1:
            payoff = now_option - self.payoff(self.underlying_prices, self.strike) - \
                     self.transaction_cost * underlying * action
            done = True
            info['msg'] = 'MATURITY'
        else:
            payoff = now_option - option

        raw_reward = payoff + price_gain - cost
        reward = self.reward_fn(raw_reward, **self.reward_fn_kwargs)
        new_raw_reward = raw_reward + self.raw_reward
        info['mean_square_reward'] = np.std(new_raw_reward) - np.std(self.raw_reward)
        self.raw_reward = new_raw_reward

        return self.get_obs(), reward, done, info

    def cashflow_pnl(self, action: np.ndarray) -> GymStepReturn:
        done, info = False, {}
        now_underlying, underlying = self.underlying_prices[self.now:self.now + 2]

        price_gain = now_underlying * (self.hold - action)
        cost = self.transaction_cost * now_underlying * np.abs(action - self.hold)
        payoff = 0

        self.now += 1
        self.hold = action

        if self.now == self.n_periods - 1:
            payoff = (1 - self.transaction_cost) * underlying * action - \
                     self.payoff(self.underlying_prices, self.strike) + self.option_prices[0]

            done = True
            info['msg'] = 'MATURITY'

        raw_reward = payoff + price_gain - cost
        reward = self.reward_fn(raw_reward, **self.reward_fn_kwargs)
        new_raw_reward = raw_reward + self.raw_reward
        info['mean_square_reward'] = np.std(new_raw_reward) - np.std(self.raw_reward)
        self.raw_reward = new_raw_reward

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
            return rewards.mean_variance_reward
        elif reward_fn == 'entropy':
            return rewards.pnl_entropic_reward
        elif reward_fn == 'raw':
            return rewards.raw_reward
        elif reward_fn == 'cvar':
            return rewards.cvar_reward
        else:
            raise ValueError(f"reward function not found: {reward_fn}")


class BSMarketEval(BSMarket):
    def __init__(self, **env_kwargs):
        super(BSMarketEval, self).__init__(**env_kwargs)

    def eval(self, model=None, reward_mode='cash', n=1):
        tmp = self.reward_mode
        self.reward_mode = reward_mode

        result = []
        for _ in range(n):
            obs = self.reset()
            reward, done, info = 0, False, {}
            while not done:
                if model:
                    action, _ = model.predict(obs, deterministic=False)
                else:
                    action = self.action_space.sample()
                obs, reward, done, info = self.step(action)
            result.append(self.raw_reward.copy())

        self.reward_mode = tmp

        return np.mean(result, axis=0)

    def delta_eval(self, reward_mode='cash', n=1):
        tmp = self.reward_mode
        self.reward_mode = reward_mode

        result = []
        for _ in range(n):
            obs = self.reset()
            reward, done, info = 0, False, {}
            i = 0
            while not done:
                # action = self.delta[i].copy()
                moneyness, expiry, volatility, drift = [obs['obs'][..., j] for j in range(4)]
                action = european_call_delta(moneyness, expiry, volatility, drift)
                # assert np.all(abs(action - self.delta[i]) < 1e-6)
                obs, reward, done, info = self.step(action)
                i += 1
            result.append(self.raw_reward)

        self.reward_mode = tmp

        return np.mean(result, axis=0)

    def delta_eval2(self, reward_mode='cash', n=1):
        tmp = self.reward_mode
        self.reward_mode = reward_mode

        result = []
        for _ in range(n):
            obs = self.reset()
            reward, done, info = 0, False, {}
            i = 1
            while not done:
                action = self.delta[i].copy()
                obs, reward, done, info = self.step(action)
                i += 1
            result.append(self.raw_reward)

        self.reward_mode = tmp

        return np.mean(result, axis=0)
