import numpy as np
import torch as th
import gym
import gym.spaces as spaces

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from typing import Dict, Any, Optional

from Env import rewards
from Utils.prices import european_option_payoff, lookback_option_payoff
from Utils.prices import geometric_brownian_motion, european_call_price, european_call_delta

class BSMarketFlat(gym.Env):
    def __init__(self,
                 cost: float,
                 maturity: int = 30,
                 freq: float =1 ,
                 period_unit: int = 365,
                 drift: float = 0.0,
                 volatility: float= 0.2,
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
                 random_vol: bool = False):

        self.random_drift = random_drift
        self.random_vol = random_vol

        super(BSMarketFlat, self).__init__()
        self.transaction_cost = cost

        self.period_unit = period_unit
        self.dt = freq / period_unit
        self.maturity = maturity / period_unit
        self.n_periods = int(maturity / freq) + 1

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
        self.reward_mode = reward_mode
        self.payoff_coeff = payoff_coeff

        self.now = 0

        self.underlying_prices = None
        self.option_prices = None

        self.hold = None
        self.delta = None

        self.observation_space = spaces.Dict({'obs': spaces.Box(0, np.inf, shape=(4, )),
                                              'prev_hedge': spaces.Box(0, 1, shape=(1, ))})
        self.action_space = spaces.Box(0, 1, shape=(1, ))

        self.reset()

        print("env 'BSMarket' was created!")

    def seed(self, seed=None):
        np.random.seed(seed)
        th.manual_seed(seed)

    def reset(self) -> GymObs:
        self.hold = np.zeros(self.action_space.shape)
        if self.random_drift:
            self.drift = np.random.choice(np.arange(6)*0.5)
        if self.random_vol:
            self.volatility = np.random.choice(np.arange(1, 6)*0.5)

        self.now = 0
        self.underlying_prices = self.price_generator(1,
                                                      self.n_periods,
                                                      self.drift,
                                                      self.volatility,
                                                      self.init_price,
                                                      self.dt)

        moneyness = self.underlying_prices / self.strike
        expiry = np.linspace(self.maturity, 0, self.n_periods)[:, None]
        expiry[np.where(expiry == 0)] = 1e-6

        self.option_prices, self.delta = european_call_price(
            moneyness, expiry, self.volatility,
            risk_free_interest=self.risk_free_interest if self.risk_free_interest else self.drift,
            strike=self.strike,
            dividend=self.dividend,
            delta_return=True
        )

        return self.get_obs()

    def get_obs(self) -> GymObs:
        moneyness = self.underlying_prices[self.now]
        prev_hedge = self.hold.copy()
        expiry = self.maturity - self.now * self.dt
        if expiry == 0:
            expiry = 1e-6

        obs = {'obs': np.hstack([moneyness, expiry, self.volatility, self.drift]),
               'prev_hedge': prev_hedge}

        return obs

    def step(self, action: np.ndarray, reward_mode='pnl') -> GymStepReturn:
        assert np.all(action >= 0) and np.all(action <= 1), f'min:{np.min(action)}, max:{np.max(action)}'

        if self.reward_mode == 'pnl':
            next_obs, raw_reward, done, info = self.step_pnl(action)
        elif self.reward_mode == 'cash':
            next_obs, raw_reward, done, info = self.step_cashflow(action)
        else:
            raise ValueError(f'Env step reward mode error: {self.reward_mode}')

        reward = raw_reward.mean()
        return next_obs, reward, done, info

    def step_pnl(self, action: np.ndarray) -> GymStepReturn:
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
        return self.get_obs(), raw_reward, done, info

    def step_cashflow(self, action: np.ndarray) -> GymStepReturn:
        done, info = False, {}
        now_underlying, underlying = self.underlying_prices[self.now:self.now + 2]

        price_gain = now_underlying * (self.hold - action)
        cost = self.transaction_cost * now_underlying * np.abs(action - self.hold)
        payoff = 0

        self.now += 1
        self.hold = action

        if self.now == self.n_periods - 1:
            payoff = (1-self.transaction_cost) * underlying * action - \
                     self.payoff(self.underlying_prices, self.strike) + self.option_prices[0]

        done = True
        info['msg'] = 'MATURITY'

        raw_reward = payoff + price_gain - cost
        return self.get_obs(), raw_reward, done, info

    def eval(self, model=None, reward_mode='pnl', n=1):
        tmp = self.reward_mode
        self.reward_mode = reward_mode
        result = []

        for _ in range(n):
            obs = self.reset()
            pnl, done, info = 0, False, {}
            while not done:
                if model:
                    action, _ = model.predict(obs, deterministic=False)
                else:
                    action = self.action_space.sample()
                obs, reward, done, info = self.step(action)
                pnl += reward

            result.append(pnl)

        self.reward_mode = tmp
        return np.mean(result, axis=0)

    def delta_eval(self, model=None, reward_mode='pnl', n=1):
        tmp = self.reward_mode
        self.reward_mode = reward_mode
        result = []

        for _ in range(n):
            obs = self.reset()
            pnl, done, info = 0, False, {}
            while not done:
                moneyness, expiry, volatility, drift = [obs['obs'][..., j] for j in range(4)]
                action = european_call_delta(moneyness, expiry, volatility, drift)
                obs, reward, done, info = self.step(action)
                pnl += reward

            result.append(pnl)

        self.reward_mode = tmp
        return np.mean(result, axis=0)

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