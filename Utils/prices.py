import numpy as np
import torch as th
from scipy.stats import norm

from typing import Union, Tuple


def geometric_brownian_motion(n_paths: int, n_periods: int,
                              drift: Union[float, np.ndarray],
                              volatility: Union[float, np.ndarray],
                              init_price: Union[float, np.ndarray], dt: float) -> np.ndarray:

    # z = np.random.standard_normal((n_periods, n_paths))
    z = th.randn(size=(n_periods, n_paths)).numpy()
    z[0, :] = 0.0

    noise_term = volatility * np.sqrt(dt) * z.cumsum(axis=0)
    t = np.linspace(0, (n_periods - 1)*dt, n_periods).reshape(-1, 1)
    return init_price*np.exp((drift - 0.5 * volatility ** 2) * t + noise_term)


def european_option_d1(moneyness: np.ndarray,
                       expiry: np.ndarray,
                       volatility: float,
                       risk_free_interest: float=0.0,
                       dividend: float=0.0) -> np.ndarray:
    """
    Black-Scholes Model d1
    :param moneyness:
    :param expiry: T * dt
    :param volatility:
    :param risk_free_interest:
    """

    a = np.log(moneyness) + (risk_free_interest - dividend + 0.5 * volatility ** 2) * expiry
    b = volatility * np.sqrt(expiry)
    return a / b


def european_option_d2(moneyness: np.ndarray,
                       expiry: np.ndarray,
                       volatility: float,
                       risk_free_interest: float=0.0) -> np.ndarray:

    return european_option_d1(moneyness, expiry, volatility, risk_free_interest) - volatility * np.sqrt(expiry)


def european_call_price(moneyness: np.ndarray,
                        expiry: np.ndarray,
                        volatility: float,
                        risk_free_interest: float = 0.0,
                        strike: float = 1.0, dividend: float = 0.0,
                        delta_return: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    normal = norm(loc=0.0, scale=1.0)
    d1 = european_option_d1(moneyness, expiry, volatility, risk_free_interest)
    d2 = d1 - volatility * np.sqrt(expiry)

    delta = normal.cdf(d1) * np.exp(-dividend * expiry)
    price = moneyness * strike * delta - \
            strike * normal.cdf(d2) * np.exp(-risk_free_interest * expiry)

    return (price, delta) if delta_return else price


def european_call_delta(moneyness: np.ndarray,
                        expiry: np.ndarray,
                        volatility: float,
                        risk_free_interest: float = 0.0,
                        strike: float = 1.0, dividend: float = 0.0) -> np.ndarray:

    normal = norm(loc=0.0, scale=1.0)
    d1 = european_option_d1(moneyness, expiry, volatility, risk_free_interest)
    return normal.cdf(d1) * np.exp(-dividend * expiry)


def european_option_payoff(prices: np.ndarray, strike: float = 1.0) -> np.ndarray:
    return np.clip(prices[-1] - strike, 0.0, np.inf)


def lookback_option_payoff(prices: np.ndarray, strike: float = 1.0) -> np.ndarray:
    return np.clip(np.max(prices, axis=0) - strike, 0.0, np.inf)

def pnl_entropic_loss(pnl, aversion=1.0) -> th.Tensor:
    return -np.mean(-np.exp(-aversion*pnl), axis=-1)