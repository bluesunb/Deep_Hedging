import torch as th
import torch.nn.functional as F

from typing import Union, Tuple


def geometric_brownian_motion(n_paths: int, n_periods: int,
                              drift: float, volatility: float, init_price: float, dt: float) -> th.Tensor:

    # z = np.random.standard_normal((n_periods, n_paths))
    z = th.randn(size=(n_periods, n_paths))
    z[0, :] = 0.0

    noise_term = volatility * (dt ** 0.5) * z.cumsum(dim=0)
    t = th.linspace(0, (n_periods - 1)*dt, n_periods)[:, None].to(noise_term)
    return init_price*th.exp((drift - 0.5 * volatility ** 2) * t + noise_term)


def european_option_d1(moneyness: th.Tensor,
                       expiry: th.Tensor,
                       volatility: float,
                       risk_free_interest: float=0.0,
                       dividend: float=0.0) -> th.Tensor:
    """
    Black-Scholes Model d1
    :param moneyness:
    :param expiry: T * dt
    :param volatility:
    :param risk_free_interest:
    """

    a = th.log(moneyness) + (risk_free_interest - dividend + 0.5 * volatility ** 2) * expiry
    b = volatility * th.sqrt(expiry)
    return a / b


def european_option_d2(moneyness: th.Tensor,
                       expiry: th.Tensor,
                       volatility: float,
                       risk_free_interest: float=0.0) -> th.Tensor:

    return european_option_d1(moneyness, expiry, volatility, risk_free_interest) - volatility * th.sqrt(expiry)


def european_call_price(moneyness: th.Tensor,
                        expiry: th.Tensor,
                        volatility: float,
                        risk_free_interest: float = 0.0,
                        strike: float = 1.0, dividend: float = 0.0,
                        delta_return: bool = False) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:

    normal = th.distributions.Normal(loc=0.0, scale=1.0)
    d1 = european_option_d1(moneyness, expiry, volatility, risk_free_interest)
    d2 = d1 - volatility * th.sqrt(expiry)

    delta = normal.cdf(d1) * th.exp(-dividend * expiry)
    price = moneyness * strike * delta - \
            strike * normal.cdf(d2) * th.exp(-risk_free_interest * expiry)

    return (price, delta) if delta_return else price


def european_call_delta(moneyness: th.Tensor,
                        expiry: th.Tensor,
                        volatility: float,
                        risk_free_interest: float = 0.0,
                        strike: float = 1.0, dividend: float = 0.0) -> th.Tensor:

    normal = th.distributions.Normal(loc=0.0, scale=1.0)
    d1 = european_option_d1(moneyness, expiry, volatility, risk_free_interest)
    return normal.cdf(d1) * th.exp(-dividend * expiry)


def european_option_payoff(prices: th.Tensor, strike: float = 1.0) -> th.Tensor:
    return F.relu(prices[-1] - strike)


def lookback_option_payoff(prices: th.Tensor, strike: float = 1.0) -> th.Tensor:
    return F.relu(th.max(prices, dim=0).values - strike)

def pnl_entropic_loss(pnl, aversion=1.0) -> th.Tensor:
    return -th.mean(-th.exp(-aversion*pnl), dim=-1)