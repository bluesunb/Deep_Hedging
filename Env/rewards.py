import numpy as np
import torch as th


def mean_variance_reward(pnl, std_coeff=0.0) -> float:
    return pnl.mean() - std_coeff * pnl.std()

def pnl_entropic_reward(pnl, aversion=1.0) -> float:
    return np.mean(-np.exp(-aversion*pnl), axis=-1)