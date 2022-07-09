import numpy as np
import torch as th


def raw_reward(pnl, **kwargs):
    return pnl

def mean_variance_reward(pnl, std_coeff=0.0) -> float:
    return np.mean(pnl, axis=-1) - std_coeff * np.std(pnl, axis=-1)

def pnl_entropic_reward(pnl, aversion=1.0) -> float:
    return np.mean(-np.exp(-aversion*pnl), axis=-1)

def var_reward(pnl, ratio=0.95) -> float:
    losses = np.sort(-pnl, axis=-1)
    boundary = int(np.ceil(pnl.shape[-1] * ratio))
    return -losses[..., boundary]

def cvar_reward(pnl, ratio=0.95):
    losses = np.sort(-pnl, axis=-1)
    boundary = int(np.ceil(pnl.shape[-1] * ratio))
    return -np.mean(losses[..., boundary:], axis=-1)