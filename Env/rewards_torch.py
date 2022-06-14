import numpy as np
import torch as th

def raw_reward(pnl, **kwargs):
    return pnl

def mean_variance_reward(pnl, std_coeff=0.0):
    return pnl.mean() - std_coeff * pnl.std()

def pnl_entropic_reward(pnl, aversion=1.0):
    return th.mean(-th.exp(-aversion*pnl), dim=-1)