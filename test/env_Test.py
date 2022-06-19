import numpy as np

class Tmp:
    def predict(self, obs, deterministic=False):
        return np.full(1000, 0.5), None

from Utils.tensors import set_seed
from Env.env import BSMarketEval

env = BSMarketEval(n_assets=1000, cost=0.02, maturity=2)
tmp = Tmp()

set_seed()
pnl = env.eval(tmp, reward_mode='pnl', n=1)
set_seed()
cash = env.eval(tmp, reward_mode='cash', n=1)

print(pnl.mean())
print(cash.mean())