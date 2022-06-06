import numpy as np

from pprint import pprint
from Algorithms.ddpg import config
from Algorithms import DDPG


best_path = '../logs/tb_logs/ddpg_220526-1356_1'
env_kwargs, model_kwargs, learn_kwargs = config.load_config(best_path + '/config.yaml')
config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)

pprint(env_kwargs)
pprint(model_kwargs)
pprint(learn_kwargs)

env = model_kwargs['env']
model = DDPG(**model_kwargs)

total_pnl = env.pnl_eval(model)
print(f'total_pnl: (mean) {np.mean(total_pnl)},  (std) {np.std(total_pnl)}')

model = model.load(best_path + '/best_model')

total_pnl_after = env.pnl_eval(model)
print(f'total_pnl: (mean) {np.mean(total_pnl_after)},  (std) {np.std(total_pnl_after)}')

f = lambda x: np.mean(x) - 0.02 * np.std(x)
print(f'imporved: {(f(total_pnl_after) - f(total_pnl))/abs(f(total_pnl_after))}')