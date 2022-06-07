import torch as th
import torch.nn as nn

from Algorithms.ddpg import config
from Algorithms import DDPG
from Env.feature_extractor import MarketObsExtractor

from pprint import pprint

default_config = config.default_config()
env_kwargs = default_config['env_kwargs']
model_kwargs = default_config['model_kwargs']
learn_kwargs = default_config['learn_kwargs']

# env_kwargs, model_kwargs, learn_kwargs = config.load_config('tmp_config.yaml')

actor_name = "mlp"  #@param ["mlp", "ntb"]

model_kwargs.update({
    'buffer_size': 60,
    'learning_starts': 50,
    'batch_size': 15,
    'std_coeff': env_kwargs['cost']
})

model_kwargs['policy_kwargs'].update({
    'features_extractor_class': MarketObsExtractor,
    'ntb_mode': False,
})

learn_kwargs.update({
    'total_timesteps': 100
})

config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)
config.save_config('tmp_config.yaml', env_kwargs, model_kwargs, learn_kwargs)

pprint(env_kwargs)
pprint(model_kwargs)
pprint(learn_kwargs)

_ = input()

model = DDPG(**model_kwargs)

print(model.policy)

_ = input()

model = model.learn(**learn_kwargs)

config_path = learn_kwargs['eval_log_path'] + '/config.yaml'
config.save_config(config_path, env_kwargs, model_kwargs, learn_kwargs)