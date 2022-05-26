from Algorithms.learn.utils import config
from Algorithms import DDPG
from Env.feature_extractor import MarketObsExtractor

from pprint import pprint

env_kwargs, model_kwargs, learn_kwargs = config.load_config('tmp_config.yaml')

model_kwargs.update({
    'buffer_size': 300,
    'learning_starts': 300,
    'batch_size': 15,
    'std_coeff': env_kwargs['cost']
})

model_kwargs['policy_kwargs'].update({
    'features_extractor_class': MarketObsExtractor
})

# model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
#     'net_arch': [32],
#     'features_out': 64
# })

learn_kwargs.update({
    'total_timesteps': 1500
})

config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)

pprint(env_kwargs)
pprint(model_kwargs)
pprint(learn_kwargs)

_ = input()

model = DDPG(**model_kwargs)
model = model.learn(**learn_kwargs)

config_path = learn_kwargs['eval_log_path'] + '/config.yaml'
config.save_config(config_path, env_kwargs, model_kwargs, learn_kwargs)