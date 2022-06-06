from pprint import pprint
from Algorithms.sac import config
from Env.feature_extractor import MarketObsExtractor

from pprint import pprint

from stable_baselines3.sac import SAC

default_config = config.default_config()
env_kwargs = default_config['env_kwargs']
model_kwargs = default_config['model_kwargs']
learn_kwargs = default_config['learn_kwargs']

actor_name = "mlp"  #@param ["mlp", "ntb"]

model_kwargs.update({
    'buffer_size': 300,
    'learning_starts': 300,
    'batch_size': 15,
    # 'std_coeff': env_kwargs['cost']
})

model_kwargs['policy_kwargs'].update({
    'features_extractor_class': MarketObsExtractor,
    # 'actor': actor_name,
})

# if actor_name=='ntb':
#     model_kwargs['policy_kwargs'].update({
#         'net_arch': {'pi': [16, 16],  # actor net arch
#                      'qf': [8]}  # critic net arch
#     })
#
#     model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
#         'features_out': 16,
#         'net_arch': [32, 32]
#     })

learn_kwargs.update({
    'total_timesteps': 1500
})

config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)

pprint(env_kwargs)
pprint(model_kwargs)
pprint(learn_kwargs)

_ = input()

model = SAC(**model_kwargs)

model = model.learn(**learn_kwargs)

config_path = learn_kwargs['eval_log_path'] + '/config.yaml'
config.save_config(config_path, env_kwargs, model_kwargs, learn_kwargs)