import torch as th
import torch.nn as nn

from Algorithms.ddpg import config
from Algorithms.ddpg import DDPG

from pprint import pprint

env_kwargs, model_kwargs, learn_kwargs = config.load_config('tmp_config.yaml')

ntb_mode = True

model_kwargs.update({
    'buffer_size': 300,
    'learning_starts': 300,
    'batch_size': 15,
    'std_coeff': env_kwargs['cost']
})

model_kwargs['policy_kwargs'].update({
    'ntb_mode': ntb_mode
})

learn_kwargs.update({
    'total_timesteps': 1500
})

if ntb_mode:
    actor_net_kwargs = {'bn_kwargs': {'num_features': env_kwargs['n_assets']}}
    critic_net_kwargs = {'bn_kwargs': {'num_features': env_kwargs['n_assets']}}

    model_kwargs['policy_kwargs'].update({
        'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 32, 32],
                     'qf': [(nn.BatchNorm1d, 'bn'), 2]},
        'actor_net_kwargs': actor_net_kwargs,
        'critic_net_kwargs': critic_net_kwargs,
    })

    model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
        'features_out': 64,
        'net_arch': [32]
    })

else:
    model_kwargs['policy_kwargs'].update({
        'net_arch': [],
    })

    model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
        'features_out': 2,
        'net_arch': [32, 64]
    })

config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)
# config.save_config('tmp_config.yaml', env_kwargs, model_kwargs, learn_kwargs)

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