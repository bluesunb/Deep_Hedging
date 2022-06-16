import torch as th
import torch.nn as nn

from Algorithms.ddpg import config
from Algorithms.ddpg import DoubleDDPG
from Algorithms.ddpg.double_ddpg import QDDPG
from stable_baselines3.ddpg import DDPG

from pprint import pprint

env_kwargs, model_kwargs, learn_kwargs = config.load_config('tmp_config.yaml')

ntb_mode = False
double_ddpg = True

env_kwargs.update({
    'n_assets': 1,
    'reward_fn': 'mean var',
    'reward_fn_kwargs': {},
    'reward_mode': 'pnl'
})

model_kwargs.update({
    'buffer_size': 100*100,
    'learning_starts': 100*10,
    'batch_size': 1000,
    'std_coeff': 0.05,
    'train_freq': (30, 'episode'),
})

model_kwargs['policy_kwargs'].update({
    'ntb_mode': ntb_mode,
    'double_ddpg': double_ddpg,
})

learn_kwargs.update({
    'total_timesteps': 1500
})

# del model_kwargs['std_coeff']

if ntb_mode:
    features_out = 64
    model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
        'features_out': features_out,
        'net_arch': [32]
    })

    actor_net_kwargs = {'bn_kwargs': {'num_features': features_out}}
    critic_net_kwargs = {'bn_kwargs': {'num_features': features_out+1}}

    model_kwargs['policy_kwargs'].update({
        'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 32, 32],
                     'qf': [(nn.BatchNorm1d, 'bn'), 2]},
        'actor_net_kwargs': actor_net_kwargs,
        'critic_net_kwargs': critic_net_kwargs,
    })

else:
    model_kwargs['policy_kwargs'].update({
        'net_arch': [],
    })

    model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
        'features_out': 2,
        'net_arch': [32, 64]
    })

model_kwargs['policy_kwargs']['one_asset'] = (env_kwargs['n_assets']==1)
if model_kwargs['policy_kwargs']['one_asset']:
    model_kwargs['replay_buffer_class'] = None
    model_kwargs['replay_buffer_kwargs'] = None

config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)
# config.save_config('tmp_config.yaml', env_kwargs, model_kwargs, learn_kwargs)

pprint(env_kwargs)
pprint(model_kwargs)
pprint(learn_kwargs)

# _ = input()

# model = DDPG(**model_kwargs)
model = QDDPG(**model_kwargs)
# model = DoubleDDPG(**model_kwargs)

print(model.policy)

# _ = input()

model = model.learn(**learn_kwargs)

config_path = learn_kwargs['eval_log_path'] + '/config.yaml'
config.save_config(config_path, env_kwargs, model_kwargs, learn_kwargs)