import torch as th
import torch.nn as nn

from Algorithms.ddpg import config
from Algorithms.ddpg import DoubleDDPG
from stable_baselines3.ddpg import DDPG

from pprint import pprint
from Env.buffers import CustomDictReplayBuffer

env_kwargs, model_kwargs, learn_kwargs = config.load_config('tmp_config.yaml')

ntb_mode = False
double_ddpg = True

env_kwargs.update({
    'drift': 1.0,
    'reward_fn': 'mean var',
    'reward_fn_kwargs': {},
    'reward_mode': 'pnl'
})

model_kwargs.update({
    'replay_buffer_class': CustomDictReplayBuffer,
    'buffer_size': 300,
    'learning_starts': 300,
    'batch_size': 15,
    'mean_coeff': 1.5,
    'std_coeff': 0.02 if ntb_mode else 0.01
})

model_kwargs['policy_kwargs'].update({
    'ntb_mode': ntb_mode,
    'double_ddpg': double_ddpg,
    'n_critics': 2,
})

learn_kwargs.update({
    'total_timesteps': 2000
})

# del model_kwargs['std_coeff']

if ntb_mode:
    actor_net_kwargs = {'bn_kwargs': {'num_features': env_kwargs['n_assets']}}
    critic_net_kwargs = {'bn_kwargs': {'num_features': env_kwargs['n_assets']}}

    model_kwargs['policy_kwargs'].update({
        'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 32, 32],
                     'qf': [(nn.BatchNorm1d, 'bn'), 16],
                     'qf2': [(nn.BatchNorm1d, 'bn'), 4]},
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

model_kwargs['policy_kwargs']['one_asset'] = (env_kwargs['n_assets']==1)

config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)
# config.save_config('tmp_config.yaml', env_kwargs, model_kwargs, learn_kwargs)

pprint(env_kwargs)
pprint(model_kwargs)
pprint(learn_kwargs)

# _ = input()

model = DoubleDDPG(**model_kwargs)
# model = DDPG(**model_kwargs)

print(model.policy)

# _ = input()

model = model.learn(**learn_kwargs)

config_path = learn_kwargs['eval_log_path'] + '/config.yaml'
config.save_config(config_path, env_kwargs, model_kwargs, learn_kwargs)