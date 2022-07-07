import torch as th
import torch.nn as nn
import numpy as np

from Algorithms.ddpg import config
from Algorithms.ddpg import QDDPG, DoubleDDPG
from stable_baselines3.ddpg import DDPG

from pprint import pprint
from Env.buffers import CustomDictReplayBuffer

env_kwargs, model_kwargs, learn_kwargs = config.load_config('tmp_config.yaml')

from Env.buffers import CustomDictReplayBuffer

from Env.buffers import CustomDictReplayBuffer

ntb_mode = True  # @param {type:"boolean"}
double_ddpg = True  # @param {type:"boolean"}

random_drift = False  # @param {type:"boolean"}
random_vol = False  # @param {type:"boolean"}

drift = 0.4  # @param ["0.0", "0.4", "0.8", "1.2", "1.6"] {type:"raw"}
volatility = 0.2  # @param ["0.2", "0.4", "0.6", "0.8", "1.0"] {type:"raw"}
n_assets = 1000
env_kwargs.update({
    'n_assets': n_assets,
    'drift': drift,
    'volatility': volatility,
    'cost': 0.02,
    'reward_fn': 'mean var',
    'reward_fn_kwargs': {},
    'reward_mode': 'pnl',
    'random_drift': random_drift,
    'random_vol': random_vol,
    'ntb_mode': ntb_mode
})

# state_name = f"d{int(env_kwargs['drift']*10)}v{int(env_kwargs['volatility']*10)}"
state_name = f"drift_test"


# state_name = f"vol_test"
# state_name = f"mixed_test"

def lr_schedule(left: float):
    return 6e-4 * (0.1 ** (1 - left ** 2))


model_kwargs.update({
    'replay_buffer_class': CustomDictReplayBuffer,
    'buffer_size': 300,
    'learning_starts': 300,
    'learning_rate': lr_schedule,
    'batch_size': 15,
    'mean_coeff': 1.0,
    'std_coeff': 0.5
})

model_kwargs['policy_kwargs'].update({
    'ntb_mode': ntb_mode,
    'double_ddpg': double_ddpg,
    'n_critics': 1
})

learn_kwargs.update({
    'total_timesteps': 2000
})

# del model_kwargs['std_coeff']

actor_net_kwargs = {'bn_kwargs': {'num_features': env_kwargs['n_assets']}}
critic_net_kwargs = {'bn_kwargs': {'num_features': env_kwargs['n_assets']}}
if ntb_mode:

    # model_kwargs['policy_kwargs'].update({
    #     'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 4,4],
    #                  'qf': [(nn.BatchNorm1d, 'bn'), 2],
    #                  'qf2': [(nn.BatchNorm1d, 'bn'), 2]},
    #     'actor_net_kwargs': actor_net_kwargs,
    #     'critic_net_kwargs': critic_net_kwargs,
    # })

    # model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
    #     'features_out': 128,
    #     'net_arch': [32, 64]
    # })
    model_kwargs['policy_kwargs'].update({
        'net_arch': [],
        # 'actor_net_kwargs': actor_net_kwargs,
        # 'critic_net_kwargs': critic_net_kwargs,
    })

    model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
        'features_out': 32,
        'net_arch': [32, 32]
    })

else:
    # model_kwargs['policy_kwargs'].update({
    #     'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 16, 16, 4],
    #                  'qf': [(nn.BatchNorm1d, 'bn'), 48, 32],
    #                  'qf2': [(nn.BatchNorm1d, 'bn'), 32, 16]},
    #     'actor_net_kwargs': actor_net_kwargs,
    #     'critic_net_kwargs': critic_net_kwargs,
    # })

    # model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
    #     'features_out': 128,
    #     'net_arch': [32]
    # })
    model_kwargs['policy_kwargs'].update({
        'net_arch': [],
    })

    model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
        'features_out': 2,
        'net_arch': [32, 64]
    })

model_kwargs['policy_kwargs']['one_asset'] = (env_kwargs['n_assets'] == 1)

config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)

pprint(env_kwargs)
pprint(model_kwargs)
pprint(learn_kwargs)

# _ = input()

model = DoubleDDPG(**model_kwargs)
# model = DDPG(**model_kwargs)

print(model.policy)

# _ = input()

print(f'double_ddpg: {double_ddpg}')
if double_ddpg:
    model = DoubleDDPG(**model_kwargs)
else:
    model = QDDPG(**model_kwargs)

model = model.learn(**learn_kwargs)

# config_path = learn_kwargs['eval_log_path'] + '/config.yaml'
# config.save_config(config_path, env_kwargs, model_kwargs, learn_kwargs)