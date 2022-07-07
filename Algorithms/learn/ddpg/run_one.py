import torch as th
import torch.nn as nn

from Algorithms.ddpg import config
from Algorithms.ddpg import DoubleDDPG
from Algorithms.ddpg.double_ddpg import QDDPG
from Env.buffers import FlatBuffer
from stable_baselines3.ddpg import DDPG

from pprint import pprint

env_kwargs, model_kwargs, learn_kwargs = config.load_config('tmp_config_flatten.yaml')

ntb_mode = True    #@param {type:"boolean"}
double_ddpg = False  #@param {type:"boolean"}

random_drift = False #@param {type:"boolean"}
random_vol = False  #@param {type:"boolean"}

env_kwargs.update({
    'drift': 0.0,
    'volatility': 0.2,
    'cost': 0.02,
    'reward_fn': 'mean var',
    'reward_fn_kwargs': {},
    'reward_mode': 'pnl',
    'random_drift': random_drift,
    'random_vol': random_vol,
})

# state_name = f"d{int(env_kwargs['drift']*10)}v{int(env_kwargs['volatility']*10)}"
state_name = f"flat_obs"
# state_name = f"vol_test"

def lr_schedule(left: float):
    return 1e-3 * (0.1 ** (1 - left ** 2))

model_kwargs.update({
    'buffer_size': 3000,
    'learning_starts': 200,
    'batch_size': 100,
    'learning_rate': lr_schedule,
    'mean_coeff': 1.0,
    'std_coeff': 1.0 if ntb_mode else 1.0
})

model_kwargs['policy_kwargs'].update({
    'n_critics': 1,
    'ntb_mode':ntb_mode
})

learn_kwargs.update({
    'total_timesteps': 10000
})

# del model_kwargs['std_coeff']

actor_net_kwargs = {'bn_kwargs': {'num_features':128}}
critic_net_kwargs = {'bn_kwargs': {'num_features': 129}}
if ntb_mode:

    model_kwargs['policy_kwargs'].update({
        'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 4],
                     'qf': [(nn.BatchNorm1d, 'bn'), 4,],
                     'qf2': [(nn.BatchNorm1d, 'bn'), 4,]},
        'actor_net_kwargs': actor_net_kwargs,
        'critic_net_kwargs': critic_net_kwargs,
    })

    model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
        'features_out': 128,
        'net_arch': [32]
    })

else:
    # model_kwargs['policy_kwargs'].update({
    #     'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 16],
    #                  'qf': [(nn.BatchNorm1d, 'bn'), 32],
    #                  'qf2': [(nn.BatchNorm1d, 'bn'), 24]},
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
        'net_arch': [32, 64],
        'flat_obs': True,
    })

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