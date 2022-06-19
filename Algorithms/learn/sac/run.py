import torch as th
import torch.nn as nn

from Algorithms.sac import config
from Algorithms.sac import SAC

from pprint import pprint

env_kwargs, model_kwargs, learn_kwargs = config.load_config('tmp_config.yaml')

ntb_mode = True

env_kwargs.update({
    'drift': 1.0,
    'reward_fn': 'mean var',
    'reward_fn_kwargs': {},
    'reward_mode': 'pnl'
})

model_kwargs.update({
    'buffer_size': 300,
    'learning_starts': 300,
    'batch_size': 15,
    'target_entropy' : 50.0,      # target log_prob => log_prob는 entropy가 작을수록 커진다
    'ent_coef': "auto_1.0",
    'mean_coeff': 1.0,
    'std_coeff': 0.02,
})

model_kwargs['policy_kwargs'].update({
    'ntb_mode': ntb_mode,

})

learn_kwargs.update({
    'total_timesteps': 5000

})

actor_net_kwargs = {'bn_kwargs': {'num_features': env_kwargs['n_assets']}}
critic_net_kwargs = {'bn_kwargs': {'num_features': env_kwargs['n_assets']}}

if ntb_mode:
    model_kwargs['policy_kwargs'].update({
        'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 32, 32,],
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
        'net_arch': {'pi': [(nn.BatchNorm1d, 'bn'), 16, 16, None],
                     'qf': [(nn.BatchNorm1d, 'bn'), 16],
                     'qf2': [(nn.BatchNorm1d, 'bn'),4]},
        'actor_net_kwargs': actor_net_kwargs,
        'critic_net_kwargs': critic_net_kwargs,
    })

    model_kwargs['policy_kwargs']['features_extractor_kwargs'].update({
        'features_out': 64,
        'net_arch': [32,]
    })

config.reconstruct_config(env_kwargs, model_kwargs, learn_kwargs)
# config.save_config('tmp_config.yaml', env_kwargs, model_kwargs, learn_kwargs)

pprint(env_kwargs)
pprint(model_kwargs)
pprint(learn_kwargs)

# _ = input()

model = SAC(**model_kwargs)

print(model.policy)

# _ = input()

model = model.learn(**learn_kwargs)

config_path = learn_kwargs['eval_log_path'] + '/config.yaml'
config.save_config(config_path, env_kwargs, model_kwargs, learn_kwargs)