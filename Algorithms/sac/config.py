import torch as th
import torch.nn as nn
import yaml

from easydict import EasyDict
from datetime import datetime
from typing import Optional, Tuple, List

from Env.env import BSMarket
from Env.feature_extractor import MarketObsExtractor
from Algorithms.sac.callbacks import ReportCallbacks
from Algorithms.sac.policies import SACPolicy

from stable_baselines3.common.noise import NormalActionNoise

MODEL_NAME = "sac_"

def easydict_to_dict(edict):
    if not isinstance(edict, (dict, EasyDict)):
        return edict
    return {k: easydict_to_dict(v) for k, v in edict.items()}


def lr_schedule(left: float):
    # return 5e-3 * (0.95 ** (30 * (1 - left)))
    return 5e-3 * (0.1 ** (1 - left**2))


def get_now(f_string='%y%m%d-%H%M'):
    return datetime.now().strftime(f_string)

def get_config_copy(*args: List[dict]):
    args_copy = [arg.copy() for arg in args]
    return args_copy

def save_config(path: str,
                env_kwargs_src: dict,
                model_kwargs_src: dict,
                learn_kwargs_src: dict,
                **kwargs) -> None:

    env_kwargs, model_kwargs, learn_kwargs =\
        get_config_copy(env_kwargs_src, model_kwargs_src, learn_kwargs_src)

    # custom env 또는 custom class는 class type + class kwargs로 따로 저장된다.
    def return_type(config, key):
        obj = config[key]
        if not isinstance(obj, (type, str)):
            config[key] = type(obj)

            log_str = f'{obj} will be save as name. '
            if key + '_kwargs' not in kwargs:
                log_str += f"{key}_kwargs not in kwargs!"
            print(log_str)

    return_type(model_kwargs, 'env')
    return_type(learn_kwargs, 'eval_env')
    return_type(learn_kwargs, 'callback')

    config = {'env_kwargs': env_kwargs,
              'model_kwargs': model_kwargs,
              'learn_kwargs': learn_kwargs,
              'kwargs': kwargs}

    config = easydict_to_dict(config)

    with open(path, 'w') as f:
        f.write(yaml.dump(config))

    print(f'{path} was saved.')


def load_config(path: Optional[str] = None) -> Tuple[dict, ...]:
    if path is None:
        config = default_config()
    else:
        config = yaml.load(open(path, 'r'), Loader=yaml.Loader)

    env_kwargs = config['env_kwargs']
    model_kwargs = config['model_kwargs']
    learn_kwargs = config['learn_kwargs']
    kwargs = config['kwargs']

    # restore env
    model_kwargs['env'] = model_kwargs['env'](**env_kwargs)

    eval_env_kwargs = kwargs.get('eval_env_kwargs', env_kwargs)
    learn_kwargs['eval_env'] = learn_kwargs['eval_env'](**eval_env_kwargs)

    # restore callbacks
    callback_kwargs = kwargs.get('callback_kwargs', dict())
    learn_kwargs['callback'] = learn_kwargs['callback'](**callback_kwargs)

    return env_kwargs, model_kwargs, learn_kwargs


def reconstruct_config(env_kwargs, model_kwargs, learn_kwargs, **kwargs):
    env = type(model_kwargs['env'])(**env_kwargs)
    model_kwargs['env'] = env
    print(f"model_kwargs['env']: {env}")

    eval_env_kwargs = kwargs.get('eval_env_kwargs', env_kwargs)
    eval_env = type(learn_kwargs['eval_env'])(**eval_env_kwargs)
    learn_kwargs['eval_env'] = eval_env
    print(f"learn_kwargs['eval_env']: {eval_env}")

    learn_kwargs['tb_log_name'] = MODEL_NAME + get_now()
    print(f"learn_kwargs['tb_log_name']: {learn_kwargs['tb_log_name']}")

    learn_kwargs['eval_log_path'] = \
        model_kwargs['tensorboard_log'] + '/' + learn_kwargs['tb_log_name'] + '_1'
    print(f"learn_kwargs['eval_log_path']: {learn_kwargs['eval_log_path']}")


def default_config() -> dict:
    env_kwargs = {'n_assets': 1000,
                  'cost': 0.02,
                  'n_periods': 30,
                  'freq': 1,
                  'period_unit': 365,
                  'drift': 0.0,
                  'volatility': 0.2,
                  'init_price': 1.0,
                  'risk_free_interest': 0.0,
                  'strike': 1.0,
                  'dividend': 0.0,
                  'payoff': 'european',
                  'gen_name': 'gbm',
                  'reward_mode': 'pnl'}

    env = BSMarket(**env_kwargs)
    eval_env = BSMarket(**env_kwargs)

    features_extractor_kwargs = {'features_in': 4,
                                 'features_out': 32,
                                 'net_arch': [32],
                                 'activation_fn': nn.ReLU,
                                 'last_activation_fn': nn.ReLU}
    optimizer_kwargs = None

    policy_kwargs = {'net_arch': [16],  # None으로 설정하면 deafult net arch가 설정됨
                     'activation_fn': nn.ReLU,
                     'use_sde': False,      # gSDE
                     'log_std_init': -3,    # gSDE
                     'sde_net_arch': None,  # gSDE
                     'use_expln': False,    # gSDE
                     'clip_mean': 2.0,      # gSDE
                     'features_extractor_class': MarketObsExtractor,
                     'features_extractor_kwargs': features_extractor_kwargs,
                     'normalize_images': False,
                     'optimizer_class': th.optim.Adam,
                     'optimizer_kwargs': optimizer_kwargs,
                     'n_critics': 1,
                     'share_features_extractor': True}

    replay_buffer_kwargs = None

    model_kwargs = {'policy': SACPolicy,
                    'env': env,
                    'learning_rate': lr_schedule,
                    'buffer_size': 200,
                    'learning_starts': 100,
                    'batch_size': 15,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': (1, "episode"),
                    'gradient_steps': -1,
                    'action_noise': None,
                    'replay_buffer_class': None,
                    'replay_buffer_kwargs': replay_buffer_kwargs,
                    'optimize_memory_usage': False,
                    'ent_coef': "auto",
                    'target_update_interval': 1,
                    'target_entropy': "auto",
                    'use_sde': False,
                    'sde_sample_freq': -1,
                    'use_sde_at_warmup': False,
                    'tensorboard_log': 'logs/tb_logs',
                    'create_eval_env': False,
                    'policy_kwargs': policy_kwargs,
                    'verbose': 1,
                    'seed': 42,
                    'device': 'auto'}

    now = get_now('%y%m%d-%H%M')
    callback = ReportCallbacks(verbose=2)

    learn_kwargs = {'total_timesteps': 1000,
                    'callback': callback,
                    'log_interval': 30,
                    'eval_env': eval_env,
                    'eval_freq': 30,
                    'n_eval_episodes': 1,
                    'tb_log_name': MODEL_NAME + now,
                    'eval_log_path': f'learn/logs/tb_logs/{MODEL_NAME}{now}_1',
                    'reset_num_timesteps': True}

    config = {'env_kwargs': env_kwargs,
              'model_kwargs': model_kwargs,
              'learn_kwargs': learn_kwargs}

    return config