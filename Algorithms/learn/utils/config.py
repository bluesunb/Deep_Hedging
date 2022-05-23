import torch as th
import torch.nn as nn
import yaml

from easydict import EasyDict
from datetime import datetime
from typing import Optional, Tuple

from Env.env import BSMarket
from Env.feature_extractor import BatchNormExtractor
from Algorithms.learn.utils.callbacks import ReportCallbacks
from Algorithms.policies import DoubleTD3Policy

from stable_baselines3.common.noise import NormalActionNoise


def easydict_to_dict(edict):
    if not isinstance(edict, (dict, EasyDict)):
        return edict
    return {k: easydict_to_dict(v) for k, v in edict.items()}


def lr_schedule(left: float):
    return 5e-3 * (0.95 ** (30 * (1 - left)))


def get_now(f_string='%y%m%d-%H%M'):
    return datetime.now().strftime(f_string)


def save_config(path: str,
                env_kwargs: dict,
                model_kwargs: dict,
                learn_kwargs: dict,
                **kwargs) -> None:
    # custom env 또는 custom class는 class type + class kwargs로 따로 저장된다.
    model_kwargs['env'] = type(model_kwargs['env'])
    learn_kwargs['eval_env'] = type(learn_kwargs['eval_env'])

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

    return env_kwargs, model_kwargs, learn_kwargs


def reconstruct_config(env_kwargs, model_kwargs, learn_kwargs, **kwargs):
    env = type(model_kwargs['env'])(**env_kwargs)
    model_kwargs['env'] = env
    print(f"model_kwargs['env']: {env}")

    eval_env_kwargs = kwargs.get('eval_env_kwargs', env_kwargs)
    eval_env = type(learn_kwargs['eval_env'])(**eval_env_kwargs)
    learn_kwargs['eval_env'] = eval_env
    print(f"learn_kwargs['eval_env']: {eval_env}")

    learn_kwargs['tb_log_name'] = "ddpg_" + get_now()
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
                                 'features_out': 2,
                                 'net_arch': [32, 64],
                                 'activation_fn': nn.ReLU}
    optimizer_kwargs = None

    policy_kwargs = {'net_arch': [],  # None으로 설정하면 deafult net arch가 설정됨
                     'activation_fn': nn.ReLU,
                     'features_extractor_class': BatchNormExtractor,
                     'features_extractor_kwargs': features_extractor_kwargs,
                     'normalize_images': False,
                     'optimizer_class': th.optim.Adam,
                     'optimizer_kwargs': optimizer_kwargs,
                     'n_critics': 1,
                     'share_features_extractor': True}

    replay_buffer_kwargs = None

    model_kwargs = {'policy': DoubleTD3Policy,
                    'env': env,
                    'learning_rate': lr_schedule,
                    'buffer_size': 200,
                    'learning_starts': 100,
                    'batch_size': 15,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': (1, "episode"),
                    'gradient_steps': -1,
                    'action_noise': NormalActionNoise(mean=0.0, sigma=0.1),
                    'replay_buffer_class': None,
                    'replay_buffer_kwargs': replay_buffer_kwargs,
                    'optimize_memory_usage': False,
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
                    'tb_log_name': "ddpg_" + now,
                    'eval_log_path': f'logs/tb_logs/ddpg_{now}_1',
                    'reset_num_timesteps': True}

    config = {'env_kwargs': env_kwargs,
              'model_kwargs': model_kwargs,
              'learn_kwargs': learn_kwargs}

    return config
