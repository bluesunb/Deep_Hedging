from Env.env import BSMarket

env_kwargs = {'cost': 0.02,
              'dividend': 0.0,
              'drift': 0.0,
              'freq': 1,
              'gen_name': 'gbm',
              'init_price': 1.0,
              'maturity': 30,
              'n_assets': 1000,
              'payoff': 'european',
              'payoff_coeff': 1.0,
              'period_unit': 365,
              'reward_fn': 'mean var',
              'reward_fn_kwargs': {},
              'reward_mode': 'cash',
              'risk_free_interest': 0.0,
              'strike': 1.0,
              'volatility': 0.2}

env = BSMarket(**env_kwargs)

def evaluate(env):
    obs = env.reset()
    done, info = False, {}
    total_pnl = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_pnl += reward

    return total_pnl

total_pnl = evaluate(env)