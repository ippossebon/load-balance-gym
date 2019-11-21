from gym.envs.registration import register

register(
    id='load-balance-v0',
    entry_point='load_balance_gym.envs:LoadBalanceEnv',
)
