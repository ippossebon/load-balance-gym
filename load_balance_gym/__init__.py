from gym.envs.registration import register

register(
    id='Load-Balance-v1',
    entry_point='load_balance_gym.envs:LoadBalanceEnv',
    kwargs={'usage': {}}
)
