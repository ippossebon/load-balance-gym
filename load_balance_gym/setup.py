from setuptools import setup

setup(
    name='load_balance_gym',
    version='0.1',
    install_requires=['gym']
)

# from gym.envs.registration import register
#
# register(
#     id='load-balance-v1',
#     entry_point='load_balance_gym.envs:LoadBalanceEnv',
# )
