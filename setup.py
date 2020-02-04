from 'load_balance_gym' import LoadBalanceEnv

env = LoadBalanceEnv()
obs = env.reset()
n_steps = 10

for step in range(n_steps):
    # Random action
    action = env.action_space.sample()
    print('step = ', step)
    print('action = ', action)

    state, reward, done, info = env.step(action)
    print('state = ', state)
    print('reward = ', reward)
