import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd

# based on tutorial from: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

DATAFRAME_FILE = './dataframe/dataframe-h1-client-h2-server-usage-rate.csv'

LINK_CAPACITY = 1000 # TODO: confirmar capacidade maxima do link

class LoadBalanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dataframe):
        super(LoadBalanceEnv, self).__init__()

        self.switches = ['S1', 'S2', 'S3', 'S4', 'S5']
        self.links = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.topology = {
            'S1': ['B', 'C'], # links entre switches,
            'S2': ['B', 'D', 'E', 'F'],
            'S3': ['F', 'G', 'H', 'I'],
            'S4': ['C', 'E', 'G'],
            'S5': ['D', 'H']
        }

        # Estado inicial = utilização de cada link
        self.usage = {
         'A': 0,
         'B': 0,
         'C': 0,
         'D': 0,
         'F': 0,
         'G': 0,
         'H': 0,
         'I': 0
        }

        self.possible_routes = [
            ['B', 'D', 'H', 'I'],
            ['B', 'F', 'I'],
            ['B', 'E', 'G', 'I'],
            ['C', 'G', 'I'],
            ['C', 'E', 'D', 'H', 'I'],
            ['C', 'E', 'F', 'I']
        ]

        # Utilização dos links da rede
        self.observation_space = spaces.Box(
            low=0,
            high=LINK_CAPACITY, # utilização maxima do link
            shape=(len(self.links), 1), # é um array de utilização dos links
            dtype=np.float16
        )

        # A ação é escolher o switch sobre o qual vai agir. Isto é, o switch que
        # terá o fluxo dividido entre 2 caminhos
        self.action_space = spaces.Discrete(len(self.switches)) # array com o índice do switch sobre o qual vamos agir
        self.action_space = spaces.Box(
            low=0,
            high=len(self.switches), # # TODO: rever
            shape=(len(self.switches), len(self.possible_routes), len(self.possible_routes)) # [switch_id, rota1, rota2]
            dtype=np.uint8)
        )

        # Se tornou o uso da rede MAIS homogêneo, recompensa = 1
        # Se tornou o uso da rede MENOS homogêneo, recompensa = -1
        self.reward_range = (-1, 1)

    def reset(self):
        """
         - Reset the state of the environment to an initial state
        It  will be called to periodically reset the environment to an initial
        state. This is followed by many steps through the environment, in which
        an action will be provided by the model and must be executed, and the next
        observation returned. This is also where rewards are calculated, more on this later.

        - Called any time a new environment is created or to reset an existing environment’s state.
        """
        # Reset the state of the environment to an initial state
        self.usage = {
         'A': 0,
         'B': 0,
         'C': 0,
         'D': 0,
         'F': 0,
         'G': 0,
         'H': 0,
         'I': 0
        }

        # Set the current step to a random point within the data frame
        # We set the current step to a random point within the data frame, because
        # it essentially gives our agent’s more unique experiences from the same data set.
        # The _next_observation method compiles the stock data for the last five time steps,
        # appends the agent’s account information, and scales all the values to between 0 and 1

        # self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
        self.current_step = random.randint(0, len(self.dataframe))


    def step(self, action):
        # Execute one time step within the environment
        # It will take an action variable and will return a list of four things:
        # the next state, the reward for the current state, a boolean representing
        # whether the current episode of our model is done and some additional info on our problem.
        switch_id = action[0]
        route1 = action[1]
        route2 = action[2]

        # action corresponde ao switch sobre o qual vamos agir, isto é: S1, S2, S3, S4 ou S5
        if switch_id == 0:
            # Atua sobre S1
            # Pega o que temos nos links
            # TODO
        else if switch_id == 1:
            # Atua sobre S2
            # TODO
        else if switch_id == 2:
            # Atua sobre S3
            # TODO
        else if switch_id == 3:
            # Atua sobre S4
            # TODO
        else if switch_id == 4:
            # Atua sobre S5
            # TODO
        else:
            # ação inválida
            # TODO: o que fazer?


        return obs, reward, done, {}


    def render(self, mode='human', close=False):
        """
        It may be called periodically to print a rendition of the environment. This could
        be as simple as a print statement, or as complicated as rendering a 3D
        environment using openGL. For this example, we will stick with print statements.
        """
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
