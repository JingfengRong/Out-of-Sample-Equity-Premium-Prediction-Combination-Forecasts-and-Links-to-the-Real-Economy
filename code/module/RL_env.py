import gym
import numpy as np
import pandas as pd
from gym import spaces

class TimeSeriesEnvBase(gym.Env):
    metadata = {"render_modes": "human"}

    def __init__(self):
        '''
        Initialize the environment.
        
        Description
        -----------
        We use the OpenAI gym.Env class as the parent class of our environment.
        Here we initialize some key properties of the environment.
        We set the observation space and action space for our environment using OpenAI Gym's space API.
        '''
        self.action_space = None
        self.state_space = None
        self.state = None
        self.reward = 0
        self.terminated = False
        self.info = {}

        self.start_tick = None
        self.end_tick = None
        self.current_tick = None

    def reset(self, seed=None):
        '''
        Reset the environment`s state. 
        Set the random seed once at the beginning of the experiment. 
        Then keep the same seed for the whole experiment.
        Also, we need to reset the state to the initial state.
        Clear the cumulative reward and other information.
        '''
        super().reset(seed=seed)
        state = None # reset the state to the initial state
        info = None

        return state, info

    def step(self, action):
        '''
        Take an action and return the next state, reward, done, info.
        We use the term "state" instead of "observation" in this function.
        The observation is the state plus some other information.
        The state is the representation of the environment.
        For research purpose, we want to keep the setting as simple as possible.
        Therefore, we only use the state as the input.
        '''

        state_prime = self._trans_func(action)
        reward = self._calculate_reward(action)
        terminated = self._is_terminated()
        info = self._get_info()

        return state_prime, reward, terminated, info
    
    def _trans_func(self, action):
        '''
        Move to the next state based on the current state and action.
        In the time series data, the transition is represented by the transition of the timestep.
        We get the timestep of current state and action value to calculate the next state.
        
        '''
        raise NotImplementedError

    def _calculate_reward(self, action):
        '''
        Since this class is built for time series data, the transition is represented by the transition of the timestep.
        Then we only need the timestep of state and action value to calculate the reward.
        '''
        raise NotImplementedError
    
    def _is_terminated(self):
        '''
        Check if the episode is terminated.
        '''
        raise NotImplementedError
    
    def _get_info(self):
        raise NotImplementedError
        
    

class EconMarketEnv(gym.Env):
    metadata = {"render_modes": "human"}
    """
    This class defines an OpenAI Gym environment for simulating a simple trading market.
    The market is modeled as a time series of econ factors.
    The environment is used for training trading agents to learn how to select and manage a portfolio of equities.

    The EconMarketEnv class is an implementation of the OpenAI Gym Env class.
    It has two main functions:
        - reset: reset the environment to its initial state and return the first observation
        - step: take an action, update the state of the environment, and return the new observation, reward, done, and info.

    The environment's state is a vector of econ factors, representing the market at a particular point in time.
    The observation is the state vector plus some additional information, (TBD).
    The action is a vector representing the portfolio weights for each equity.
    The reward is computed based on the portfolio returns.
    The environment is considered "done" if the episode reaches the last tick of the data.
    """

    def __init__(self, data: pd.DataFrame, portfolio: pd.DataFrame, features: dict):
        """
        Initialize the environment.

        Parameters
        ----------
        data : pandas.DataFrame
            The input time series data of econ factors.
        portfolio : pandas.DataFrame
            The portfolio time series of each individual equity return.

        Description
        -----------
        We set the observation space and action space for our environment using OpenAI Gym's space API.
        We also convert the data and portfolio to numpy arrays for performance reasons.
        """
        super(EconMarketEnv, self).__init__()
        self.obs_space_size = 18
        self.num_equities = 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_size,),dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float64)
        self.data = data.values
        self.index = data.index
        self.portfolio = portfolio.values
        self.features = features
        self.equitites_vol = self.features['volatility'].values
        self._initial_check()

        self.start_tick = 0
        self.end_tick = data.shape[0] - 2
        self.current_tick = None
        
        self.total_reward = 0
        self.port_ret = 0
        self.terminated = False
        self.info = {}
        self.reward = 0.0

        self.gamma = 0.0


    def reset(self, seed=None):
        '''
        Reset the environment's state. 
    
        Parameters
        ----------
        seed : int, optional
            Random seed for the experiment, by default None.
        
        Returns
        -------
        state : ndarray
            The initial state of the environment.
        info : dict
            A dictionary of additional information for the initial state.

        Notes
        -----
        Set the random seed once at the beginning of the experiment. 
        Then keep the same seed for the whole experiment.
        '''
        super().reset(seed=seed)
        self.current_tick = self.start_tick
        state = self.data[self.current_tick]
        state = np.append(state, [0.5, 0.5, 0.0]) # add weights and portfolio return

        self.reward = 0.0
        self.total_reward = 0.0
        self.terminated = False
        self.port_ret = 0.0
        self.info = {}

        return state
    
    def step(self, action):
        '''
        Takes an action and returns the next state, reward, done, info.

        Parameters:
        -----------
        state : numpy.ndarray
            The current state of the environment.
        action : int
            The action taken by the agent.

        Returns:
        --------
        state_prime : numpy.ndarray
            The next state of the environment.
        reward : float
            The reward received by the agent.
        terminated : bool
            True if the episode is over, False otherwise.
        info : dict
            A dictionary containing any additional information about the transition.

        Notes:
        ------
        We use the term "state" instead of "observation" in this function.
        The observation is the state plus some other information.
        The state is the representation of the environment.
        For research purposes, we want to keep the setting as simple as possible.
        Therefore, we only use the state as the input.
        '''
        self.terminated = self._is_terminated()
        self.reward = self._calculate_reward(action)
        state_prime = self._trans_func(action)
        info = self._get_info()
        self.total_reward += self.reward

        return state_prime, self.reward, self.terminated, info

    
    def _trans_func(self, action):
        '''
        Move to the next state based on the current state and action.

        Parameters:
        ----------
        state : ndarray
            A numpy array that represents the current state of the environment.
        action : float
            An float that represents the action taken by the agent.

        Returns:
        -------
        state_prime : ndarray
            A numpy array that represents the next state of the environment.

        Description:
        ------------
        In the time series data, the transition is represented by the transition of the timestep.
        We get the timestep of current state and action value to calculate the next state.    
        '''
        state_prime = self.data[self.current_tick + 1]
        state_prime = np.append(state_prime, [action[0], 1 - action[0], np.sign(self.reward)])
        self.current_tick += 1

        return state_prime
        
    def _calculate_reward(self, action):
        '''
        Calculate the reward based on the current state and action.
        
        Parameters
        ----------
        state : numpy array
            The current state of the environment.
        action : numpy array
            The action taken by the agent.

        Returns
        -------
        float
            The reward for the current state and action.
        
        Description:
        ------------
        For time series data, the transition is represented by the transition of the timestep.
        We get the timestep of current state and action value to calculate the reward.
        We need wait untile the next timestep to get the equity return.
        Then we calculate the reward by the portfolio weight and the equity return.
        '''
        w = np.array([action, 1 - action]).reshape(-1,)
        self.port_ret = w @ self.portfolio[self.current_tick + 1]
        port_std = self.equitites_vol[self.current_tick + 1]
        utility = self.port_ret - self.gamma * w ** 2 @ port_std ** 2
        # reward = np.log(self.port_ret + 1)
    
        return utility

    def _is_terminated(self):
        '''
        Check if the episode is terminated.
        
        Returns
        -------
        bool
            True if the episode is terminated, False otherwise.
        
        Description:
        ------------
            An episode is terminated if the current tick is equal to the end tick.
        '''
        return self.current_tick == self.end_tick

    def _get_info(self):
        return {'portfolio return': self.port_ret, 'date': self.index[self.current_tick]}

    def _initial_check(self):
        assert self.data.shape[0] == self.portfolio.shape[0], 'The data and portfolio should have the same length.'
        # assert self.data.shape[1] == self.obs_space_size, f'The data should have {self.num_equities} columns.'
        assert self.portfolio.shape[1] == self.num_equities, f'The portfolio should have {self.num_equities} columns.'