import numpy as np
from collections import namedtuple
import random
import copy
import torch

"""
Transition object summarizing changes to the environment at each time step
:param state:       State object representing observable features of the current state
:param action:      Array of actions of all agents
:param next_state:  State object representing observable features of the resultant state
:param reward:      Array of rewards obtained by all agents
"""
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

"""
State object summarizing observable features of the environment at each time step
:param t:  Time step number
:param p:  Price of stock
:param q:  Array of inventory levels of all agents
"""
State = namedtuple('State', ('t', 'p', 'q'))

class State(State):
    """
    Game State Class.
    __init__ takes arguments (t,p,q)
    Inherits from namedtuple class
    """

    def getNormalizedState(self, toTensor=True):
        """
        Returned Normalized State Values
        :return: Array of concatenated values
        """

        norm_q = self.q / 200
        norm_p = (self.p - 110) / 100
        norm_t = (self.t - 12) / 12
        out = copy.deepcopy(np.concatenate( (np.array([norm_t,norm_p]), norm_q) ))

        if toTensor:
            return out
        else:
            return torch.from_numpy(out)
    
    def getState (self):
        """
        Returned Non-Normalized State Values
        :return: Array of concatenated values
        """
        return copy.deepcopy(np.concatenate( (np.array([self.t,self.p]), self.q) ))


class MarketSimulator(object):
    """
    Class representing the market environment
    
    :param p_imp:   Price Impact Coefficent
    :param t_cost:  Transaction cost coefficent
    :param L_cost:  Liquidation cost coefficent
    :param phi:     Risk penalty (b_3)
    :param T:       Total number of time steps
    :param dt:      Change in time per time step in seconds
    :param N:       Total number of agents
    :param mu:      Drift function of process
    :param sigma:   Volatility of the stock
    :param sigma0:  Volatility of initial stock price
    :param r:       Reward function for the agents
    """
    def __init__(self, param_dict):
        # Unpack Parameter Dictionary
        # Game-Specific Parameters
        self.p_imp = param_dict['price_impact']
        self.t_cost = param_dict['transaction_cost']
        self.L_cost = param_dict['liquidation_cost']
        self.phi = param_dict['running_penalty']
        self.T = param_dict['T']
        self.dt = param_dict['dt']
        self.N = param_dict['N_agents']

        # Simulation Parameters
        self.mu = param_dict['drift_function']
        self.sigma = param_dict['volatility']
        self.sigma0 = param_dict['initial_price_var']

        # Define Reward Function
        self.r = lambda Q, S, nu: - nu * (S + self.t_cost * nu) - self.phi * Q ** 2

        # Allocating Memory for Game Variables
        self.Q = np.random.normal(0, self.sigma0, self.N)
        self.S = np.float32(10 + np.random.normal(0, self.sigma))
        self.dS = np.float32(0)
        self.dF = np.float32(0)
        self.t = np.float32(0)

        # Variable Containing Total Accumulated Score
        self.last_reward = np.zeros(self.N, dtype=np.float32)
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        # Variable Containing BM increments
        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                   int(round(np.ceil(self.T / self.dt) + 2)))
        
    def check_price(self):
        """
        Ensure price process does not fall below 0
        """
        if self.S <= 0:
            self.S = 0.01
            
    def setInv(self,inv):
        """
        Ensure price process does not fall below 0
        :param inv: Array containing the inventory levels of all agents being set to
        """
        self.Q = inv

    def reset(self):
        """
        Reset the simulation and reinitialize inventory and price levels
        """
        self.Q = np.random.normal(0, 10, self.N)
        self.S = np.float32(10 + np.random.normal(0, self.sigma))
        self.t = np.float32(0)

        self.last_reward = np.zeros(self.N, dtype=np.float32)
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                   int(round(np.ceil(self.T / self.dt) + 2)))
        
        self.check_price()

    def step(self, nu):
        """
        Calculate and update all environment parameters given all agent's actions.
        Returns a Transition tuple holding observable state values of the previous
         state of the environment, actions executed by all agents, observable state 
         values of the resultant state and the rewards obtained by all agents.
         
        :param nu: Array containing the actions of all agents
        :returns:  Transition object containing summary of the change to the environment
        """
        last_state = State(self.t,self.S,self.Q)
        if self.t < self.T:
            # Advance Inventory & Time
            self.Q += nu
            self.t += self.dt

            # Compute Action Reward
            self.last_reward = self.r(self.Q, self.S, nu)
            self.total_reward += self.last_reward

            # Advance Asset Price
            self.dF = self.mu(self.t, self.S) * self.dt + self.sigma * self.dW[int(round(self.t))]
            #self.dS = self.dF + self.dt * (self.p_imp * np.sign(np.mean(nu))*np.sqrt(np.abs(np.mean(nu))))
            self.dS = self.dF + self.dt * (self.p_imp * np.mean(nu))
            self.S += self.dS

        return Transition(last_state, nu, State(self.t, self.S, self.Q), self.last_reward)

    def get_state(self):
        """
        Returns the observable features of the current state
        :return: State object summarizing observable features of the current state
        """
        return State(copy.deepcopy(self.t), copy.deepcopy(self.S), copy.deepcopy(self.Q)), copy.deepcopy(self.last_reward), copy.deepcopy(self.total_reward)

    def __str__(self):
        state, last_reward, total_reward = self.get_state()
        str = "Simulation -- Last State: {}, \
        Last Reward: {}, Total Reward: {}".format(state,last_reward,total_reward)
        return str


class ExperienceReplay:
    """
    Class for storing objects in the experience replay buffer
    :param buffer:          List containing all objects in the replay buffer
    :param max_buffer_size: Max size of the buffer
    :param buffer_size:     Current size of the buffer
    """
    def __init__(self, buffer_size):
        self.buffer = []
        self.max_buffer_size = buffer_size
        self.buffer_size = 0

    def add(self, experience):
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
        self.buffer_size = len(self.buffer)

    def sample(self, size):
        return random.sample(self.buffer, min(size, self.buffer_size))

    def __len__(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer = []
