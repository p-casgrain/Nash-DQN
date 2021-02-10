import numpy as np
from collections import namedtuple
import random
import torch
from copy import deepcopy as dc

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
ProtoState = namedtuple('State', ('t', 'p', 'i', 'q'))
class State(ProtoState):
    def to_numpy(self):
        return np.concatenate([x for x in self], axis=None)
    def to_tensor(self, **kwargs):
        return torch.tensor(self.to_numpy(), **kwargs)
    def to_sep_numpy(self, idx):
        # i is agent index (start from 0) to include is invariant
        return np.array( (self.t, self.p, self.q[idx], self.i) ), np.concatenate( (self.q[:idx], self.q[idx+1:]) , axis = None) 
    def to_sep_tensor(self, idx, **kwargs):
        non_inv, inv = self.to_sep_numpy(idx)
        return torch.tensor(non_inv, **kwargs), torch.tensor(inv, **kwargs)



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
    def __init__(self, param_dict_in):
        # Fill in default input arguments
        def_dict = {'trans_impact_scale': 0.0,
                    'trans_impact_decay': 0.0,
                    'perm_price_impact': 0.0 }

        param_dict = {**def_dict, **param_dict_in}

        # Unpack Parameter Dictionary
        # Game-Specific Parameters
        self.perm_imp = param_dict['perm_price_impact']
        self.tmp_scale = param_dict['trans_impact_scale']
        self.tmp_decay = param_dict['trans_impact_decay']
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
        self.r = lambda Q, S, nu: - nu * (S + self.t_cost * nu) - self.phi * ( Q ** 2 )

        # Allocating Memory for Game Variables & Resetting
        self.dS = np.float32(0)
        self.dF = np.float32(0)
        self.dI = np.float32(0)

        self.reset()
        
    def reset(self):
        """
        Reset the simulation and reinitialize inventory and price levels
        """
        self.Q = np.random.normal(0, 25, self.N)
        self.S = np.float32(10 + np.random.normal(0, self.sigma))
        self.I = np.random.normal(0, 0.25*self.sigma)
        self.t = np.float32(0)

        self.last_reward = np.zeros(self.N, dtype=np.float32)
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                   int(round(np.ceil(self.T / self.dt) + 2)))
        
        self.check_price()
    def check_price(self):
        """
        Ensure price process does not fall below 0
        """
        if self.S <= 0:
            self.S = 0.0001
            
    def setInv(self,inv):
        """
        Manually set inventory level
        :param inv: Array containing the inventory levels of all agents being set to
        """
        self.Q = inv


    def step(self, nu):
        """
        Calculate and update all environment parameters given all agent's actions.
        Returns a Transition tuple holding observable state values of the previous
         state of the environment, actions executed by all agents, observable state 
         values of the resultant state and the rewards obtained by all agents.
         
        :param nu: Array containing the actions of all agents
        :returns:  Transition object containing summary of the change to the environment
        """
        last_state, _, _ = self.get_state()

        if self.t < self.T:
            # Advance Inventory & Time
            self.Q += nu
            self.t += self.dt

            # Compute Action Reward
            self.last_reward = self.r(self.Q, self.S, nu)
            self.total_reward += self.last_reward

            # Advance Asset Price
            self.dF = self.mu(self.t, self.S) * self.dt + self.sigma * self.dW[int(round(self.t))]
            #self.dS = self.dF + self.dt * (self.perm_imp * np.sign(np.mean(nu))*np.sqrt(np.abs(np.mean(nu))))
            self.dI = self.I * (np.exp(-self.tmp_decay*self.dt) - 1) + \
                self.dt * self.tmp_scale * self.impact_scale(np.mean(nu))
            self.I += self.dI
            self.dS = self.dF + self.dI + self.dt * (self.perm_imp * np.mean(nu))
            self.S += self.dS

        cur_state, _, _ = self.get_state()

        return dc( Transition(last_state, nu, cur_state, self.last_reward) )
    
    def impact_scale(self,v):
        return np.sign(v)*np.sqrt(np.abs(v))

    def get_state(self):
        """
        Returns the observable features of the current state
        :return: State object summarizing observable features of the current state
        """
        return dc(State( self.T-self.t, self.S, self.I, self.Q)), dc(self.last_reward), dc(self.total_reward)

    def __str__(self):
        state, last_reward, total_reward = self.get_state()
        str = "Simulation -- Last State: {}, \
        Last Reward: {}, Total Reward: {}".format(state,last_reward,total_reward)
        return str

    def __repr__(self):
        return self.__str__()


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




if __name__=='__main__':

    sim_dict = {'perm_price_impact':.3,
                'trans_impact_scale':0.15,
                'trans_impact_decay':0.25*np.log(2),
                'transaction_cost':.5,
                'liquidation_cost':.5,
                'running_penalty':0,
                'T':8,
                'dt':1.0/60.0,
                'N_agents': 25,
                'drift_function':(lambda x,y: 0.1*(10-y)) , #x -> time, y-> price
                'volatility':1,
                'initial_price_var':20}

    test_sim = MarketSimulator(sim_dict)
    test_sim.step(5*np.random.randn(25))
    print(test_sim)
