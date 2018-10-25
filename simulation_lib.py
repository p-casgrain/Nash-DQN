import numpy as np
from collections import namedtuple
import random
from itertools import count
from per.prioritized_memory import *

# Define Transition Class as Named Tuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



class MarketSimulator(object):
    def __init__(self,param_dict):

        # Unpack Parameter Dictionary

        # Game-Specific Parameters
        self.p_imp  = param_dict['price_impact']
        self.t_cost = param_dict['transaction_cost']
        self.L_cost = param_dict['liquidation_cost']
        self.phi    = param_dict['running_penalty']

        self.T      = param_dict['T']
        self.dt     = param_dict['dt']

        self.N      = param_dict['N_agents']

        # Simulation Parameters
        self.mu      = param_dict['drift_function']
        self.sigma   = param_dict['volatility']

        # Define Reward Functions for t<T and t=T (Revise, perhaps)
        self.r  = lambda Q,S,nu : - nu*(S + self.t_cost*nu) - self.phi * Q**2
        self.rT = lambda Q,S,nu : Q*(S - Q*self.L_cost)

        # Allocating Memory for Game Variables
        self.Q = np.random.normal(0,10,self.N)
        self.S = np.float32(10+np.random.normal(0,self.sigma))
        self.dS = np.float32(0)
        self.dF = np.float32(0)
        # self.F = np.float32(0)
        self.t = np.float32(0)

        # Variable Containing Total Accumulated Score
        self.last_reward = np.zeros( self.N, dtype=np.float32 )
        self.total_reward = np.zeros( self.N, dtype=np.float32 )

        # Variable Containing BM increments
        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                      int(round(np.ceil(self.T / self.dt) + 2 )))

        # Variable Indicating Whether Done
        self.done = False

    def reset(self):
        # Reset Game Values
        self.Q = np.random.normal(0,10,self.N)
        self.S = np.float32(10+np.random.normal(0,self.sigma))
        self.t = np.float32(0)

        self.last_reward = np.zeros( self.N, dtype=np.float32 )
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                      int(round(np.ceil(self.T / self.dt) + 2 )))

        self.done = False

    def step(self,nu):

        last_state = (self.Q,self.S)

        if self.t < self.T:
            # Advance Inventory & Time
            self.Q += nu
            self.t += self.dt

            # Compute Action Reward
            self.last_reward = self.r(self.Q,self.S,nu)
            self.total_reward += self.last_reward

            # Advance Asset Price
            self.dF = self.mu(self.t,self.S) * self.dt + self.sigma * self.dW[int(round(self.t))]
            self.dS = self.dF + self.dt * ( self.p_imp * np.mean(nu) )
            self.S += self.dS

#        elif (not self.done):
#
#            # Compute Action Reward
#            self.last_reward = self.rT(self.Q, self.S, self.nu)
#            self.total_reward += self.last_reward
#
#            # Update Variables
#            self.Q = np.zeros(self.N, dtype=np.float32)

        return Transition( last_state, nu, (self.Q,self.S), self.last_reward )

    def get_state(self):
        return (self.Q,self.S), self.last_reward, self.total_reward


# Define an object that summarizes all state variables
class State(object):
    def __init__(self, param_dict):
        self.t = param_dict['time_step']
        self.q = param_dict['current_inventory']
        self.p = param_dict['current_price']

    # Returns list representation of all state variables normalized to be [-1,1]
    # Not done yet
    def getNormalizedState(self):
        norm_q = self.q / 10
        norm_p = (self.p - 10) / 10
        norm_t = self.t / 4 - 1
        return np.array(np.append(np.append(norm_q, norm_p), norm_t))

    def print_state(self):
        print("t =", self.t, "q = ", self.q, "p = ", self.p)


class ExperienceReplay:
    # each experience is a list of with each tuple having:
    # first element: state,
    # second element: array of actions of each agent,
    # third element: array of rewards received for each agent
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
        return random.sample(self.buffer, min(size,self.buffer_size))

    def __len__(self):
        return len(self.buffer)


class PriorityExperienceReplay(ExperienceReplay):
    def __init__(self,buffer_size):
        self.buffer = []