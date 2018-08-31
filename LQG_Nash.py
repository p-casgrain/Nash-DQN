import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as T

# -------------------------------------------------------------------
# This file defines some of the necessary objects and classes needed
# for the LQ-Nash Reinforcement Learning Algorithm
# -------------------------------------------------------------------



# Define Transition Class as Named Tuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Define object for estimated elements via NN
class FittedValues(object):
    # Initialized via a single numpy vector
    def __init__(self,value_vector):
        self.num_players = 4 #number of players in game
        self.V = value_vector[0:self.num_players] #nash value vector
        value_vector = value_vector[self.num_players:]
        
        self.mu = value_vector[0:self.num_players] #mean of each player
        value_vector = value_vector[self.num_players:]
        
        self.P1 = np.exp(value_vector[0:self.num_players]) #P1 matrix for each player, exp transformation to ensure positive value
        value_vector = value_vector[self.num_players:]
        
        self.a = value_vector[0:self.num_players] #a of each player
        value_vector = value_vector[self.num_players:]
        
        #A matrix where each row is the vector P_i2
        #Note that currently there is no symmetry assumption and each P_i2 is estimated separately
        self.P2matrix = (value_vector[0:self.num_players*(self.num_players-1)]).reshape((self.num_players,self.num_players-1)) 
        value_vector = value_vector[self.num_players*(self.num_players-1):]
        
        #A matrix where each row is the vector mu_i{-1}
        #Note that currently there is no symmetry assumption and each mu_i{-1} is estimated separately
        self.muNeg1 = (value_vector[0:self.num_players*(self.num_players-1)]).reshape((self.num_players,self.num_players-1)) 
        
# Define an object that summarizes all state variables
class State(object):
    def __init__(self,param_dict):
        self.t = param_dict['time_step']
        self.q = param_dict['current_inventory']
        self.p = param_dict['current_price']
        
    # Returns list representation of all state variables normalized to be [-1,1]
    # Not done yet
    def getNormalizedState(self):
        return [self.p,self.q,self.t]

# Defines basic network parameters and functions
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        
        # Define basic fully connected network
        self.main = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, output_dim)
        )
        
        # Define optimizer used (SGD, etc)
        self.optimizer = optim.RMSprop(self.main_dqn.parameters(),lr=0.001)
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        
    #Since only single network, forward prop is simple evaluation of network
    def forward(self, input):
        return self.main(input)
    
    # Predicts resultant values, input a State object, outputs a FittedValues object
    def predict(self, input):
        return self.tensorTransform(self.main(self.stateTransform(input)))
        
    # Transforms state object into tensor
    def stateTransform(self, s):
        return Variable(torch.FloatTensor(s.getNormalizedState())).view(1, -1)
        
    # Transforms output tensor into FittedValues Object
    def tensorTransform(self, output):
        return FittedValues(output.data.numpy()[0])

# Define Replay Buffer Class using Transition Object
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define Market Game Simulation Object
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
        self.Q = np.zeros( self.N, dtype=np.float32 )
        self.S = np.float32(0)
        self.dS = np.float32(0)
        self.dF = np.float32(0)
        # self.F = np.float32(0)
        self.t = np.float32(0)

        # Variable Containing Total Accumulated Score
        self.last_reward = np.zeros( self.N, dtype=np.float32 )
        self.total_reward = np.zeros( self.N, dtype=np.float32 )

        # Variable Containing BM increments
        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                      np.ceil(self.T / self.dt) + 2 )

        # Variable Indicating Whether Done
        self.done = False

    def reset(self,S0,Q0):
        # Reset Game Values
        self.Q = np.float32(Q0)
        self.S = np.float32(S0)
        self.t = np.float32(0)

        self.last_reward = np.zeros( self.N, dtype=np.float32 )
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                      np.ceil(self.T / self.dt) + 2 )

        self.done = False

    def step(self,nu):

        last_state = (self.Q,self.S)

        if self.t < T:
            # Advance Inventory & Time
            self.Q += nu
            self.t += self.dt

            # Compute Action Reward
            self.last_reward = self.r(self.Q,self.S,self.nu)
            self.total_reward += self.last_reward

            # Advance Asset Price
            self.dF = self.mu(self.t,self.S) * self.dt + self.sigma * self.dW
            self.dS = self.dF + self.dt * ( self.p_imp * np.mean(nu) )
            self.S += self.dS

        elif (not self.done):

            # Compute Action Reward
            self.last_reward = self.rT(self.Q, self.S, self.nu)
            self.total_reward += self.last_reward

            # Update Variables
            self.Q = np.zeros(self.N, dtype=np.float32)

        return Transition( last_state, nu, (self.Q,self.S), self.last_reward )

    def get_state(self):
        return (self.Q,self.S), self.last_reward, self.total_reward
