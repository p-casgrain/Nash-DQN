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
import matplotlib.pyplot as plt
#import torchvision.transforms as T

# -------------------------------------------------------------------
# This file defines some of the necessary objects and classes needed
# for the LQ-Nash Reinforcement Learning Algorithm
# -------------------------------------------------------------------



# Define Transition Class as Named Tuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Define object for estimated elements via NN
# ***NOTE*** All elements are tensors
class FittedValues(object):
    # Initialized via a single numpy vector
    def __init__(self,value_vector):
        self.num_players = 4 #number of players in game
        self.V = value_vector[0:self.num_players] #nash value vector
        value_vector = value_vector[self.num_players:]
        
        self.mu = value_vector[0:self.num_players] #mean of each player
        value_vector = value_vector[self.num_players:]
        
        self.P1 = torch.exp(value_vector[0:self.num_players]) #P1 matrix for each player, exp transformation to ensure positive value
        value_vector = value_vector[self.num_players:]
        
        self.a = value_vector[0:self.num_players] #a of each player
        value_vector = value_vector[self.num_players:]
        
        #A matrix where each row is the vector P_i2
        #Note that currently there is no symmetry assumption and each P_i2 is estimated separately
        self.P2matrix = (value_vector[0:self.num_players*(self.num_players-1)]).view((self.num_players,self.num_players-1)) 
        value_vector = value_vector[self.num_players*(self.num_players-1):]
        
        #A matrix where each row is the vector mu_i{-1}
        #Note that currently there is no symmetry assumption and each mu_i{-1} is estimated separately
        self.muNeg1 = (value_vector[0:self.num_players*(self.num_players-1)]).view((self.num_players,self.num_players-1)) 
        
# Define an object that summarizes all state variables
class State(object):
    def __init__(self,param_dict):
        self.t = param_dict['time_step']
        self.q = param_dict['current_inventory']
        self.p = param_dict['current_price']
        
    # Returns list representation of all state variables normalized to be [-1,1]
    # Not done yet
    def getNormalizedState(self):
        return np.array(np.append(np.append(self.q,self.p),self.t))

# Defines basic network parameters and functions
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Define basic fully connected network
        self.main = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            #nn.Linear(20, 20),
            #nn.ReLU(),
            nn.Linear(20, output_dim)
        )
        
    #Since only single network, forward prop is simple evaluation of network
    def forward(self, input):
        return self.main(input)
    
class NashNN():
    def __init__(self, input_dim, output_dim):
        self.main_net = DQN(input_dim, output_dim)
                # Define optimizer used (SGD, etc)
        self.optimizer = optim.RMSprop(self.main_net.parameters(),lr=0.001)
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        
        # Predicts resultant values, input a State object, outputs a FittedValues object
    def predict(self, input):
        return self.tensorTransform(self.main_net(self.stateTransform(input)))
        
    # Transforms state object into tensor
    def stateTransform(self, s):
        return Variable(torch.from_numpy(s.getNormalizedState()).float()).view(1, -1)
        
    # Transforms output tensor into FittedValues Object
    def tensorTransform(self, output):
        return FittedValues(output[0])
    
    
    #ignore these functions for now... was trying to do batch predictions 
    def predict_batch(self, input):
        return self.tensorsTransform(self.main_net(self.statesTransform(input),batch_size = 3))
        
    # Transforms state object into tensor
    def statesTransform(self, s):
        print(np.array([st.getNormalizedState() for st in s]))
        return Variable(torch.from_numpy(np.array([st.getNormalizedState() for st in s])).float()).view(1, -1)
        
    # Transforms output tensor into FittedValues Object
    def tensorsTransform(self, output):
        return np.apply_along_axis(FittedValues(),1,output.data.numpy())

# Define Replay Buffer Class using Transition Object
#class ReplayMemory(object):
#
#    def __init__(self, capacity):
#        self.capacity = capacity
#        self.memory = []
#        self.position = 0
#
#    def push(self, *args):
#        """Saves a transition."""
#        if len(self.memory) < self.capacity:
#            self.memory.append(None)
#        self.memory[self.position] = Transition(*args)
#        self.position = (self.position + 1) % self.capacity
#
#    def sample(self, batch_size):
#        return random.sample(self.memory, batch_size)
#
#    def __len__(self):
#        return len(self.memory)


class ExperienceReplay:
    #each experience is a list of with each tuple having:
    #first element: state,
    #second element: array of actions of each agent,
    #third element: array of rewards received for each agent
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self,size):
        return random.sample(self.buffer,size)

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
        self.Q = np.zeros( self.N, dtype=np.float32 )
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

        elif (not self.done):

            # Compute Action Reward
            self.last_reward = self.rT(self.Q, self.S, self.nu)
            self.total_reward += self.last_reward

            # Update Variables
            self.Q = np.zeros(self.N, dtype=np.float32)

        return Transition( last_state, nu, (self.Q,self.S), self.last_reward )

    def get_state(self):
        return (self.Q,self.S), self.last_reward, self.total_reward
    
if __name__ == '__main__':
    num_players = 4
    T = 5
    replay_stepnum = 3
    batch_update_size = 100
    num_sim = 1000
    
    #number of parameters that need to be estimated by the network
    parameter_number = 4*num_players + 2*num_players*(num_players-1)
    
    #network object
    net = NashNN(2+num_players,parameter_number)
    
    #simulation object
    sim_dict = {'price_impact': 0.05,
                'transaction_cost':0.01,
                'liquidation_cost':0.1,
                'running_penalty':0,
                'T':T,
                'dt':1/T,
                'N_agents':num_players,
                'drift_function':(lambda x,y: 0) ,
                'volatility':0.01}
    sim = MarketSimulator(sim_dict)
    
    #Initialize storage variables
    prices = np.zeros(T)
    Qs = np.zeros((T,num_players))
    Actions = np.zeros((T,num_players))
    rewards = np.zeros((T,num_players))
    
    [state,lr,tr] = sim.get_state()
    state_dict = {'time_step':0,
                  'current_inventory':state[0],
                  'current_price':state[1]}
    current_state = State(state_dict)    
    states = [current_state]
    
    #exploration chance
    ep = 0.5
    
    #intialize relay memory
    replay = ExperienceReplay(200)
    
    sum_loss = np.zeros(num_sim)
    
    for k in range (0,num_sim):
        epsilon = ep - ep*(k/(num_sim-1))
        total_l = 0
        for i in range(0,T):   
            #print(net.predict(current_state).mu)
            #takes action
            if i == T-1:
                a = -current_state.q
            else:
                if np.random.random() < epsilon:
                    #random action between buying/selling 5 shares for all players
                    a = np.random.rand(num_players)*10-5
                else:
                    #else take predicted nash action
                    a = net.predict(current_state).mu.data.numpy()
                    
            #take chosen action and update new state
            sim.step(a)
            [state,lr,tr] = sim.get_state()
            state_dict = {'time_step':i,
                          'current_inventory':state[0],
                          'current_price':state[1]}
            new_state = State(state_dict)
            
            #updates storage variables
            states.append(new_state)
            prices[i] = new_state.p
            Qs[i,:] = new_state.q
            Actions[i,:] = a
            rewards[i] = lr
            
            #adds experience to replay memory
            replay.add((current_state,a,new_state,lr))
    #        explist = []
    #        for j in range(max(i-(replay_stepnum-1),0),i+1):
    #            explist.append((states[j],Actions[i,:],rewards[i,:]))
    #        replay.add(explist)
            
            replay_sample = replay.sample(min(i+1,batch_update_size))
            
            #samples from priority and calculate total loss
            estimated_q = []
            actual_q = []
            for sample in (replay_sample):
                #obtain estimated current state nash value
                output = net.predict(sample[0])
                #obtain estimated next state nash value
                next_NN_value = net.predict(sample[2])
                
                #cycle through all agents
                for agent_num in range(0,num_players):
                    
                    #convert experience replay action/reward to auto_grad variables
                    sample_a = Variable(torch.from_numpy(sample[1]).float())
                    sample_lr= Variable(torch.from_numpy(sample[3]).float())
                    
                    #define the vector mu_{-1}
                    if agent_num == 0:
                        other_agenta = sample_a[1:]
                    elif agent_num == num_players-1:
                        other_agenta = sample_a[0:-1]
                    else:
                        other_agenta = torch.cat((sample_a[0:agent_num],sample_a[agent_num+1:]))
                    
                    #calculate actual value of state-action pair for agent (based on experienced reward + next state estimated nash)
                    actual_q.append(next_NN_value.V[agent_num] + sample_lr[agent_num])
                    
                    #calculate estimated value of state-action pair for agent(based on network output using advantage function)
                    estimated_q.append(output.V[agent_num]-0.5*(sample_a[agent_num]-output.mu[agent_num])*output.P1[agent_num]*(sample_a[agent_num]-output.mu[agent_num])+
                                       (sample_a[agent_num]-output.a[agent_num])*output.P2matrix[agent_num].dot(other_agenta-output.muNeg1[agent_num]))
            
            #convert list of tensors into tensor and set actual_q to non-autograd variable (since its fixed value)
            estimated_q = torch.stack(estimated_q)
            actual_q = torch.stack(actual_q)
            actual_q = actual_q.detach()
            
            #define loss function, calc gradients and update
            loss = net.criterion(estimated_q, actual_q)
            net.optimizer.zero_grad()
            loss.backward()
            net.optimizer.step()
            
            #totals loss (of experience replay) per time step
            total_l += sum(map(lambda a,b:(a-b)*(a-b),estimated_q,actual_q))

        #defines loss per period
        sum_loss[k] = total_l
        #for param in net.main_net.parameters():
        #    print(param.data) 
        #print(prices)
        #print(Qs)
        #print(Actions)
        
        #resets simulation
        sim.reset()
        
    plt.plot(sum_loss)
        
    