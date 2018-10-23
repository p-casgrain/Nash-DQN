import math
import copy
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T

from simulation_lib import *
from NashRL_Lib import *

# -------------------------------------------------------------------
# This file defines some of the necessary objects and classes needed
# for the LQ-Nash Reinforcement Learning Algorithm
# -------------------------------------------------------------------


# Define object for estimated elements via NN
# ***NOTE*** All elements are tensors
class FittedValues(object):
    # Initialized via a single numpy vector
    def __init__(self,value_vector,v_value,num_players):

        self.num_players = num_players #number of players in game

        self.V = v_value #nash value vector
        
        self.mu = value_vector[0:self.num_players] #mean of each player
        value_vector = value_vector[self.num_players:]
        
        self.P1 = (value_vector[0])**2 #P1 matrix for each player, exp transformation to ensure positive value
        value_vector = value_vector[1:]
        
        self.a = value_vector[0:self.num_players] #a of each player
        value_vector = value_vector[self.num_players:]
        
        #p2 vector
        self.P2 = value_vector[0] 
        
        self.P3 = value_vector[1]
        
# Define an object that summarizes all state variables
class State(object):
    def __init__(self,param_dict):
        self.t = param_dict['time_step']
        self.q = param_dict['current_inventory']
        self.p = param_dict['current_price']
        
    # Returns list representation of all state variables normalized to be [-1,1]
    # Not done yet
    def getNormalizedState(self):
        norm_q = self.q/10
        norm_p = (self.p-10)/10
        norm_t = self.t/4-1
        return np.array(np.append(np.append(norm_q,norm_p),norm_t))
    
    def print_state(self):
        print("t =", self.t, "q = ", self.q, "p = ", self.p)

# Defines basic network parameters and functions
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim,nump):
        super(DQN, self).__init__()
        self.num_players = nump
        # Define basic fully connected network for parameters in Advantage function
        self.main = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 60),
            nn.ReLU(),
            nn.Linear(60, 160),
            nn.ReLU(),
            nn.Linear(160, 60),
            nn.ReLU(),
            nn.Linear(60, output_dim)
        )
        
        # Define basic fully connected network for estimating nash value of each state
        self.main_V = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, num_players)
        )
        
    #Since only single network, forward prop is simple evaluation of network
    def forward(self, input):
        return self.main(input), self.main_V(input)
    
class NashNN():
    def __init__(self, input_dim, output_dim, nump,t):
        self.num_players = nump
        self.T = t
        self.main_net = DQN(input_dim, output_dim,num_players)
        #self.target_net = copy.deepcopy(self.main_net)
        self.num_sim = 10000
        # Define optimizer used (SGD, etc)
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),lr=0.001)
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        self.counter = 0
        
        # Predicts resultant values, input a State object, outputs a FittedValues object
    def predict(self, input):
        a,b = self.main_net.forward(self.stateTransform(input))
        return self.tensorTransform(a,b)
    
#    def predict_targetNet(self, input):
#        a,b = self.target_net.forward(self.stateTransform(input))
#        return self.tensorTransform(a,b)
        
    # Transforms state object into tensor
    def stateTransform(self, s):
        return torch.tensor(s.getNormalizedState()).float()
        
    # Transforms output tensor into FittedValues Object
    def tensorTransform(self, output1, output2):
        return FittedValues(output1, output2, self.num_players)
    
    #takes a tuple of transitions and outputs loss
    def compute_Loss(self,state_tuple):
        currentState, action, nextState, reward = state_tuple[0], torch.tensor(state_tuple[1]).float(), state_tuple[2], state_tuple[3]
        A = lambda u, uNeg, mu, muNeg, a, v, c1, c2, c3: v - 0.5*c1*(u-mu)**2 + c2*(u -a)*torch.sum(uNeg - muNeg) + c3*(uNeg - muNeg)**2
        flag = 1
        nextVal = self.predict(nextState).V
        
        #set next nash value to be 0 if last time step
        if nextState.t >= self.T:
            nextVal = torch.zeros(self.num_players)
            flag = 0
            
        curVal = self.predict(currentState)
        loss = []
        for i in range(0,self.num_players):
            r = lambda T : torch.cat([T[0:i], T[i+1:]])
            loss.append(nextVal[i] + reward[i] - flag*A(action[i],r(action),curVal.mu[i],r(curVal.mu),curVal.a[i],curVal.V[i],curVal.P1,curVal.P2,curVal.P3)-(1-flag)*curVal.V[i])
        
        return torch.sum(torch.stack(loss)**2)
        
    def updateLearningRate(self):
        self.counter += 1
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),lr=0.005-(0.005-0.0005)*self.counter/self.num_sim)
    
#    #ignore these functions for now... was trying to do batch predictions 
#    def predict_batch(self, input):
#        return self.tensorsTransform(self.main_net(self.statesTransform(input),batch_size = 3))
#        
#    # Transforms state object into tensor
#    def statesTransform(self, s):
#        print(np.array([st.getNormalizedState() for st in s]))
#        return Variable(torch.from_numpy(np.array([st.getNormalizedState() for st in s])).float()).view(1, -1)
#        
#    # Transforms output tensor into FittedValues Object
#    def tensorsTransform(self, output):
#        return np.apply_along_axis(FittedValues(),1,output.data.numpy())

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
if __name__ == '__main__':
    num_players = 2
    T = 2
    #replay_stepnum = 3
    batch_update_size = 200
    num_sim = 10000
    max_action = 20
    update_net = 50
    buffersize = 500
    
    #number of parameters that need to be estimated by the network
    parameter_number = 3*num_players +3
    
    #network object
    net = NashNN(2+num_players,parameter_number,num_players,T)
    
    #simulation object
    sim_dict = {'price_impact': 0.5,
                'transaction_cost':1,
                'liquidation_cost':100,
                'running_penalty':0,
                'T':T,
                'dt':1,
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
    all_states = []
    states = []
    
    all_predicts = []
    preds = []
    
    #exploration chance
    ep = 0.5
    
    #intialize relay memory
    replay = ExperienceReplay(buffersize)
    
    sum_loss = np.zeros(num_sim)
    total_l = 0
    
    for k in range (0,num_sim):
        epsilon = ep - ep*(k/(num_sim-1))
        total_l = 0
        
        if k%50 == 0:
            print("Starting Inventory: ", sim.get_state()[0][0])
            flag = True
            
#        if k%update_net == 0:
#                net.target_net = copy.deepcopy(net.main_net)
            
        for i in range(0,T):   
            state,lr,tr = sim.get_state()
            #print("funny state:", state[0])
            state_dict = {'time_step':i,
                          'current_inventory':np.copy(state[0]),
                          'current_price':state[1]}
            current_state = State(state_dict)
            
            #print(net.predict(current_state).mu)
            #takes action
            if i == T-1:
                a = -current_state.q
                #print (net.predict(current_state).V.data.numpy())
            else:
                if np.random.random() < epsilon:
                    #random action between buying/selling 5 shares for all players
                    a = np.random.rand(num_players)*40-20
                    #print("r")
                else:
                    #else take predicted nash action
                    a = net.predict(current_state).mu.data.numpy()
                    #print(a)
                super_threshold_indices = a > max_action
                a[super_threshold_indices] = max_action
                sub_threshold_indices = a < -max_action
                a[sub_threshold_indices] = -max_action
            
            #take chosen action and update new state
            #current_state.print_state()
            sim.step(a)
            state,lr,tr = sim.get_state()
            state_dict = {'time_step':i+1,
                          'current_inventory':state[0],
                          'current_price':state[1]}
            new_state = State(state_dict)
            #print("current state:")
            #current_state.print_state()
            #print("action:", a)
            #print("new state:")
            #new_state.print_state()
            
            #updates storage variables
            states.append(new_state)
            prices[i] = new_state.p
            Qs[i,:] = new_state.q
            Actions[i,:] = a
            rewards[i] = lr
            #preds.append[net.predict(current_state).V.data()]
            
            #adds experience to replay memory
            replay.add((current_state,a,new_state,lr))
    #        explist = []
    #        for j in range(max(i-(replay_stepnum-1),0),i+1):
    #            explist.append((states[j],Actions[i,:],rewards[i,:]))
    #        replay.add(explist)
            
            if (k-1)*T > batch_update_size:
                replay_sample = replay.sample(batch_update_size)
                
                #samples from priority and calculate total loss
                replay_sample.append((current_state,a,new_state,lr))
                
                loss = []
                
                for sample in replay_sample:              
                    loss.append(net.compute_Loss(sample))
                    
                if (flag):
                    print(current_state.p,a,current_state.q)
                    new_V = net.predict(new_state).V.data.numpy()
                    if i == T-1:
                        new_V = np.zeros(num_players)
                    print(new_V+lr)
                    print(net.predict(current_state).V.data.numpy())
                
                #loss = net.criterion(estimated_q, actual_q)
                loss = torch.sum(torch.stack(loss))
                
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                
                #totals loss (of experience replay) per time step
                #total_l += sum(map(lambda a,b:(a-b)*(a-b),estimated_q,actual_q))
                #current_state = new_state
                
                
        flag = False
        #defines loss per period
        sum_loss[k] = total_l
        #for param in net.main_net.parameters():
        #    print(param.data) 
        #print(prices)
        #print(Qs)
        #print(Actions)
        
        #resets simulation
        sim.reset()
        all_states.append(states)
        all_predicts.append(preds)
        net.updateLearningRate()
        
    plt.plot(sum_loss)
        
    