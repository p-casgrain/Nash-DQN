import math
import copy
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    def __init__(self,value_vector,v_value):
        self.num_players = 3 #number of players in game
        self.V = v_value #nash value vector
        
        self.mu = value_vector[0:self.num_players] #mean of each player
        value_vector = value_vector[self.num_players:]
        
        self.P1 = torch.exp(value_vector[0:self.num_players]) #P1 matrix for each player, exp transformation to ensure positive value
        value_vector = value_vector[self.num_players:]
        
        self.a = value_vector[0:self.num_players] #a of each player
        value_vector = value_vector[self.num_players:]
        
        #p2 vector
        self.P2matrix = value_vector[0:self.num_players] 
        
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
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.num_players = 3
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
    def __init__(self, input_dim, output_dim):
        self.main_net = DQN(input_dim, output_dim)
        self.target_net = copy.deepcopy(self.main_net)
        
        # Define optimizer used (SGD, etc)
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),lr=0.001)
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        
        # Predicts resultant values, input a State object, outputs a FittedValues object
    def predict(self, input):
        a,b = self.main_net.forward(self.stateTransform(input))
        return self.tensorTransform(a,b)
    
    def predict_targetNet(self, input):
        a,b = self.target_net.forward(self.stateTransform(input))
        return self.tensorTransform(a,b)
        
    # Transforms state object into tensor
    def stateTransform(self, s):
        return torch.tensor(s.getNormalizedState()).float()
        
    # Transforms output tensor into FittedValues Object
    def tensorTransform(self, output1, output2):
        return FittedValues(output1, output2)
    
    
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
        self.total_reward = np.zeros(self.N, dtype=np.float32 )

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
    
if __name__ == '__main__':
    num_players = 3
    T = 5
    #replay_stepnum = 3
    batch_update_size = 100
    num_sim = 10000
    max_action = 20
    update_net = 50
    
    #number of parameters that need to be estimated by the network
    parameter_number = 4*num_players
    
    #network object
    net = NashNN(2+num_players,parameter_number)
    
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
    states = [current_state]
    
    #exploration chance
    ep = 0.5
    
    #intialize relay memory
    replay = ExperienceReplay(200)
    
    sum_loss = np.zeros(num_sim)
    total_l = 0
    
    for k in range (0,num_sim):
        epsilon = ep - ep*(k/(num_sim-1))
        total_l = 0
        
        if k%50 == 0:
            print("Starting Inventory: ", sim.get_state()[0][0])
            flag = True
            
        if k%update_net == 0:
                net.target_net = copy.deepcopy(net.main_net)
            
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
            
            #adds experience to replay memory
            replay.add((current_state,a,new_state,lr))
    #        explist = []
    #        for j in range(max(i-(replay_stepnum-1),0),i+1):
    #            explist.append((states[j],Actions[i,:],rewards[i,:]))
    #        replay.add(explist)
            
            if (k-1)*T > batch_update_size:
                replay_sample = replay.sample(batch_update_size)
                
                #samples from priority and calculate total loss
                estimated_q = []
                actual_q = []
                replay_sample.append((current_state,a,new_state,lr))
                #print("entire sample", replay_sample)
                
                for sample in (replay_sample):
                    #obtain estimated current state nash value
                    output = net.predict(sample[0])
                    #obtain estimated next state nash value
                    next_NN_value = net.predict_targetNet(sample[2])
                    #sample[0].print_state()
                    #cycle through all agents
                    for agent_num in range(0,num_players):
                        
                        #convert experience replay action/reward to auto_grad variables
                        sample_a = torch.tensor(sample[1]).float()
                        sample_lr= sample[3]
                        #print("current sample:",sample)
                        
                        #define the vector mu_{-1} and estimatibles vector p_{i,2}, and mu_{-1}
                        if agent_num == 0:
                            other_agenta = sample_a[1:]
                            p2 = output.P2matrix[1:]
                            other_agentmu = output.mu[1:]
                        elif agent_num == num_players-1:
                            other_agenta = sample_a[0:-1]
                            p2 = output.P2matrix[0:-1]
                            other_agentmu = output.mu[0:-1]
                        else:
                            other_agenta = torch.cat((sample_a[0:agent_num],sample_a[agent_num+1:]))
                            p2 = torch.cat((output.P2matrix[0:agent_num],output.P2matrix[agent_num+1:]))
                            other_agentmu = torch.cat((output.mu[0:agent_num],output.mu[agent_num+1:]))
                        
                        #calculate actual value of state-action pair for agent (based on experienced reward + next state estimated nash)
                        if sample[0].t == T-1:
                            actual_q.append(sample_lr[agent_num])
                            #calculate estimated value of state-action pair for agent(based on network output using advantage function)
                            estimated_q.append(output.V[agent_num])
                            #flag = True
                            #print(sample_lr[agent_num])
                        else:
                            actual_q.append(next_NN_value.V[agent_num] + sample_lr[agent_num])

                            #actual_q.append(next_NN_value.V[agent_num].data.numpy() + sample_lr[agent_num])
                            #calculate estimated value of state-action pair for agent(based on network output using advantage function)
                            estimated_q.append(output.V[agent_num]-0.5*(sample_a[agent_num]-output.mu[agent_num])*output.P1[agent_num]*(sample_a[agent_num]-output.mu[agent_num])+
                                           (sample_a[agent_num]-output.a[agent_num])*p2.dot(other_agenta-other_agentmu))
                        
                
                #convert list of tensors into tensor and set actual_q to non-autograd variable (since its fixed value)
                estimated_q = torch.stack(estimated_q)
                actual_q = torch.tensor(actual_q)
                #actual_q = Variable(torch.from_numpy(np.array(actual_q, dtype=np.float32)))
                #actual_q = actual_q.detach()
                
                #define loss function, calc gradients and update
                #if flag:
                    #print(net.predict(current_state).V.data.numpy())
                    #print(lr)
                    #print(estimated_q,actual_q)
                
                if (flag):
                    print(current_state.p,a)
                    print("Estimated Reward",estimated_q[-num_players:])
                    print("Actual Reward",actual_q[-num_players:])
                
                loss = net.criterion(estimated_q, actual_q)
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                
                #totals loss (of experience replay) per time step
                total_l += sum(map(lambda a,b:(a-b)*(a-b),estimated_q,actual_q))
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
        
    plt.plot(sum_loss)
        
    