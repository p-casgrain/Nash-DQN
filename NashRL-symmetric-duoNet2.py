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
from nashRL_netlib import *
from nashRL_DQlib import *
from prioritized_memory import *
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
        
        self.mu = value_vector[0] #mean of current player
        value_vector = value_vector[1:]
        
        self.P1 = (value_vector[0])**2 #P1 matrix for each player, exp transformation to ensure positive value
        value_vector = value_vector[1:]
        
        self.a = value_vector[0:self.num_players] #a of each player
        value_vector = value_vector[self.num_players:]
        
        #p2 vector
        self.P2 = value_vector[0] 
        
        self.P3 = value_vector[1]
        
class NashFittedValues(object):
    # Initialized via a single numpy vector
    def __init__(self, fittedval, mu_vec):
        self.num_players = fittedval.num_players #number of players in game
        self.V = fittedval.V #nash value vector
        self.a = fittedval.a
        self.P1 = fittedval.P1
        self.P2 = fittedval.P2
        self.P3 = fittedval.P3
        self.mu = mu_vec
        
# Define an object that summarizes all state variables
class State(object):
    def __init__(self,param_dict):
        self.t = param_dict['time_step']
        self.q = param_dict['current_inventory']
        self.p = param_dict['current_price']
        self.original_q = param_dict['current_inventory']
        
    # Returns list representation of all state variables normalized to be [-1,1]
    # Not done yet
    def getNormalizedState(self):
        norm_q = self.q/10
        norm_p = (self.p-10)/10
        norm_t = self.t/4-1
        return np.array(np.append(np.append(norm_q,norm_p),norm_t))
    
    def print_state(self):
        print("t =", self.t, "q = ", self.q, "p = ", self.p)
        
    def permute(self,player_num):
        self.q = np.delete(np.insert(self.q,0,self.q[player_num]),player_num+1)
        
    def reset_q(self):
        self.q = self.original_q
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
        self.num_sim = 5000
        # Define optimizer used (SGD, etc)
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),lr=0.005)
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        self.counter = 0
        
        # Predicts resultant values, input a State object, outputs a FittedValues object
    def predict(self, state):
        mu = []
        #first_output = False
        for i in range(0,num_players):
            state.permute(i)
            #print(state.q)
            a,b = self.main_net.forward(self.stateTransform(state))
            output = self.tensorTransform(a,b)
            mu.append(output.mu)
            if i == 0:
                first_output = output
            state.reset_q()
            #print(state.q)
        
        return NashFittedValues(first_output,torch.stack(mu))
    
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
        currentState, action, nextState, reward, isNash = state_tuple[0], torch.tensor(state_tuple[1]).float(), state_tuple[2], state_tuple[3], state_tuple[4]
        #Q = lambda u, uNeg, mu, muNeg, a, v, c1, c2, c3: v - 0.5*c1*(u-mu)**2 + c2*(u -a)*torch.sum(uNeg - muNeg) + c3*(uNeg - muNeg)**2
        nextVal = self.predict(nextState).V
        flag = 0
        
        #set next nash value to be 0 if last time step
        if nextState.t > self.T-1:
            flag = 1
        
        curVal = self.predict(currentState)
        loss = []
        
        for i in range(0,self.num_players):
            r = lambda T : torch.cat([T[0:i], T[i+1:]])
            A = lambda u, uNeg, mu, muNeg, a, c1, c2, c3: 0.5*c1*(u-mu)**2 + c2*(u -a)*torch.sum(uNeg - muNeg) + c3*(uNeg - muNeg)**2
            loss.append((1-flag)*nextVal[i] + flag*nextState.q[i]*(nextState.p-50*nextState.q[i])  + reward[i] - curVal.V[i] + A(action[i],r(action),curVal.mu[i],r(curVal.mu),curVal.a[i],curVal.P1,curVal.P2,curVal.P3))
        
#        if all(isNash):
#            for i in range(0,self.num_players):
#                loss.append(nextVal[i] + reward[i] - curVal.V[i])
#        else:
#            #note that this assumes that at most one person did not take nash action
#            for i in range(0,self.num_players):
#                r = lambda T : torch.cat([T[0:i], T[i+1:]])
#                if isNash[i]:
#                    loss.append(nextVal[i] + reward[i] - curVal.V[i].detach() - curVal.P2*(action[i] -curVal.a[i])*torch.sum(r(action) - r(curVal.mu)) - curVal.P3*(torch.sum(r(action) - r(curVal.mu))**2))
#                else:
#                    loss.append(nextVal[i] + reward[i] - curVal.V[i].detach() + 0.5*curVal.P1*(action[i]-curVal.mu[i])**2)
#                    #A(action[i],r(action),curVal.mu[i],r(curVal.mu),curVal.a[i],curVal.V[i].detach(),curVal.P1,curVal.P2,curVal.P3)
                    

        
        return torch.sum(torch.stack(loss)**2)
        
    def updateLearningRate(self):
        self.counter += 1
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),lr=0.005-(0.005-0.001)*self.counter/self.num_sim)
    
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


#class ExperienceReplay:
#    #each experience is a list of with each tuple having:
#    #first element: state,
#    #second element: array of actions of each agent,
#    #third element: array of rewards received for each agent
#    def __init__(self, buffer_size):
#        self.buffer = []
#        self.buffer_size = buffer_size
#    
#    def add(self,experience):
#        if len(self.buffer) > self.buffer_size:
#            self.buffer.pop(0)
#        self.buffer.append(experience)
#        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),lr=0.001-(0.001-0.0005)*self.counter/self.num_sim)


# Define Market Game Simulation Object
if __name__ == '__main__':
    num_players = 2
    T = 2
    #replay_stepnum = 3
    batch_update_size = 100
    num_sim = 5000
    max_action = 20
    update_net = 50
    buffersize = 400
    
    #number of parameters that need to be estimated by the network
    parameter_number = 2*num_players +4
    
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
    # replay = PrioritizedMemory(buffersize)
    replay = ExperienceReplay(buffersize)

    sum_loss = np.zeros(num_sim)
    total_l = 0
    
    for k in range (0,num_sim):
        epsilon = ep - (ep-0.05)*(k/(num_sim-1))
        total_l = 0
        
        if k%20 == 0:
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
                if np.random.random() < epsilon:
                    #random action between buying/selling 20 shares for all players
                    a = net.predict(current_state).mu.data.numpy()
                    a = a + np.random.normal(0, 2.5, num_players)
#                    rand_player = np.random.randint(0,num_players)
#                    a[rand_player] = np.random.rand()*40-20
                    isNash = [True] * num_players
#                    isNash[rand_player] = False
                else:
                    #else take predicted nash action
                    a = net.predict(current_state).mu.data.numpy()
                    isNash = [True] * num_players
                #a = -current_state.q
                #isNash = [True] * num_players
                #print (net.predict(current_state).V.data.numpy())
            else:
                if np.random.random() < epsilon:
                    #random action between buying/selling 20 shares for all players
                    a = net.predict(current_state).mu.data.numpy()
                    a = a + np.random.normal(0, 2.5, num_players)
#                    rand_player = np.random.randint(0,num_players)
#                    a[rand_player] = np.random.rand()*40-20
                    isNash = [True] * num_players
#                    isNash[rand_player] = False
                else:
                    #else take predicted nash action
                    a = net.predict(current_state).mu.data.numpy()
                    isNash = [True] * num_players
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
            
            #creates experience element
            experience = (current_state,a,new_state,lr,isNash)
            #computes loss on new experience
            new_loss = net.compute_Loss(experience).data.numpy()

            #adds new experience to replay memory
            # replay.add(new_loss,experience)
            replay.add(experience)

    #        explist = []
    #        for j in range(max(i-(replay_stepnum-1),0),i+1):
    #            explist.append((states[j],Actions[i,:],rewards[i,:]))
    #        replay.add(explist)
            
            #if (k-1)*T > batch_update_size:
            # replay_sample, index, weights = replay.sample(batch_update_size)
            replay_sample = replay.sample(batch_update_size)

            
            
            loss = []
            #defined new list for replay - can delete and switch to loss later
            pro_loss = []
            
            for sample in replay_sample: 
                cur_loss = net.compute_Loss(sample)
                loss.append(cur_loss)
                pro_loss.append(cur_loss.data.numpy().item())
                
                
#            if (flag):
#                print("Transition: ",current_state.p,a,current_state.q)
#                new_V = net.predict(new_state).V.data.numpy()
#                if i == T-1:
#                    new_V = np.zeros(num_players)
#                print("Actual Value: ",new_V+lr)
#                print("Last Reward: ",lr)
#                print("Next V: ", new_V)
#                print("Predicted State Value: ", net.predict(current_state).V.data.numpy())
#                curVal = net.predict(current_state)
#                temp = []
#                action = torch.tensor(a).float()
#                for j in range(0,num_players):
#                    r = lambda w : torch.cat([w[0:j], w[j+1:]])
#                    A = lambda u, uNeg, mu, muNeg, a, v, c1, c2, c3: v - 0.5*c1*(u-mu)**2 + c2*(u -a)*torch.sum(uNeg - muNeg) + c3*(uNeg - muNeg)**2
#                    temp.append(A(action[j],r(action),curVal.mu[j],r(curVal.mu),curVal.a[j],0,curVal.P1,curVal.P2,curVal.P3).data.numpy().item())
#                formTemp = [ '%.2f' % elem for elem in temp ]
#                if i == T-1:
#                    formTemp = net.predict(current_state).V.data.numpy()
#                print("Predicted Q Value: ", formTemp)
#                print("Predicted Nash Action: ", curVal.mu.data.numpy())
#                print("")
            
            #loss = net.criterion(estimated_q, actual_q)
            loss = torch.sum(torch.stack(loss))
            
            net.optimizer.zero_grad()
            loss.backward()
            net.optimizer.step()
            
            #update priority replay
            # replay.update(index,pro_loss)

            
            cur_loss = net.compute_Loss(experience).data.numpy()
            total_l += cur_loss
            
            if (flag):
                curVal = net.predict(current_state)
                print("Transition: ",current_state.p,current_state.q,a,new_state.q)
                print("Predicted Nash Action: ", curVal.mu.data.numpy())
                print(cur_loss)

                
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
        
    