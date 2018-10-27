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
        
    # Predicts resultant values, input a State object, outputs a NashFittedValues object
    def predict(self, state):
        mu = [] # list of Nash actions
        
        for i in range(0,num_players):
            norm_state = state.getNormalizedState()
            #Moves the i'th agent's inventory to the front, rest of the list order is preserved
            norm_state = np.delete(np.insert(norm_state,0,norm_state[i]),i+1)
            #Evaluate net
            a,b = self.main_net.forward(torch.tensor(norm_state).float())
            output = self.tensorTransform(a,b)
            #Adds i'th agent Nash action 
            mu.append(output.mu)
            #For all other estimated variables, take only the first (ie using only natural ordered inventories) agent's predicted values
            if i == 0:
                first_output = output
        
        return NashFittedValues(first_output,torch.stack(mu))
        
    # Transforms output tensor into FittedValues Object
    def tensorTransform(self, output1, output2):
        return FittedValues(output1, output2, self.num_players)
    
    #takes a tuple of transitions and outputs loss
    def compute_Loss(self,state_tuple):
        currentState, action, nextState, reward = state_tuple[0], torch.tensor(state_tuple[1]).float(), state_tuple[2], state_tuple[3]
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
        
        return torch.sum(torch.stack(loss)**2)
        
    def updateLearningRate(self):
        self.counter += 1
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),lr=0.005-(0.005-0.001)*self.counter/self.num_sim)
    
# Set global digit printing options
np.set_printoptions(precision=4)

# Define truncation function
def trunc_array(a,max_a):
    """
    Truncate a into [-max_a,max_a]
    :param a: array / tensor to truncate
    :param max_a: truncation threshold (max_a > 0)
    :return: Truncated Array
    """
    lt, gt = a < -max_a, a > max_a
    return a * (1 - lt - gt) - lt * max_a + gt*max_a

# Define Market Game Simulation Object
if __name__ == '__main__':

    # Define Training and Model Parameters
    num_players = 2
    T = 2
    
    #number of parameters that need to be estimated by the network
    parameter_number = 2*num_players +4
    batch_update_size = 200
    num_sim = 5000
    max_action = 20
    update_net = 50
    buffersize = 300

    # Random Action Probability
    eps = 0.5
    
    # Set number of output variables needed from net:
    # Supposed to be = 1 x ( V + c1 + c2 + c3 + mu + a). Note: Need to make more robust
    parameter_number = 3 * num_players + 3
    # parameter_number = 6
    
    # Initialize NashNN Agents
    nash_agent = NashNN(2+num_players,parameter_number,num_players,T)
    
    # Package Simulation Parameters
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
    
    current_state = sim.get_state()[0]

    all_states = []
    states = []
    
    all_predicts = []
    preds = []
    
    #exploration chance
    ep = 0.5

    # Intialize relay memory
    replay = ExperienceReplay(buffersize)

    sum_loss = np.zeros(num_sim)
    total_l = 0
    
    for k in range (0,num_sim):


        eps = max( eps - (eps-0.05)*(k/(num_sim-1)), 0 )
        total_l = 0        
#        if k%20 == 0:
#            print("Starting Inventory: ", sim.get_state()[0][0])
#            flag = True
            
#        if k%update_net == 0:
#                net.target_net = copy.deepcopy(net.main_net)
            
        for i in range(0,T):   
            current_state,lr,tr = sim.get_state()
            #print("funny state:", state[0])
#            state_dict = {'time_step':i,
#                          'current_inventory':np.copy(state[0]),
#                          'current_price':state[1]}
#            current_state = State(state_dict)

        print_flag = not k % 20

        if print_flag : print("New Simulation: \n", sim.get_state()[0])

        for i in range(0,T):

            current_state,lr,tr = sim.get_state()
            
            #### Begin Action Generation Block ####
            ## Note to self: Will need to take deeper look

            if np.random.random() < eps:
                #random action between buying/selling 20 shares for all players
                a = nash_agent.predict(current_state).mu.data.numpy()
                a = a + np.random.normal(0, 2.5, num_players)
            else:
                #else take predicted nash action
                a = nash_agent.predict(current_state).mu.data.numpy()

            a = trunc_array(a, max_action)

            #### End Action Generation Block ####
            
            # Take Chosen Actions and Take Step
            sim.step(a)
            new_state,lr,tr = sim.get_state()

            #updates storage variables
            states.append(new_state)
            prices[i] = new_state.p
            Qs[i,:] = new_state.q
            Actions[i,:] = a
            rewards[i] = lr
            #preds.append[nash_agent.predict(current_state).V.data()]
            
            #creates experience element
            experience = (current_state,a,new_state,lr)
            #computes loss on new experience
            new_loss = nash_agent.compute_Loss(experience).data.numpy()

        
            # Append Transition to replay memory
            replay.add(experience)
            # replay.add(new_loss,experience)

            # Sample from replay buffer
            replay_sample = replay.sample(batch_update_size)
            # replay_sample, index, weights = replay.sample(batch_update_size)

            
            
            loss = []
            #defined new list for replay - can delete and switch to loss later
            pro_loss = []
            
            for sample in replay_sample: 
                cur_loss = nash_agent.compute_Loss(sample)
                loss.append(cur_loss)
                pro_loss.append(cur_loss.data.numpy().item())
                

            loss = torch.sum(torch.stack(loss))
            
            nash_agent.optimizer.zero_grad()
            loss.backward()
            nash_agent.optimizer.step()
            
            # Update priority replay
            # replay.update(index,pro_loss)

            
            cur_loss = nash_agent.compute_Loss(experience).data.numpy()
            total_l += cur_loss

            if (print_flag):
                curVal = nash_agent.predict(current_state)
                print("{} , Action: {}, Loss: {}".\
                      format( current_state, curVal.mu.data.numpy(), cur_loss ) )


                
        #defines loss per period
        sum_loss[k] = total_l

        #resets simulation
        sim.reset()
        all_states.append(states)
        all_predicts.append(preds)
        nash_agent.updateLearningRate()
        
    plt.plot(sum_loss)
        
    