import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nashRL_netlib import *

#-----------------------
# Define Object containing estimated coefficients and state value for 
# an individual player (i.e. v_1, mu_1, c1_1, c2_1, c3_1)
# ***NOTE*** All elements are tensors
#-----------------------
class FittedValues(object):
    def __init__(self, value_vector, num_players):
        """
        :param value_vector:
        :param num_players:
        """
        self.num_players = num_players  # number of players in game

        self.mu = value_vector[0] #mean of current player
        
        self.c1 = (value_vector[1])**2 #P1 matrix for each player, exp transformation to ensure positive value
        
        #self.a = value_vector[0] #a of current player
        
        self.c2 = value_vector[2] 
        
        self.c3 = value_vector[3]
        
#-----------------------
# Define Object containing estimated coefficients and state value for 
# ALL players
# ***NOTE*** All elements are tensors
#-----------------------
class NashFittedValues(object):
    # Initialized via a single numpy vector
    def __init__(self, FV_list):
        """
        :param FV_list:
        """
        self.num_players = FV_list[0].num_players #number of players in game
        #self.a = [] #vector of all agents
        self.c1 = []
        self.c2 = []
        self.c3 = []
        self.mu = []
        
        for item in FV_list:
            #self.a.append(item.a)
            self.c1.append(item.c1)
            self.c2.append(item.c2)
            self.c3.append(item.c3)
            self.mu.append(item.mu)
        
        #self.a = torch.stack(self.a)
        self.c1 = torch.stack(self.c1)
        self.c2 = torch.stack(self.c2)
        self.c3 = torch.stack(self.c3)
        self.mu = torch.stack(self.mu)

#--------------------
# Class containing all functions related to nash agent, including loss functions,
# and prediction functions
#--------------------
class NashNN():
    def __init__(self, input_dim, output_dim, nump, t, t_cost, term_cost):
        """
        :param input_dim: 
        :param output_dim:
        :param nump:
        :param t:
        :param t_cost:
        :param term_cost:
        """
        # Simulation Parameters
        self.num_players = nump         # number of agents
        self.T = t                      # total number of time steps
        self.num_sim = 5000             # total number of simulations - unused
        self.transaction_cost = t_cost  # estimated transaction cost coefficient
                                        # user inputed
        self.term_costs = term_cost     # estimated terminal costs coefficent
                                        # user inputed
        
        # Initialize Networks
        ####### Action Network
        self.action_net = DQN(input_dim, output_dim, self.num_players)
        ####### Value Network
        self.value_net = ValueNet(input_dim,output_dim,self.num_players)
        
        # Define optimizer used (SGD, etc)
        ####### Optimizer for Action Net
        self.optimizer_DQN = optim.RMSprop(self.action_net.main.parameters(),
                                       lr=0.005)
        ####### Optimizer for Value Net
        self.optimizer_value = optim.RMSprop(self.value_net.main.parameters(),
                                       lr=0.01) 
        
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        
        # Counter for decaying learning rate, currently not being used
        self.counter = 0

    def slice(self,X,i):
        "Return all items in array except for i^th item"
        return torch.cat([X[0:i], X[i + 1:]])

    #-------------------
    # Outputs the estimated nash action and related coefficients for each agent:
    # i.e. mu, c1, c2, c3 
    # in the form of a NashFittedValues Object
    #-------------------
    def predict_action(self, state):
        FV_list = [] # list of Nash actions
        
        for i in range(0,self.num_players):
            norm_state = state.getNormalizedState()
            #Moves the i'th agent's inventory to the front, rest of the list order is preserved
            norm_state = np.delete(np.insert(norm_state,2,norm_state[i+2]),i+3)
            #Evaluate net
            output = FittedValues(self.action_net.forward(torch.tensor(norm_state).float()), self.num_players)
            #Adds i'th agent Nash action 
            FV_list.append(output)
        
        return NashFittedValues(FV_list)
    
    #-------------------
    # Outputs the estimated nash action and related coefficients for each agent:
    # i.e. mu, c1, c2, c3 
    # in the form of a NashFittedValues Object
    # *****NOTE: SAME FUNCTION AS ABOVE, ONLY USED FOR DEBUGGING PURPOSES******
    #-------------------
    def predict_action_print(self, state):
        FV_list = [] # list of Nash actions
        
        for i in range(0,self.num_players):
            norm_state = state.getNormalizedState()
            #Moves the i'th agent's inventory to the front, rest of the list order is preserved
            norm_state = np.delete(np.insert(norm_state,2,norm_state[i+2]),i+3)
            #Evaluate net
            output = FittedValues(self.action_net.forward(torch.tensor(norm_state).float()), self.num_players)
            #Adds i'th agent Nash action 
            FV_list.append(output)
            
        out = NashFittedValues(FV_list)
        print("Nash Actiion: {}, Constant 1: {}, Constant 2: {}, Constant 3: {}" .\
              format(out.mu.data.numpy(),out.c1.data.numpy(),out.c2.data.numpy(),out.c3.data.numpy()))
        return out

    #--------------------
    # Returns the estimated Nash Value of each agent in a given state
    #--------------------
    def predict_value(self, state):
        norm_state = state.getNormalizedState()
        return self.value_net.forward(torch.tensor(norm_state).float())

    #--------------------
    # Returns the squared loss of Nash Value estimates at a particular state
    #--------------------
    def compute_value_Loss(self,state_tuple):
        currentState, action, nextState, reward = state_tuple[0], torch.tensor(state_tuple[1]).float(), \
                                                          state_tuple[2], state_tuple[3]
        # Indicator of whether current state is last state or not
        # Flag = 1 if last state
        # Flag = 0 otherwise
        flag = 0
        if nextState.t > self.T - 1:
            flag = 1
            
        # Returns the squared loss with target being:
        # Sum of current state's reward + estimated next state's nash value     --- if not last state
        # Reward obtained if agent executes action to net out current inventory --- if last state
        return self.criterion(self.predict_value(currentState), torch.tensor(flag*(currentState.q*currentState.p \
                              - self.transaction_cost*currentState.q**2)).float() + (1-flag)*(self.predict_value(nextState).detach()+torch.tensor(reward).float()))  
    
    #--------------------
    # Returns the loss of Action Coefficient estimates at a particular state
    #--------------------
    def compute_action_Loss(self, state_tuple):
        currentState, action, nextState, reward = state_tuple[0], torch.tensor(state_tuple[1]).float(), \
                                                          state_tuple[2], state_tuple[3]
        # Squared Loss penalty for Difference in Coefficients (c1,c2,c3) between agents 
        penalty = 50
        
        # Indicator of whether current state is last state or not
        # Flag = 1 if last state
        # Flag = 0 otherwise
        flag = 0
        if nextState.t > self.T - 1:
            flag = 1

        # Predict some values
        curVal = self.predict_action(currentState)          # Nash Action Coefficient Estimates of current state
        curNash = self.predict_value(currentState).detach() # Nash Value of Current state
        nextVal = self.predict_value(nextState).detach()    # Nash Value of Next state
        
        # Initialize Loss
        loss = []

        # Loops through each agent and appends to loss vector, then stack and sum 
        # determine final loss
        for i in range(0, self.num_players):
            # Define advantage function
            A = lambda u, uNeg, mu, muNeg, c1, c2, c3: -0.5 * c1 * (u - mu) ** 2 - c2 * (u - mu) * torch.sum(
                uNeg - muNeg) - c3 * (uNeg - muNeg) ** 2
                    
            # Returns squared Loss of the Difference between (Advantage Function + Estimated Current Nash Value) and :
            #   Estimated next state's Nash Value + current rewards               ---- if not last state
            #   Rewards obtained by netting out all remaining inventory 
            #       with terminal penalty + current rewards                         ---- if last state
            # + Tuning Penalty applied to squared difference of coefficients (c1,c2,c3) between agents
            loss.append(((1 - flag) * nextVal[i] + flag * (nextState.q[i]) * (nextState.p - self.term_costs * (nextState.q[i]))
                        + reward[i]
                        - A(action[i], self.slice(action,i), curVal.mu[i], self.slice(curVal.mu,i), curVal.c1[i], curVal.c2[i],
                            curVal.c3[i])
                        - curNash[i])**2 
                        + penalty*(curVal.c1[0]-curVal.c1[1])**2
                        + penalty*(curVal.c2[0]-curVal.c2[1])**2
                        + penalty*(curVal.c3[0]-curVal.c3[1])**2)
        return torch.sum(torch.stack(loss))

    #--------------------
    # Updates Learning Rate --- Currently not being used
    #--------------------
    def updateLearningRate(self):
        self.counter += 1
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),
                                       lr=0.01 - (0.01 - 0.003) * self.counter / self.num_sim)

#------- Currently not being used
class NashNN2:
    def __init__(self, num_players=2, control_size=1, state_size=2):

        # Store Relevant Variables
        self.num_players = num_players
        self.control_size = control_size
        self.state_size = state_size

        # Compute Output Size (V + mu + c1 + c2 + c3 + a)
        self.output_dim = 6

        # Compute Input Size : X = (t,S,Q1,...,QK)
        self.input_size = self.state_size + self.num_players*self.control_size

        # Initialize DQN
        self.NNmodel = DQN2(input_dim,output_dim)

        # # Initialize Permutation Invariant NN-Model. Takes arguments (S,t,Q)
        # self.NNmodel = \
        #     PermInvariantQNN(self.num_players * self.control_size,
        #                      2, 3 + self.num_players * 2, block_size=1, num_moments=5)

    def predict(self, input):
        """
        :param input: Tensor of State Vectors, assumed to be in order (t,p,q)
        :return:
        """
        self.NNmodel.forward()

