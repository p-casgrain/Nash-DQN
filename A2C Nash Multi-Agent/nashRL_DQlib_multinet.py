import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nashRL_netlib import *
        
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
        self.mu = FV_list[:,0]
        self.c1 = FV_list[:,1]
        self.c2 = FV_list[:,2]
        self.c3 = FV_list[:,3]
        

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
        self.action_net = PermInvariantQNN(in_invar_dim = self.num_players - 1, non_invar_dim = 3, out_dim = output_dim)
        ####### Value Network
        #self.value_net = ValueNet(input_dim,output_dim,self.num_players)
        self.value_net = PermInvariantQNN(in_invar_dim = self.num_players - 1, non_invar_dim = 3, out_dim = 1)
        
        # Define optimizer used (SGD, etc)
        ####### Optimizer for Action Net
        self.optimizer_DQN = optim.RMSprop(list(self.action_net.moment_encoder_net.parameters()) + list(self.action_net.decoder_net.parameters()),
                                       lr=0.005)
        ####### Optimizer for Value Net
        self.optimizer_value = optim.RMSprop(list(self.value_net.moment_encoder_net.parameters()) + list(self.value_net.decoder_net.parameters()),
                                       lr=0.01) 
        
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        
        # Counter for decaying learning rate, currently not being used
        self.counter = 0

    def slice(self,X,i):
        "Return all items in array except for i^th item"
        return torch.cat([X[0:i], X[i + 1:]])
    
    #------------------
    #Returns a matrix of uNeg or muNeg
    #------------------
    def matrix_slice(self,X):
        mat = []
        for i in range(0,len(X)):
            for j in range(0,self.num_players):
                mat.append(self.slice(X[i,:],j))
        return torch.stack(mat)
    
    #-------------------
    # Expands a list of states into a matrix with rows being blocks of agents for each state 
    # for each state columns being normalized state parameters
    #-------------------
    def expand_list(self,state_list):
        expanded_states = []
        for cur_s in state_list:
            s = cur_s.getNormalizedState()
            for j in range(0,self.num_players):
                expanded_states.append(np.append(np.append(np.array([s[0],s[1],s[2+j]]),s[2:2+j]),s[2+j+1:]))
        return np.array(expanded_states)
    
    #-------------------
    # Expands a list of states into a matrix with rows being blocks of agents for each state 
    # for each state columns being normalized state parameters
    #-------------------
    def expand_list_untransformed(self,state_list):
        expanded_states = []
        for cur_s in state_list:
            s = cur_s.getState()
            for j in range(0,self.num_players):
                expanded_states.append(np.append(np.append(np.array([s[0],s[1],s[2+j]]),s[2:2+j]),s[2+j+1:]))
        return np.array(expanded_states)
    
    #-------------------
    # Outputs the estimated nash action and related coefficients for each agent:
    # i.e. mu, c1, c2, c3 
    # in the form of a NashFittedValues Object
    #-------------------
    def predict_action(self, states):
        expanded_states = torch.tensor(self.expand_list(states)).float()
        action_list = self.action_net.forward(invar_input = expanded_states[:,3:], non_invar_input = expanded_states[:,0:3])
        
        NFV_list = []
        for i in range(0,len(states)):
            NFV_list.append(NashFittedValues(action_list[i*self.num_players:(i+1)*self.num_players,:]))
        
        return NFV_list
    
    #--------------------
    # Returns the estimated Nash Value of each agent in a list of states
    #--------------------
    def predict_value(self, states):
        expanded_states = torch.tensor(self.expand_list(states)).float()
        return self.value_net.forward(invar_input = expanded_states[:,3:], non_invar_input = expanded_states[:,0:3])

    #--------------------
    # Returns the squared loss of Nash Value estimates at a particular state
    #--------------------
    def compute_value_Loss(self,state_tuples):
        cur_state_list = [tup[0] for tup in state_tuples]
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = [tup[3] for tup in state_tuples]
        # Indicator of whether current state is last state or not
        # Flag = 1 if last state
        # Flag = 0 otherwise
        flag = np.repeat(np.array([s.t > self.T - 1 for s in next_state_list]).astype(int),self.num_players)
        
        target = self.predict_value(cur_state_list).view(-1)
        
        expanded_states = self.expand_list_untransformed(cur_state_list)
        expanded_next_states = self.expand_list_untransformed(next_state_list)

        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_states[:,1],expanded_states[:,2]/2,self.transaction_cost*np.ones(len(expanded_states))))) \
                    + np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_states[:,2]/2,self.transaction_cost*np.ones(len(expanded_states)))))
        nextstate_val = self.predict_value(next_state_list).detach().view(-1).data.numpy()
        
        # Returns the squared loss with target being:
        # Sum of current state's reward + estimated next state's nash value     --- if not last state
        # Reward obtained if agent executes action to net out current inventory --- if last state
        return self.criterion(target, torch.tensor(np.multiply(flag,term_list) + np.multiply(np.ones(len(expanded_states)) - flag, nextstate_val + np.array(reward_list).flatten())).float())
        
        
    def compute_value_Loss_print(self,state_tuples):
        cur_state_list = [tup[0] for tup in state_tuples]
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = [tup[3] for tup in state_tuples]
        # Indicator of whether current state is last state or not
        # Flag = 1 if last state
        # Flag = 0 otherwise
        flag = np.repeat(np.array([s.t > self.T - 1 for s in next_state_list]).astype(int),self.num_players)
        
        target = self.predict_value(cur_state_list).view(-1)
        
        expanded_states = self.expand_list_untransformed(cur_state_list)
        expanded_next_states = self.expand_list_untransformed(next_state_list)
        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_states[:,1],expanded_states[:,2]/2,self.transaction_cost*np.ones(len(expanded_states))))) \
                    + np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_states[:,2]/2,self.transaction_cost*np.ones(len(expanded_states)))))
        nextstate_val = self.predict_value(next_state_list).detach().view(-1).data.numpy()
        
        # Returns the squared loss with target being:
        # Sum of current state's reward + estimated next state's nash value     --- if not last state
        # Reward obtained if agent executes action to net out current inventory --- if last state
        return self.criterion(target, torch.tensor(np.multiply(flag,term_list) + np.multiply(np.ones(len(expanded_states)) - flag, nextstate_val + np.array(reward_list).flatten())).float())
        
    #--------------------
    # Returns the loss of Action Coefficient estimates at a particular state
    #--------------------
    def compute_action_Loss(self, state_tuples):
        # Squared Loss penalty for Difference in Coefficients (c1,c2,c3) between agents 
        penalty = 25
        
        cur_state_list = [tup[0] for tup in state_tuples]
        action_list = torch.tensor([tup[1] for tup in state_tuples])
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = torch.tensor([tup[3] for tup in state_tuples]).float()
        
        # Indicator of whether current state is last state or not
        # Flag = 1 if last state
        # Flag = 0 otherwise
        flag = np.repeat(np.array([s.t > self.T - 1 for s in next_state_list]).astype(int),self.num_players)
        
        # Predict some values
        curAct = self.predict_action(cur_state_list)                                    # Nash Action Coefficient Estimates of current state
        curVal = self.predict_value(cur_state_list).detach().view(-1)      # Nash Value of Current state
        nextVal = self.predict_value(next_state_list).detach().view(-1).data.numpy()    # Nash Value of Next state
        
        #Makes Matrix of Terminal Values
        expanded_next_states = self.expand_list_untransformed(next_state_list)
        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_next_states[:,2],self.term_costs*np.ones(len(expanded_next_states)))))
        
        # Create Lists for predicted Values
        c1_list = torch.stack([nfv.c1 for nfv in curAct]).view(-1)
        c2_list = torch.stack([nfv.c2 for nfv in curAct]).view(-1)
        c3_list = torch.stack([nfv.c3 for nfv in curAct]).view(-1)
        mu_list = torch.stack([nfv.mu for nfv in curAct])
        
        #Creates the Mu_neg and u_Neg Matrices
        uNeg_list = torch.tensor(self.matrix_slice(action_list)).float()
        muNeg_list = self.matrix_slice(mu_list)
        act_list = torch.tensor(action_list.view(-1)).float()
        mu_list = mu_list.view(-1)

        #Computes the Advantage Function using matrix operations
        A = - c1_list * (act_list-mu_list)**2 / 2 - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list,
                        dim = 1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2,dim = 1) / 2

        return torch.sum((torch.tensor(np.multiply(np.ones(len(curVal))-flag, nextVal) + np.multiply(flag, term_list) + reward_list.view(-1)).float()
                          - curVal - A)**2 
                        + penalty*torch.var(c1_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                        + penalty*torch.var(c2_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                        + penalty*torch.var(c3_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1))
        
    #--------------------
    # Returns the loss of Action Coefficient estimates at a particular state
    #--------------------
    def compute_action_Loss_print(self, state_tuples):
        # Squared Loss penalty for Difference in Coefficients (c1,c2,c3) between agents 
        penalty = 25
        
        cur_state_list = [tup[0] for tup in state_tuples]
        action_list = torch.tensor([tup[1] for tup in state_tuples])
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = torch.tensor([tup[3] for tup in state_tuples]).float()
        
        # Indicator of whether current state is last state or not
        # Flag = 1 if last state
        # Flag = 0 otherwise
        flag = np.repeat(np.array([s.t > self.T - 1 for s in next_state_list]).astype(int),self.num_players)
        
        # Predict some values
        curAct = self.predict_action(cur_state_list)                                    # Nash Action Coefficient Estimates of current state
        curVal = self.predict_value(cur_state_list).detach().view(-1)      # Nash Value of Current state
        nextVal = self.predict_value(next_state_list).detach().view(-1).data.numpy()    # Nash Value of Next state
        
        expanded_next_states = self.expand_list_untransformed(next_state_list)
        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_next_states[:,2],self.term_costs*np.ones(len(expanded_next_states)))))
        
        # Create Lists for predicted Values
        c1_list = torch.stack([nfv.c1 for nfv in curAct]).view(-1)
        c2_list = torch.stack([nfv.c2 for nfv in curAct]).view(-1)
        c3_list = torch.stack([nfv.c3 for nfv in curAct]).view(-1)
        mu_list = torch.stack([nfv.mu for nfv in curAct])
        uNeg_list = torch.tensor(self.matrix_slice(action_list)).float()
        muNeg_list = self.matrix_slice(mu_list)
        act_list = torch.tensor(action_list.view(-1)).float()
        mu_list = mu_list.view(-1)

        A = - c1_list * (act_list-mu_list)**2 / 2 - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list,
                        dim = 1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2,dim = 1) / 2
        #print(np.multiply(np.ones(len(curVal))-flag, nextVal))
        print(np.multiply(np.ones(len(curVal))-flag, nextVal) + np.multiply(flag, term_list))
        #print(np.multiply(flag, term_list))
        return torch.sum((torch.tensor(np.multiply(np.ones(len(curVal))-flag, nextVal) + np.multiply(flag, term_list) + reward_list.view(-1)).float()
                          - curVal - A)**2 
                        + penalty*torch.var(c1_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                        + penalty*torch.var(c2_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                        + penalty*torch.var(c3_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1))

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

