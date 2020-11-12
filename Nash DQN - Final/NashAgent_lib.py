import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nashRL_netlib import *
        
class NashFittedValues(object):
    """
    Object summarizing estimated parameters of the advantage function, initiated 
    through a vector of inputs
    :param mu: Array of nash actions for all agents
    :param c1: Array of estimated coefficients for first term in A function
    :param c2: Array of estimated coefficients for second term in A function
    :param c3: Array of estimated coefficients for third term in A function
    """
    def __init__(self, FV_list):
        self.mu = FV_list[:,0]
        self.c1 = FV_list[:,1]
        self.c2 = FV_list[:,2]
        self.c3 = FV_list[:,3]
        
#--------------------
# Class containing all functions related to nash agent, including loss functions,
# and prediction functions
#--------------------
class NashNN():
    """
    Object summarizing estimated parameters of the advantage function, initiated 
    through a vector of inputs
    :param input_dim:    Number of total input features
    :param output_dim:   Number of total parameters to be estimated via NN
    :param nump:         Number of agents
    :param t:            Number of total time steps
    :param t_cost:       Transaction costs (estimated or otherwise)
    :param term_cost:    Terminal costs (estimated or otherwise)
    """
    def __init__(self, input_dim, output_dim, n_players, max_steps, trans_cost, terminal_cost, num_moms = 5):
        # Simulation Parameters
        self.num_players = n_players         
        self.T = max_steps                      
        self.transaction_cost = trans_cost  
        self.terminal_cost = terminal_cost     
        
        # Initialize Networks
        if torch.cuda.is_available():
            self.action_net = PermInvariantQNN(in_invar_dim = self.num_players - 1, non_invar_dim = 3, out_dim = output_dim, num_moments=num_moms).cuda()
            self.value_net = PermInvariantQNN(in_invar_dim = self.num_players - 1, non_invar_dim = 3, out_dim = 1).cuda()
        else:
            self.action_net = PermInvariantQNN(in_invar_dim=self.num_players - 1, non_invar_dim=3, out_dim=output_dim, num_moments=num_moms)
            self.value_net = PermInvariantQNN(in_invar_dim=self.num_players - 1, non_invar_dim=3, out_dim=1)


        # Define optimizer used (SGD, etc)
        self.optimizer_DQN = optim.RMSprop(list(self.action_net.moment_encoder_net.parameters()) + list(self.action_net.decoder_net.parameters()),
                                       lr=0.005)
        self.optimizer_value = optim.RMSprop(list(self.value_net.moment_encoder_net.parameters()) + list(self.value_net.decoder_net.parameters()),
                                       lr=0.01) 
        
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()

    def slice(self,X,i):
        "Return all items in array except for i^th item"
        return torch.cat([X[0:i], X[i + 1:]])
    
    def matrix_slice(self,X):
        """
        Returns a matrix where each row in X is replicated N number of times where N
        is the number of total agents. Then the value of the j'th element of the 
        (i*N + j)'th row is moved to the front of the row. Effectively creating
        a stacked version of the u^(-1) or mu(-1) for batched inputs.
        :param X:    Matrix of actions/nash actions where each row corresponds to one transition from a batch input
        :return:     Matrix of u^(-1) or mu(-1) as described above
        """
        mat = []
        for i in range(0,len(X)):
            for j in range(0,self.num_players):
                mat.append(self.slice(X[i,:],j))
        return torch.stack(mat)
    
    def expand_list(self,state_list, norm = True):
        """
        Creates a matrix of features for a batch of input states of the environment.
        
        Specifically, given a list of states, returns a matrix of features structured as follows:
            - Blocks of matrices stacked vertically where each block represents the features for one element
              in the batch
            - Each block is separated into N rows where N is the number of players
            - Each i'th row in each block is structured such that the first three elements are the non-permutation-invariant features
              i.e. time, price, and inventory of the i'th agent, followed by the permutation invariant features
              i.e. the inventory of all the other agents. 
        :param state_list:    List of states to pass pass into NN later
        :param norm:          Whether or not to use normalized features
        :return:              Matrix of the batch of features structured to be pass into NN
        """
        expanded_states = []
        for cur_s in state_list:
            # if norm:
            #     s = cur_s.getNormalizedState()
            # else:
            #     s = cur_s.getState()
            
            for _ in range(0,self.num_players):
                # expanded_states.append(np.append(np.append(np.array([s[0],s[1],s[2+j]]),s[2:2+j]),s[2+j+1:]))
                expanded_states.append(cur_s.to_numpy())
                
        return np.array(expanded_states)
    
    def predict_action(self, states):
        """
        Predicts the parameters of the advantage function of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          List of NashFittedValue objects representing the estimated parameters
        """
        expanded_states = torch.tensor(self.expand_list(states)).float()

        if torch.cuda.is_available():
            action_list = self.action_net.forward(invar_input = expanded_states[:,3:].cuda(), non_invar_input = expanded_states[:,0:3].cuda())
        else:
            action_list = self.action_net.forward(invar_input=expanded_states[:, 3:], non_invar_input=expanded_states[:, 0:3])
            
        NFV_list = []
        for i in range(0,len(states)):
            NFV_list.append(NashFittedValues(action_list[i*self.num_players:(i+1)*self.num_players,:]))
        
        return NFV_list
    
    def predict_value(self, states):
        """
        Predicts the nash value of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          Tensor of estimated nash values of all agents for the batch of states
        """
        expanded_states = torch.tensor(self.expand_list(states)).float()

        if torch.cuda.is_available():
            values = self.value_net.forward(invar_input = expanded_states[:,3:].cuda(), non_invar_input = expanded_states[:,0:3].cuda())
        else:
            values = self.value_net.forward(invar_input = expanded_states[:,3:], non_invar_input = expanded_states[:,0:3])
        return values

    def compute_value_Loss(self,state_tuples):
        """
        Computes the loss function for the value network
        :param state_tuples:    List of a batch of transition tuples
        :return:                Total loss of batch
        """
        cur_state_list = [tup[0] for tup in state_tuples]
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = [tup[3] for tup in state_tuples]
        
        # Indicator of whether current state is last state or not
        isLastState = np.repeat(np.array([s.t > self.T - 1 for s in next_state_list]).astype(int),self.num_players)
        
        target = self.predict_value(cur_state_list).view(-1)
        expanded_states = self.expand_list(cur_state_list, norm = False)
        expanded_next_states = self.expand_list(next_state_list, norm = False)

        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_states[:,1],expanded_states[:,2]/2,self.transaction_cost*np.ones(len(expanded_states))))) \
                    + np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_states[:,2]/2,self.transaction_cost*np.ones(len(expanded_states)))))
        nextstate_val = self.predict_value(next_state_list).detach().view(-1).data.cpu().numpy()
        
        # Returns the squared loss with target being:
        # Sum of current state's reward + estimated next state's nash value     --- if not last state
        # Reward obtained if agent executes action to net out current inventory --- if last state
        if torch.cuda.is_available():
            return self.criterion(target, torch.tensor(np.multiply(isLastState,term_list) + np.multiply(np.ones(len(expanded_states)) - isLastState, nextstate_val + np.array(reward_list).flatten())).float().cuda())
        else:
            return self.criterion(target, torch.tensor(np.multiply(isLastState, term_list) + np.multiply(np.ones(len(expanded_states)) - isLastState, nextstate_val + np.array(reward_list).flatten())).float())
    
    def compute_action_Loss(self, state_tuples):
        """
        Computes the loss function for the action/advantage-function network
        :param state_tuples:    List of a batch of transition tuples
        :return:                Total loss of batch
        """
        
        # Squared Loss penalty for Difference in Coefficients (c1,c2,c3) between agents 
        # to ensure consistency among agents 
        penalty = 25
        
        cur_state_list = [tup[0] for tup in state_tuples]
        action_list = torch.tensor([tup[1] for tup in state_tuples])
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = torch.tensor([tup[3] for tup in state_tuples]).float()
        
        # Indicator of whether current state is last state or not
        isLastState = np.repeat(np.array([s.t > self.T - 1 for s in next_state_list]).astype(int),self.num_players)
        
        curAct = self.predict_action(cur_state_list)                             # Nash Action Coefficient Estimates of current state
        curVal = self.predict_value(cur_state_list).detach().view(-1).cpu()                    # Nash Value of Current state
        nextVal = self.predict_value(next_state_list).detach().view(-1).cpu().data.numpy()    # Nash Value of Next state
        
        #Makes Matrix of Terminal Values
        expanded_next_states = self.expand_list(next_state_list, norm = False)
        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_next_states[:,2],self.terminal_cost*np.ones(len(expanded_next_states)))))
        
        # Create Lists for predicted Values
        c1_list = torch.stack([nfv.c1 for nfv in curAct]).view(-1).cpu()
        c2_list = torch.stack([nfv.c2 for nfv in curAct]).view(-1).cpu()
        c3_list = torch.stack([nfv.c3 for nfv in curAct]).view(-1).cpu()
        mu_list = torch.stack([nfv.mu for nfv in curAct]).cpu()
        
        #Creates the Mu_neg and u_Neg Matrices
        uNeg_list = torch.tensor(self.matrix_slice(action_list)).float().cpu()
        muNeg_list = self.matrix_slice(mu_list).cpu()
        act_list = torch.tensor(action_list.view(-1)).float().cpu()
        mu_list = mu_list.view(-1).cpu()

        #Computes the Advantage Function using matrix operations
        A = - c1_list * (act_list-mu_list)**2 / 2 - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list,
                        dim = 1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2,dim = 1) / 2

        if torch.cuda.is_available():
            return torch.sum((torch.tensor(np.multiply(np.ones(len(curVal))-isLastState, nextVal) + np.multiply(isLastState, term_list) + reward_list.view(-1)).float()
                            - curVal - A)**2 
                            + penalty*torch.var(c1_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                            + penalty*torch.var(c2_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                            + penalty*torch.var(c3_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)).cuda()
        else:
            return torch.sum((torch.tensor(np.multiply(np.ones(len(curVal))-isLastState, nextVal) + np.multiply(isLastState, term_list) + reward_list.view(-1)).float()
                              - curVal - A)**2
                             + penalty*torch.var(c1_list.view(-1, self.num_players),1).view(-1, 1).repeat(1, self.num_players).view(-1)
                             + penalty*torch.var(c2_list.view(-1, self.num_players), 1).view(-1, 1).repeat(1, self.num_players).view(-1)
                             + penalty*torch.var(c3_list.view(-1, self.num_players), 1).view(-1, 1).repeat(1, self.num_players).view(-1))


