import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from copy import deepcopy as dc


class PermInvariantQNN(torch.nn.Module):
    """
    Permutation Invariant Network

    :param in_invar_dim:   Number of total features across all agents that needs to be perm invariant 
    :param non_invar_dim:  Number of total features constant across all agents
    :param out_dim:        Dimension of output
    :param block_size:     Number of invariant features of each agent
    :param num_moments:    Number of features/moments to summarize invariant features of each agent
    :raises assertError:   Raise assertion error if in_invar_dim not multiple of block size
    """

    block_size: int
    in_invar_dim: int
    non_invar_dim: int
    num_moments: int
    out_dim: int

    def __init__(self, in_invar_dim, non_invar_dim,
                 out_dim, block_size=1, num_moments=1, lat_dims=32, layers=4):
        super(PermInvariantQNN, self).__init__()

        # Store input and output dimensions
        self.in_invar_dim = in_invar_dim
        self.non_invar_dim = non_invar_dim
        self.block_size = block_size
        self.num_moments = num_moments
        self.out_dim = out_dim

        # Verify invariant dimension is multiple of block size
        assert not self.in_invar_dim % self.block_size, "in_invar_dim must be a multiple of block size."

        # Compute Number of blocks
        self.num_blocks = self.in_invar_dim / self.block_size

        nets = []
        nets.append(nn.Linear(self.num_moments + self.non_invar_dim, lat_dims))
        nets.append(nn.SiLU())
        
        for i in range(layers):
            nets.append(nn.Linear(lat_dims, lat_dims))
            nets.append(nn.SiLU())
            
        nets.append(nn.Linear(lat_dims, self.out_dim))
        self.decoder_net = nn.Sequential(*nets)

    def forward(self, invar_input, non_invar_input, inv_split_dim=1):
        if invar_input is not None:
            # Use first moment only
            invar_moments = torch.mean(invar_input, dim = 1).unsqueeze(1)

            # Concat moment vector with non-invariant input and pipe into next layer
            cat_input = torch.cat((invar_moments, non_invar_input), dim=1)

            # Output Final Tensor
            out_tensor = self.decoder_net(cat_input)
        else:
            out_tensor = self.decoder_net(non_invar_input)
        return out_tensor


class NashNN():
    """
    Object summarizing estimated parameters of the advantage function, initiated 
    through a vector of inputs
    :param non_invar_dim:Number of total non invariant (i.e. market state) input features
    :param output_dim:   Number of total parameters to be estimated via NN
    :param nump:         Number of agents
    :param t:            Number of total time steps
    :param t_cost:       Transaction costs (estimated or otherwise)
    :param term_cost:    Terminal costs (estimated or otherwise)
    """

    def __init__(self, non_invar_dim, output_dim, n_players, max_steps, terminal_cost, num_moms=5, lr=0.001, lat_dims=32, c_cons=0.1, c2_cons=True, c3_pos=True, c_pen=True, layers=4, weighted_adam=False):
        # Simulation Parameters
        self.num_players = n_players
        self.T = max_steps
        self.terminal_cost = terminal_cost
        self.non_invar_dim = non_invar_dim
        self.lr = lr
        self.c_pen = c_pen

        # Initialize Networks
        if torch.cuda.is_available():
            print("CUDA IS AVAILABLE!")
            self.use_cuda = True
            self.action_net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1, non_invar_dim=self.non_invar_dim, out_dim=output_dim, num_moments=num_moms, lat_dims=lat_dims, layers=layers).cuda()
            self.value_net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1, non_invar_dim=self.non_invar_dim, out_dim=1, num_moments=num_moms, lat_dims=lat_dims, layers=layers).cuda()
            self.slow_val_net=PermInvariantQNN(
                in_invar_dim=self.num_players - 1, non_invar_dim=self.non_invar_dim, out_dim=1, num_moments=num_moms, lat_dims=lat_dims, layers=layers).cuda()
        else:
            self.use_cuda = False
            print("CUDA IS NOT AVAILABLE!")
            self.action_net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1, non_invar_dim=self.non_invar_dim, out_dim=output_dim, num_moments=num_moms, lat_dims=lat_dims, layers=layers)
            self.value_net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1, non_invar_dim=self.non_invar_dim, out_dim=1, num_moments=num_moms, lat_dims=lat_dims, layers=layers)
            self.slow_val_net=PermInvariantQNN(
                in_invar_dim=self.num_players - 1, non_invar_dim=self.non_invar_dim, out_dim=1, num_moments=num_moms, lat_dims=lat_dims, layers=layers)
            
        # Define optimizer used (SGD, etc)
        if weighted_adam:
            self.optimizer_DQN = optim.AdamW(
                self.action_net.parameters(), lr=self.lr)

            self.optimizer_value = optim.AdamW(
                self.value_net.parameters(), lr=self.lr)
        else:
            self.optimizer_DQN = optim.Adam(
                self.action_net.parameters(), lr=self.lr)

            self.optimizer_value = optim.Adam(
                self.value_net.parameters(), lr=self.lr)

        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        
        # Define constant L-2 penalty
        self.c_cons = c_cons
        self.c2_cons = c2_cons
        self.c3_pos = c3_pos

    def __repr__(self):
        return "NashNN Object:\n \# Players:%i\nT:%i\nNon Invariant Dim Size:%i" % (self.num_players, self.T, self.non_invar_dim)

    def matrix_slice(self, X):
        """
        Returns a matrix where each row in X is replicated N number of times where N
        is the number of total agents. Then the value of the j'th element of the 
        (i*N + j)'th row is removed. Effectively creating
        a stacked version of the u^(-1) or mu(-1) for batched inputs.
        :param X:    Matrix of actions/nash actions where each row corresponds to one transition from a batch input
        :return:     Matrix of u^(-1) or mu(-1) as described above
        """
        num_entries = len(X)
        arr = X.repeat_interleave(self.num_players, dim = 0)
        
        if torch.cuda.is_available():
            ids = torch.tensor(torch.arange(self.num_players)).tile(num_entries).cuda()
            #mask = torch.ones_like(arr).scatter_(1, ids.unsqueeze(1), torch.tensor(0.).cuda())
        else:
            ids = torch.tensor(torch.arange(self.num_players)).tile(num_entries)
        
        mask = torch.ones_like(arr).scatter_(1, ids.unsqueeze(1), 0.)
            
        res = arr[mask.bool()].view(-1, self.num_players - 1)
        
        return res

    def predict_action(self, states, invt_states):
        """
        Predicts the parameters of the advantage function of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          List of NashFittedValue objects representing the estimated parameters
        """
        if torch.cuda.is_available():
            action_list = self.action_net.forward(
                invar_input=invt_states, non_invar_input=states.cuda())
        else:
            action_list = self.action_net.forward(
                invar_input=invt_states, non_invar_input=states)
            
        if self.c3_pos:
            action_list = torch.hstack([torch.abs(action_list[:, 0]).view(-1,1), action_list[:, 1].view(-1,1), torch.abs(action_list[:, 2]).view(-1,1), action_list[:, 3:]])
        else:
            action_list = torch.hstack([torch.abs(action_list[:, 0]).view(-1,1), action_list[:, 1:]])

        #action_factor = 10.0
        action_factor = 1.0
        action_list[:,4:] = action_list[:,4:] * action_factor

        
        return action_list
    
    def update_slow(self):
        self.slow_val_net.load_state_dict(dc(self.value_net.state_dict()))

    def predict_value(self, states, invt_states, slow=False):
        """
        Predicts the nash value of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          Tensor of estimated nash values of all agents for the batch of states
        """
        if slow:
            if torch.cuda.is_available():
                values = self.slow_val_net.forward(
                    invar_input=invt_states, non_invar_input=states.cuda())
            else:
                values = self.slow_val_net.forward(
                    invar_input=invt_states, non_invar_input=states)
        else:
            if torch.cuda.is_available():
                values = self.value_net.forward(
                    invar_input=invt_states, non_invar_input=states.cuda())
            else:
                values = self.value_net.forward(
                    invar_input=invt_states, non_invar_input=states)
        return values

    def compute_value_Loss(self, state_tuples):
        """
        Computes the loss function for the value network
        :param state_tuples:    List of a batch of transition tuples
        :return:                Total loss of batch
        """
        
        cur_state_list = state_tuples[0]
        cur_ivt_state_list = state_tuples[1]
        next_state_list = state_tuples[2]
        next_ivt_state_list = state_tuples[3]
        isLastState = state_tuples[4].view(-1)
        reward_list = state_tuples[5].view(-1)
        action_list = state_tuples[6].view(-1)
        
        # Nash Action Coefficient Estimates of current state
        curAct = self.predict_action(cur_state_list, cur_ivt_state_list).detach()
        curVal = self.predict_value(cur_state_list, cur_ivt_state_list).view(-1) # Nash Value of Current state
        nextVal = self.predict_value(next_state_list, next_ivt_state_list, slow=True).detach().view(-1) # Nash Value of Next state

        # Create Lists for predicted Values
        c1_list = curAct[:, 0]
        c2_list = curAct[:, 1]
        c3_list = curAct[:, 2]
        c4_list = curAct[:, 3]
        mu_list = curAct[:, 4]
        
        #print(curAct)
        
        if torch.cuda.is_available():
            act_list = action_list.cuda()
        else:
            act_list = action_list
            
        if self.num_players > 1:
            # Creates the Mu_neg and u_Neg Matrices
            uNeg_list = self.matrix_slice(act_list.view(-1, self.num_players))
            muNeg_list = self.matrix_slice(mu_list.view(-1, self.num_players))

            # Computes the Advantage Function using matrix operations
            A = - c1_list * (act_list-mu_list)**2 - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list, dim=1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2, dim=1) + c4_list * torch.sum((uNeg_list - muNeg_list), dim=1)
        else:
            A = - c1_list * (act_list-mu_list)**2

        if torch.cuda.is_available():
            if self.c_pen:
                return torch.sum(((torch.ones(len(isLastState)).cuda()-isLastState).cuda().detach() * nextVal.detach() + reward_list.cuda().detach() - curVal - A.detach())**2) + self.c_cons * torch.sum((c4_list)**2) + self.c2_cons * self.c_cons *  torch.sum((c2_list)**2)
            else:
                return torch.sum(((torch.ones(len(isLastState)).cuda()-isLastState).cuda().detach() * nextVal.detach() + reward_list.cuda().detach() - curVal - A.detach())**2)
        else:
            raise Exception("NEED CUDA")
            return torch.sum(((torch.ones(len(isLastState))-isLastState) * nextVal + reward_list - curVal - A)**2) + torch.sum(c4_list**2)

    def compute_action_Loss(self, state_tuples):
        """
        Computes the loss function for the action/advantage-function network
        :param state_tuples:    List of a batch of transition tuples
        :return:                Total loss of batch
        """

        cur_state_list = state_tuples[0]
        cur_ivt_state_list = state_tuples[1]
        next_state_list = state_tuples[2]
        next_ivt_state_list = state_tuples[3]
        isLastState = state_tuples[4].view(-1)
        reward_list = state_tuples[5].view(-1)
        action_list = state_tuples[6].view(-1)

        # Nash Action Coefficient Estimates of current state
        curAct = self.predict_action(cur_state_list, cur_ivt_state_list)
        curVal = self.predict_value(cur_state_list, cur_ivt_state_list).detach().view(-1) # Nash Value of Current state
        nextVal = self.predict_value(next_state_list, next_ivt_state_list, slow=True).detach().view(-1) # Nash Value of Next state

        # Create Lists for predicted Values
        c1_list = curAct[:, 0]
        c2_list = curAct[:, 1]
        c3_list = curAct[:, 2]
        c4_list = curAct[:, 3]
        mu_list = curAct[:, 4]
        
        #print(curAct)
        
        if torch.cuda.is_available():
            act_list = action_list.cuda()
        else:
            act_list = action_list
            
        if self.num_players > 1:
            # Creates the Mu_neg and u_Neg Matrices
            uNeg_list = self.matrix_slice(act_list.view(-1, self.num_players))
            muNeg_list = self.matrix_slice(mu_list.view(-1, self.num_players))

            # Computes the Advantage Function using matrix operations
            A = - c1_list * (act_list-mu_list)**2  - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list, dim=1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2, dim=1) + c4_list * torch.sum((uNeg_list - muNeg_list), dim=1)
        else:
            A = - c1_list * (act_list-mu_list)**2

        if torch.cuda.is_available():
            return torch.sum(((torch.ones(len(isLastState)).cuda()-isLastState).cuda().detach() * nextVal.detach() + reward_list.cuda().detach() - curVal.detach() - A)**2) + self.c_cons * torch.sum((c4_list)**2)  + self.c2_cons * self.c_cons * torch.sum((c2_list)**2)
        else:
            return torch.sum(((torch.ones(len(isLastState))-isLastState) * nextVal + reward_list - curVal - A)**2) + torch.sum(c4_list**2)