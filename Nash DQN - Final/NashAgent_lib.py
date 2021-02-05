import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
        
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
                 out_dim, block_size=1, num_moments=1):
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

        # Define Networks
        self.moment_encoder_net = nn.Sequential(
            nn.Linear(self.block_size, 25),
            nn.ReLU(),
            # nn.Linear(10, 10),
            # nn.ReLU(),
            nn.Linear(25, self.num_moments)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.num_moments + self.non_invar_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, self.out_dim)
        )

    def forward(self, invar_input, non_invar_input, inv_split_dim=1):
        # Reshape invar_input into blocks and compute "moments"
        invar_split = torch.split(
            invar_input, self.block_size, dim=inv_split_dim)
        invar_moments = \
            sum((self.moment_encoder_net(ch)
                             for ch in invar_split)) / len(invar_split)

        # Concat moment vector with non-invariant input and pipe into next layer
        cat_input = torch.cat((invar_moments, non_invar_input), dim=1)

        # Output Final Tensor
        out_tensor = self.decoder_net(cat_input)
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
    def __init__(self, non_invar_dim, output_dim, n_players, max_steps, trans_cost, terminal_cost, num_moms = 5):
        # Simulation Parameters
        self.num_players = n_players         
        self.T = max_steps                      
        self.transaction_cost = trans_cost  
        self.terminal_cost = terminal_cost   
        self.non_invar_dim = non_invar_dim 
        
        # Initialize Networks
        if torch.cuda.is_available():
            self.action_net = PermInvariantQNN(in_invar_dim = self.num_players - 1, non_invar_dim = self.non_invar_dim, out_dim = output_dim, num_moments=num_moms).cuda()
            self.value_net = PermInvariantQNN(in_invar_dim = self.num_players - 1, non_invar_dim = self.non_invar_dim, out_dim = 1).cuda()
        else:
            self.action_net = PermInvariantQNN(in_invar_dim=self.num_players - 1, non_invar_dim = self.non_invar_dim, out_dim=output_dim, num_moments=num_moms)
            self.value_net = PermInvariantQNN(in_invar_dim=self.num_players - 1, non_invar_dim = self.non_invar_dim, out_dim=1)


        # Define optimizer used (SGD, etc)
        self.optimizer_DQN = optim.RMSprop(list(self.action_net.moment_encoder_net.parameters()) + list(self.action_net.decoder_net.parameters()),
                                       lr=0.005)
        self.optimizer_value = optim.RMSprop(list(self.value_net.moment_encoder_net.parameters()) + list(self.value_net.decoder_net.parameters()),
                                       lr=0.01) 
        
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
    
    def __repr__(self):
        return "NashNN Object:\n \# Players:%i\nT:%i\nNon Invariant Dim Size:%i" % (self.num_players, self.T, self.non_invar_dim)

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
    
    def expand_list(self,state_list, as_tensor = True):
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
        expanded_ivt_states = []
        if as_tensor:
            for cur_s in state_list:
                for i in range(0, self.num_players):
                    s, s_inv = cur_s.to_sep_numpy(i)
                    expanded_states.append(s)
                    expanded_ivt_states.append(s_inv)

        if as_tensor:
            return torch.tensor(expanded_states, dtype=torch.float32), torch.tensor(expanded_ivt_states, dtype=torch.float32)
        else:
            return np.array(expanded_states), np.array(expanded_ivt_states)
    
    def predict_action(self, states):
        """
        Predicts the parameters of the advantage function of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          List of NashFittedValue objects representing the estimated parameters
        """
        expanded_states, inv_states = self.expand_list(states,as_tensor=True)

        if torch.cuda.is_available():
            action_list = self.action_net.forward(invar_input = inv_states.cuda(), non_invar_input = expanded_states.cuda())
        else:
            action_list = self.action_net.forward(invar_input = inv_states, non_invar_input = expanded_states)
            
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
        expanded_states, inv_states = self.expand_list(states, as_tensor=True)

        if torch.cuda.is_available():
            values = self.value_net.forward(invar_input = inv_states.cuda(), non_invar_input = expanded_states.cuda())
        else:
            values = self.value_net.forward(invar_input = inv_states, non_invar_input = expanded_states)
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
        isLastState = np.repeat(np.array([s.t <= 0 for s in next_state_list]).astype(int),self.num_players)
        
        target = self.predict_value(cur_state_list).view(-1)
        expanded_states, _ = self.expand_list(cur_state_list)
        expanded_next_states, _ = self.expand_list(next_state_list)

        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_next_states[:,2]/2,self.terminal_cost*np.ones(len(expanded_next_states))))) \
                    + np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_next_states[:,2]/2,self.terminal_cost*np.ones(len(expanded_next_states)))))
        
        nextstate_val = self.predict_value(next_state_list).detach().view(-1).data.cpu().numpy()
        
        # Returns the squared loss with target being:
        # Sum of current state's reward + estimated next state's nash value     --- if not last state
        # Sum of current state's reward + terminal costs of netting out position  --- if last state
        if torch.cuda.is_available():
            return self.criterion(target, torch.tensor(np.multiply(isLastState,term_list) + np.multiply(np.ones(len(expanded_states)) - isLastState, nextstate_val) + np.array(reward_list).flatten(),dtype=torch.float32).cuda())
        else:
            return self.criterion(target, torch.tensor(np.multiply(isLastState, term_list) + np.multiply(np.ones(len(expanded_states)) - isLastState, nextstate_val) + np.array(reward_list).flatten(), dtype=torch.float32))
    
    def compute_action_Loss(self, state_tuples):
        """
        Computes the loss function for the action/advantage-function network
        :param state_tuples:    List of a batch of transition tuples
        :return:                Total loss of batch
        """
        
        # Squared Loss penalty for Difference in Coefficients (c1,c2,c3) between agents 
        # to ensure consistency among agents 
        penalty = 25
        
        # if torch.cuda.is_available():
        #     cur_state_list = [tup[0].cuda() for tup in state_tuples]
        #     action_list = torch.stack([tup[1].cuda() for tup in state_tuples])
        #     next_state_list = [tup[2].cuda() for tup in state_tuples]
        #     reward_list = torch.tensor(
        #         [tup[3].cuda() for tup in state_tuples], dtype=torch.float32)
        # else:
        
        cur_state_list = [tup[0] for tup in state_tuples]
        action_list = torch.stack([tup[1] for tup in state_tuples])
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = torch.tensor(
            [tup[3] for tup in state_tuples], dtype=torch.float32)
        
        # Indicator of whether current state is last state or not
        isLastState = np.repeat(np.array([s.t <= 0 for s in next_state_list]).astype(int),self.num_players)
        
        curAct = self.predict_action(cur_state_list)                             # Nash Action Coefficient Estimates of current state
        curVal = self.predict_value(cur_state_list).detach().view(-1).cpu()                    # Nash Value of Current state
        nextVal = self.predict_value(next_state_list).detach().view(-1).cpu().data.numpy()    # Nash Value of Next state
        
        #Makes Matrix of Terminal Values
        expanded_next_states, _ = self.expand_list(next_state_list)
        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_next_states[:,2],self.terminal_cost*np.ones(len(expanded_next_states)))))
        
        # Create Lists for predicted Values
        c1_list = torch.stack([nfv.c1 for nfv in curAct]).view(-1).cpu()
        c2_list = torch.stack([nfv.c2 for nfv in curAct]).view(-1).cpu()
        c3_list = torch.stack([nfv.c3 for nfv in curAct]).view(-1).cpu()
        mu_list = torch.stack([nfv.mu for nfv in curAct]).cpu()
        
        #Creates the Mu_neg and u_Neg Matrices
        uNeg_list = torch.tensor(self.matrix_slice(
            action_list), dtype=torch.float32).cpu()
        muNeg_list = self.matrix_slice(mu_list).cpu()
        act_list = torch.tensor(action_list.view(-1),
                                dtype=torch.float32).cpu()
        mu_list = mu_list.view(-1).cpu()

        #Computes the Advantage Function using matrix operations
        A = - c1_list * (act_list-mu_list)**2 / 2 - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list,
                        dim = 1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2,dim = 1) / 2

        if torch.cuda.is_available():
            return torch.sum((torch.tensor(np.multiply(np.ones(len(curVal))-isLastState, nextVal) + np.multiply(isLastState, term_list) + reward_list.view(-1).cpu().numpy(), dtype=torch.float32)
                            - curVal - A)**2 
                            + penalty*torch.var(c1_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                            + penalty*torch.var(c2_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                            + penalty*torch.var(c3_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)).cuda()
        else:
            return torch.sum((torch.tensor(np.multiply(np.ones(len(curVal))-isLastState, nextVal) + np.multiply(isLastState, term_list) + reward_list.view(-1).numpy(), dtype=torch.float32)
                              - curVal - A)**2
                             + penalty*torch.var(c1_list.view(-1, self.num_players),1).view(-1, 1).repeat(1, self.num_players).view(-1)
                             + penalty*torch.var(c2_list.view(-1, self.num_players), 1).view(-1, 1).repeat(1, self.num_players).view(-1)
                             + penalty*torch.var(c3_list.view(-1, self.num_players), 1).view(-1, 1).repeat(1, self.num_players).view(-1))


