from collections import namedtuple
import random
import torch
from copy import deepcopy as dc
import numpy as np

"""
Transition object summarizing changes to the environment at each time step
:param state:       State object representing observable features of the current state
:param action:      Array of actions of all agents
:param next_state:  State object representing observable features of the resultant state
:param reward:      Array of rewards obtained by all agents
"""
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

"""
State object summarizing observable features of the environment at each time step
:param t:  Time step number
:param p:  Price of stock
:param q:  Array of inventory levels of all agents
"""
ProtoState = namedtuple('State', ('t', 'p', 'i', 'q', 'q0'))


class State(ProtoState):

    def to_tensor(self, **kwargs):
        return torch.tensor(self.to_numpy(), **kwargs)
    
    def to_sep_numpy(self, idx, nm, ns):
        """
        :param nm & ns: (t, p, q, i) normalized mean and std
        """
        dq = self.q - self.q0
        #dq = self.q
        # i is agent index (start from 0) to include is invariant
        
        if len(self.q) > 1:
#             return torch.tensor([ (self.t - nm[0])/ns[0] , 
#                               ((self.p - self.i) - nm[1])/ns[1], 
#                               #((self.p) - nm[1])/ns[1], 
#                               (self.q[idx] - nm[2])/ns[2], 
#                               (self.i - nm[3])/ns[3] ]).float().cuda(), torch.cat([(dq[:idx] - nm[2])/ns[2],(dq[idx+1:] - nm[2])/ns[2]], dim =0)
            return torch.tensor([ (self.t - nm[0])/ns[0] , 
                              ((self.p - self.i) - nm[1])/ns[1], 
                              #((self.p) - nm[1])/ns[1], 
                              (self.q[idx] - nm[2])/ns[2], 
                              (self.i - nm[3])/ns[3], 
                              ((torch.sum(dq) - dq[idx])/(len(dq)-1) - nm[4])/ns[4]]
                               ).float().cuda(), None
        else:
            return torch.tensor([ (self.t - nm[0])/ns[0] , 
                              ((self.p - self.i) - nm[1])/ns[1], 
                              #((self.p) - nm[1])/ns[1], 
                              (self.q[idx] - nm[2])/ns[2], 
                              (self.i - nm[3])/ns[3] ]).float().cuda(), None
        
    def to_combine_tens(self, nm, ns):
        return torch.cat([ torch.unsqueeze((self.t - nm[0])/ns[0],dim=0) , 
                              torch.unsqueeze(((self.p - self.i) - nm[1])/ns[1],dim=0), 
                              (self.q - nm[2])/ns[2], 
                              torch.unsqueeze((self.i - nm[3])/ns[3],dim=0)
                            ]).float().cuda(), None

    def to_sep_tensor(self, idx, **kwargs):
        non_inv, inv = self.to_sep_numpy(idx)
        return torch.tensor(non_inv, **kwargs), torch.tensor(inv, **kwargs)
    
    def to_sep_tensor_less(self, idx, nm, ns, mean = False, q0=True):
        """
        :param nm & ns: (t, p, q, i) normalized mean and std
        """
        # i is agent index (start from 0) to include is invariant
        #dq = self.q - self.q0
        if q0:
            dq = self.q - self.q0
        else:
            dq = self.q
        if mean:
            #print(self.t)
            #print(self.p)
            #print(self.q[idx])
            #print(self.i)
            #print((torch.sum(self.q) - self.q[idx])/(len(self.q)-1))
            if len(self.q) > 1:
                return torch.stack([(self.t - nm[0])/ns[0] , 
                                  (self.p - self.i - nm[1])/ns[1], 
                                  (self.q[idx] - nm[2])/ns[2], 
                                  (self.i - nm[3])/ns[3], ((torch.sum(dq) - dq[idx])/(len(dq)-1) - nm[2])/ns[2]
                                    ]).float()
            else:
                return torch.stack([(self.t - nm[0])/ns[0] , 
                              (self.p - self.i - nm[1])/ns[1], 
                              (self.q[idx] - nm[2])/ns[2], 
                              (self.i - nm[3])/ns[3]]).float()
        else:
            return None


class MarketSimulator(object):
    """
    Class representing the market environment

    :param p_imp:   Price Impact Coefficent
    :param t_cost:  Transaction cost coefficent
    :param L_cost:  Liquidation cost coefficent
    :param phi:     Risk penalty (b_3)
    :param T:       Total number of time steps
    :param dt:      Change in time per time step in seconds
    :param N:       Total number of agents
    :param mu:      Drift function of process
    :param sigma:   Volatility of the stock
    :param sigma0:  Volatility of initial stock price
    :param r:       Reward function for the agents
    """

    def __init__(self, param_dict_in, store_hist=False, detach_output=True, impact='linear'):
        # Fill in default input arguments
        def_dict = {'trans_impact_scale': 0.0,
                    'trans_impact_decay': 0.0,
                    'perm_price_impact': 0.0,
                    'init_inv_var': 10.0 }

        #param_dict = {**def_dict, **param_dict_in}
        param_dict = param_dict_in
        self.store_hist = store_hist

        # Unpack Parameter Dictionary
        # Game-Specific Parameters
        self.perm_imp = param_dict['perm_price_impact']
        self.tmp_scale = param_dict['trans_impact_scale']
        self.tmp_decay = param_dict['trans_impact_decay']
        self.t_cost = param_dict['transaction_cost']
        self.L_cost = param_dict['liquidation_cost']
        self.phi = param_dict['running_penalty']
        self.T = param_dict['T']
        self.dt = param_dict['dt']
        self.N = param_dict['N_agents']

        # Simulation Parameters
        self.mu = param_dict['drift_function']
        self.sigma = param_dict['volatility']
        self.sigma0 = torch.sqrt(param_dict['initial_price_var'])
        self.sigma_Q0 = torch.sqrt(param_dict['init_inv_var'])
        self.impact = impact
        
        # Define Reward Function: Transaction cost + Running cost + Trade profit + Terminal Penalty
#        self.r = lambda t, Q, S, dS, nu: - nu * (S + self.t_cost * nu) * self.dt + (self.t >= self.T) * ( (S+dS) * Q - self.L_cost * Q**2)
        #self.r = lambda t, Q, S, dS, nu: - nu * (S + self.t_cost * nu) * self.dt + ( -(Q-nu*self.dt)*S + Q*(S+dS) ) + (self.t >= self.T) * ( - self.L_cost * Q**2)
            

        # Allocating Memory for Game Variables & Resetting
        self.dS=torch.tensor(0.0).cuda()
        self.dF=torch.tensor(0.0).cuda()
        self.dI=torch.tensor(0.0).cuda()
        
        self.total_reward = torch.tensor(0.0).cuda()
        self.Q0=torch.tensor(0.0).cuda()
        
        self.reset()
        
    def r(self, t, Q, S, dS, nu, to_print=False):
        if to_print:
            print(- nu * (S + self.t_cost * nu) * self.dt)
            print(( -(Q-nu*self.dt)*S + Q*(S+dS) ))
            print((t >= self.T) * ( - self.L_cost * Q**2))
            print(-(Q-nu*self.dt)*S)
            print(Q*(S+dS))
            print(Q)
            print(S)
            print(dS)
            print(S+dS)
        
        return - nu * (S + self.t_cost * nu) * self.dt + ( -(Q-nu*self.dt)*S + Q*(S+dS) ) + (t >= self.T) * ( - self.L_cost * Q**2) - self.phi * self.dt * Q**2
        

    def reset(self):
        """
        Reset the simulation and reinitialize inventory and price levels
        """
        self.Q=(torch.randn(self.N).cuda() * self.sigma_Q0)
        self.Q0=self.Q.clone()
        self.S=torch.squeeze((10 + torch.randn(1).cuda() * self.sigma0))
        
        self.I=torch.squeeze((torch.randn(1).cuda() * 0.5 * self.sigma))
        self.t=torch.tensor(0.0).cuda()

        self.last_reward=torch.zeros(self.N).cuda()
        self.total_reward=torch.zeros(self.N).cuda()

        temp = (torch.ceil(self.T / self.dt) + 2).detach().cpu().numpy()
        self.dW=(torch.randn(int(np.round(temp))).cuda() * torch.sqrt(self.dt))

        self.done=False

    def setInv(self, inv):
        """
        Manually set inventory level
        :param inv: Array containing the inventory levels of all agents being set to
        """
        self.Q=inv


    def step(self, nu, to_print=False):
        """
        Calculate and update all environment parameters given all agent's actions.
        Returns a Transition tuple holding observable state values of the previous
         state of the environment, actions executed by all agents, observable state
         values of the resultant state and the rewards obtained by all agents.

        :param nu: Array containing the actions of all agents
        :returns:  Transition object containing summary of the change to the environment
        """
        last_state, _, _ =self.get_state()

        self.done=(self.t >= self.T)
        
        #print(last_state, nu)
        
        data = {}
        
        #nu_c = nu.clone().detach()

        if not self.done:
            # Rescale Shares Purchased as Rate
            #nu=nu*self.dt
            
            # Advance Inventory & Time
            self.Q = self.Q.clone() + nu*self.dt
            self.t = self.t.clone() + self.dt

            # Advance Asset Price
            steps = int(torch.round(self.t/self.dt))
            self.dF=self.mu(self.t, self.S) * self.dt + \
                            self.sigma * self.dW[steps]
            
            # self.dS = self.dF + self.dt * (self.perm_imp * np.sign(np.mean(nu))*np.sqrt(np.abs(np.mean(nu))))
            self.dI=self.I * (torch.exp(-self.tmp_decay*self.dt) - 1) + \
                self.tmp_scale * self.impact_scale(torch.sum(nu))* self.dt            
            self.I = self.I.clone() + self.dI
            
            self.dS=self.dF + self.dI + (self.perm_imp * torch.sum(nu)* self.dt)
                        
            # Compute Action Reward
            self.last_reward = self.r(self.t.clone(), self.Q.clone(), self.S.clone(), self.dS.clone(), nu, to_print)
            
            if to_print:
                print("reward: " + str(self.last_reward))
                print(self.t.clone(), self.Q.clone(), self.S.clone(), self.dS.clone(), nu)
                            
            self.S = self.S.clone() + self.dS
            
            self.S = torch.clip(self.S.clone(), min=0.01)

        cur_state, _, _=self.get_state()
            
         
        return Transition(last_state, nu, cur_state, self.last_reward)

    def impact_scale(self, v):
        if self.impact == 'linear':
            return v
        elif self.impact == 'sqrt':
            return torch.sign(v)*torch.sqrt(torch.abs(v))
        elif self.impact == 'none':
            return torch.tensor(0.0).cuda()
        else:
            raise Exception('Bad impact value')

    def get_state(self):
        """
        Returns the observable features of the current state
        :return: State object summarizing observable features of the current state
        """
        return State(
            (self.T-self.t).clone(), 
            self.S.clone(), 
            self.I.clone(), 
            self.Q.clone(),
            self.Q0.clone(),
        ), self.last_reward.clone(), self.total_reward.clone()
       
    def __str__(self):
        state, last_reward, total_reward=self.get_state()
        str="Simulation -- Last State: {}, \
        Last Reward: {}, Total Reward: {}".format(state, last_reward, total_reward)
        return str

    def __repr__(self):
        return self.__str__()


class ExperienceReplay:
    """
    Class for storing objects in the experience replay buffer
    :param buffer:          List containing all objects in the replay buffer
    :param max_buffer_size: Max size of the buffer
    :param buffer_size:     Current size of the buffer
    """
    def __init__(self, buffer_size):
        self.cur_s_buffer=torch.empty(0).cuda()
        self.next_s_buffer=torch.empty(0).cuda()
        self.term_flag_buffer=torch.empty(0).cuda()
        self.rewards_buffer=torch.empty(0).cuda()
        self.action_buffer=torch.empty(0).cuda()
        self.max_buffer_size=buffer_size
        self.buffer_size=0

    def add(self, cur_s, next_s, term_flag, rewards, action):
        if self.buffer_size >= self.max_buffer_size:
            self.cur_s_buffer=self.cur_s_buffer[1:,:]
            self.next_s_buffer=self.next_s_buffer[1:,:]
            self.term_flag_buffer=self.term_flag_buffer[1:]
            self.rewards_buffer=self.rewards_buffer[1:,:]
            self.action_buffer=self.action_buffer[1:,:]
            
            self.buffer_size -= 1
        
        self.cur_s_buffer=torch.cat([self.cur_s_buffer, torch.unsqueeze(cur_s,0)], dim = 0)
        self.next_s_buffer=torch.cat([self.next_s_buffer, torch.unsqueeze(next_s,0)], dim = 0)
        self.term_flag_buffer=torch.cat([self.term_flag_buffer, torch.unsqueeze(term_flag,0)], dim = 0)
        self.rewards_buffer=torch.cat([self.rewards_buffer, torch.unsqueeze(rewards,0)], dim = 0)
        self.action_buffer=torch.cat([self.action_buffer, torch.unsqueeze(action,0)], dim = 0)
        self.buffer_size += 1

    def sample(self, size):
        sample_idx = torch.tensor(np.random.choice(self.buffer_size, min(size, self.buffer_size), replace=False), dtype = torch.long)
        
        return self.cur_s_buffer[sample_idx], self.next_s_buffer[sample_idx], self.term_flag_buffer[sample_idx], self.rewards_buffer[sample_idx], self.action_buffer[sample_idx]
    
    def query(self, idx):
        sample_idx = idx
        return self.cur_s_buffer[sample_idx], self.next_s_buffer[sample_idx], self.term_flag_buffer[sample_idx], self.rewards_buffer[sample_idx], self.action_buffer[sample_idx]

    def __len__(self):
        return self.buffer_size

    def reset(self):
        self.cur_s_buffer=torch.empty(0).cuda()
        self.next_s_buffer=torch.empty(0).cuda()
        self.term_flag_buffer=torch.empty(0).cuda()
        self.rewards_buffer=torch.empty(0).cuda()
        self.action_buffer=torch.empty(0).cuda()
        self.buffer_size=0
        
