import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc

from NashAgent_lib import *

import os

# -------------------------------------------------------------------
# This file executes the Nash-DQN Reinforcement Learning Algorithm
# -------------------------------------------------------------------

# Define truncation function

def expand_list(state, norm_mean, norm_std, num_players, is_numpy):
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
    :param norm_mean & norm_std:          (t, p, q, i)
    :return:              Matrix of the batch of features structured to be pass into NN
    """
    expanded_states = []
    expanded_ivt_states = []
    for i in range(0, num_players):
        s, s_inv = state.to_sep_numpy(i, norm_mean, norm_std)
        expanded_states.append(s)
        expanded_ivt_states.append(s_inv)
    
    if is_numpy:
        if expanded_ivt_states[0] is not None:
            return torch.tensor(expanded_states).float(), torch.tensor(expanded_ivt_states).float()
        else:
            return torch.tensor(expanded_states).float(), None
    else:
        if expanded_ivt_states[0] is not None:
            return torch.stack(expanded_states), torch.stack(expanded_ivt_states)
        else:
            return torch.stack(expanded_states) , None


def run_Nash_Agent(sim_obj, sim_dict, max_steps, nash_agent=None, num_sim=15000, batch_update_size=100, buffersize=5000, AN_file_name="Action_Net", VN_file_name="Value_Net", norm_mean=np.zeros((4,1)), norm_std=np.ones((4,1)), rv_min=.01, rv_max=2.5, is_numpy=False, path='', early_stop=False, early_lim=1000, mini_batch=10):
    """
    Runs the nash RL algothrim and outputs two files that hold the network parameters
    for the estimated action network and value network
    :param num_sim:           Number of Simulations
    :param batch_update_size: Number of experiences sampled at each time step
    :param buffersize:        Maximum size of replay buffer
    :return: Truncated Array
    """
    # number of parameters that need to be estimated by the network
    max_a = 100         # Size of Largest Action that can be taken

    # Package Simulation Parameters
    if sim_obj is None:
        sim_obj = MarketSimulator()

    # Estimate/actual transaction costs (used to improve convergence of nash value)
    est_tr_cost = sim_dict['transaction_cost']

    # Estimated Liquidation cost (of selling/buying shares past last timestep)
    term_cost = sim_dict['liquidation_cost']

    # Load Game Parameters
    st0, _, _ = sim_obj.get_state()
    n_agents = sim_obj.N
    max_T = max_steps
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    good_action_net_w = nash_agent.action_net.state_dict()
    good_value_net_w = nash_agent.value_net.state_dict()
    
    best_action_net = None
    best_value_net = None
    best_loss = None
    best_idx = None
    impv_counter = 0

    if nash_agent is None:
        # Set number of output variables needed from net:
        # (c1 + c2 + c3 + mu)
        parameter_number = 4

        # all state variables but other agent's inventories
        net_non_inv_dim = st0.to_numpy().shape[0] - (n_agents - 1)

        nash_agent = NashNN(non_invar_dim=net_non_inv_dim, n_players=n_agents,
                            output_dim=parameter_number, max_steps=max_T,
                            trans_cost=est_tr_cost, terminal_cost=term_cost,
                            num_moms=5)
        
    use_cuda = nash_agent.use_cuda

    # exploration chance
    ep = 0.9  # Initial chance
    min_ep = 0.1  # Minimum chance

    sum_loss = np.zeros(num_sim)
    total_l = 0

    # Set feasibility exploration space of inventory levels:
    #q_space = torch.range(start=-20, end=20, step=1)
    q_space = torch.range(start=-10, end=10, step=1)
    q_span = torch.max(q_space) - torch.min(q_space)
    q_m = torch.mean(q_space)
    explore_dist = \
        torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.ones(n_agents)*q_m,
            covariance_matrix=torch.eye(n_agents)*(q_span/5)**2
        )

    # ---------- Main simulation Block -----------------
    for k in range(0, num_sim):

        # Decays Exploration rate Linearly and Resets Loss
        eps = max(max(ep - (ep- min_ep )*(k/(num_sim-1)), 0), min_ep)
        total_l = 0

        # Sets Print Flag - Prints simulation results every 20 simuluations
        print_flag = (not k % 50) and k > 0
        if print_flag:
            #update slow value network
            last_loss = sum_loss[k-1]
            print(last_loss)
            
            if k<1000 or (last_loss < 1e4 and last_loss > 100):
                # update slow network
                nash_agent.update_slow()
            elif last_loss < 100:
                # record last good point
                nash_agent.update_slow()
                good_value_net_w = dc(nash_agent.value_net.state_dict())
                good_action_net_w = dc(nash_agent.action_net.state_dict())
            else:
                # reset
                print("RESETTING WEIGHTS")
                if good_action_net_w is not None and good_value_net_w is not None:
                    nash_agent.value_net.load_state_dict(good_value_net_w)
                    nash_agent.action_net.load_state_dict(good_action_net_w)
                    nash_agent.update_slow()
                    print(nash_agent.value_net.state_dict())
                    print(nash_agent.action_net.state_dict())
                else:
                    print("CANNOT RESET, NO SAVE POINT")
                
            
            print(" \n New Simulation:{} \n Starting State: {} \n ".format(
                k, sim_obj.get_state()))
            
        cur_s_buffer=torch.empty(0).cuda()
        cur_ivt_buffer=torch.empty(0).cuda()
        next_s_buffer=torch.empty(0).cuda()
        next_ivt_buffer=torch.empty(0).cuda()
        term_flag_buffer=torch.empty(0).cuda()
        rewards_buffer=torch.empty(0).cuda()
        action_buffer=torch.empty(0).cuda()
        if n_agents > 1:
            cur_ivt_buffer=torch.empty(0).cuda()
            next_ivt_buffer=torch.empty(0).cuda()
        else:
            cur_ivt_buffer = None
            next_ivt_buffer = None
        minibatch=mini_batch
        print_idx = minibatch-1

        for i in range(0, minibatch):
            sim_obj.reset()
            
            for _ in range(0, max_T):
                current_state, lr, _ = sim_obj.get_state()
                #always rdm
                eps = 1
                rand_action_flag = np.random.random() < eps
                if rand_action_flag:

                    # two types of random actions, one inventory based, one perturb based on nash action
                    #if np.random.random() < 0.8 * (1 - k/num_sim):
                    if np.random.random() < 0:
                        # Set target level of inventory level to cover feasible exploration space
                        # then select action so it results in that inventory level
                        if (print_flag) and i == print_idx:
                            print("Taking random inv action")
                        target_q = explore_dist.sample().cuda()
                        a = target_q - torch.tensor(current_state.q).cuda()
                    else:
                        cur_s, cur_ivt = expand_list(current_state, norm_mean, norm_std, n_agents, is_numpy=is_numpy)
                        if use_cuda:
                            nash_a = nash_agent.predict_action(cur_s, cur_ivt)[:, 4].detach().cpu()
                        else:
                            nash_a = nash_agent.predict_action(cur_s, cur_ivt)[:, 4].detach()

                        if (print_flag) and i == print_idx:
                            print("Taking random delta action")

                        a = nash_a + torch.randn(nash_a.size()) * (rv_max - (rv_max-rv_min)*k/num_sim)

                else:
                    if (print_flag) and i == print_idx:
                        print("Taking nash action")

                    cur_s, cur_ivt = expand_list(current_state, norm_mean, norm_std, n_agents, is_numpy=is_numpy)

                    if use_cuda:
                        a = nash_agent.predict_action(cur_s, cur_ivt)[:, 4].detach()
                    else:
                        a = nash_agent.predict_action(cur_s, cur_ivt)[:, 4].detach()

                #a = torch.clamp(a, -max_a, max_a)
                
                a = a.cuda()

                # Take Chosen Actions and Take Step
                
                if is_numpy :
                    current_state, a, new_state, lr = sim_obj.step(a.detach().cpu().numpy())
                else:
                    current_state, a, new_state, lr = sim_obj.step(a.detach())
                cur_s, cur_ivt = expand_list(current_state, norm_mean, norm_std, n_agents, is_numpy=is_numpy)
                next_s, next_ivt = expand_list(new_state, norm_mean, norm_std, n_agents, is_numpy=is_numpy)

                if new_state.t <= 0:
                    isLastState = torch.ones(n_agents,dtype=torch.float32).cuda()
                else:
                    isLastState = torch.zeros(n_agents,dtype=torch.float32).cuda()

                if is_numpy:
                    rewards = torch.tensor(lr).float().cuda()
                    action = torch.tensor(a).float().cuda()
                    cur_s = cur_s.detach().cuda()
                    next_s = next_s.detach().cuda()
                    if cur_ivt is not None:
                        cur_ivt = cur_ivt.detach().cuda()
                        next_ivt = next_ivt.detach().cuda()
                else:
                    rewards = lr.detach()
                    action = a.detach()
                    cur_s = cur_s.detach()
                    next_s = next_s.detach()
                    if cur_ivt is not None:
                        cur_ivt = cur_ivt.detach()
                        next_ivt = next_ivt.detach()

                cur_s_buffer=torch.cat([cur_s_buffer, cur_s], dim = 0)
                next_s_buffer=torch.cat([next_s_buffer, next_s], dim = 0)
                term_flag_buffer=torch.cat([term_flag_buffer, torch.unsqueeze(isLastState,0)], dim = 0)
                rewards_buffer=torch.cat([rewards_buffer, torch.unsqueeze(rewards,0)], dim = 0)
                action_buffer=torch.cat([action_buffer, torch.unsqueeze(action,0)], dim = 0)
                
                if cur_ivt is not None:
                    cur_ivt_buffer=torch.cat([cur_ivt_buffer, cur_ivt], dim = 0)
                    next_ivt_buffer=torch.cat([next_ivt_buffer, next_ivt], dim = 0)
                else:
                    cur_ivt_buffer=None
                    next_ivt_buffer=None


                # Prints Some Information
                if (print_flag) and i == print_idx:
                    #print(cur_s, cur_ivt)
                    cur = nash_agent.predict_action(cur_s, cur_ivt)[:, 4]
                    c4 = nash_agent.predict_action(cur_s, cur_ivt)[:, 3]
                    curNashVal = np.transpose(nash_agent.predict_value(
                        cur_s, cur_ivt).cpu().data.numpy())
                    print("Current State: {}".format(current_state))
                    print("Action taken: {}, is random:{}".format(a, rand_action_flag))
                    print("Rewards: {}".format(rewards))
                    print("Ending State: {}".format(new_state))
                    print("Nash Action: {}, Nash Value: {}\n".
                          format(cur.cpu().data.numpy(), curNashVal))
                    print("c4 values: {}\n".
                          format(c4))
               
            
        nash_agent.value_net.train()
        #nash_agent.action_net.train()
        nash_agent.action_net.eval()
        
        # Computes value loss and updates Value network
        replay_sample = (cur_s_buffer, cur_ivt_buffer, next_s_buffer, next_ivt_buffer, term_flag_buffer, rewards_buffer, action_buffer)
            
        for _ in range(1):
            nash_agent.optimizer_value.zero_grad()
            vloss = nash_agent.compute_value_Loss(replay_sample)
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(nash_agent.value_net.parameters(), 1e-1)
            nash_agent.optimizer_value.step()

        nash_agent.action_net.train()
        nash_agent.value_net.eval()

        # Computes action loss and updates Action network
        
        # Computes action loss multiple times
        for _ in range(1):
            nash_agent.optimizer_DQN.zero_grad()
            loss = nash_agent.compute_action_Loss(replay_sample)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nash_agent.action_net.parameters(), 1e-1)
            nash_agent.optimizer_DQN.step()
        
        nash_agent.value_net.eval()
        nash_agent.action_net.eval()

        # Calculat Current Step's final Total Loss
        total_l += vloss + loss

        sum_loss[k] = total_l

        if print_flag:
            print("Iteration {} Loss: {}".format(k, total_l))
            print("Iteration {} V_Loss: {} A_Loss {}".format(k, vloss, loss))
           

        # Set Save Flag
        save_flag = not (k+1) % 500
        if save_flag:
            print("Saving weights to disk")
            torch.save(nash_agent.action_net.state_dict(),
                       AN_file_name + "_" + str(k) + ".pt")
            torch.save(nash_agent.value_net.state_dict(), VN_file_name + "_" + str(k) + ".pt")
            print("Weights saved to disk")
            
        if early_stop:
            if best_loss is None or total_l.item() < best_loss:
                print("New best loss: " + str(total_l.item()))
                best_loss = dc(total_l.item())
                best_action_net = dc(nash_agent.action_net.state_dict())
                best_value_net = dc(nash_agent.value_net.state_dict())
                best_idx = k
                impv_counter = 0
                
                if k > 1000:
                    print("Saving temp weights to disk")
                    torch.save(best_action_net, AN_file_name +"_" + str(best_idx) + "_best.pt")
                    torch.save(best_value_net, VN_file_name + "_" + str(best_idx) + "_best.pt")
                    print("Weights saved to disk")
                
            else:
                impv_counter += 1
                
                if impv_counter > early_lim:
                    print("EARLY STOPPING ON ITERATION " + str(k))
                    print("Saving final weights to disk")
                    torch.save(best_action_net, AN_file_name +"_" + str(best_idx) + "_best.pt")
                    torch.save(best_value_net, VN_file_name + "_" + str(best_idx) + "_best.pt")
                    print("Weights saved to disk")
                    
                    return nash_agent, sum_loss
                
                

    print("Saving final weights to disk")
    torch.save(nash_agent.action_net.state_dict(), AN_file_name + ".pt")
    torch.save(nash_agent.value_net.state_dict(), VN_file_name + ".pt")
    print("Weights saved to disk")

    return nash_agent, sum_loss
