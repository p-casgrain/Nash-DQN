import numpy as np
import torch
from datetime import date

from simulation_lib import *
from NashAgent_lib import *

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# -------------------------------------------------------------------
# This file executes the Nash-DQN Reinforcement Learning Algorithm
# -------------------------------------------------------------------

# Define truncation function


def run_Nash_Agent(sim_dict, nash_agent=None, num_sim=15000, batch_update_size=100, buffersize=5000, AN_file_name="Action_Net", VN_file_name="Value_Net"):
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
    sim_obj = MarketSimulator(sim_dict)

    # Estimate/actual transaction costs (used to improve convergence of nash value)
    est_tr_cost = sim_dict['transaction_cost']

    # Estimated Liquidation cost (of selling/buying shares past last timestep)
    term_cost = sim_dict['liquidation_cost']

    # Load Game Parameters
    st0, _, _ = sim_obj.get_state()
    n_agents = sim_obj.N
    max_T = sim_obj.T

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

    # exploration chance
    ep = 0.5  # Initial chance
    min_ep = 0.05  # Minimum chance

    # Intialize relay memory
    replay = ExperienceReplay(buffersize)

    sum_loss = np.zeros(num_sim)
    total_l = 0

    # Set feasibility exploration space of inventory levels:
    q_space = torch.range(start=-100, end=100, step=1)
    q_span = torch.max(q_space) - torch.min(q_space)
    q_m = torch.mean(q_space)
    explore_dist = \
        torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.ones(n_agents)*q_m,
            covariance_matrix=torch.eye(n_agents)*q_span/5
        )

    # ---------- Main simulation Block -----------------
    for k in range(0, num_sim):

        # Decays Exploration rate Linearly and Resets Loss
        eps = max(max(ep - (ep-0.05)*(k/(num_sim-1)), 0), min_ep)
        total_l = 0

        # Sets Print Flag - Prints simulation results every 20 simuluations
        print_flag = not k % 30
        if print_flag:
            print(" \n New Simulation:{} \n Starting State: {} \n ".format(
                k, sim_obj.get_state()))

        for _ in range(0, max_T):
            current_state, lr, _ = sim_obj.get_state()

            rand_action_flag = np.random.random() < eps
            if rand_action_flag:
                # Set target level of inventory level to cover feasible exploration space
                # then select action so it results in that inventory level
                target_q = explore_dist.sample()
                a = target_q - torch.tensor(current_state.q)

            else:
                a = nash_agent.predict_action([current_state])[0].mu

            a = torch.clamp(a, -max_a, max_a)

            # Take Chosen Actions and Take Step
            sim_obj.step(a.clone().detach().cpu().numpy())
            new_state, lr, tr = sim_obj.get_state()
            experience = (current_state, a, new_state, lr)
            replay.add(experience)

            # Sample from replay buffer
            replay_sample = replay.sample(batch_update_size)

            # Computes value loss and updates Value network
            vloss = nash_agent.compute_value_Loss(replay_sample)
            nash_agent.optimizer_value.zero_grad()
            vloss.backward()
            nash_agent.optimizer_value.step()

            # Computes action loss and updates Action network
            loss = nash_agent.compute_action_Loss(replay_sample)
            nash_agent.optimizer_DQN.zero_grad()
            loss.backward()
            nash_agent.optimizer_DQN.step()

            # Calculations Current Step's Total Loss
            cur_loss = nash_agent.compute_action_Loss(
                [experience]).cpu().data.numpy()
            cur_val_loss = nash_agent.compute_value_Loss(
                [experience]).cpu().data.numpy()
            total_l += cur_loss + cur_val_loss

            # Prints Some Information
            if (print_flag):
                cur = nash_agent.predict_action([current_state])[0]
                curNashVal = np.transpose(nash_agent.predict_value(
                    [current_state]).cpu().data.numpy())
                print("Current State: {}".format(current_state))
                print("Action taken: {}, is random:{}".format(a, rand_action_flag))
                print("Ending State: {}".format(new_state))
                print("Nash Action: {}, Nash Value: {}\n".
                      format(cur.mu.cpu().data.numpy(), curNashVal))

        sum_loss[k] = total_l
        sim_obj.reset()

        if print_flag:
            print("Iteration {} Loss: {}".format(k, total_l))

        # Set Save Flag
        save_flag = not (k+1) % 500
        if save_flag:
            print("Saving weights to disk")
            torch.save(nash_agent.action_net.state_dict(),
                       AN_file_name + ".pt")
            torch.save(nash_agent.value_net.state_dict(), VN_file_name + ".pt")
            print("Weights saved to disk")

    print("Saving final weights to disk")
    torch.save(nash_agent.action_net.state_dict(), AN_file_name + ".pt")
    torch.save(nash_agent.value_net.state_dict(), VN_file_name + ".pt")
    print("Weights saved to disk")

    return nash_agent, sum_loss


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Set global digit printing options
    np.set_printoptions(precision=4)

    # Define Training and Model Parameters
    num_players = 5           # Total number of agents
    T = 8                    # Total number of time steps

    # Default simulation parameters
    sim_dict = {'perm_price_impact': .3,
                'transaction_cost': .5,
                'liquidation_cost': .5,
                'running_penalty': 0,
                'T': T,
                'dt': 1,
                'N_agents': num_players,
                'drift_function': (lambda x, y: 0.1*(10-y)),
                'volatility': 1,
                'initial_price_var': 20}

    sim_obj = MarketSimulator(sim_dict)
    net_non_inv_dim = len(sim_obj.get_state()[0].to_numpy())
    net_non_inv_dim -= sim_obj.N-1
    out_dim = 4

    nash_agent = NashNN(non_invar_dim=net_non_inv_dim, n_players=sim_obj.N,
                        output_dim=4, max_steps=T, trans_cost=0.5,
                        terminal_cost=0.5, num_moms=5)

    # current_state = sim_obj.get_state()[0]
    # expanded_states, inv_states = nash_agent.expand_list(
    #     [current_state], as_tensor=True)

    # invar_split = torch.split(inv_states, 1, dim=1)

    # nash_agent.action_net.moment_encoder_net(invar_split[0])

    # nash_agent.action_net.forward(
    #     invar_input=inv_states,
    #     non_invar_input=expanded_states)

    # run_Nash_Agent(sim_dict,num_sim=15000, AN_file_name="Action_Net")

    # Load saved network parameters from file
    # net_file_name = "Action_Net"

    # net_file_name = "./ADA_Trained_Files/Action_Net_ADA.pt"
    # nash_agent.action_net.load_state_dict(torch.load(net_file_name))

    # %load_ext line_profiler
    # %lprun -f NashNN.compute_value_Loss my_func()

    str_dt = date.today().strftime("%d%m%Y")
    nash_agent, loss_data = \
        run_Nash_Agent(sim_dict, nash_agent=nash_agent, num_sim=2000,
                       AN_file_name="./pt_files/Action_Net_ADA"+str_dt, 
                       VN_file_name="./pt_files/Value_Net_ADA"+str_dt )
