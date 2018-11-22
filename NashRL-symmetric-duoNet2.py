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

    for k in range(0, num_sim):

        # Update Exploration Probability and Total Loss
        eps = max(eps - (eps - 0.05) * (k / (num_sim - 1)), 0)
        total_l = 0

        # Set print flag and print initial sim values
        print_flag = not k % 50
        if print_flag: print("Simulation #{}:\n{}".format(k, sim.get_state()[0]))

        for i in range(0, T):

            current_state, lr, tr = sim.get_state()

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
            experience = sim.step(a)
            new_state, lr, tr = sim.get_state()
            new_state,lr,tr = sim.get_state()

            # updates storage variables
            states.append(new_state)
            prices[i] = new_state.p
            Qs[i,:] = new_state.q
            Actions[i,:] = a
            rewards[i] = lr
            # preds.append[nash_agent.predict(current_state).V.data()]

            # creates experience element
            # experience = (current_state, a, new_state, lr, isNash)
            # computes loss on new experience
            new_loss = nash_agent.compute_Loss(experience).data.numpy()

            # Append Transition to replay memory
            replay.add(experience)
            # replay.add(new_loss,experience)

            # Sample from replay buffer
            replay_sample = replay.sample(batch_update_size)
            # replay_sample, index, weights = replay.sample(batch_update_size)

            loss = []
            # defined new list for replay - can delete and switch to loss later
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
                curVal = nash_agent.predict_print(current_state)
                print("{} , Action: {}, Loss: {}".\
                      format( current_state, curVal.mu.data.numpy(), cur_loss ) )


        # defines loss per period
        sum_loss[k] = total_l

        # resets simulation
        sim.reset()
        all_states.append(states)
        all_predicts.append(preds)
        nash_agent.updateLearningRate()


    plt.plot(sum_loss)
