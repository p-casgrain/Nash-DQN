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
from nashRL_DQlib_multinet import *
# -------------------------------------------------------------------
# This file defines some of the necessary objects and classes needed
# for the LQ-Nash Reinforcement Learning Algorithm
# -------------------------------------------------------------------

# Set global digit printing options
np.set_printoptions(precision=4)

# Define truncation function
def trunc_array(a,max_a):
    """
    Truncate a into [-max_a,max_a]
    :param a: array / tensor to truncate
    :param max_a: truncation threshold (max_a > 0)6
    :return: Truncated Array
    """
    lt, gt = a < -max_a, a > max_a
    return a * (1 - lt - gt) - lt * max_a + gt*max_a

# Define Market Game Simulation Object
if __name__ == '__main__':

    # Define Training and Model Parameters
    num_players = 2          # Total number of agents
    T = 25                    # Total number of time steps
    
    #number of parameters that need to be estimated by the network
    batch_update_size = 100  # Number of experiences sampled at each time step
    num_sim = 20000          # Number of Simulations
    max_action = 1000         # Size of Largest Action that can be taken
    buffersize = 5000        # Maximum size of replay buffer

    # Random Action Probability
    eps = 0.5
    
    # Set number of output variables needed from net:
    # Supposed to be = 1 x ( V + c1 + c2 + c3 + mu). Note: Need to make more robust
    parameter_number = 5
    
    # Package Simulation Parameters
    sim_dict = {'price_impact': 0.1,
                'transaction_cost':0.1,
                'liquidation_cost':0.1,
                'running_penalty':0,
                'T':T,
                'dt':1,
                'N_agents':num_players,
                'drift_function':(lambda x,y: 0) ,
                'volatility':1}
    sim = MarketSimulator(sim_dict)
    
    #Estimate/actual transaction costs (used to improve convergence of nash value)
    est_tr_cost = sim_dict['transaction_cost']
    
    #Estimated Liquidation cost (of selling/buying shares past last timestep)
    term_cost = sim_dict['liquidation_cost']
    
    # Initialize NashNN Agents
    nash_agent = NashNN(2+num_players,parameter_number,num_players,T,est_tr_cost,term_cost)
        
    current_state = sim.get_state()[0]
    
    #exploration chance
    ep = 0.5

    # Intialize relay memory
    replay = ExperienceReplay(buffersize)

    sum_loss = np.zeros(num_sim)
    total_l = 0
    
    #Set feasibility space:
    space = np.array([-100,100])
    
    #---------- Main simulation Block -----------------
    for k in range (0,num_sim):

        # Decays Exploration rate Linearly and Resets Loss
        eps = max( ep - (ep-0.05)*(k/(num_sim-1)), 0 )
        total_l = 0        

        # Sets Print Flag - Prints simulation results every 20 simuluations
        print_flag = not k % 20
        if print_flag : print("New Simulation:", k,  "\n", sim.get_state()[0])
        #lastval = np.zeros(2)
        
        for i in range(0,T):
            # Get Current state of the simulation
            current_state,lr,tr = sim.get_state()
            
            #### Begin Action Generation Block ####
            if np.random.random() < eps:
                # Takes random action by perturbing current estimated nash action
                #b = nash_agent.predict_action([current_state])[0].mu.data.numpy()
                #a = a + np.random.multivariate_normal(np.zeros(num_players), np.diag(np.maximum(4*np.ones(num_players),np.abs(current_state.q/2))))
                
                #Set feasibility space:
                target_q = np.random.multivariate_normal(np.ones(num_players)*(space[1]+space[0])/2,np.diag(np.ones(num_players)*(space[1]-space[0])/4))
                a = target_q - current_state.q
                if (print_flag):
                    #print("Nash A: ",b)
                    print("Random A: ",a)
                    #print(np.ones(num_players)*(space[1]+space[0])/2)
                    #print(np.diag(np.ones(num_players)*(space[1]-space[0])/4))
                    #print(np.maximum(T*np.ones(num_players),np.abs(current_state.q/2)))
            else:
                # Else take predicted nash action
                a = nash_agent.predict_action([current_state])[0].mu.data.numpy()
            # Truncate Action if Exceeds Max allowable action
            a = trunc_array(a, max_action)
            #### End Action Generation Block ####
            
            # Take Chosen Actions and Take Step
            sim.step(a)
            new_state,lr,tr = sim.get_state()

            #creates experience element
            experience = (current_state,a,new_state,lr)
            #computes loss on new experience
            #new_loss = nash_agent.compute_Loss(experience).data.numpy()

            # Append Transition to replay memory
            replay.add(experience)
            # replay.add(new_loss,experience)

            # Sample from replay buffer
            replay_sample = replay.sample(batch_update_size)
            # replay_sample, index, weights = replay.sample(batch_update_size)

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
            
            # Update priority
            # replay.update(index,pro_loss)
            
            # Calculations Current Step's Total Loss
            cur_loss = nash_agent.compute_action_Loss([experience]).data.numpy()
            cur_val_loss = nash_agent.compute_value_Loss([experience]).data.numpy()
            total_l += cur_loss + cur_val_loss

            # Prints Some Information
            if (print_flag):
                cur_loss = nash_agent.compute_action_Loss_print([experience]).data.numpy()
                cur_val_loss = nash_agent.compute_value_Loss_print([experience]).data.numpy()
                cur = nash_agent.predict_action([current_state])[0]
                curNashVal = nash_agent.predict_value([current_state]).data.numpy()
                #print("Current Nash Value: ", nash_agent.predict_value(current_state).data.numpy())
                print("{} , Action: {}, Action Loss: {}, Value Loss: {}, Nash Value: {}".\
                      format( current_state, cur.mu.data.numpy(), cur_loss, cur_val_loss, curNashVal ) )
                #print("Difference in Nash Value: ", curNashVal - lastval)
                print("")
                #lastval = curNashVal

                
        #defines loss per period
        sum_loss[k] = total_l

        #resets simulation
        sim.reset()
        #nash_agent.updateLearningRate()
        
    plt.clf()
    plt.plot(np.log(sum_loss))
    plt.savefig(str(num_sim) + ' Losses')
    
    sample_path_num = 10
    inv_paths1 = np.zeros((T,sample_path_num))
    inv_paths2 = np.zeros((T,sample_path_num))
    for k in range(0,sample_path_num):
        sim.reset()
        for i in range(0,T):
            current_state,lr,tr = sim.get_state()
            a = nash_agent.predict_action([current_state])[0].mu.data.numpy()
            a = trunc_array(a, max_action)
            
            sim.step(a)
            new_state,lr,tr = sim.get_state()
            inv_paths1[i,k] = new_state.q[0]
            inv_paths2[i,k] = new_state.q[1]
            
    plt.clf()
    plt.plot(inv_paths1,color = 'b')
    plt.plot(inv_paths2,color = 'r')
    plt.savefig(str(num_sim) + ' Paths')