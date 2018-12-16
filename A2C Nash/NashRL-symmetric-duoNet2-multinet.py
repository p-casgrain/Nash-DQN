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
    T = 5                    # Total number of time steps
    
    #number of parameters that need to be estimated by the network
    batch_update_size = 100  # Number of experiences sampled at each time step
    num_sim = 15000          # Number of Simulations
    max_action = 100         # Size of Largest Action that can be taken
    buffersize = 5000        # Maximum size of replay buffer

    # Random Action Probability
    eps = 0.5
    
    # Set number of output variables needed from net:
    # Supposed to be = 1 x ( V + c1 + c2 + c3 + mu). Note: Need to make more robust
    parameter_number = 5
    
    # Package Simulation Parameters
    sim_dict = {'price_impact': 0.1,
                'transaction_cost':1,
                'liquidation_cost':20,
                'running_penalty':0,
                'T':T,
                'dt':1,
                'N_agents':num_players,
                'drift_function':(lambda x,y: 0) ,
                'volatility':.1}
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
    
    #-----------
    # Following Block relates to training network on border cases
    # Improves convergence rates if number of simulations are low
    # Currently unused
    #-----------
#    for l in range (0,30):
#        for z in range (0,4):
#            if z == 0:
#                a1,a2 = 1,0
#                pos = 1
#            elif z == 1:
#                a1,a2 = 1,0
#                pos = -1
#            elif z == 2:
#                a1,a2 = 0,1
#                pos = 1
#            else:
#                a1,a2 = 0,1
#                pos = -1
#                
#            for k in range (0,20):
#                total_l = 0        
#                print_flag = not (k+1) % 20
#                testinv = pos*np.array([a1*2*k,a2*2*k]) + np.array([(1-a1)*np.random.normal(0,2),(1-a2)*np.random.normal(0,2)])
#                sim.Q = testinv
#                if print_flag : print("New Simulation:", k,  "\n", sim.get_state()[0])
#                for i in range(0,T):
#        
#                    current_state,lr,tr = sim.get_state()
#                    
#                    #### Begin Action Generation Block ####                        
##                    if i == 0:
##                        a = testinv-current_state.q
#                    if i == T - 1:
#                        if np.random.random() < 0.5:
#                            a = -testinv*min(np.random.uniform()+0.5,1)
#                        else:
#                            a = nash_agent.predict_action(current_state).mu.data.numpy()
#                    else:
#                        a = np.array([0,0])
#        
#                    #### End Action Generation Block ####
#                    
#                    # Take Chosen Actions and Take Step
#                    sim.step(a)
#                    new_state,lr,tr = sim.get_state()
#                    
#                    #creates experience element
#                    experience = (current_state,a,new_state,lr)
#                    
#                    # Append Transition to replay memory
#                    replay.add(experience)
#        
#                    # Sample from replay buffer
#                    replay_sample = replay.sample(batch_update_size)
#        
#                    #computes value loss and updates value network
#                    vloss = 0
#                    for sample in replay_sample:
#                        vloss += nash_agent.compute_value_Loss(sample)
#                    nash_agent.optimizer_value.zero_grad()
#                    vloss.backward()
#                    nash_agent.optimizer_value.step()
#                    
#                    #computes action loss and updates action network
#                    loss = []
#                    for sample in replay_sample: 
#                        cur_loss = nash_agent.compute_action_Loss(sample)
#                        loss.append(cur_loss)                
#        
#                    loss = torch.sum(torch.stack(loss))
#                    
#                    nash_agent.optimizer_DQN.zero_grad()
#                    loss.backward()
#                    nash_agent.optimizer_DQN.step()
#                    
#                    # Update priority replay
#                    # replay.update(index,pro_loss)
#        
#                    
#                    cur_loss = nash_agent.compute_action_Loss(experience).data.numpy()
#                    total_l += cur_loss
#                    
#                    if (print_flag):
#                        print(nash_agent.predict_value(current_state))
#                        curVal = nash_agent.predict_action_print(current_state)
#                        terminal = new_state.q*new_state.p- 20 * new_state.q**2
#                        print(lr,terminal, a)
#                        print("{} , Action: {}, Loss: {}".\
#                              format( current_state, curVal.mu.data.numpy(), cur_loss ) )
#        
#                #resets simulation
#                sim.reset()        
#    replay.reset()
    
    #---------- Main simulation Block -----------------
    for k in range (0,num_sim):

        # Decays Exploration rate Linearly and Resets Loss
        eps = max( eps - (eps-0.05)*(k/(num_sim-1)), 0 )
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
                a = nash_agent.predict_action(current_state).mu.data.numpy()
                a = a + np.random.normal(0, 2.5, num_players)
            else:
                # Else take predicted nash action
                a = nash_agent.predict_action(current_state).mu.data.numpy()
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
            vloss = 0
            for sample in replay_sample:
                vloss += nash_agent.compute_value_Loss(sample)
            nash_agent.optimizer_value.zero_grad()
            vloss.backward()
            nash_agent.optimizer_value.step()
            
            # Computes action loss and updates Action network
            loss = []
            for sample in replay_sample: 
                cur_loss = nash_agent.compute_action_Loss(sample)
                loss.append(cur_loss)                
            loss = torch.sum(torch.stack(loss))
            nash_agent.optimizer_DQN.zero_grad()
            loss.backward()
            nash_agent.optimizer_DQN.step()
            
            # Update priority
            # replay.update(index,pro_loss)
            
            # Calculations Current Step's Total Loss
            cur_loss = nash_agent.compute_action_Loss(experience).data.numpy()
            cur_val_loss = nash_agent.compute_value_Loss(experience).data.numpy()
            total_l += cur_loss + cur_val_loss

            # Prints Some Information
            if (print_flag):
                cur = nash_agent.predict_action(current_state)
                curNashVal = nash_agent.predict_value(current_state).data.numpy()
                #print("Current Nash Value: ", nash_agent.predict_value(current_state).data.numpy())
                print("{} , Action: {}, Loss: {}".\
                      format( current_state, cur.mu.data.numpy(), cur_loss ) )
                #print("Difference in Nash Value: ", curNashVal - lastval)
                print("")
                #lastval = curNashVal

                
        #defines loss per period
        sum_loss[k] = total_l

        #resets simulation
        sim.reset()
        #nash_agent.updateLearningRate()
        
    plt.plot(sum_loss)
        
    