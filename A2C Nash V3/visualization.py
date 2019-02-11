import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from NashRL import *
from nashRL_netlib import *
from nashRL_DQlib_multinet import *

#Creates a Mesh with Inventory on Y-axis, Time on X-axis at a specified price
def to_State_mesh(t_list, q_list, p, net):
    default_inventory = 30
    state_list = []
    for q in q_list:
        for t in t_list:
            state_list.append(State(t,p,np.array([q,default_inventory])))
    
    act_list = net.predict_action(state_list)
    mu_list = torch.stack([nfv.mu for nfv in act_list])
    out = mu_list[:,0].view((len(q_list),len(t_list))).data.numpy()
    return out

#Creates a series of heatmaps of Inventory x Time, with each subplot
# representing a separate price point
def heatmap_old(net, t_step, q_step, p_step, t_range, q_range, p_range):
    counter = 1
    p_list = np.linspace(p_range[0], p_range[1], p_step)
    for p in p_list:
        plt.subplot(1,p_step,counter)
        counter += 1
        q_list = np.linspace(q_range[0], q_range[1], q_step)
        t_list = np.linspace(t_range[0], t_range[1], t_step)
        plt.contourf(t_list, q_list, to_State_mesh(t_list,q_list,p,net), 20, cmap='RdGy')
        plt.colorbar();
        
if __name__ == '__main__':
    nash_agent = NashNN(input_dim=4, output_dim=5, nump = 2, t = 5, t_cost = .1, term_cost = .1)
    nash_agent.action_net.load_state_dict(torch.load("ActionNet_0.1tc_1v_0.1lc_25T"))
    heatmap_old(nash_agent,25,50,5,[0,24],[-25,25],[0,20])