import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from NashRL import *
from nashRL_netlib import *
from nashRL_DQlib_multinet import *

#Creates a Mesh with Inventory on Y-axis, Time on X-axis at a specified price
def to_State_mesh(t_list, q_list, p, net, nump, other_inv):
    state_list = []
    for q in q_list:
        for t in t_list:
            state_list.append(State(t,p,np.append(q,other_inv*np.ones(nump-1))))
    
    act_list = net.predict_action(state_list)
    mu_list = torch.stack([nfv.mu for nfv in act_list])
    out = mu_list[:,0].view((len(q_list),len(t_list))).data.numpy()
    return out

#Creates a series of heatmaps of Inventory x Time, with each subplot
# representing a separate price point
def heatmap_old(net, t_step, q_step, p_step, t_range, q_range, p_range, nump):
    counter = 1
    default_inventory = 50
    p_list = np.linspace(p_range[0], p_range[1], p_step)
    fig, axes = plt.subplots(nrows=1, ncols=5)
    levels = np.linspace(-30, 30, 60)
    for p in p_list:
        plt.subplot(1,p_step,counter)
        counter += 1
        q_list = np.linspace(q_range[0], q_range[1], q_step)
        t_list = np.linspace(t_range[0], t_range[1], t_step)
        im = plt.contourf(t_list, q_list, to_State_mesh(t_list,q_list,p,net,nump,default_inventory), 20, cmap='gnuplot', vmin = -30, vmax = 30, levels = levels)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Inventory', va='center', rotation='vertical')
    plt.suptitle('Heatmap of Optimal Action for Agent 1 (Other Agents Inventory: ' + str(default_inventory) + ')')
    #plt.colorbar(vmin = -20, vmax = 20);
        
if __name__ == '__main__':
    num_players = 5
    nash_agent = NashNN(input_dim=2+num_players, output_dim=4, nump = num_players, t = 15, t_cost = .1, term_cost = .1)
    nash_agent.action_net.load_state_dict(torch.load("Action_Net_single3"))
    nash_agent.action_net.eval()
    heatmap_old(nash_agent,15,50,5,[0,14],[-25,25],[5,15],nump = num_players)
    #for name, param in nash_agent.action_net.named_parameters():
    #    if param.requires_grad:
    #        print (name, param.data)