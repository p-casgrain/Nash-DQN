import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import copy

from NashRL import expand_list
from textwrap import wrap

from simulation_lib import State


font = {'size'   : 12}

def to_State_mesh(t_list, q_list, p, net, nump, other_inv, i_val, norm_mean, norm_std, T=5, is_numpy=False, norm_input=False, uniq_agent=False, all_output=False):
    """
    Creates a Mesh with Inventory on Y-axis, Time on X-axis at a specified price
    :param t_list:      List of time values to be evaluated at
    :param q_list:      List of inventory values to be evaluated at
    :param p:           Price point to be evaluated at
    :param net:         NashAgent class object containing the action/value nets
    :param nump:        Number of total agents
    :param other_inv:   Average Inventory level of all other agents
    :return: 2D mesh of optimal action over the grid t by q
    """
    state_list = []
    for q in q_list:
        for t in t_list:
            if is_numpy:
                if norm_input:
                    state_list.append(State(t=T-t,p=p,i=i_val,q0=0,q=np.append(q,other_inv*np.ones(nump-1))))
                else:
                    state_list.append(State(t=T-t,p=p,i=i_val,q0=0,q=np.append(q,other_inv*np.ones(nump-1))))
            else:
                if norm_input:
                    state_list.append(State(t=torch.tensor(T-t).cuda().float(),
                                              p=torch.tensor(p).cuda().float(),
                                              i=torch.tensor(i_val).cuda().float(),
                                              q0=torch.tensor(0.0).cuda().float(),
                                              q=torch.tensor(np.append(q,other_inv*np.ones(nump-1))).cuda().float()
                                                            ))
                    #print(state_list[-1])
                else:
                    state_list.append(State(t=T-t,p=p,i=i_val,q0=0,q=np.append(q,other_inv*np.ones(nump-1))))
    
    new_state_list = []
    new_invt_state_list = []
    
    for state in state_list:
        s, invt = expand_list(state, norm_mean, norm_std, nump, is_numpy=is_numpy)
        new_state_list.append(s)
        new_invt_state_list.append(invt)
        
        
    new_state_list = torch.cat(new_state_list,dim=0)
    if new_invt_state_list[0] is not None:
        new_invt_state_list = torch.cat(new_invt_state_list,dim=0)
    else:
        new_invt_state_list = None
    
    act_list = net.predict_action(new_state_list, new_invt_state_list)
    
    if uniq_agent:
        mu_list = act_list[:,4*nump:]
    else:
        mu_list = act_list[:,4].view(-1, nump)
        
    out = mu_list[:,0].view((len(q_list),len(t_list))).cpu().data.numpy()
    
    if all_output:
        return act_list
    else:
        return out
    
    
def to_State_mesh_simple(t_list, q_list, p, net, nump, other_inv, i_val, norm_mean, norm_std, T=5):
    """
    Creates a Mesh with Inventory on Y-axis, Time on X-axis at a specified price
    :param t_list:      List of time values to be evaluated at
    :param q_list:      List of inventory values to be evaluated at
    :param p:           Price point to be evaluated at
    :param net:         NashAgent class object containing the action/value nets
    :param nump:        Number of total agents
    :param other_inv:   Average Inventory level of all other agents
    :return: 2D mesh of optimal action over the grid t by q
    """
    state_list = []
    for q in q_list:
        for t in t_list:
            state_list.append(State(t=T-t,p=p+i_val,i=i_val,q0=0, q=torch.tensor(np.append(q,other_inv*np.ones(nump-1))).cuda()))
    
    cur_state = torch.vstack([torch.tensor(s.to_sep_tensor_less(0, norm_mean, norm_std, mean = True)) for s in state_list])
    
    act_list = net(cur_state) * 4.512414940762905
    out = act_list.view((len(q_list),len(t_list))).cpu().data.numpy()
    return out

#Creates a series of heatmaps of Inventory x Time, with each subplot
# representing a separate price point
def draw_heatmap(net, t_step, q_step, p_step, t_range, q_range, p_range, n_agents, other_agent_inv,i_val, norm_mean, norm_std, a_range=[-20,20],T=5, is_numpy=False, norm_input=False, uniq_agent=False):
    """
    Creates a heatmap panel at a fixed average other agent inventory level, across
     different price levels with price and inventory axis within each price level
    :param net:                 NashAgent class object containing the action/value nets
    :param t_step:              Number of blocks over the time axis
    :param q_step:              Number of blocks over the inventory axis
    :param p_step:              Number of subplots for different price points in the panel
    :param t_range:             Range of the time axis
    :param q_range:             Range of the inventory axis
    :param p_range:             Range of the price levels
    :param i_range:             Range of impact state levels
    :param n_agents:                Number of total agents
    :param other_agent_inv:     Average Inventory level of all other agents
    """
    counter = 1
    default_inventory = other_agent_inv
    matplotlib.rc('font', **font)
    
    # Create price levels
    p_list = np.linspace(p_range[0], p_range[1], p_step)
    print(p_list)
    levels = np.linspace(a_range[0], a_range[1], a_range[1] - a_range[0] + 1)
    
    fig, axes = plt.subplots(nrows=1, ncols=5,sharex='col', sharey='row')
        
    for i, p in enumerate(p_list):
        plt.subplot(1,p_step,counter)
        counter += 1
        
        # Creates mesh over each individual price subplot and plot contours
        q_list = np.linspace(q_range[0], q_range[1], q_step)
        t_list = np.linspace(t_range[0], t_range[1], t_step)
        im = plt.contourf(t_list, q_list, to_State_mesh(t_list,q_list,p,net,n_agents,default_inventory,i_val, norm_mean, norm_std, T=T, is_numpy=is_numpy, norm_input=norm_input, uniq_agent=uniq_agent), cmap='RdBu', vmin = a_range[0], vmax = a_range[1], levels = levels)
        im2 = plt.contour(t_list, q_list, to_State_mesh(t_list,q_list,p,net,n_agents,default_inventory,i_val, norm_mean, norm_std, T=T, is_numpy=is_numpy, norm_input=norm_input, uniq_agent=uniq_agent), levels = [0])
        im2.collections[0].set_linewidth(2)
        im2.collections[0].set_color('black')
        im2.collections[0].set_linestyle('dashed')
        
        xtick_loc = [0, 3]
        axes[i].set_xticks(xtick_loc)
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=16)
        if counter > 2:
            ax.yaxis.set_visible(False)
            
    # Create labels and axis
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax,ticks=np.linspace(a_range[0], a_range[1], 5))
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
    fig.text(0.5, 0.01, 'Time', ha='center')
    fig.text(0.01, 0.5, 'Inventory', va='center', rotation='vertical')
    plt.savefig('Heatmap_OA_' + str(default_inventory))   
    
def draw_heatmap_simple(net, t_step, q_step, p_step, t_range, q_range, p_range, n_agents, other_agent_inv,i_val, norm_mean, norm_std, a_range=[-20,20],T=5):
    """
    Creates a heatmap panel at a fixed average other agent inventory level, across
     different price levels with price and inventory axis within each price level
    :param net:                 NashAgent class object containing the action/value nets
    :param t_step:              Number of blocks over the time axis
    :param q_step:              Number of blocks over the inventory axis
    :param p_step:              Number of subplots for different price points in the panel
    :param t_range:             Range of the time axis
    :param q_range:             Range of the inventory axis
    :param p_range:             Range of the price levels
    :param i_range:             Range of impact state levels
    :param n_agents:                Number of total agents
    :param other_agent_inv:     Average Inventory level of all other agents
    """
    counter = 1
    default_inventory = other_agent_inv
    matplotlib.rc('font', **font)
    
    # Create price levels
    p_list = np.linspace(p_range[0], p_range[1], p_step)
    levels = np.linspace(a_range[0], a_range[1], a_range[1] - a_range[0] + 1)
    
    fig, axes = plt.subplots(nrows=1, ncols=5,sharex='col', sharey='row')
        
    for p in p_list:
        plt.subplot(1,p_step,counter)
        counter += 1
        
        # Creates mesh over each individual price subplot and plot contours
        q_list = np.linspace(q_range[0], q_range[1], q_step)
        t_list = np.linspace(t_range[0], t_range[1], t_step)
        im = plt.contourf(t_list, q_list, to_State_mesh_simple(t_list,q_list,p,net,n_agents,default_inventory,i_val, norm_mean, norm_std, T=T), cmap='RdBu', vmin = a_range[0], vmax = a_range[1], levels = levels)
        im2 = plt.contour(t_list, q_list, to_State_mesh_simple(t_list,q_list,p,net,n_agents,default_inventory,i_val, norm_mean, norm_std, T=T), levels = [0])
        im2.collections[0].set_linewidth(2)
        im2.collections[0].set_color('black')
        im2.collections[0].set_linestyle('dashed')
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=20)
        if counter > 2:
            ax.yaxis.set_visible(False)
            
    # Create labels and axis
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax,ticks=np.linspace(a_range[0], a_range[1], 5))
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
    fig.text(0.5, 0.01, 'Time', ha='center')
    fig.text(0.01, 0.5, 'Inventory', va='center', rotation='vertical')
  
    