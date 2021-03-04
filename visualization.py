import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from copy import deepcopy as dc


from NashRL import *
# from nashRL_netlib import *
from NashAgent_lib import *
from textwrap import wrap


font = {'size'   : 17}

def to_State_mesh(t_list, q_list, p, net, nump, other_inv, i_val):
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
            state_list.append(State(t=t,p=p,i=i_val,q=np.append(q,other_inv*np.ones(nump-1))))
    
    act_list = net.predict_action(state_list)
    mu_list = torch.stack([nfv.mu for nfv in act_list])
    out = mu_list[:,0].view((len(q_list),len(t_list))).cpu().data.numpy()
    return out

#Creates a series of heatmaps of Inventory x Time, with each subplot
# representing a separate price point
def heatmap_old(net, t_step, q_step, p_step, t_range, q_range, p_range, n_agents, other_agent_inv,i_val):
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
    levels = np.linspace(-20, 20, 41)
    
    fig, axes = plt.subplots(nrows=1, ncols=5,sharex='col', sharey='row')
        
    for p in p_list:
        plt.subplot(1,p_step,counter)
        counter += 1
        
        # Creates mesh over each individual price subplot and plot contours
        q_list = np.linspace(q_range[0], q_range[1], q_step)
        t_list = np.linspace(t_range[0], t_range[1], t_step)
        im = plt.contourf(t_list, q_list, to_State_mesh(t_list,q_list,p,net,n_agents,default_inventory,i_val), 20, cmap='RdBu', vmin = -20, vmax = 20, levels = levels)
        im2 = plt.contour(t_list, q_list, to_State_mesh(t_list,q_list,p,net,n_agents,default_inventory,i_val), levels = [0])
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
    cb = fig.colorbar(im, cax=cbar_ax,ticks=np.linspace(-20, 20, 5))
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
    fig.text(0.5, 0.01, 'Time', ha='center')
    fig.text(0.01, 0.5, 'Inventory', va='center', rotation='vertical')
    plt.savefig('Heatmap_OA_' + str(default_inventory))   
    
def sample_paths(net,num_plots,nump,T,sim_dict):
    """
    Creates a group of sample plots of price and inventory trajectories of all agents
    :param net:                 NashAgent class object containing the action/value nets
    :param num_plots:           Number of sample plots to plot
    :param nump:                Number of total agents
    :param T:                   Total number of time steps
    :param sim_dict:            Dictionary of simulation parameters
    """
    # Fixes number of columns to be 3
    fig, axes = plt.subplots(nrows=int(np.ceil(num_plots/3)), ncols=3)
    plt.tight_layout()
    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    
    
    for i in range(0,num_plots):
        # Simulate trajectory using estimated net
        sim = MarketSimulator(sim_dict)
        inv_list = []
        p_list = []
        plt.subplot(int(np.ceil(num_plots/3)), 3, i+1,aspect='equal', adjustable='box-forced')
        for t in range(0, T):
            current_state,lr,tr = sim.get_state()
            inv_list.append(current_state.q)
            p_list.append(current_state.p)
            a = net.predict_action([current_state])[0].mu.cpu().data.numpy()
            sim.step(a)
            new_state,lr,tr = sim.get_state()
        inv_list = np.array(inv_list)
        p_list = np.array(p_list)
        
        # Label axis
        ax = plt.gca()
        ax2 = ax.twinx()
        ax.plot(inv_list)
        ax2.plot(p_list, 'k--')
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins = 4))
        
    # More labeling
    fig.text(0.5, 0.01, 'Time', ha='center')
    fig.text(0.01, 0.5, 'Inventory', va='center', rotation='vertical')
    fig.text(1, 0.5, 'Price', va='center', rotation='vertical')
    plt.savefig('SamplePaths')   
    
    
def fixed_sample_paths(net,num_plots,nump,T,sim_dict,random_seed):
    fig, axes = plt.subplots(nrows=int(np.ceil(num_plots/3)), ncols=3,dpi=160)
    matplotlib.rc('font', **font)
    for i in range(0,3):
        np.random.seed(random_seed*(i+1))
        q0 = np.random.normal(0,10,nump)
        for j in range(0,3):
            np.random.seed(111*(j+1))
            sim = MarketSimulator(sim_dict)
            sim.setInv(copy.deepcopy(q0))
            current_state,lr,tr = sim.get_state()
            inv_list = []
            p_list = []
            q_list = []
            plt.subplot(int(np.ceil(num_plots/3)), 3, int(i*3+j)+1)
            for t in range(0, T):
                current_state,lr,tr = sim.get_state()
                #if t == 0:
                #    q_list.append(current_state.q)
                inv_list.append(current_state.q)
                p_list.append(current_state.p)
                a = net.predict_action([current_state])[0].mu.cpu().data.numpy()
                sim.step(a)
                new_state,lr,tr = sim.get_state()
            inv_list = np.array(inv_list)
            p_list = np.array(p_list)
            ax = plt.gca()
            ax.grid(False)
            ax2 = ax.twinx()
            ax2.grid(False)
            ax.plot(inv_list)
            ax.set_ylim((-20, 20))
            ax2.set_ylim((7, 13))
            ax2.plot(p_list, 'k--')
            ax.tick_params(axis='both', which='major', labelsize=5)
            ax2.tick_params(axis='both', which='major', labelsize=5)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins = 10))
            
            #remove inner ticks
            if j < 2:
                ax2.yaxis.set_visible(False)
            if j > 0:
                ax.yaxis.set_visible(False)
            if i < 2:
                ax.xaxis.set_visible(False)
                
    fig.text(0.5, 0.03, 'Time', ha='center', fontdict = {'size': 5})
    fig.text(0.05, 0.5, 'Inventory', va='center', rotation='vertical',fontdict = {'size': 5})
    fig.text(.95, 0.5, 'Price', va='center', rotation='vertical',fontdict = {'size': 5})
    plt.savefig('SamplePaths')
        
if __name__ == '__main__':
    num_players = 5
    sim_dict = {'perm_price_impact': .3,
                'transaction_cost':.5,
                'liquidation_cost':.5,
                'running_penalty':0,
                'T':15,
                'dt':1,
                'N_agents':num_players,
                'drift_function':(lambda x,y: 0.1*(10-y)) , #x -> time, y-> price
                'volatility':1,
                'initial_price_var':5}
    nash_agent = NashNN(input_dim=2+num_players, output_dim=4, nump = num_players, t = 15, t_cost = .1, term_cost = .1,num_moms = 5)
    nash_agent.action_net.load_state_dict(torch.load("Action_Net"))
    nash_agent.action_net.eval()
    
    # Creates three optimal action heat maps at different levels of other agent's inventory (-20, 0, 20)
    heatmap_old(nash_agent,15,50,5,[0,14],[-25,25],[6,14],nump = num_players, other_agent_inv = -40)
    heatmap_old(nash_agent,15,50,5,[0,14],[-25,25],[6,14],nump = num_players, other_agent_inv = 0)
    heatmap_old(nash_agent,15,50,5,[0,14],[-25,25],[6,14],nump = num_players, other_agent_inv = 40)

    # Creates plot with subplots of sample trajectories
    sample_paths(nash_agent,9,num_players,15, sim_dict)
