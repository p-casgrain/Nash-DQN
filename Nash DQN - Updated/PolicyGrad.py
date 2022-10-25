import torch
import torch.nn as nn
from simulation_lib import ExperienceReplay
import numpy as np
import timeit
from copy import deepcopy as dc


class Policy(torch.nn.Module):

    def __init__(self, input_dim, nodes=32, layers=4):
        super(Policy, self).__init__()
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, nodes),
                #nn.BatchNorm1d(nodes),
                #nn.ReLU())
                nn.SiLU())
        )

        for i in range(layers):
            modules.append(
                nn.Sequential(
                    nn.Linear(nodes, nodes),
                    #nn.BatchNorm1d(nodes),
                    #nn.ReLU())
                    nn.SiLU())
            )

        modules.append(nn.Linear(nodes, 1))

        self.policy_net = nn.Sequential(*modules)

    def forward(self, input):
        return self.policy_net(input)


class Value(torch.nn.Module):

    def __init__(self, input_dim, nodes=32, layers=4):
        super(Value, self).__init__()
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, nodes),
                #nn.BatchNorm1d(nodes),
                nn.SiLU())
        )

        for i in range(layers):
            modules.append(
                nn.Sequential(
                    nn.Linear(nodes, nodes),
                    #nn.BatchNorm1d(nodes),
                    nn.SiLU())
            )

        modules.append(nn.Linear(nodes, 1))

        self.value_net = nn.Sequential(*modules)

    def forward(self, input):
        return self.value_net(input)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def fit_policy(sim, nm, ns, other_pol, tau, it_lim=2000, b_size=1000, b_samp=200, lr=0.001, weight_decay=1e-3, start_pol=None, start_val=None, rand_inv=0,q0 = False, minibatch=10):
    n_steps = int(sim.T/sim.dt)
    n_agents = 2
    buffer = ExperienceReplay(b_size)
    val_loss = nn.MSELoss()
    val_train_loss = []
    pol_train_loss = []
    pol_target = Policy(n_agents + 3).cuda()
    val_target = Value(n_agents + 4).cuda()
    
    if start_pol is not None and start_val is not None:
        pol_target.load_state_dict(start_pol.state_dict())
        val_target.load_state_dict(start_val.state_dict())

    pol = Policy(n_agents + 3).cuda()
    pol.load_state_dict(pol_target.state_dict())
    val = Value(n_agents + 4).cuda()
    val.load_state_dict(val_target.state_dict())

    pol_opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=weight_decay)
    val_opt = torch.optim.AdamW(val.parameters(), lr=lr, weight_decay=weight_decay)

    ep_min = 0.05
    ep_max = 0.5
    
    # Set all target/other agent policies to eval only
    pol.eval()
    val.eval()
    if other_pol is not None:
        other_pol.eval()
    pol_target.eval()
    val_target.eval()

    for epoch in range(it_lim):
        val_ep_l = []
        pol_ep_l = []

        ep = (ep_max - ep_min) * (it_lim - epoch)/it_lim + ep_min

        buffer.reset()
        
        for n in range(minibatch):
            sim.reset()
            
            for j in range(n_steps):
                # Get initial state and actions for all agents
                s, _, _ = sim.get_state()
                a = torch.zeros(sim.N).cuda()
                #print(s.to_sep_tensor_less(1, nm, ns, mean=True))

                for k in range(sim.N):
                    s_t = s.to_sep_tensor_less(k, nm, ns, mean=True, q0=q0).cuda()
                    s_t = torch.unsqueeze(s_t, axis = 0)

                    if k == 0:
                        if torch.rand(1) < ep:
                            if rand_inv == 0:
                                inv = torch.randn(1).cuda() * 10
                                a[k] = inv - s.q[0]
                            elif rand_inv == 1:
                                a[k] = torch.randn(1).cuda() * 5
                            else:
                                a[k] = pol(s_t).detach() * 4.512414940762905 + torch.randn(1).cuda() * 10.0
                        else:
                            a[k] = pol(s_t).detach() * 4.512414940762905
                    else:
                        if other_pol is not None:
                            a[k] = other_pol(s_t).detach() * 4.512414940762905
                        else:
                            a[k] = torch.tensor(0.0).cuda()

                a = torch.clip(a, -50, 50).detach()

                # Advance sim and update buffer
                if n == minibatch - 1 and epoch % 100 == 0:
                    trans = sim.step(a, False)
                else:
                    trans = sim.step(a, False)
                    
                cur_s = trans[0].to_sep_tensor_less(0, nm, ns, mean=True, q0=q0).detach()
                cur_a = trans[1].detach()
                next_s = trans[2].to_sep_tensor_less(0, nm, ns, mean=True, q0=q0).detach()
                r = trans[3].detach()

                if trans[2].t <= 0:
                    isLastState = torch.tensor(0.0).cuda()
                else:
                    isLastState = torch.tensor(1.0).cuda()

                if n == minibatch - 1 and epoch % 500 == 0:
                    print(s)
                    print("Actions:")
                    print(a)
                    print("Rewards:")
                    print(r)
                    print("Ending state:")
                    s1, _, _ = sim.get_state()
                    print(s1)
                    print("")
                    
                buffer.add(cur_s, next_s, isLastState, r, cur_a)

        # Sample from buffer and prep state vals
        s_cur, s_next, s_flag, s_reward, s_action = buffer.sample(buffer.buffer_size)

        rewards = s_reward[:,0].view(-1,1)
        cur_state = s_cur
        next_state =s_next
        actions = s_action[:,0].view(-1,1)

        norm_actions = actions/4.512414940762905
        is_term = s_flag

        # Value update
        val.train()
        val.zero_grad()
       # print(rewards.size())
        #print((val_target(torch.cat([pol_target(next_state), next_state], dim=1)).detach().squeeze() * is_term).size())

        target_q = (rewards.squeeze() + val_target(torch.cat([pol_target(next_state), next_state], dim=1)).detach().squeeze() * is_term).view(-1,1)
        val_l = val_loss(val(torch.cat([norm_actions, cur_state], dim=1)), target_q)
        val_l.backward()
        torch.nn.utils.clip_grad_norm_(val.parameters(), 1e4)
        val_opt.step()
        val.eval()

        # Actor Update
        pol.train()
        pol.zero_grad()
        pol_l = -val(torch.cat([pol(cur_state), cur_state], dim=1))
        pol_l = pol_l.mean()
        pol_l.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1e4)
        pol_opt.step()
        pol.eval()

        # Update targets
        soft_update(pol_target, pol, tau)
        soft_update(val_target, val, tau)

        val_ep_l.append(val_l.item())
        pol_ep_l.append(pol_l.item())

        if np.mean(val_ep_l) > 1e7:
            print(np.mean(val_ep_l))
            return val_train_loss, pol_train_loss, None, None
            
        val_train_loss.append(val_ep_l)
        pol_train_loss.append(pol_ep_l)

        if epoch % 500 == 0:
            print("Finishing epoch: " + str(epoch))
            print("Pol: " + str(np.mean(pol_ep_l)))
            print("Val: " + str(np.mean(val_ep_l)))

    return val_train_loss, pol_train_loss, pol_target, val_target

def sim_policy(sim, nm, ns, nm2 , ns2, pol, other_pol, it_lim=2000, nash_a0=False, nash_a1=False, sim_tens=True, cmb=False):
    n_steps = int(sim.T/sim.dt)
    n_agents = sim.N
    state_arr = []
    action_arr = []
    reward_arr = []

    for epoch in range(it_lim):
        sim.reset()

        state_list = []
        action_list = []
        reward_list = []

        for j in range(n_steps):
            # Get initial state and actions for all agents
            s, _, _ = sim.get_state()
            a = torch.zeros(n_agents).cuda()
            for k in range(n_agents):
                if k == 0:
                    if  nash_a0:
                        if cmb:
                            in_ivt, ivt = s.to_combine_tens(nm, ns)
                            a[k] = pol.predict_action(in_ivt.unsqueeze(0), ivt).squeeze()[4*n_agents].detach()
                        else:
                            in_ivt, ivt = s.to_sep_numpy(k, nm, ns)
                            if ivt is not None:
                                a[k] = pol.predict_action(in_ivt.unsqueeze(0), ivt.unsqueeze(0))[0, 4].detach()
                            else:
                                a[k] = pol.predict_action(in_ivt.unsqueeze(0), ivt)[0, 4].detach()
                    else:
                        a[k] = pol(s.to_sep_tensor_less(k, nm, ns, mean=True)) * 4.512414940762905
                else:
                    if  nash_a1:
                        if cmb:
                            in_ivt, ivt = s.to_combine_tens(nm, ns)
                            a[k] = other_pol.predict_action(in_ivt.unsqueeze(0), ivt).squeeze()[4*n_agents+k].detach()
                        else:
                            in_ivt, ivt = s.to_sep_numpy(k, nm2, ns2)
                            if ivt is not None:
                                a[k] = other_pol.predict_action(in_ivt.unsqueeze(0), ivt.unsqueeze(0))[0, 4].detach()
                            else:
                                a[k] = other_pol.predict_action(in_ivt.unsqueeze(0), ivt)[0, 4].detach()
                    else:
                        a[k] = other_pol(s.to_sep_tensor_less(k, nm2, ns2, mean=True)) * 4.512414940762905
                    #a[k] = 0.0    
            #print(a)
            
            # Advance sim and update buffer
            #print(a)
            if sim_tens:
                trans = sim.step(a.detach())
            else:
                trans = sim.step(a.detach().cpu().numpy())

            # Update loss tracker
            state_list.append(s)
            action_list.append(a[0])
            reward_list.append(trans[3])


        state_arr.append(state_list)
        action_arr.append(action_list)
        reward_arr.append(reward_list)

        if epoch % 100 == 0:
            print("Finishing epoch: " + str(epoch))

    return state_arr, action_arr, reward_arr

def fic_replay_bp_ddqn(sim, nm, ns, its=20, it_lim=2000, lr=0.001, weight_decay=1e-3, rand_inv=0, other_pol_init=None, starting_i=0, wdir='', start_itn=None, q0=True, tau=.5, minibatch=10):
    start = timeit.default_timer()
    
    # Initial other agents policy
    n_agents = sim.N
    
    if other_pol_init is None:
        if n_agents > 1:
            other_pol = Policy(2+3).cuda()
        else:
            other_pol = Policy(1+3).cuda()
    else:
        other_pol = other_pol_init
        
    if start_itn is None:
        start_itn = it_lim

    pol_t_loss_arr = []
    val_t_loss_arr = []

    last_pol = None
    last_val = None

    for i in range(starting_i, its):
        if i == 0 and other_pol_init is None:
            val_train_loss, pol_train_loss, pol, val = fit_policy(sim, nm, ns, None, tau=tau, it_lim=start_itn, lr=lr, weight_decay=weight_decay, q0=q0, minibatch=minibatch)
        elif i == 0 and other_pol_init is not None:
            val_train_loss, pol_train_loss, pol, val  = fit_policy(sim, nm, ns, other_pol,  tau=tau,it_lim=start_itn, lr=lr, weight_decay=weight_decay, q0=q0, minibatch=minibatch)
        else:
            val_train_loss, pol_train_loss, pol, val  = fit_policy(sim, nm, ns, other_pol,  tau=tau, it_lim=it_lim, lr=lr, weight_decay=weight_decay,start_pol=last_pol, start_val=last_val, q0=q0, minibatch=minibatch)

                
        pol_t_loss_arr.append(np.sum(np.array(pol_train_loss),axis=1))
        val_t_loss_arr.append(np.sum(np.array(val_train_loss),axis=1))

        # Copy agent 0's policy and set as standard other agent's policy
        other_pol.load_state_dict(dc(pol.state_dict()))

        print("Finishing Fictitious Replay Iteration: " + str(i))
        plt.plot(pol_t_loss_arr[-1])
        plt.show()
        plt.plot(val_t_loss_arr[-1])
        plt.show()
        
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        
        torch.save(pol.state_dict(), wdir + 'pol_net_it_' + str(i) + ".pt")
        torch.save(val.state_dict(), wdir + 'val_net_it_' + str(i) + ".pt")
        
        last_pol=dc(pol)
        last_val=dc(val)

    return pol, val, pol_t_loss_arr, val_t_loss_arr
