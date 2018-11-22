import numpy as np
from collections import namedtuple
import random
from itertools import count
from per.prioritized_memory import *

import copy

from per.SumTree import SumTree
import torch



# Define Transition Class as Named Tuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

State = namedtuple('State', ('t', 'p', 'q'))

class State(State):
    """
    Game State Class.
    __init__ takes arguments (t,p,q)
    Inherits from namedtuple class
    """

    def getNormalizedState(self, toTensor=True):
        """
        Returned Normalized State Values
        :return: Array of concatenated values
        """

        norm_q = self.q / 10
        norm_p = (self.p - 10) / 10
        norm_t = self.t - 1
        out = copy.deepcopy(np.concatenate( (np.array([norm_t,norm_p]), norm_q) ))

        if toTensor:
            return out
        else:
            return torch.from_numpy(out)


class MarketSimulator(object):
    def __init__(self, param_dict):
        # Unpack Parameter Dictionary

        # Game-Specific Parameters
        self.p_imp = param_dict['price_impact']
        self.t_cost = param_dict['transaction_cost']
        self.L_cost = param_dict['liquidation_cost']
        self.phi = param_dict['running_penalty']

        self.T = param_dict['T']
        self.dt = param_dict['dt']

        self.N = param_dict['N_agents']

        # Simulation Parameters
        self.mu = param_dict['drift_function']
        self.sigma = param_dict['volatility']

        # Define Reward Functions for t<T and t=T (Revise, perhaps)
        self.r = lambda Q, S, nu: - nu * (S + self.t_cost * nu) - self.phi * Q ** 2
        self.rT = lambda Q, S, nu: Q * (S - Q * self.L_cost)

        # Allocating Memory for Game Variables
        self.Q = np.random.normal(0, 10, self.N)
        self.S = np.float32(10 + np.random.normal(0, self.sigma))
        self.dS = np.float32(0)
        self.dF = np.float32(0)
        # self.F = np.float32(0)
        self.t = np.float32(0)

        # Variable Containing Total Accumulated Score
        self.last_reward = np.zeros(self.N, dtype=np.float32)
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        # Variable Containing BM increments
        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                   int(round(np.ceil(self.T / self.dt) + 2)))

        # Variable Indicating Whether Done
        self.done = False

    def reset(self):
        # Reset Game Values
        self.Q = np.random.normal(0, 10, self.N)
        self.S = np.float32(10 + np.random.normal(0, self.sigma))
        self.t = np.float32(0)

        self.last_reward = np.zeros(self.N, dtype=np.float32)
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                   int(round(np.ceil(self.T / self.dt) + 2)))

        self.done = False

    def step(self, nu):

        last_state = State(self.t,self.S,self.Q)

        if self.t < self.T:
            # Advance Inventory & Time
            self.Q += nu
            self.t += self.dt

            # Compute Action Reward
            self.last_reward = self.r(self.Q, self.S, nu)
            self.total_reward += self.last_reward

            # Advance Asset Price
            self.dF = self.mu(self.t, self.S) * self.dt + self.sigma * self.dW[int(round(self.t))]
            self.dS = self.dF + self.dt * (self.p_imp * np.mean(nu))
            self.S += self.dS

        #        elif (not self.done):
        #
        #            # Compute Action Reward
        #            self.last_reward = self.rT(self.Q, self.S, self.nu)
        #            self.total_reward += self.last_reward
        #
        #            # Update Variables
        #            self.Q = np.zeros(self.N, dtype=np.float32)

        return Transition(last_state, nu, State(self.t, self.S, self.Q), self.last_reward)

    def get_state(self):
        return State(copy.deepcopy(self.t), copy.deepcopy(self.S), copy.deepcopy(self.Q)), copy.deepcopy(self.last_reward), copy.deepcopy(self.total_reward)


    def __str__(self):
        state, last_reward, total_reward = self.get_state()
        str = "Simulation -- Last State: {}, \
        Last Reward: {}, Total Reward: {}".format(state,last_reward,total_reward)
        return str


class ExperienceReplay:
    # each experience is a list of with each tuple having:
    # first element: state,
    # second element: array of actions of each agent,
    # third element: array of rewards received for each agent
    def __init__(self, buffer_size):
        self.buffer = []
        self.max_buffer_size = buffer_size
        self.buffer_size = 0

    def add(self, experience):
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

        self.buffer_size = len(self.buffer)

    def sample(self, size):
        return random.sample(self.buffer, min(size, self.buffer_size))

    def __len__(self):
        return len(self.buffer)


class PrioritizedMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        #        p = self._get_priority(error)
        #        self.tree.update(idx, p)

        for i in range(0, len(idx)):
            p = self._get_priority(error[i])
            self.tree.update(idx[i], p)