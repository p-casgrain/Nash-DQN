import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# -------------------------------------------------------------------
# This file defines all of the necessary objects and classes needed
# for the LQ-Nash Reinforcement Learning Algorithm
# -------------------------------------------------------------------



# Define Transition Class as Named Tuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Define Replay Buffer Class using Transition Object

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define Market Simulator Object

class MarketSimulator(object):
    def __init__(self,param_dict):
        # Unpack Parameter Dict
        pass

    def reset(self,x0):
        pass

    def step(self,actions):
        pass